# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""agent6 command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import argcomplete

from agent6 import __version__
from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import (
    AnthropicProviderEntry,
    Config,
    ConfigError,
    NotifyConfig,
    RoleName,
    load_config,
)
from agent6.config_fix import FixKind, apply_fixes, propose_fixes
from agent6.detect import Environment, detect, select_profile
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    commit_paths,
    create_branch,
    is_git_repo,
    make_run_branch_name,
    revert_head,
    slugify,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.graph.client import GraphClient, spawn_curator
from agent6.graph.curator import GraphCurator
from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, load_graph
from agent6.init import init_workspace
from agent6.machine import (
    TOML_PAYLOAD_KEY,
    AgentExecResult,
    AgentFact,
    AgentRequest,
    EngineError,
    JournalError,
    LiveWorld,
    MachineError,
    MachineJournal,
    MachineSpec,
    StepEvent,
    build_authoring_prompt,
    drive,
    extract_toml,
    load_machine,
    machine_lock,
    render,
    write_source,
)
from agent6.mcp_server import run_server as _mcp_run_server
from agent6.memory import (
    MemoryError as Agent6MemoryError,
)
from agent6.memory import (
    MemoryScope,
)
from agent6.memory import (
    add as memory_add,
)
from agent6.memory import (
    invalidate as memory_invalidate,
)
from agent6.memory import (
    list_entries as memory_list,
)
from agent6.providers import (
    AnthropicProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
)
from agent6.providers.anthropic import ANTHROPIC_URL
from agent6.providers.egress import clear_routes, parse_endpoint, register_route
from agent6.run_id import RunIdError, new_friendly_id, resolve_run_id
from agent6.sandbox import (
    BrokerHandle,
    EgressBrokerError,
    Endpoint,
    JailUnavailableError,
    LandlockNotSupportedError,
    apply_agent_landlock,
    enter_network_isolation,
    landlock_abi,
    run_in_jail,
    start_egress_broker,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.mcp_client import MCPManager
from agent6.types import JailPolicy, SandboxProfile, SandboxReport
from agent6.workflows.loop import ResumeError, RunResult, Workflow
from agent6.workflows.review import CodeReviewError, run_review


def _build_role_provider(
    cfg: Config,
    role: RoleName,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    model_override: str = "",
) -> Provider:
    """Construct the configured provider for `role`.

    `model_override` (if truthy) replaces the model string from
    `[models.<role>].model`; provider routing is unchanged. Caller is
    responsible for env-var presence checks via
    `_check_provider_env_vars(cfg)` BEFORE this is called.
    """
    rm = cfg.models.all()[role]
    model = model_override or rm.model
    entry = cfg.providers.get(rm.provider)
    if entry is None:  # pragma: no cover - blocked by config validation
        raise ProviderError(
            f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}] missing"
        )
    if isinstance(entry, AnthropicProviderEntry):
        return AnthropicProvider.from_env(
            env_var=entry.api_key_env,
            model=model,
            prompt_caching=entry.prompt_caching,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
        )
    return OpenAIProvider.from_env(
        env_var=entry.api_key_env,
        model=model,
        base_url=entry.base_url,
        extra_headers=entry.extra_headers,
        timeout_s=entry.http_timeout_s,
        transcript_sink=transcript_sink,
        budget=budget,
    )


def _provider_endpoints(cfg: Config) -> set[Endpoint]:
    """The set of ``host:port`` endpoints every configured provider dials.

    Used to build the provider-only egress allow-list: one broker socket
    per endpoint. Anthropic's endpoint is fixed; OpenAI-compatible
    providers carry it in ``base_url``.
    """
    eps: set[Endpoint] = set()
    for entry in cfg.providers.values():
        url = ANTHROPIC_URL if isinstance(entry, AnthropicProviderEntry) else entry.base_url
        host, port = parse_endpoint(url)
        eps.add(Endpoint(host=host, port=port))
    return eps


def _maybe_start_egress(
    cfg: Config, selected_profile: SandboxProfile
) -> tuple[BrokerHandle | None, Path | None, str | None]:
    """Establish provider-only egress confinement, if configured.

    Returns ``(broker, sock_dir, error)``. When ``error`` is non-None the
    caller must refuse the run (the message is ready to print). When
    ``sandbox.network != "provider_only"`` returns ``(None, None, None)``
    and nothing is confined.

    Must be called before any network-using object is built and while the
    process is single-threaded (``unshare(CLONE_NEWUSER)`` requires it).
    On success this process is left inside an empty network namespace and
    the egress routes are registered so provider calls reach the broker.
    """
    if cfg.sandbox.network != "provider_only":
        return None, None, None
    if selected_profile != "strict":
        return (
            None,
            None,
            (
                "sandbox.network = 'provider_only' requires the strict profile "
                "(unprivileged user namespaces) to confine egress, but this host "
                "only supports the hardened profile. Set sandbox.network = 'allow' "
                "or 'no', or enable user namespaces."
            ),
        )
    endpoints = _provider_endpoints(cfg)
    sock_dir = Path(tempfile.mkdtemp(prefix="agent6-egress-"))
    try:
        broker = start_egress_broker(endpoints, sock_dir=sock_dir)
        enter_network_isolation()
    except EgressBrokerError as exc:
        shutil.rmtree(sock_dir, ignore_errors=True)
        return None, None, f"could not establish provider-only egress: {exc}"
    for ep in endpoints:
        uds = broker.uds_for(ep.host, ep.port)
        if uds is not None:
            register_route(ep.host, ep.port, uds)
    return broker, sock_dir, None


def _stop_egress(broker: BrokerHandle | None, sock_dir: Path | None) -> None:
    """Tear down the egress broker and clear its routes. Idempotent."""
    if broker is not None:
        broker.close()
    clear_routes()
    if sock_dir is not None:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _maybe_apply_agent_landlock(
    cfg: Config, selected_profile: SandboxProfile, env: Environment
) -> str | None:
    """Confine the agent's OWN process with Landlock on hardened hosts.

    Returns ``None`` when nothing is to be done or confinement succeeds, or a
    ready-to-print error message when the run must be refused.

    Only the ``hardened`` profile takes this path. The ``strict`` profile
    instead runs every child command in its own user+mount+pid+net namespace
    (a stronger boundary) and confines provider egress with the broker;
    Landlocking the agent there would break the jail's ``pivot_root(2)`` /
    ``mount(2)`` on kernels at ABI >= 7. Irrevocable, and applied before any
    provider or network object is built so it covers the whole run and every
    child it spawns.
    """
    if selected_profile != "hardened" or not env.kernel.supports_landlock_fs:
        return None
    cwd = Path.cwd().resolve()
    # Landlock allow-root, not a temp file we create: children (git, the jail
    # launcher, the curator socket dir) legitimately read and write under /tmp.
    tmp = Path("/tmp")  # noqa: S108
    dev_files = tuple(
        p
        for p in (
            Path("/dev/null"),
            Path("/dev/zero"),
            Path("/dev/urandom"),
            Path("/dev/random"),
            Path("/dev/tty"),
        )
        if p.exists()
    )
    run_paths = (Path("/run"),) if Path("/run").exists() else ()
    proc_paths = (Path("/proc"),) if Path("/proc").exists() else ()
    read_paths = (
        cwd,
        Path.home(),
        Path("/usr"),
        Path("/etc"),
        tmp,
        *dev_files,
        *run_paths,
        *proc_paths,
    )
    write_paths = (cwd, tmp, *dev_files, *proc_paths)
    # Allow connecting only to the ports the configured providers dial,
    # rather than blanket-allowing 443: a self-hosted gateway on another
    # port still works, and nothing else can open a TCP connection.
    ports = tuple(sorted({ep.port for ep in _provider_endpoints(cfg)}))
    try:
        report = apply_agent_landlock(
            read_paths=read_paths,
            write_paths=write_paths,
            tcp_connect_ports=ports,
        )
    except LandlockNotSupportedError:
        print(
            "[agent6] WARNING: Landlock unavailable; agent process is NOT "
            "filesystem/network confined",
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        return f"could not apply agent Landlock confinement: {exc}"
    tcp_note = (
        f", tcp connect ports {report.tcp_connect_ports}"
        if report.tcp_supported
        else " (kernel too old for Landlock TCP rules)"
    )
    print(
        f"[agent6] agent-process Landlock: ABI {report.abi}, "
        f"{len(report.fs_read)} read / {len(report.fs_write)} write roots{tcp_note}",
        file=sys.stderr,
    )
    return None


@dataclass(frozen=True, slots=True)
class _InstrumentedProvider:
    """Wraps any Provider with role.call / role.result / budget.update emission.

    Pure decoration; the inner provider is unchanged. Lives in cli.py
    because that is the only place that owns the EventSink and the
    BudgetTracker and the role -> model mapping all at once.
    """

    inner: Provider
    role: str
    model: str
    provider_name: str
    events: EventSink
    budget: BudgetTracker
    stream_text: bool = False

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        self.events.emit(
            "role.call",
            role=self.role,
            model=self.model,
            provider=self.provider_name,
        )
        # When the inner provider streams, fan text deltas
        # out as `role.text_delta` events. The TUI can subscribe to
        # these for a live-typing render; non-TUI consumers ignore the
        # event and see no behaviour change.
        role_for_event = self.role
        events = self.events

        def _on_delta(piece: str) -> None:
            events.emit("role.text_delta", role=role_for_event, text=piece)

        effective_delta_cb: Callable[[str], None] | None
        if text_delta_callback is not None:
            # Caller already passed one — chain through ours too.
            outer = text_delta_callback

            def _both(piece: str) -> None:
                _on_delta(piece)
                outer(piece)

            effective_delta_cb = _both
        else:
            effective_delta_cb = _on_delta if self.stream_text else None
        try:
            resp = self.inner.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                text_delta_callback=effective_delta_cb,
            )
        except Exception as exc:
            self.events.emit("role.result", role=self.role, ok=False, error=str(exc)[:200])
            raise
        self.events.emit(
            "role.result",
            role=self.role,
            ok=True,
            tokens_in=resp.input_tokens,
            tokens_out=resp.output_tokens,
            cache_read=resp.cache_read_tokens,
            cache_creation=resp.cache_creation_tokens,
            stop_reason=resp.stop_reason,
        )
        snap = self.budget.snapshot()
        usd_total, usd_partial = self.budget.estimate_usd()
        self.events.emit(
            "budget.update",
            input_total=snap["input_total"],
            output_total=snap["output_total"],
            input_cap=snap["max_input_tokens"],
            output_cap=snap["max_output_tokens"],
            usd_total=usd_total,
            usd_partial=usd_partial,
        )
        return resp


def _default_stdin_approver(prompt: str) -> bool:
    """Plain TTY fallback for tool approval (used when no TUI is live)."""
    try:
        ans = input(f"{prompt} [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in {"y", "yes"}


_REPL_HELP = (
    "  /continue  (empty enter) - let the agent take another iteration\n"
    "  /cost                    - print the running token + USD summary\n"
    "  /diff                    - git diff: base_sha -> this run's HEAD\n"
    "                              (read-only; same as `agent6 diff`)\n"
    "  /watch                   - print the last 20 events from this run\n"
    "                              (snapshot; not a live tail)\n"
    "  /mcp                     - list MCP servers + tools currently wired\n"
    "                              into the agent's tool surface\n"
    "  /init                    - run `agent6 init` in the current cwd to\n"
    "                              (re)write agent6.toml/AGENTS.md scaffolds\n"
    "  /undo                    - git revert HEAD (forward revert of the\n"
    "                              last auto-commit; safe under git policy).\n"
    "                              History is preserved: a NEW commit is\n"
    "                              added that inverts the last one. Nothing\n"
    "                              is destroyed; ``git reset --hard`` is\n"
    "                              forbidden by agent6's git policy.\n"
    "  /help                    - show this help\n"
    "  /quit                    - stop the agent cleanly after this commit\n"
)


def _build_repl_hook(
    root: Path,
    budget: BudgetTracker,
    *,
    run_id: str = "",
    mcp_manager: MCPManager | None = None,
) -> Callable[[int, str], Literal["continue", "stop"]]:
    """Build the after_auto_commit hook for ``agent6 run -i``.

    Captures the budget tracker (for ``/cost``), the repo root (for
    ``/undo`` and ``/diff``), the current run id (for ``/diff`` and
    ``/watch``), and the live MCP manager (for ``/mcp``) in a closure
    so Workflow stays agnostic of the CLI's extra state.
    extends with /diff, /watch, /mcp, /init.
    """

    def hook(iteration: int, sha: str) -> Literal["continue", "stop"]:
        print(
            f"\n[agent6] iter {iteration} committed {sha[:12]}. "
            f"REPL: /continue /cost /diff /watch /mcp /init /undo /help /quit",
            file=sys.stderr,
        )
        while True:
            try:
                raw = input("agent6> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[agent6] EOF - stopping interactively.", file=sys.stderr)
                return "stop"
            cmd = raw.lower()
            if cmd in {"", "/continue", "/c"}:
                return "continue"
            if cmd in {"/quit", "/q", "/stop"}:
                return "stop"
            if cmd in {"/help", "/h", "?"}:
                print(_REPL_HELP, file=sys.stderr)
                continue
            if cmd == "/cost":
                print(budget.format_summary(), file=sys.stderr)
                continue
            if cmd == "/diff":
                _repl_run_diff(run_id)
                continue
            if cmd == "/watch":
                _repl_show_recent_events(root, run_id, n=20)
                continue
            if cmd == "/mcp":
                _repl_list_mcp(mcp_manager)
                continue
            if cmd == "/init":
                _repl_run_init(root)
                continue
            if cmd == "/undo":
                try:
                    revert_sha = revert_head(root)
                except GitError as exc:
                    print(f"[agent6] /undo failed: {exc}", file=sys.stderr)
                    continue
                print(
                    f"[agent6] /undo: reverted {sha[:12]} via new commit {revert_sha[:12]}",
                    file=sys.stderr,
                )
                continue
            print(
                f"[agent6] unknown command {raw!r}; try /help",
                file=sys.stderr,
            )

    return hook


def _repl_run_diff(run_id: str) -> None:
    """REPL /diff: print `git diff base_sha..HEAD` for the live run."""
    try:
        _cmd_diff(run_id=run_id, stat=False, paths=())
    except Exception as exc:
        print(f"[agent6] /diff failed: {exc}", file=sys.stderr)


def _repl_show_recent_events(root: Path, run_id: str, *, n: int) -> None:
    """REPL /watch: snapshot the last n events from this run's events.jsonl.

    Intentionally NOT a live tail - the REPL is between turns of the
    agent loop; a tail would block the next iteration. Operators who
    want continuous tail use ``agent6 watch --plain`` in another shell.
    """
    if not run_id:
        print("[agent6] /watch: no run id available", file=sys.stderr)
        return
    events_path = root / ".agent6" / "runs" / run_id / "events.jsonl"
    if not events_path.is_file():
        print(f"[agent6] /watch: no events.jsonl at {events_path}", file=sys.stderr)
        return
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[agent6] /watch failed: {exc}", file=sys.stderr)
        return
    run_start_ts: float | None = None
    if lines:
        try:
            obj0 = json.loads(lines[0])
            if isinstance(obj0, dict) and isinstance(obj0.get("ts"), (int, float)):
                run_start_ts = float(obj0["ts"])
        except json.JSONDecodeError:
            run_start_ts = None
    tail = lines[-n:]
    print(f"[agent6] /watch: last {len(tail)} events from {run_id}", file=sys.stderr)
    for raw in tail:
        print(_format_plain_event(raw, run_start_ts=run_start_ts))


def _repl_list_mcp(mcp_manager: MCPManager | None) -> None:
    """REPL /mcp: print configured MCP servers + their tool surface."""
    if mcp_manager is None:
        print(
            "[agent6] /mcp: no MCP servers configured (set [mcp] in agent6.toml)",
            file=sys.stderr,
        )
        return
    descriptors = mcp_manager.descriptors()
    if not descriptors:
        print("[agent6] /mcp: 0 tools (servers started but exposed nothing)", file=sys.stderr)
        return
    by_server: dict[str, list[str]] = {}
    for d in descriptors:
        by_server.setdefault(d.server_name, []).append(d.tool_name)
    print(f"[agent6] /mcp: {len(descriptors)} tools across {len(by_server)} server(s)")
    for server, tools in sorted(by_server.items()):
        print(f"  {server}: {len(tools)} tool(s)")
        for t in sorted(tools):
            print(f"    - {t}")


def _repl_run_init(root: Path) -> None:
    """REPL /init: run init_workspace (non-destructive without --force)."""
    try:
        rc = init_workspace(root, force=False, profile="py")
    except Exception as exc:
        print(f"[agent6] /init failed: {exc}", file=sys.stderr)
        return
    if rc == 0:
        print("[agent6] /init: ok", file=sys.stderr)
    else:
        print(f"[agent6] /init: exit {rc} (existing files left in place)", file=sys.stderr)


@dataclass
class _SteerState:
    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]


def _build_critic_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider | None:
    """critic-in-loop. Routes the reviewer role as the critic
    provider when ``workflow.critic != "off"``. Returns None when
    disabled so Workflow leaves the critic path inert."""
    if cfg.workflow.critic == "off":
        return None
    critic_inner = _build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
    )
    rm = cfg.models.reviewer
    return _InstrumentedProvider(
        inner=critic_inner,
        role="critic",
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


def _build_prompt_reviser_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider | None:
    """Route the reviewer role as a one-shot prompt reviser."""
    if cfg.workflow.revise_prompt == "off":
        return None
    reviser_inner = _build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
    )
    rm = cfg.models.reviewer
    return _InstrumentedProvider(
        inner=reviser_inner,
        role="prompt_reviser",
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


def _build_summariser_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider:
    """Route the reviewer role as the tier-2 context summariser. Always
    available (context compaction can fire on any run) and cheaper than the
    worker model."""
    summariser_inner = _build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
    )
    rm = cfg.models.reviewer
    return _InstrumentedProvider(
        inner=summariser_inner,
        role="summariser",
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


def _select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
) -> str | None:
    """Interactive accept/edit/skip prompt for workflow.revise_prompt."""
    print("\n[agent6] prompt revision proposed:", file=sys.stderr)
    print("\n--- revised ---", file=sys.stderr)
    print(revised, file=sys.stderr)
    if questions:
        print("\n--- clarifying questions ---", file=sys.stderr)
        for question in questions:
            print(f"- {question}", file=sys.stderr)
    print("\n--- original ---", file=sys.stderr)
    print(original, file=sys.stderr)
    while True:
        try:
            choice = (
                input("[agent6] revise_prompt: [a]ccept, [o]riginal, [e]dit, [q]uit? ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            return None
        if choice in {"", "a", "accept", "y", "yes"}:
            return revised
        if choice in {"o", "orig", "original", "s", "skip"}:
            return original
        if choice in {"q", "quit", "abort"}:
            return None
        if choice in {"e", "edit"}:
            editor = os.environ.get("EDITOR", "vi")
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix="agent6-revised-task-",
                suffix=".md",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(revised.rstrip() + "\n")
            try:
                result = subprocess.run([editor, str(tmp_path)], check=False)
                if result.returncode != 0:
                    print(
                        f"[agent6] editor exited {result.returncode}; choose again.",
                        file=sys.stderr,
                    )
                    continue
                edited = tmp_path.read_text(encoding="utf-8").strip()
            finally:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            if edited:
                return edited
            print("[agent6] edited prompt was empty; choose again.", file=sys.stderr)
            continue
        print("[agent6] choose accept, original, edit, or quit.", file=sys.stderr)


def _install_steer_sigint(events: EventSink) -> _SteerState:
    """Install a SIGINT handler that asks the workflow to steer.

    * 1st SIGINT — set the "steer requested" flag. The workflow notices at
      its next safe boundary (between steps) and prompts on stdin.
    * 2nd SIGINT within 2 s — raise KeyboardInterrupt to abort the run.

    Returns callables for the workflow plus a ``restore`` hook to put the
    previous handler back when the run is done.
    """
    state: dict[str, Any] = {"requested": False, "last_ts": 0.0}
    window_s = 2.0

    def _handler(_signum: int, _frame: Any) -> None:
        now = time.monotonic()
        if state["requested"] and (now - state["last_ts"]) < window_s:
            raise KeyboardInterrupt
        state["requested"] = True
        state["last_ts"] = now
        events.emit("run.steer_requested", source="sigint")
        with contextlib.suppress(Exception):
            print(
                "\n[agent6] steer requested — finishing current step, then will prompt. "
                "Press Ctrl-C again within 2s to abort.",
                file=sys.stderr,
                flush=True,
            )

    previous = signal.signal(signal.SIGINT, _handler)

    def requested() -> bool:
        return bool(state["requested"])

    def clear() -> None:
        state["requested"] = False

    def prompt() -> str | None:
        try:
            return input("[agent6] steer (blank=continue, 'abort'=stop, else=instruction): ")
        except (EOFError, KeyboardInterrupt):
            return None

    def restore() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, previous)

    return _SteerState(requested=requested, clear=clear, prompt=prompt, restore=restore)


def _check_provider_env_vars(cfg: Config) -> str | None:
    """Return an error message if any required API key env var is unset.

    Only providers actually referenced by `[models.<role>]` are checked.
    OpenAI-compat providers with `api_key_env = None` are skipped
    (unauthenticated local endpoints like Ollama).
    """
    needed = {rm.provider for rm in cfg.models.all().values()}
    for name, entry in cfg.providers.items():
        if name not in needed:
            continue
        env = entry.api_key_env
        if env is None:
            continue
        if not os.environ.get(env):
            return f"environment variable {env} (for [providers.{name}]) is not set."
    return None


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    parser = argparse.ArgumentParser(prog="agent6", description="Sandboxed coding agent.")
    parser.add_argument("--version", action="version", version=f"agent6 {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent6.toml"),
        help="Path to agent6 config (default: ./agent6.toml).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the single-loop agent on a task.")
    run_p.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task description (in quotes). Omit when using --continue.",
    )
    run_p.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    run_p.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help=(
            "Resume the most recent run under .agent6/runs/ for this cwd"
            " instead of starting a new one. Mutually exclusive with a"
            " task argument."
        ),
    )
    run_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "REPL mode: after each successful auto-commit, prompt on"
            " stdin for one of /continue (default), /cost, /undo (git"
            " revert HEAD), /help, /quit. Requires a TTY."
        ),
    )
    run_p.add_argument(
        "--from-plan",
        default="",
        metavar="RUN_ID",
        help=(
            "Use the plan.md from a prior `agent6 plan` run (resolved"
            " under .agent6/runs/, exact or unambiguous prefix) as the"
            " task description. Mutually exclusive with a positional task."
        ),
    )

    plan_p = sub.add_parser(
        "plan",
        help=(
            "Planning pass: same loop, read-only tools, writes plan.md."
            " Pair with `agent6 run --from-plan <run-id>` to execute."
        ),
    )
    plan_p.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task description (in quotes). Omit when using --show/--edit.",
    )
    plan_p.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    plan_p.add_argument(
        "--show",
        default="",
        metavar="RUN_ID",
        help="Print the plan.md for a prior plan run and exit.",
    )
    plan_p.add_argument(
        "--edit",
        default="",
        metavar="RUN_ID",
        help="Open the plan.md for a prior plan run in $EDITOR (default: vi) and exit.",
    )

    watch_p = sub.add_parser(
        "watch",
        help="Read-only live view of a run (defaults to most recent under .agent6/runs).",
    )
    watch_p.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id under .agent6/runs/ (omit for the most recent run).",
    )
    watch_p.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Plain text tail of events.jsonl (no textual TUI). Useful when"
            " textual is not installed or in headless terminals. Streams each"
            " event as a single line `<elapsed> <type> <key=val ...>` and"
            " follows the file like `tail -f`."
        ),
    )
    watch_p.add_argument(
        "--since",
        type=int,
        default=0,
        metavar="N",
        help=(
            "With --plain: replay the last N events before starting to follow."
            " 0 (default) starts at end-of-file."
        ),
    )

    resume_p = sub.add_parser("resume", help="Resume a paused run from its snapshot.")
    resume_p.add_argument("run_id", help="Run id under .agent6/runs/.")
    resume_p.add_argument(
        "--config",
        type=Path,
        default=Path("agent6.toml"),
        help="Path to agent6.toml (defaults to ./agent6.toml).",
    )
    resume_p.add_argument(
        "--force-resume",
        action="store_true",
        help="Resume even if snapshot commit is missing or worktree has diverged.",
    )

    check_config_p = sub.add_parser(
        "check-config", help="Validate config and print detected environment."
    )
    check_config_p.add_argument(
        "--fix",
        action="store_true",
        help=(
            "If the config is missing required fields, print the recommended"
            " additions (sourced from the starter template) and offer to"
            " insert them after confirmation."
        ),
    )
    check_config_p.add_argument(
        "--yes",
        action="store_true",
        help="With --fix, apply proposed additions without an interactive prompt.",
    )
    sub.add_parser(
        "check-sandbox",
        help="Run sandbox self-tests against the current kernel and report.",
    )

    doctor_p = sub.add_parser(
        "doctor",
        help=(
            "Consolidated pre-flight: sandbox + MCP + verify_command + config."
            " Read-only; safe to run on any clean repo."
        ),
    )
    doctor_p.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=("all", "sandbox", "mcp", "verify", "config"),
        help=(
            "Limit the report to one section. 'all' (default) runs every"
            " check in order and prints a single PASS/FAIL summary."
        ),
    )
    doctor_p.add_argument(
        "--config",
        type=Path,
        default=Path("agent6.toml"),
        help="Path to agent6.toml (defaults to ./agent6.toml).",
    )

    mem_p = sub.add_parser("memory", help="Manage persistent agent memories.")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True)
    mem_add = mem_sub.add_parser("add", help="Append a new memory entry.")
    mem_add.add_argument(
        "scope", choices=("facts", "decisions", "preferences"), help="Memory scope."
    )
    mem_add.add_argument("body", help="Entry body (in quotes).")
    mem_list = mem_sub.add_parser("list", help="List memory entries.")
    mem_list.add_argument(
        "--scope",
        choices=("facts", "decisions", "preferences"),
        default="",
        help="Limit to one scope; omit for all.",
    )
    mem_list.add_argument(
        "--all", action="store_true", help="Include invalidated entries (default: hide)."
    )
    mem_inv = mem_sub.add_parser("invalidate", help="Mark a memory entry as invalidated.")
    mem_inv.add_argument("memory_id", help="26-char ULID of the entry to invalidate.")
    mem_inv.add_argument("reason", help="Why this entry is no longer valid.")

    hist_p = sub.add_parser("history", help="Search persisted transcripts and run data.")
    hist_sub = hist_p.add_subparsers(dest="history_command", required=True)
    hist_search = hist_sub.add_parser("search", help="ripgrep-backed search over all runs.")
    hist_search.add_argument("query", help="Pattern (passed to rg --fixed-strings by default).")
    hist_search.add_argument(
        "--regex", action="store_true", help="Interpret query as a regex instead of fixed string."
    )
    hist_search.add_argument(
        "--run", default="", help="Restrict to a single run id (default: all runs)."
    )

    hist_graph = hist_sub.add_parser(
        "graph",
        help="Render the persisted task graph for a run as a DFS tree.",
    )
    hist_graph.add_argument(
        "run",
        nargs="?",
        default="",
        help="Run id (or unambiguous prefix). Defaults to the most recent run.",
    )

    init_p = sub.add_parser(
        "init",
        help="Write starter agent6.toml, AGENTS.md, and update .gitignore in the cwd.",
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files (default: refuse and write a .suggested sibling).",
    )
    init_p.add_argument(
        "--profile",
        choices=("py", "rust", "node"),
        default="py",
        help=(
            "Pick a starter verify_command for your language. "
            "py (default): `uv run pytest -x`. "
            "rust: `cargo test`. "
            "node: `npm test --silent`. "
            "Edit agent6.toml afterward to match your real pipeline."
        ),
    )

    review_p = sub.add_parser(
        "review",
        help="Read-only code review of a diff (working tree, branch-vs-base, or arbitrary range).",
    )
    review_p.add_argument(
        "--base",
        default="",
        help="Base ref. Default: review uncommitted changes (working tree vs HEAD).",
    )
    review_p.add_argument(
        "--head",
        default="HEAD",
        help="Head ref (default: HEAD). Only used when --base is set.",
    )
    review_p.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Restrict the diff to these paths (forwarded to `git diff -- PATHS`).",
    )
    review_p.add_argument(
        "--model",
        default="",
        help=(
            "Override the reviewer model for this one-shot review "
            "(e.g. claude-sonnet-4-5 for a cheaper read). "
            "Default: reviewer_model from config."
        ),
    )

    diff_p = sub.add_parser(
        "diff",
        help="Print the git diff produced by a run (manifest.base_sha -> HEAD of run branch).",
    )
    diff_p.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unique prefix). Omit to diff the most recent run.",
    )
    diff_p.add_argument(
        "--stat",
        action="store_true",
        help="Show --stat summary instead of the full patch.",
    )
    diff_p.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Restrict the diff to these paths.",
    )

    mcp_p = sub.add_parser(
        "mcp",
        help="MCP (Model Context Protocol) integration. See `agent6 mcp serve --help`.",
    )
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_sub.add_parser(
        "serve",
        help=(
            "Run agent6 as an MCP stdio server, exposing run_verify /"
            " run_in_sandbox / apply_patch_in_sandbox / query_dag / list_runs"
            " against the cwd's agent6.toml. Speaks line-delimited JSON-RPC"
            " on stdin/stdout; spawn from an MCP-aware client (e.g. VS Code"
            " Copilot's hand-off menu) and configure it with this command."
        ),
    )
    mcp_serve.add_argument(
        "--config",
        type=Path,
        default=Path("agent6.toml"),
        help="Path to agent6.toml (defaults to ./agent6.toml in cwd).",
    )

    machine_p = sub.add_parser(
        "machine",
        help="Author-time tooling for agent6 state machines (.asm.toml).",
    )
    machine_sub = machine_p.add_subparsers(dest="machine_command", required=True)
    machine_check = machine_sub.add_parser(
        "check",
        help="Validate a .asm.toml machine file (parse, type-check, reachability). Pure.",
    )
    machine_check.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_graph = machine_sub.add_parser(
        "graph",
        help="Emit the machine as a state diagram (mermaid or Graphviz dot).",
    )
    machine_graph.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_graph.add_argument(
        "--format",
        choices=("mermaid", "dot"),
        default="mermaid",
        help="Diagram format (default: mermaid).",
    )
    machine_run = machine_sub.add_parser(
        "run",
        help="Run (or resume) a machine, driving its states to a terminal one.",
    )
    machine_run.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_run.add_argument(
        "--exit-on-wait",
        action="store_true",
        help=(
            "Persist the next wake instant and exit 0 (status 'waiting') at the first"
            " not-ready wait instead of blocking, for an external scheduler to resume."
        ),
    )
    machine_status = machine_sub.add_parser(
        "status",
        help="Report a machine instance's current state, spend, and next wake. Read-only.",
    )
    machine_status.add_argument(
        "machine_id", help="Machine id (directory under .agent6/machines/)."
    )
    machine_poke = machine_sub.add_parser(
        "poke",
        help="Signal a waiting machine to wake on its next check (drops a signal file).",
    )
    machine_poke.add_argument("machine_id", help="Machine id (directory under .agent6/machines/).")
    machine_replay = machine_sub.add_parser(
        "replay",
        help="Deterministically replay a machine's journal offline (no world I/O).",
    )
    machine_replay.add_argument(
        "machine_id", help="Machine id (directory under .agent6/machines/)."
    )

    machine_create = machine_sub.add_parser(
        "create",
        help="Draft a .asm.toml machine from a natural-language task (LLM-assisted).",
    )
    machine_create.add_argument("task", help="Natural-language description of the loop to author.")
    machine_create.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Write the draft here (overwriting freely). Default: <machine-name>.asm.toml"
            " in cwd, which is never overwritten."
        ),
    )
    machine_create.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum draft->check->fix attempts before giving up (default: 3).",
    )

    # Shell tab-completion. argcomplete is a hard dependency; the call is a
    # no-op unless the shell sourced its completion script for this binary
    # (see `agent6 --help` and the README for activation instructions).
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.continue_run:
            if args.task:
                print("ERROR: pass either a task OR --continue, not both.", file=sys.stderr)
                return 2
            if args.run_id:
                print(
                    "ERROR: --run-id is incompatible with --continue"
                    " (--continue resolves the most recent run automatically).",
                    file=sys.stderr,
                )
                return 2
            target = _most_recent_run_id(Path.cwd() / ".agent6" / "runs")
            if target is None:
                print(
                    "ERROR: --continue: no prior runs under .agent6/runs/ in this cwd.",
                    file=sys.stderr,
                )
                return 2
            print(f"[agent6] --continue: resuming {target}", file=sys.stderr)
            return _cmd_resume(args.config, target, force=False)
        if args.from_plan:
            if args.task:
                print(
                    "ERROR: --from-plan is mutually exclusive with a task argument.",
                    file=sys.stderr,
                )
                return 2
            resolved = _resolve_plan_run_id(args.from_plan)
            if resolved is None:
                return 2
            plan_md = (Path.cwd() / ".agent6" / "runs" / resolved / "plan.md").read_text(
                encoding="utf-8"
            )
            task = (
                f"The following plan was prepared by a planning pass at {resolved}."
                f" Execute it.\n\n{plan_md}"
            )
        else:
            if not args.task:
                print("ERROR: 'run' needs a task argument (or --continue).", file=sys.stderr)
                return 2
            task = args.task
        return _cmd_run(
            args.config,
            task,
            run_id=args.run_id,
            interactive=args.interactive,
        )
    if args.command == "plan":
        if args.show and args.edit:
            print("ERROR: --show and --edit are mutually exclusive.", file=sys.stderr)
            return 2
        if args.show:
            return _cmd_plan_show(args.show)
        if args.edit:
            return _cmd_plan_edit(args.edit)
        if not args.task:
            print(
                "ERROR: 'plan' needs a task argument (or --show/--edit).",
                file=sys.stderr,
            )
            return 2
        return _cmd_run(
            args.config,
            args.task,
            run_id=args.run_id,
            mode="plan",
        )
    if args.command == "watch":
        return _cmd_watch(args.run_id, plain=args.plain, since=args.since)
    if args.command == "resume":
        return _cmd_resume(args.config, args.run_id, force=args.force_resume)
    if args.command == "check-config":
        return _cmd_check_config(args.config, fix=args.fix, assume_yes=args.yes)
    if args.command == "check-sandbox":
        return _cmd_check_sandbox()
    if args.command == "doctor":
        return _cmd_doctor(args.config, section=args.section)
    if args.command == "memory":
        if args.memory_command == "add":
            return _cmd_memory_add(args.scope, args.body)
        if args.memory_command == "list":
            return _cmd_memory_list(args.scope or None, include_invalidated=args.all)
        if args.memory_command == "invalidate":
            return _cmd_memory_invalidate(args.memory_id, args.reason)
    if args.command == "history" and args.history_command == "search":
        return _cmd_history_search(args.query, fixed=not args.regex, run_id=args.run)
    if args.command == "history" and args.history_command == "graph":
        return _cmd_history_graph(args.run)
    if args.command == "init":
        return _cmd_init(force=args.force, profile=args.profile)
    if args.command == "review":
        return _cmd_review(
            args.config,
            base=args.base,
            head=args.head,
            paths=tuple(args.paths),
            model_override=args.model,
        )
    if args.command == "diff":
        return _cmd_diff(
            run_id=args.run_id,
            stat=args.stat,
            paths=tuple(args.paths),
        )
    if args.command == "mcp" and args.mcp_command == "serve":
        return _cmd_mcp_serve(args.config)
    if args.command == "machine" and args.machine_command == "check":
        return _cmd_machine_check(args.file)
    if args.command == "machine" and args.machine_command == "graph":
        return _cmd_machine_graph(args.file, fmt=args.format)
    if args.command == "machine" and args.machine_command == "run":
        return _cmd_machine_run(args.file, exit_on_wait=args.exit_on_wait)
    if args.command == "machine" and args.machine_command == "status":
        return _cmd_machine_status(args.machine_id)
    if args.command == "machine" and args.machine_command == "poke":
        return _cmd_machine_poke(args.machine_id)
    if args.command == "machine" and args.machine_command == "replay":
        return _cmd_machine_replay(args.machine_id)
    if args.command == "machine" and args.machine_command == "create":
        return _cmd_machine_create(args.task, output=args.output, max_attempts=args.max_attempts)
    parser.error("unknown command")  # pragma: no cover
    return 2


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_machine_check(path: Path) -> int:
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {path} ({spec.machine}, {len(spec.states)} states)")
    return 0


def _cmd_machine_graph(path: Path, *, fmt: str) -> int:
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    render_fmt: Literal["mermaid", "dot"] = "dot" if fmt == "dot" else "mermaid"
    print(render(spec, render_fmt), end="")
    return 0


def _build_machine_agent_runner(
    cfg: Config, root: Path, profile: SandboxProfile, transcript_dir: Path
) -> Callable[[AgentRequest], AgentExecResult]:
    """Build the live runner an `agent` state uses to drive a normal agent6 loop.

    Each invocation gets a fresh budget slice, provider (with the state's model),
    dispatcher, and `Workflow`; it runs until the agent calls `finish_run` (or
    the loop stops for another reason) and surfaces the structured payload.

    The state's `timeout_secs` is enforced with a watchdog: the loop runs in a
    daemon thread joined for the timeout. On expiry we return the `timeout`
    outcome; the abandoned thread is bounded by its own one-shot budget slice
    (true mid-call cancellation needs out-of-process execution — Phase 4).
    """

    def run_agent(request: AgentRequest) -> AgentExecResult:
        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
        )
        transcript_sink = TranscriptSink(transcript_dir)
        provider = _build_role_provider(
            cfg,
            "worker",
            transcript_sink=transcript_sink,
            budget=budget,
            model_override=request.model,
        )
        dispatcher = ToolDispatcher(
            root=root,
            config=cfg,
            sandbox_profile=profile,
            approver=None,
            events=None,
            graph_client=None,
            run_root_node_id=None,
            mcp_manager=None,
        )
        wf = Workflow(
            root=root,
            config=cfg,
            provider=provider,
            dispatcher=dispatcher,
            logger=lambda msg: print(msg, file=sys.stderr),
        )

        box: dict[str, RunResult | BaseException] = {}

        def _target() -> None:
            try:
                box["result"] = wf.run(request.prompt)
            except Exception as exc:  # surfaced on the main thread
                box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(request.timeout_s)
        usd, _ = budget.estimate_usd()
        snap = budget.snapshot()
        input_total = snap["input_total"]
        output_total = snap["output_total"]
        assert isinstance(input_total, int)
        assert isinstance(output_total, int)
        input_tokens = input_total
        output_tokens = output_total
        if thread.is_alive():
            return AgentExecResult(
                reason="timeout",
                payload=None,
                usd=usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        error = box.get("error")
        if isinstance(error, BaseException):
            raise error
        result = box["result"]
        assert isinstance(result, RunResult)
        payload = result.finish_payload if result.reason == "finish_run" else None
        return AgentExecResult(
            reason=result.reason,
            payload=payload,
            usd=usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return run_agent


def _cmd_machine_run(path: Path, *, exit_on_wait: bool = False) -> int:  # noqa: PLR0911
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    cwd = Path.cwd()
    has_agent_state = any(getattr(state, "kind", None) == "agent" for state in spec.states.values())
    agent_runner: Callable[[AgentRequest], AgentExecResult] | None = None
    if has_agent_state:
        try:
            cfg = load_config(cwd / "agent6.toml")
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2
        missing = _check_provider_env_vars(cfg)
        if missing is not None:
            print(missing, file=sys.stderr)
            return 2
        env = detect()
        try:
            profile = select_profile(cfg.sandbox.profile, env)
        except RuntimeError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 2
        root = cwd / ".agent6" / "machines" / spec.machine
        agent_runner = _build_machine_agent_runner(cfg, cwd, profile, root / "agent_transcripts")
    root = cwd / ".agent6" / "machines" / spec.machine
    journal = MachineJournal(root)
    try:
        with machine_lock(root):
            journal.ensure_dirs()
            if not journal.exists():
                write_source(root, path.read_text(encoding="utf-8"))
            world = LiveWorld(cwd=cwd, journal=journal, agent_runner=agent_runner)
            result = drive(spec, journal, world, live=True, exit_on_wait=exit_on_wait)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.status == "waiting":
        print(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {result.transitions} transitions ({result.reason})"
        )
        return 0
    print(
        f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
        f" after {result.transitions} transitions ({result.reason})"
    )
    return 0 if result.status == "ok" else 1


def _cmd_machine_replay(machine_id: str) -> int:
    root = Path.cwd() / ".agent6" / "machines" / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"{result.status.upper()}: {spec.machine} replayed to {result.state!r}"
        f" after {result.transitions} transitions ({result.reason})"
    )
    return 0 if result.status in ("ok", "incomplete") else 1


def _cmd_machine_status(machine_id: str) -> int:
    root = Path.cwd() / ".agent6" / "machines" / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
        events = journal.read()
        snapshot = journal.latest_snapshot()
        pending = journal.read_pending_wait()
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    usd_total = 0.0
    input_total = 0
    output_total = 0
    for event in events:
        if isinstance(event, StepEvent) and isinstance(event.fact, AgentFact):
            usd_total += event.fact.usd
            input_total += event.fact.input_tokens
            output_total += event.fact.output_tokens

    print(f"machine: {spec.machine} (v{spec.version})")
    print(f"  status: {result.status}")
    print(f"  state: {result.state!r}")
    print(f"  transitions: {result.transitions}")
    print(f"  spend: ${usd_total:.4f} (in={input_total} tok, out={output_total} tok)")
    if pending is not None:
        wake = _dt.datetime.fromtimestamp(pending.wake_epoch, tz=_dt.UTC).isoformat()
        print(f"  next wake: {wake} (waiting in {pending.state!r})")
    if snapshot is not None and snapshot.blackboard:
        print("  blackboard:")
        for key, value in snapshot.blackboard.items():
            print(f"    {key} = {value!r}")
    step_events = [e for e in events if isinstance(e, StepEvent)]
    if step_events:
        print("  recent steps:")
        for event in step_events[-5:]:
            print(f"    [{event.seq}] {event.state!r} --{event.label}--> {event.goto!r}")
    return 0


def _cmd_machine_poke(machine_id: str) -> int:
    root = Path.cwd() / ".agent6" / "machines" / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    try:
        MachineJournal(root).poke()
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"poked {machine_id}: it will wake on its next signal check")
    return 0


_CREATE_TIMEOUT_S = 900.0
_CREATE_STOP_REASONS = frozenset(
    {"budget_exhausted", "timeout", "provider_error", "prompt_revision_failed"}
)


def _check_machine_text(text: str, scratch: Path) -> tuple[MachineSpec | None, list[str]]:
    """Validate a candidate `.asm.toml` source by parsing it through `load_machine`.

    Returns the parsed spec and an empty problem list on success, or `(None,
    problems)` when the source is invalid.
    """
    candidate_path = scratch / "candidate.asm.toml"
    candidate_path.write_text(text, encoding="utf-8")
    try:
        spec = load_machine(candidate_path)
    except MachineError as exc:
        return None, list(exc.problems)
    return spec, []


def _cmd_machine_create(  # noqa: PLR0911, PLR0912, PLR0915
    task: str, *, output: Path | None, max_attempts: int
) -> int:
    if max_attempts < 1:
        print("ERROR: --max-attempts must be >= 1.", file=sys.stderr)
        return 2
    cwd = Path.cwd()
    try:
        cfg = load_config(cwd / "agent6.toml")
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    missing = _check_provider_env_vars(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2
    env = detect()
    try:
        profile = select_profile(cfg.sandbox.profile, env)
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    scratch = cwd / ".agent6" / "machine-drafts" / new_friendly_id()
    scratch.mkdir(parents=True, exist_ok=True)
    runner = _build_machine_agent_runner(cfg, cwd, profile, scratch / "agent_transcripts")

    prior_toml: str | None = None
    diagnostics: list[str] | None = None
    spec: MachineSpec | None = None
    valid_toml: str | None = None
    total_usd = 0.0
    for attempt in range(1, max_attempts + 1):
        prompt = build_authoring_prompt(
            task, attempt=attempt, prior_toml=prior_toml, diagnostics=diagnostics
        )
        print(f"machine create: attempt {attempt}/{max_attempts}...", file=sys.stderr)
        result = runner(AgentRequest(model="", prompt=prompt, timeout_s=_CREATE_TIMEOUT_S))
        total_usd += result.usd
        candidate = extract_toml(result.payload)
        if candidate is None:
            diagnostics = [
                f"You did not return a draft: call finish_run with result.{TOML_PAYLOAD_KEY}"
                " set to the complete .asm.toml source as a single string."
                f" (agent loop reason: {result.reason})"
            ]
            prior_toml = None
            if result.reason in _CREATE_STOP_REASONS:
                break
            continue
        candidate_spec, problems = _check_machine_text(candidate, scratch)
        if candidate_spec is not None:
            spec = candidate_spec
            valid_toml = candidate
            break
        prior_toml = candidate
        diagnostics = problems
        if result.reason in _CREATE_STOP_REASONS:
            break

    print(f"machine create: spent ~${total_usd:.4f}", file=sys.stderr)

    if spec is None or valid_toml is None:
        print(f"FAILED: no valid machine after {max_attempts} attempt(s).", file=sys.stderr)
        if diagnostics:
            print("Last diagnostics:", file=sys.stderr)
            for problem in diagnostics:
                print(f"  - {problem}", file=sys.stderr)
        if prior_toml is not None:
            print("The last (invalid) draft is on stdout for reference.", file=sys.stderr)
            print(prior_toml if prior_toml.endswith("\n") else prior_toml + "\n", end="")
        return 1

    payload = valid_toml if valid_toml.endswith("\n") else valid_toml + "\n"
    if output is not None:
        output.write_text(payload, encoding="utf-8")
        print(
            f"OK: wrote draft to {output} ({spec.machine}, {len(spec.states)} states).",
            file=sys.stderr,
        )
        print(
            "Review and commit it; `machine run` only accepts committed machines.",
            file=sys.stderr,
        )
        return 0

    default_path = cwd / f"{spec.machine}.asm.toml"
    if default_path.exists():
        print(f"REFUSING to overwrite existing {default_path}.", file=sys.stderr)
        print(
            "The validated draft is on stdout; redirect it or re-run with -o <file>.",
            file=sys.stderr,
        )
        print(payload, end="")
        return 1
    default_path.write_text(payload, encoding="utf-8")
    print(
        f"OK: wrote draft to {default_path} ({spec.machine}, {len(spec.states)} states).",
        file=sys.stderr,
    )
    print(
        "Review and commit it; `machine run` only accepts committed machines.",
        file=sys.stderr,
    )
    return 0


def _cmd_check_config(path: Path, *, fix: bool = False, assume_yes: bool = False) -> int:
    env = detect()
    print(f"Container signals: {list(env.container_signals) or 'none'}")
    print(f"Kernel: {env.kernel.raw} (Landlock TCP: {env.kernel.supports_landlock_tcp})")
    print(f"Userns supported: {env.userns_supported}")
    print(f"Detected sandbox profile: {env.detected_profile}")
    print(f"Landlock ABI on this host: {landlock_abi()}")
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        if fix:
            return _run_fix_flow(path, original_error=str(exc), assume_yes=assume_yes)
        print(f"\nCONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    if fix:
        print(f"\nConfig file OK: {path} — no missing fields to fix.")
    else:
        print(f"\nConfig file OK: {path}")
    print(f"  sandbox.profile = {cfg.sandbox.profile}")
    print(f"  sandbox.network = {cfg.sandbox.network}")
    print(f"  sandbox.run_commands = {cfg.sandbox.run_commands}")
    print(f"  workflow.verify_command = {' '.join(cfg.workflow.verify_command)}")

    try:
        selected = select_profile(cfg.sandbox.profile, env)
    except RuntimeError as exc:
        print(f"\nREFUSE: {exc}", file=sys.stderr)
        return 1
    print(f"  -> selected profile: {selected}")
    return 0


def _run_fix_flow(path: Path, *, original_error: str, assume_yes: bool) -> int:
    """Interactive --fix flow invoked when initial validation failed.

    Prints the original error, lists each proposed insertion sourced from
    the starter template, asks for confirmation (unless `assume_yes`),
    then writes the file and re-validates. Returns 0 only if the file
    validates after the edits.
    """
    print(f"\nCONFIG ERROR:\n{original_error}", file=sys.stderr)
    result = propose_fixes(path)
    if not result.fixes:
        print(
            "\n--fix: no automatic repair available for the errors above.",
            file=sys.stderr,
        )
        for line in result.remaining_errors:
            print(f"  - {line}", file=sys.stderr)
        return 2
    print("\n--fix: proposed additions (sourced from the starter template):\n")
    for index, item in enumerate(result.fixes, start=1):
        kind_label = "new section" if item.kind is FixKind.NEW_SECTION else "new field"
        print(f"  [{index}] {kind_label}: {item.description}")
        for line in item.render_preview().splitlines():
            print(f"        {line}")
    if result.remaining_errors:
        print("\nThese errors are NOT addressable by --fix:")
        for line in result.remaining_errors:
            print(f"  - {line}")
    if not assume_yes:
        try:
            answer = input(f"\nApply all {len(result.fixes)} additions to {path}? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("--fix: aborted, no changes written.")
            return 2
    apply_fixes(path, result.fixes)
    print(f"\n--fix: wrote {len(result.fixes)} additions to {path}.")
    try:
        load_config(path)
    except ConfigError as exc:
        print(
            f"\n--fix: file still does not validate after edits:\n{exc}",
            file=sys.stderr,
        )
        return 2
    print("--fix: file now validates cleanly.")
    return 0


def _cmd_check_sandbox() -> int:
    """Run the sandbox boundary self-tests on the host's kernel."""
    reports: list[SandboxReport] = []

    # Landlock probe
    abi = landlock_abi()
    reports.append(
        SandboxReport(
            name="landlock_abi",
            ok=abi > 0,
            detail=f"abi={abi}; TCP={'yes' if abi >= 4 else 'no (need Linux 6.7)'}",
        )
    )

    # Try running `/bin/true` in the jail.
    cwd = Path.cwd()
    try:
        res = run_in_jail(
            JailPolicy(cwd=cwd, argv=("/usr/bin/true",), allow_network=False, timeout_s=10.0)
        )
        reports.append(SandboxReport(name="jail_true", ok=res.ok, detail=f"rc={res.returncode}"))
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_true", ok=False, detail=str(exc)))

    # Confirm child cannot reach the network (when allow_network=False).
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=cwd,
                argv=("/usr/bin/getent", "hosts", "example.com"),
                allow_network=False,
                timeout_s=10.0,
            )
        )
        ok = res.returncode != 0
        reports.append(
            SandboxReport(
                name="jail_blocks_network",
                ok=ok,
                detail=f"rc={res.returncode} (nonzero = blocked, as expected)",
            )
        )
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_blocks_network", ok=False, detail=str(exc)))

    # Confirm child cannot write outside /workspace.
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=cwd,
                argv=("/bin/sh", "-c", "echo x > /etc/agent6-escape || true"),
                allow_network=False,
                timeout_s=10.0,
            )
        )
        # /etc was bind-mounted RO and Landlock confines writes to /workspace, so the
        # file should not exist on the host.
        ok = not Path("/etc/agent6-escape").exists()
        reports.append(
            SandboxReport(
                name="jail_blocks_etc_write",
                ok=ok,
                detail=f"rc={res.returncode}; host /etc/agent6-escape exists: {not ok}",
            )
        )
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_blocks_etc_write", ok=False, detail=str(exc)))

    overall_ok = True
    for r in reports:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        overall_ok = overall_ok and r.ok
    return 0 if overall_ok else 1


@dataclass(frozen=True, slots=True)
class _DoctorCheck:
    name: str
    ok: bool
    detail: str


def _cmd_doctor(config_path: Path, *, section: str) -> int:
    """Consolidated pre-flight (sandbox + MCP + verify + config).

    All checks are read-only. The command never spawns the agent loop,
    never makes a network call to the configured providers, and never
    writes to the repo. MCP servers are started just long enough to
    enumerate their tool descriptors and then closed.

    Returns 0 when every selected check passes, 1 otherwise.
    """
    print(f"agent6 doctor: section={section}")
    print()

    checks: list[_DoctorCheck] = []
    if section in {"all", "sandbox"}:
        print("== sandbox ==")
        rc = _cmd_check_sandbox()
        checks.append(
            _DoctorCheck(
                name="sandbox",
                ok=(rc == 0),
                detail="all jail probes passed" if rc == 0 else f"check-sandbox exit {rc}",
            )
        )
        print()

    try:
        cfg = load_config(config_path) if section in {"all", "mcp", "verify", "config"} else None
    except (ConfigError, OSError) as exc:
        cfg = None
        if section in {"all", "mcp", "verify", "config"}:
            print(f"== config ==\n[FAIL] cannot load {config_path}: {exc}\n")
            checks.append(_DoctorCheck(name="config_load", ok=False, detail=str(exc)))

    if cfg is not None and section in {"all", "mcp"}:
        print("== mcp ==")
        checks.extend(_doctor_check_mcp(cfg))
        print()

    if cfg is not None and section in {"all", "verify"}:
        print("== verify ==")
        checks.extend(_doctor_check_verify(cfg))
        print()

    if cfg is not None and section in {"all", "config"}:
        print("== config ==")
        checks.extend(_doctor_check_config(cfg))
        print()

    print("== summary ==")
    overall_ok = True
    for c in checks:
        flag = "PASS" if c.ok else "FAIL"
        print(f"[{flag}] {c.name}: {c.detail}")
        overall_ok = overall_ok and c.ok
    return 0 if overall_ok else 1


def _doctor_check_mcp(cfg: Config) -> list[_DoctorCheck]:
    """Start configured MCP servers, enumerate tools, then close them.

    Returns one check per configured server plus a summary check. When
    ``[mcp]`` is disabled or empty, returns a single skip-style PASS so
    the doctor doesn't fail an unconfigured-by-design feature.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        print("(MCP disabled or no servers configured; skipping)")
        return [
            _DoctorCheck(
                name="mcp",
                ok=True,
                detail="not configured (cfg.mcp.enabled=False or empty servers)",
            )
        ]
    manager = _start_mcp_manager_if_enabled(cfg)
    if manager is None:
        return [_DoctorCheck(name="mcp", ok=True, detail="no enabled servers")]
    out: list[_DoctorCheck] = []
    try:
        descriptors = manager.descriptors()
        by_server: dict[str, list[str]] = {}
        for d in descriptors:
            by_server.setdefault(d.server_name, []).append(d.tool_name)
        configured = {srv.name for srv in cfg.mcp.servers if srv.enabled}
        for name in sorted(configured):
            tools = by_server.get(name, [])
            ok = bool(tools)
            detail = f"{len(tools)} tool(s)" if ok else "started but exposed no tools"
            print(f"[{'PASS' if ok else 'FAIL'}] mcp.{name}: {detail}")
            out.append(_DoctorCheck(name=f"mcp.{name}", ok=ok, detail=detail))
    finally:
        manager.close()
    return out


def _doctor_check_verify(cfg: Config) -> list[_DoctorCheck]:
    """Verify command sanity: argv non-empty and the head executable resolves.

    Does NOT execute the verify command — that would run an arbitrary
    test suite on every doctor call. Operators can do
    ``./$(verify_command)`` themselves when they want a live run.
    """
    argv = list(cfg.workflow.verify_command)
    if not argv:
        print("[FAIL] verify.argv: empty")
        return [_DoctorCheck(name="verify.argv", ok=False, detail="empty")]
    head = argv[0]
    resolved = shutil.which(head)
    ok = resolved is not None
    detail = f"resolves to {resolved}" if resolved else f"not found on PATH: {head!r}"
    print(f"[{'PASS' if ok else 'FAIL'}] verify.head: {detail}")
    print(f"       argv = {argv}")
    print(f"       timeout = {cfg.workflow.verify_timeout_s}s")
    return [_DoctorCheck(name="verify.head", ok=ok, detail=detail)]


def _doctor_check_config(cfg: Config) -> list[_DoctorCheck]:
    """Static config sanity checks: provider env vars + worktree git policy."""
    out: list[_DoctorCheck] = []
    env_err = _check_provider_env_vars(cfg)
    ok_env = env_err is None
    detail_env = "all required API key env vars set" if ok_env else env_err or ""
    print(f"[{'PASS' if ok_env else 'FAIL'}] config.provider_env: {detail_env}")
    out.append(_DoctorCheck(name="config.provider_env", ok=ok_env, detail=detail_env))

    ok_git = cfg.git.allow_push is False
    detail_git = "git.allow_push=False (push is blocked, as required)"
    if not ok_git:
        detail_git = "git.allow_push=True — agent6 never pushes; set this back to false"
    print(f"[{'PASS' if ok_git else 'FAIL'}] config.git_policy: {detail_git}")
    out.append(_DoctorCheck(name="config.git_policy", ok=ok_git, detail=detail_git))
    return out


def _ensure_agent6_gitignored(
    root: Path,
    *,
    identity: CommitIdentity | None = None,
    logger: Callable[[str], None] = print,
) -> None:
    """Make sure `.agent6/` is in `.gitignore` before we write anything under it.

    `agent6 run` and `agent6 plan` create `.agent6/runs/<id>/` early in startup
    (transcripts, run log). If the project's `.gitignore` doesn't already
    exclude `.agent6/`, those files become untracked content and the
    `require_clean_worktree` pre-flight check then refuses to proceed — a
    self-DoS that confuses first-time users.

    Append the entry, then commit `.gitignore` immediately so the worktree
    stays clean for the subsequent dirty-tree check. We commit on the user's
    current branch *before* `branch_per_run` cuts the agent's working branch,
    so this single housekeeping commit lands on the parent branch where it
    belongs.
    """
    gitignore = root / ".gitignore"
    entry = ".agent6/"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    if any(line.strip() in {entry, "/.agent6/", ".agent6"} for line in existing.splitlines()):
        return
    suffix = "" if existing.endswith("\n") or not existing else "\n"
    gitignore.write_text(
        existing + suffix + "# agent6 run state (transcripts, run logs, graph)\n" + entry + "\n",
        encoding="utf-8",
    )
    # Commit on the current branch only if we are inside a git repo; otherwise
    # writing the file is enough (the workflow's git pre-flight will refuse
    # to proceed anyway, with a clearer error than "dirty worktree").
    try:
        if is_git_repo(root):
            commit_paths(
                root,
                "chore: ignore .agent6/ run state (added by agent6)",
                (".gitignore",),
                identity=identity,
            )
            logger(f"[agent6] added {entry!r} to {gitignore.name} (committed)")
            return
    except GitError as exc:
        logger(f"[agent6] WARNING: wrote {entry!r} to .gitignore but commit failed: {exc}")
        return
    logger(f"[agent6] added {entry!r} to {gitignore.name}")


# inline-resolved file references in user task strings.
#
# A token of the form `@PATH` that resolves to a regular file inside `root`
# is replaced with the file's contents wrapped in a `<file path=...>` block.
# Anything that doesn't match (missing files, paths that escape root, email
# addresses, decorators copied from code, etc.) is left untouched so the
# transformation never corrupts a hand-written task string.
_TASK_FILE_REF_RE = re.compile(r"(?<![\w@/])@([A-Za-z0-9_./\-]+)")
_TASK_FILE_REF_MAX_BYTES = 64 * 1024  # cap per file - bigger reads need an explicit tool call.


def _expand_task_file_refs(task: str, root: Path) -> str:
    """Inline `@path` references in `task` that resolve to files under `root`.

    Behaviour:
      - The match must start at a word boundary that excludes `@` and `/`
        (so `user@example.com` and `//@noqa` are not touched).
      - The path must resolve (via ``Path.resolve``) to a regular file
        whose resolved path is inside ``root``. Symlinks that escape are
        rejected the same way the sandbox would reject them.
      - File contents are truncated to ``_TASK_FILE_REF_MAX_BYTES`` and
        decoded as UTF-8 with replacement; binary files therefore appear
        as garbled text rather than crashing the run.
      - Unresolved references are left as-is. We never raise.
    """
    root_resolved = root.resolve()

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(1)
        try:
            candidate = (root / rel).resolve()
        except (OSError, RuntimeError):
            return match.group(0)
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            return match.group(0)
        if not candidate.is_file():
            return match.group(0)
        try:
            raw = candidate.read_bytes()
        except OSError:
            return match.group(0)
        truncated = raw[:_TASK_FILE_REF_MAX_BYTES]
        text = truncated.decode("utf-8", errors="replace")
        suffix = ""
        if len(raw) > _TASK_FILE_REF_MAX_BYTES:
            suffix = (
                f"\n... (truncated, {len(raw) - _TASK_FILE_REF_MAX_BYTES} bytes omitted; "
                "use read_file with an explicit range for the rest)"
            )
        return f'\n<file path="{rel}">\n{text}{suffix}\n</file>\n'

    return _TASK_FILE_REF_RE.sub(_replace, task)


def _start_mcp_manager_if_enabled(cfg: Config) -> MCPManager | None:
    """Spawn all enabled MCP servers from ``cfg.mcp``. Returns None when
    MCP is disabled or no servers are configured (so callers can skip
    teardown entirely). Each server's startup failure is logged and
    silently skipped; one bad server doesn't poison the run.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        return None
    configs = [
        (srv.name, srv.command, srv.startup_timeout_s, srv.call_timeout_s)
        for srv in cfg.mcp.servers
        if srv.enabled
    ]
    if not configs:
        return None
    return MCPManager.start(configs, logger=lambda m: print(m, file=sys.stderr))


def _write_run_manifest(
    layout: RunLayout,
    *,
    run_id: str,
    user_task: str,
    base_sha: str,
    base_branch: str,
    run_branch: str | None,
    cfg: Config,
) -> None:
    """Write the canonical manifest.json for a run.

    This is the only thing that reads/writes ``layout.manifest_path``.
    Format is JSON for the same reason logs.jsonl is JSON: trivially
    grep-able from a shell and easy to consume from any language. The
    on-disk shape is *liquid* until 1.0 - bump ``version`` only when
    the new shape genuinely improves a downstream consumer.
    """
    manifest: dict[str, Any] = {
        "version": 1,
        "agent6_version": __version__,
        "run_id": run_id,
        "start_ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        "user_task": user_task[:4000],
        "base_sha": base_sha,
        "base_branch": base_branch,
        "run_branch": run_branch,
        "models": {
            "worker": {
                "provider": cfg.models.worker.provider,
                "model": cfg.models.worker.model,
            },
            "reviewer": {
                "provider": cfg.models.reviewer.provider,
                "model": cfg.models.reviewer.model,
            },
        },
        "workflow": {
            "critic": cfg.workflow.critic,
            "revise_prompt": cfg.workflow.revise_prompt,
        },
    }
    layout.manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _cmd_run(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path,
    task: str,
    *,
    run_id: str = "",
    interactive: bool = False,
    mode: Literal["run", "plan"] = "run",
) -> int:
    """Single-loop agent: one provider, one LLM driving via tool
    calls over the audited tool surface, deterministic harness (jail +
    budget + verify timeout + DAG curator for persistence/resume).
    Sole ``agent6 run`` path.

    When ``mode="plan"`` the same harness drives a planning
    pass instead of an execution pass: planning system prompt,
    edit-tools filtered out, ``finish_planning`` instead of
    ``finish_run``, no auto-commit. The plan markdown lands at
    ``<run-dir>/plan.md`` and is consumed by ``agent6 run --from-plan``.
    """
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    # Resolve @path references in the task string before the
    # workflow ever sees it. Lets the user write "fix the bug in @src/x.py
    # described in @notes.md" and have those files inlined verbatim.
    task = _expand_task_file_refs(task, Path.cwd())

    env = detect()
    try:
        selected_profile = select_profile(cfg.sandbox.profile, env)
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    missing = _check_provider_env_vars(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2

    # Git pre-flight (verify identity, ignore .agent6/).
    # The auto-commit-on-verify-pass behaviour requires a clean working tree,
    # so the same git assumptions apply. Skipping these left first-time runs
    # crashing on dirty-tree or missing-identity errors deep into a paid run.
    cwd = Path.cwd()
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        coauthor=cfg.git.commit.coauthor,
    )
    try:
        verify_git_identity(cwd, identity)
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Capture base sha + branch BEFORE we (optionally) cut a run branch
    # so `agent6 diff <run-id>` knows where the run started.
    try:
        pre_status = git_status(cwd)
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    base_sha = pre_status.head_sha
    base_branch = pre_status.branch

    # Layout: standard run-dir scaffolding for transcripts + logs.
    effective_run_id = run_id or new_friendly_id()
    layout = RunLayout(root=cwd, run_id=effective_run_id)
    _ensure_agent6_gitignored(cwd, identity=identity)
    layout.ensure()

    # Optionally cut a fresh branch for the run so the human can later
    # `git diff <base_branch>..agent6/...` or just delete the branch to
    # discard everything the agent did. Skipped silently when disabled.
    run_branch: str | None = None
    if cfg.git.branch_per_run:
        run_branch = make_run_branch_name(task_slug=slugify(task))
        try:
            create_branch(cwd, run_branch)
        except GitError as exc:
            print(f"ERROR: could not cut run branch {run_branch}: {exc}", file=sys.stderr)
            return 2

    # Write the run manifest. This is the canonical record of where the
    # run started (base_sha + base_branch), which model+provider drove
    # it, and the user_task it was given. `agent6 diff <run-id>` and
    # any future tooling that wants to reproduce a run reads from here.
    _write_run_manifest(
        layout,
        run_id=effective_run_id,
        user_task=task,
        base_sha=base_sha,
        base_branch=base_branch,
        run_branch=run_branch,
        cfg=cfg,
    )

    transcript_sink = TranscriptSink(layout.transcripts_dir)
    events = EventSink(layout.logs_path)

    egress_broker, egress_sock_dir, egress_err = _maybe_start_egress(cfg, selected_profile)
    if egress_err is not None:
        print(f"REFUSING: {egress_err}", file=sys.stderr)
        return 2
    if egress_broker is not None:
        print(
            f"[agent6] provider-only egress: confined to host network "
            f"namespace via broker pid {egress_broker.pid}",
            file=sys.stderr,
        )

    landlock_err = _maybe_apply_agent_landlock(cfg, selected_profile, env)
    if landlock_err is not None:
        print(f"REFUSING: {landlock_err}", file=sys.stderr)
        return 2

    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )

    # Workflow uses ONE provider (worker role) for everything. No
    # critic/triage/planner/reviewer/escalation cascade.
    worker_inner = _build_role_provider(
        cfg, "worker", transcript_sink=transcript_sink, budget=budget
    )
    rm_worker = cfg.models.worker
    # Enable SSE streaming when stderr is a TTY (covers TUI
    # and interactive shell use). Bench/CI runs pipe stderr, so they
    # stay on the audited non-streaming code path UNLESS the operator
    # sets AGENT6_FORCE_STREAM=1 — the Kimi/OpenRouter bench needs
    # streaming on because the gateway emits SSE keep-alive comment
    # heartbeats during long requests, which corrupt the non-streaming
    # response body (resp.json() blows up with JSONDecodeError).
    stream_text = sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1"
    provider: Provider = _InstrumentedProvider(
        inner=worker_inner,
        role="worker",
        model=rm_worker.model,
        provider_name=rm_worker.provider,
        events=events,
        budget=budget,
        stream_text=stream_text,
    )

    critic_provider = _build_critic_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )
    prompt_reviser_provider = _build_prompt_reviser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )
    summariser_provider = _build_summariser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )

    # Spawn the curator + connect a GraphClient so the agent
    # has access to the DAG-as-tool surface.
    #
    # AF_UNIX paths have a 108-char limit (Linux sun_path), which
    # bench setups with long BENCH_ROOT (and any future overlay-mount
    # paths) blew through. Bind the socket under a short /tmp dir and
    # leave a symlink under run_dir for observability. Cleaned up in
    # the finally block. See bench/improvement_plan.md audit cross-cutting.
    sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
    sock_path = sock_tmpdir / "curator.sock"
    sock_link = layout.run_dir / "curator.sock"
    with contextlib.suppress(FileNotFoundError):
        sock_link.unlink()
    sock_link.symlink_to(sock_path)
    curator_proc = spawn_curator(cwd, effective_run_id, sock_path)
    print(f"[agent6] run id: {effective_run_id}", file=sys.stderr)

    # Spawn any configured MCP servers BEFORE the workflow
    # starts so their tools are visible from iteration 1. The manager
    # owns its subprocesses; we close it in the finally block.
    mcp_manager = _start_mcp_manager_if_enabled(cfg)

    # audit finding: install the same steering SIGINT
    # handler installed in `_cmd_run`, so mid-run Ctrl-C drops a steering prompt
    # rather than aborting immediately. Double-Ctrl-C within 2s still
    # raises KeyboardInterrupt for the hard-abort path below.
    steer_state = _install_steer_sigint(events)

    result = None
    interrupted = False
    dispatcher: ToolDispatcher | None = None
    try:
        with GraphClient(sock_path) as graph_client:
            dispatcher = ToolDispatcher(
                root=cwd,
                config=cfg,
                sandbox_profile=selected_profile,
                approver=_default_stdin_approver,
                events=events,
                graph_client=graph_client,
                run_root_node_id=None,  # Workflow seeds the root + calls set_run_root_node_id
                mcp_manager=mcp_manager,
            )
            wf = Workflow(
                root=cwd,
                config=cfg,
                provider=provider,
                dispatcher=dispatcher,
                logger=print,
                events=events,
                graph_client=graph_client,
                steer_requested=steer_state.requested,
                steer_clear=steer_state.clear,
                steer_prompt=steer_state.prompt,
                budget=budget,
                resume_state_path=layout.run_dir / "loop_state.json",
                mode=mode,
                plan_output_path=(layout.run_dir / "plan.md" if mode == "plan" else None),
                after_auto_commit=(
                    _build_repl_hook(
                        cwd,
                        budget,
                        run_id=effective_run_id,
                        mcp_manager=mcp_manager,
                    )
                    if interactive and mode == "run"
                    else (lambda _i, _s: "continue")
                ),
                critic_provider=critic_provider,
                critic_mode=cfg.workflow.critic,
                critic_period=cfg.workflow.critic_period,
                prompt_reviser_provider=prompt_reviser_provider,
                revise_prompt=cfg.workflow.revise_prompt,
                temperature=cfg.models.worker.temperature,
                critic_temperature=cfg.models.reviewer.temperature,
                prompt_reviser_temperature=cfg.models.reviewer.temperature,
                prompt_revision_selector=(
                    _select_revised_prompt if cfg.workflow.revise_prompt == "interactive" else None
                ),
                summariser_provider=summariser_provider,
            )
            try:
                result = wf.run(task)
            except KeyboardInterrupt:
                interrupted = True
                print("\n[agent6] run interrupted", file=sys.stderr)
    finally:
        curator_proc.terminate()
        try:
            curator_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            curator_proc.kill()
        steer_state.restore()
        # Clean up the /tmp socket dir + symlink under run_dir.
        with contextlib.suppress(FileNotFoundError):
            sock_link.unlink()
        shutil.rmtree(sock_tmpdir, ignore_errors=True)
        if dispatcher is not None:
            dispatcher.close()
        if mcp_manager is not None:
            mcp_manager.close()
        _stop_egress(egress_broker, egress_sock_dir)

    if interrupted:
        return 130
    if result is None:
        return 1

    print()
    print(
        f"[agent6] result: completed={result.completed} reason={result.reason} "
        f"iterations={result.iterations} tool_calls={result.tool_calls}"
    )
    print(f"  summary: {result.summary[:500]}")
    print()
    print(budget.format_summary())
    _fire_notify_hook(
        cfg.notify,
        run_id=layout.run_id,
        run_dir=layout.run_dir,
        ok=result.completed,
        reason=result.reason,
    )
    return 0 if result.completed else 1


def _fire_notify_hook(
    notify: NotifyConfig,
    *,
    run_id: str,
    run_dir: Path,
    ok: bool,
    reason: str,
) -> None:
    """Run the operator-configured post-completion hook.

    The argv comes from `[notify].on_complete` in agent6.toml — operator-
    controlled, not LLM-controlled — so it does not go through the jail.
    Failures are logged to stderr and do not change the agent6 exit code.
    """
    if not notify.on_complete:
        return
    env = dict(os.environ)
    env["AGENT6_RUN_ID"] = run_id
    env["AGENT6_RUN_OK"] = "1" if ok else "0"
    env["AGENT6_RUN_REASON"] = reason
    env["AGENT6_RUN_DIR"] = str(run_dir)
    try:
        subprocess.run(
            list(notify.on_complete),
            env=env,
            timeout=notify.timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[agent6] notify.on_complete failed: {exc}", file=sys.stderr)


def _resolve_plan_run_id(run_id: str) -> str | None:
    """Resolve a (possibly prefix) run-id under .agent6/runs/.

    Prints an error and returns None on failure. Used by ``run --from-plan``,
    ``plan --show``, and ``plan --edit``.
    """
    runs_dir = Path.cwd() / ".agent6" / "runs"
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None
    plan = runs_dir / resolved / "plan.md"
    if not plan.is_file():
        print(
            f"ERROR: {resolved} has no plan.md (was it created with `agent6 plan`?)",
            file=sys.stderr,
        )
        return None
    return resolved


def _cmd_plan_show(run_id: str) -> int:
    """Print a planning run's plan.md to stdout."""
    resolved = _resolve_plan_run_id(run_id)
    if resolved is None:
        return 2
    plan = Path.cwd() / ".agent6" / "runs" / resolved / "plan.md"
    sys.stdout.write(plan.read_text(encoding="utf-8"))
    return 0


def _cmd_plan_edit(run_id: str) -> int:
    """Open a planning run's plan.md in $EDITOR (default: vi).

    Operator-controlled argv (the editor name + the resolved plan path),
    not LLM-controlled, so direct subprocess.run is allowed.
    """
    resolved = _resolve_plan_run_id(run_id)
    if resolved is None:
        return 2
    plan = Path.cwd() / ".agent6" / "runs" / resolved / "plan.md"
    editor = os.environ.get("EDITOR", "vi")
    try:
        result = subprocess.run([editor, str(plan)], check=False)
    except OSError as exc:
        print(f"ERROR: failed to spawn editor {editor!r}: {exc}", file=sys.stderr)
        return 1
    return result.returncode


def _most_recent_run_id(runs_dir: Path) -> str | None:
    """Return the directory name (= run id) of the most recently mtime'd run.

    Used by `agent6 watch` (no arg), `agent6 run --continue`, and the
    history-graph subcommand. Returns None outside an initialised workspace
    (no `.agent6/runs/`) or when the directory exists but is empty.
    """
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].name


def _cmd_watch(run_id: str, *, plain: bool = False, since: int = 0) -> int:  # noqa: PLR0911
    """Read-only live view of a run directory.

    Default is the textual TUI viewer. ``--plain`` switches to a no-deps
    line tail of ``events.jsonl``; useful in headless terminals
    or when ``textual`` isn't installed.
    """
    runs_dir = Path.cwd() / ".agent6" / "runs"
    if run_id:
        try:
            resolved = resolve_run_id(runs_dir, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        target = runs_dir / resolved
    else:
        if not runs_dir.is_dir():
            print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
            return 2
        candidates = sorted(
            (p for p in runs_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"ERROR: no runs found under {runs_dir}", file=sys.stderr)
            return 2
        target = candidates[0]
        print(f"[agent6] watching most recent run: {target.name}", file=sys.stderr)
    if not target.is_dir():
        print(f"ERROR: no such run dir: {target}", file=sys.stderr)
        return 2
    if plain:
        return _cmd_watch_plain(target, since=since)
    try:
        from agent6.ui.tui import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "HINT: pass --plain for a no-deps text tail of events.jsonl.",
            file=sys.stderr,
        )
        return 3
    return run_tui(target)


def _format_plain_event(line: str, *, run_start_ts: float | None) -> str:
    """Pretty-print one events.jsonl line as `<elapsed> <type> key=val ...`.

    Falls back to the raw line on parse error so a corrupt event doesn't
    abort the tail. ``run_start_ts`` is the wall-clock timestamp of the
    earliest event seen so far; used to render relative elapsed seconds.
    """
    raw = line.rstrip("\n")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(obj, dict):
        return raw
    ts = obj.get("ts")
    event = obj.get("event") or obj.get("type") or "?"
    if isinstance(ts, (int, float)) and run_start_ts is not None:
        elapsed = max(0.0, float(ts) - run_start_ts)
        ts_str = f"+{elapsed:7.1f}s"
    else:
        ts_str = "        "
    skip = {"ts", "event", "type", "run_id"}
    pairs: list[str] = []
    for k, v in obj.items():
        if k in skip:
            continue
        if isinstance(v, str):
            shown = v if len(v) <= 80 else v[:77] + "..."
            pairs.append(f"{k}={shown!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            pairs.append(f"{k}={v}")
        else:
            blob = json.dumps(v, default=str)
            shown = blob if len(blob) <= 80 else blob[:77] + "..."
            pairs.append(f"{k}={shown}")
    return f"{ts_str} {event:30s} {' '.join(pairs)}"


def _cmd_watch_plain(target: Path, *, since: int) -> int:  # noqa: PLR0912, PLR0915
    """Tail ``events.jsonl`` line-by-line with no extra deps.

    Polls the file with 0.25s sleeps; rotates when the inode changes.
    Pretty-prints each event with the type and key fields. Returns 0 on
    EOF (run dir gone) or KeyboardInterrupt.
    """
    events_path = target / "events.jsonl"
    if not events_path.is_file():
        print(f"ERROR: no events.jsonl in {target}", file=sys.stderr)
        return 2

    # Read the first event for the elapsed-time anchor.
    run_start_ts: float | None = None
    try:
        with events_path.open(encoding="utf-8") as fh:
            first = fh.readline()
        if first:
            obj0 = json.loads(first)
            if isinstance(obj0, dict) and isinstance(obj0.get("ts"), (int, float)):
                run_start_ts = float(obj0["ts"])
    except (OSError, json.JSONDecodeError):
        run_start_ts = None

    print(
        f"[agent6] tailing {events_path} (--plain). Ctrl-C to exit.",
        file=sys.stderr,
    )

    try:
        fh = events_path.open(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot open {events_path}: {exc}", file=sys.stderr)
        return 2

    try:
        if since > 0:
            # Replay the last `since` lines before following.
            try:
                lines = fh.readlines()
            except OSError as exc:
                print(f"ERROR: read failed: {exc}", file=sys.stderr)
                return 2
            for line in lines[-since:]:
                print(_format_plain_event(line, run_start_ts=run_start_ts))
        else:
            # Seek to end; only show new events going forward.
            fh.seek(0, 2)
        try:
            current_ino = events_path.stat().st_ino
        except OSError:
            current_ino = -1
        while True:
            line = fh.readline()
            if line:
                print(_format_plain_event(line, run_start_ts=run_start_ts), flush=True)
                continue
            # No new data: check for rotation and sleep briefly.
            try:
                new_ino = events_path.stat().st_ino
            except OSError:
                time.sleep(0.5)
                continue
            if new_ino != current_ino:
                with contextlib.suppress(OSError):
                    fh.close()
                try:
                    fh = events_path.open(encoding="utf-8")
                except OSError:
                    time.sleep(0.5)
                    continue
                current_ino = new_ino
                continue
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[agent6] watch --plain: stopped.", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            fh.close()


def _cmd_resume(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path, run_id: str, *, force: bool
) -> int:
    """Resume a paused/crashed run from its snapshot.

    Mirrors ``_cmd_run`` setup but uses the existing run id, refuses
    if no ``loop_state.json`` snapshot exists, and calls ``wf.resume()``
    instead of ``wf.run(task)``. A safety check (``compute_resume_diff``)
    refuses on snapshot-missing unless ``--force-resume`` is passed.

    NOTE: token budget on resume is a FRESH ceiling, not a continuation
    of the prior run's accounting. Each ``agent6 resume`` invocation
    starts at 0 tokens against ``[budget].max_input_tokens`` /
    ``max_output_tokens``. This is by design - the budget is a per-
    invocation runaway-cost circuit breaker.
    """
    runs_dir = Path.cwd() / ".agent6" / "runs"
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_id = resolved
    cwd = Path.cwd()
    layout = RunLayout(root=cwd, run_id=run_id)
    if not layout.run_dir.is_dir():
        print(f"ERROR: no such run dir: {layout.run_dir}", file=sys.stderr)
        return 2

    snapshot_path = layout.run_dir / "loop_state.json"
    if not snapshot_path.is_file():
        print(
            f"ERROR: no resume snapshot at {snapshot_path}; nothing to resume.",
            file=sys.stderr,
        )
        return 2

    # Safety check: refuse on snapshot-commit divergence unless --force-resume.
    curator = GraphCurator(layout)
    diff = curator.compute_resume_diff(run_id, cwd)
    print(f"Run: {run_id}")
    print(f"  snapshot head: {diff.snapshot_head}")
    print(f"  current head:  {diff.current_head}")
    if diff.committed_delta.files:
        print(f"  committed delta: {len(diff.committed_delta.files)} file(s)")
    if diff.uncommitted_diff:
        print(f"  uncommitted diverged: {len(diff.uncommitted_diff)} file(s)")
    if diff.snapshot_missing:
        print(f"\nGUARD: {diff.guard_summary}", file=sys.stderr)
        if not force:
            print(
                "REFUSING to resume. Re-run with --force-resume to override.",
                file=sys.stderr,
            )
            return 1

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    env = detect()
    try:
        selected_profile = select_profile(cfg.sandbox.profile, env)
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    missing = _check_provider_env_vars(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2

    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        coauthor=cfg.git.commit.coauthor,
    )
    try:
        verify_git_identity(cwd, identity)
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _ensure_agent6_gitignored(cwd, identity=identity)

    transcript_sink = TranscriptSink(layout.transcripts_dir)
    events = EventSink(layout.logs_path)

    egress_broker, egress_sock_dir, egress_err = _maybe_start_egress(cfg, selected_profile)
    if egress_err is not None:
        print(f"REFUSING: {egress_err}", file=sys.stderr)
        return 2
    if egress_broker is not None:
        print(
            f"[agent6] provider-only egress: confined to host network "
            f"namespace via broker pid {egress_broker.pid}",
            file=sys.stderr,
        )

    landlock_err = _maybe_apply_agent_landlock(cfg, selected_profile, env)
    if landlock_err is not None:
        print(f"REFUSING: {landlock_err}", file=sys.stderr)
        return 2

    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )

    worker_inner = _build_role_provider(
        cfg, "worker", transcript_sink=transcript_sink, budget=budget
    )
    rm_worker = cfg.models.worker
    # Streaming gated on stderr TTY (matches _cmd_run);
    # AGENT6_FORCE_STREAM=1 forces it on for bench/CI.
    stream_text = sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1"
    provider: Provider = _InstrumentedProvider(
        inner=worker_inner,
        role="worker",
        model=rm_worker.model,
        provider_name=rm_worker.provider,
        events=events,
        budget=budget,
        stream_text=stream_text,
    )

    critic_provider = _build_critic_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )
    summariser_provider = _build_summariser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )

    sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
    sock_path = sock_tmpdir / "curator.sock"
    sock_link = layout.run_dir / "curator.sock"
    with contextlib.suppress(FileNotFoundError):
        sock_link.unlink()
    sock_link.symlink_to(sock_path)
    curator_proc = spawn_curator(cwd, run_id, sock_path)
    print(f"[agent6] resume run id: {run_id}", file=sys.stderr)

    mcp_manager = _start_mcp_manager_if_enabled(cfg)

    steer_state = _install_steer_sigint(events)

    result = None
    interrupted = False
    dispatcher: ToolDispatcher | None = None
    try:
        with GraphClient(sock_path) as graph_client:
            dispatcher = ToolDispatcher(
                root=cwd,
                config=cfg,
                sandbox_profile=selected_profile,
                approver=_default_stdin_approver,
                events=events,
                graph_client=graph_client,
                run_root_node_id=None,
                mcp_manager=mcp_manager,
            )
            wf = Workflow(
                root=cwd,
                config=cfg,
                provider=provider,
                dispatcher=dispatcher,
                logger=print,
                events=events,
                graph_client=graph_client,
                steer_requested=steer_state.requested,
                steer_clear=steer_state.clear,
                steer_prompt=steer_state.prompt,
                budget=budget,
                resume_state_path=snapshot_path,
                critic_provider=critic_provider,
                critic_mode=cfg.workflow.critic,
                critic_period=cfg.workflow.critic_period,
                temperature=cfg.models.worker.temperature,
                critic_temperature=cfg.models.reviewer.temperature,
                summariser_provider=summariser_provider,
            )
            try:
                result = wf.resume()
            except ResumeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                interrupted = True
                print("\n[agent6] resume interrupted", file=sys.stderr)
    finally:
        curator_proc.terminate()
        try:
            curator_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            curator_proc.kill()
        steer_state.restore()
        with contextlib.suppress(FileNotFoundError):
            sock_link.unlink()
        shutil.rmtree(sock_tmpdir, ignore_errors=True)
        if dispatcher is not None:
            dispatcher.close()
        if mcp_manager is not None:
            mcp_manager.close()
        _stop_egress(egress_broker, egress_sock_dir)

    if interrupted:
        return 130
    if result is None:
        return 1

    print()
    print(
        f"[agent6] result: completed={result.completed} reason={result.reason} "
        f"iterations={result.iterations} tool_calls={result.tool_calls}"
    )
    print(f"  summary: {result.summary[:500]}")
    print()
    print(budget.format_summary())
    _fire_notify_hook(
        cfg.notify,
        run_id=layout.run_id,
        run_dir=layout.run_dir,
        ok=result.completed,
        reason=result.reason,
    )
    return 0 if result.completed else 1


# ---------------------------------------------------------------------------
# memory subcommands
# ---------------------------------------------------------------------------


def _cmd_memory_add(scope: MemoryScope, body: str) -> int:
    try:
        entry = memory_add(Path.cwd(), scope, body)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{entry.scope} {entry.id} created at {entry.created_at}")
    return 0


def _cmd_memory_list(scope: MemoryScope | None, *, include_invalidated: bool) -> int:
    try:
        entries = memory_list(Path.cwd(), scope)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print("(no memories)")
        return 0
    for e in entries:
        if not include_invalidated and not e.is_active:
            continue
        flag = "" if e.is_active else " [INVALIDATED]"
        print(f"[{e.scope}] {e.id} {e.created_at}{flag}")
        if not e.is_active and e.invalidation_reason:
            print(f"    invalidated_at: {e.invalidated_at}  reason: {e.invalidation_reason}")
        for line in e.body.splitlines():
            print(f"    {line}")
        print()
    return 0


def _cmd_memory_invalidate(memory_id: str, reason: str) -> int:
    try:
        entry = memory_invalidate(Path.cwd(), memory_id, reason)
    except Agent6MemoryError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"invalidated {entry.scope} {entry.id} at {entry.invalidated_at}")
    return 0


# ---------------------------------------------------------------------------
# history search
# ---------------------------------------------------------------------------


def _cmd_history_search(query: str, *, fixed: bool, run_id: str) -> int:
    rg = shutil.which("rg")
    if rg is None:
        print(
            "ERROR: `rg` (ripgrep) is required for `agent6 history search`. "
            "Install ripgrep (https://github.com/BurntSushi/ripgrep) and retry.",
            file=sys.stderr,
        )
        return 2
    runs_root = Path.cwd() / ".agent6" / "runs"
    if run_id:
        try:
            run_id = resolve_run_id(runs_root, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    target = runs_root / run_id if run_id else runs_root
    if not target.is_dir():
        print(f"ERROR: no such directory: {target}", file=sys.stderr)
        return 2
    argv: list[str] = [
        rg,
        "--color=never",
        "--with-filename",
        "--line-number",
    ]
    if fixed:
        argv.append("--fixed-strings")
    argv.extend(["--", query, str(target)])
    completed = subprocess.run(argv, check=False)
    # rg returns 1 if no matches; that's not an error for us.
    if completed.returncode in (0, 1):
        return completed.returncode
    return completed.returncode


def _cmd_history_graph(run_id: str) -> int:
    """Render the persisted TaskNode tree for a run as a DFS-ordered listing."""

    runs_dir = Path.cwd() / ".agent6" / "runs"
    if run_id:
        try:
            target_id = resolve_run_id(runs_dir, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        if not runs_dir.is_dir():
            print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
            return 2
        candidates = sorted(
            (p for p in runs_dir.iterdir() if p.is_dir() and (p / "graph").is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"ERROR: no runs with a graph under {runs_dir}", file=sys.stderr)
            return 2
        target_id = candidates[0].name
        print(f"[agent6] showing graph for most recent run: {target_id}", file=sys.stderr)

    layout = RunLayout(root=Path.cwd(), run_id=target_id)
    nodes = load_graph(layout)
    if not nodes:
        print(f"ERROR: run {target_id} has no persisted graph nodes", file=sys.stderr)
        return 2

    roots = sorted(
        (n for n in nodes.values() if n.parent_id is None),
        key=lambda n: n.created_at,
    )
    print(f"Run id: {target_id}")
    print()
    for root in roots:
        _print_node_dfs(root, nodes, depth=0)
    return 0


def _print_node_dfs(node: TaskNode, nodes: dict[str, TaskNode], *, depth: int) -> None:
    """Depth-first, left-to-right print of one TaskNode subtree."""

    indent = "  " * depth
    status = f"[{node.status}]"
    commit = f"  (commit: {node.commit_sha[:7]})" if node.commit_sha else ""
    print(f"{indent}{status} {node.title}{commit}")
    # Children are ordered by insertion (curator preserves order); walk them
    # left-to-right, recursing fully into each before moving to the next.
    for child_id in node.children:
        child = nodes.get(child_id)
        if child is None:
            print(f"{indent}  [MISSING] <child id {child_id} not found>")
            continue
        _print_node_dfs(child, nodes, depth=depth + 1)


def _cmd_init(*, force: bool, profile: str) -> int:
    return init_workspace(Path.cwd(), force=force, profile=profile)


def _cmd_diff(*, run_id: str, stat: bool, paths: tuple[str, ...]) -> int:  # noqa: PLR0911
    """Print the git diff a run produced (manifest.base_sha -> branch HEAD).

    Resolves the run id (or unique prefix; empty string means most-recent),
    reads ``manifest.json`` for ``base_sha`` and ``run_branch``, then shells
    out to ``git diff`` with operator-controlled argv (no LLM input).
    """
    cwd = Path.cwd()
    runs_dir = cwd / ".agent6" / "runs"
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return 2

    target_id = run_id
    if not target_id:
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}", file=sys.stderr)
            return 2
        target_id = latest
        print(f"[agent6] diffing most recent run: {target_id}", file=sys.stderr)
    else:
        try:
            target_id = resolve_run_id(runs_dir, target_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    layout = RunLayout(root=cwd, run_id=target_id)
    if not layout.manifest_path.is_file():
        print(
            f"ERROR: run {target_id} has no manifest.json "
            "(predates manifest support, or was killed before setup)",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest: {exc}", file=sys.stderr)
        return 2

    base_sha = str(manifest.get("base_sha") or "")
    run_branch = manifest.get("run_branch")
    if not base_sha:
        print("ERROR: manifest has no base_sha; nothing to diff against", file=sys.stderr)
        return 2

    head_ref = str(run_branch) if run_branch else "HEAD"
    argv: list[str] = ["git", "diff"]
    if stat:
        argv.append("--stat")
    argv.extend([f"{base_sha}..{head_ref}"])
    if paths:
        argv.append("--")
        argv.extend(paths)
    print(
        f"[agent6] {' '.join(argv)}  (base_branch={manifest.get('base_branch')!r})",
        file=sys.stderr,
    )
    proc = subprocess.run(argv, cwd=cwd, check=False)
    return proc.returncode


def _cmd_mcp_serve(config_path: Path) -> int:
    """Spawn an MCP stdio server against ``config_path``'s
    workspace. Thin wrapper so dispatch stays uniform with the other
    ``_cmd_*`` helpers."""
    return _mcp_run_server(config_path)


def _cmd_review(  # noqa: PLR0911
    config_path: Path,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
    model_override: str = "",
) -> int:
    """Print a freeform code review of a diff to stdout. Read-only; no jail."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    err = _check_provider_env_vars(cfg)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    root = Path.cwd()
    git = shutil.which("git")
    if git is None:
        print("ERROR: git not found on PATH.", file=sys.stderr)
        return 2

    if base:
        diff_args = [git, "diff", f"{base}..{head}"]
    else:
        # Working tree vs HEAD, including untracked files (intent-to-add).
        subprocess.run([git, "add", "-N", "--", "."], cwd=root, check=False)
        diff_args = [git, "diff", "HEAD"]
    if paths:
        diff_args.extend(["--", *paths])
    diff_proc = subprocess.run(diff_args, cwd=root, capture_output=True, text=True, check=False)
    if diff_proc.returncode != 0:
        print(f"ERROR: git diff failed: {diff_proc.stderr.strip()}", file=sys.stderr)
        return 2
    diff = diff_proc.stdout
    if not diff.strip():
        print("(no diff to review)", file=sys.stderr)
        return 0

    log_proc = subprocess.run(
        [git, "log", "-n", "10", "--oneline"], cwd=root, capture_output=True, text=True, check=False
    )
    recent_log = log_proc.stdout if log_proc.returncode == 0 else ""

    agents_md_path = root / "AGENTS.md"
    agents_md = agents_md_path.read_text(encoding="utf-8") if agents_md_path.is_file() else ""

    # Reviewer-only: route the "reviewer" role per [models.reviewer]. Budget
    # is per-invocation since this command is a one-shot.
    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )
    layout_root = root / ".agent6" / "reviews"
    layout_root.mkdir(parents=True, exist_ok=True)
    transcript_sink = TranscriptSink(layout_root)

    try:
        reviewer = _build_role_provider(
            cfg,
            "reviewer",
            transcript_sink=transcript_sink,
            budget=budget,
            model_override=model_override,
        )
    except ProviderError as exc:
        print(f"ERROR: provider init failed: {exc}", file=sys.stderr)
        return 2

    label = (
        "working tree vs HEAD"
        if not base
        else f"{base}..{head}" + (f" -- {' '.join(paths)}" if paths else "")
    )
    print(f"[agent6] reviewing: {label}", file=sys.stderr)
    try:
        text = run_review(
            reviewer,
            diff=diff,
            agents_md=agents_md,
            recent_log=recent_log,
        )
    except CodeReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except BudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3

    print(text)
    print(budget.format_summary(), file=sys.stderr)
    return 0
