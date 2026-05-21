# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""agent6 command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argcomplete

from agent6 import __version__
from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import AnthropicProviderEntry, Config, ConfigError, RoleName, load_config
from agent6.config_fix import FixKind, apply_fixes, propose_fixes
from agent6.detect import detect, select_profile
from agent6.events import EventSink, UserInputSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    commit_paths,
    is_git_repo,
    verify_git_identity,
)
from agent6.graph.client import GraphClient, spawn_curator
from agent6.graph.curator import GraphCurator
from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, load_graph
from agent6.init import init_workspace
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
from agent6.models import Plan
from agent6.providers import (
    AnthropicProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
)
from agent6.run_id import RunIdError, new_friendly_id, resolve_run_id
from agent6.sandbox import (
    JailUnavailableError,
    LandlockNotSupportedError,
    apply_agent_landlock,
    landlock_abi,
    run_in_jail,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import JailPolicy, SandboxReport
from agent6.ui.approval import read_answer, tui_is_live
from agent6.workflows import (
    ImplementWorkflow,
    ManifestError,
    PlanModeError,
    PlanModeQuestionsPending,
    PlanModeWorkflow,
    WorkflowError,
    format_plan,
    read_answers_file,
    read_manifest,
)
from agent6.workflows._context import load_repo_summary
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
            transcript_sink=transcript_sink,
            budget=budget,
        )
    return OpenAIProvider.from_env(
        env_var=entry.api_key_env,
        model=model,
        base_url=entry.base_url,
        extra_headers=entry.extra_headers,
        transcript_sink=transcript_sink,
        budget=budget,
    )


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

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.events.emit(
            "role.call",
            role=self.role,
            model=self.model,
            provider=self.provider_name,
        )
        try:
            resp = self.inner.call(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
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
        self.events.emit(
            "budget.update",
            input_total=snap["input_total"],
            output_total=snap["output_total"],
            input_cap=snap["max_input_tokens"],
            output_cap=snap["max_output_tokens"],
        )
        return resp


def _instrument(
    cfg: Config,
    role: RoleName,
    provider: Provider,
    *,
    events: EventSink,
    budget: BudgetTracker,
    model_override: str = "",
) -> Provider:
    rm = cfg.models.all()[role]
    return _InstrumentedProvider(
        inner=provider,
        role=role,
        model=model_override or rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


def _default_stdin_approver(prompt: str) -> bool:
    """Plain TTY fallback for tool approval (used when no TUI is live)."""
    try:
        ans = input(f"{prompt} [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in {"y", "yes"}


def _make_tui_approver(
    layout: RunLayout,
    events: EventSink,
    *,
    fallback: Callable[[str], bool],
    user_inputs: UserInputSink | None = None,
) -> Callable[[str], bool]:
    counter = {"n": 0}

    def approver(prompt: str) -> bool:
        counter["n"] += 1
        prompt_id = f"a{counter['n']:03d}"
        events.emit("approval.prompt", id=prompt_id, prompt=prompt)
        if tui_is_live(layout.run_dir):
            answer = read_answer(layout.run_dir, prompt_id)
            if answer is None:
                approved = fallback(prompt)
                source = "stdin-fallback"
            else:
                approved = answer
                source = "tui"
        else:
            approved = fallback(prompt)
            source = "stdin"
        events.emit("approval.answer", id=prompt_id, approved=approved, source=source)
        if user_inputs is not None:
            user_inputs.record(
                kind="tool_approval",
                prompt=prompt,
                answer="yes" if approved else "no",
                source=source,
                prompt_id=prompt_id,
            )
        return approved

    return approver


def _maybe_spawn_tui(layout: RunLayout, *, enabled: bool) -> subprocess.Popen[bytes] | None:
    """Spawn `python -m agent6.ui --watch <run-dir>` as a subprocess.

    Returns None (and prints a hint to stderr) when:
    - --no-tui was passed,
    - stdout is not a TTY,
    - the `textual` optional dep is not installed.

    The TUI only reads files; the workflow keeps writing JSONL events
    regardless. Tearing down the TUI does not affect the workflow.
    """
    if not enabled:
        return None
    if not sys.stdout.isatty():
        return None
    try:
        if importlib.util.find_spec("textual") is None:
            print(
                "[agent6] TUI disabled: install 'agent6[tui]' for a live dashboard.",
                file=sys.stderr,
            )
            return None
    except Exception:
        return None
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "agent6.ui", "--watch", str(layout.run_dir)],
        )
    except OSError as exc:
        print(f"[agent6] failed to spawn TUI: {exc}", file=sys.stderr)
        return None


@dataclass
class _SteerState:
    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]


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

    run_p = sub.add_parser("run", help="Run the default workflow on a task.")
    run_p.add_argument("task", help="Task description (in quotes).")
    run_p.add_argument(
        "--yes", action="store_true", help="Auto-confirm the plan (no interactive prompt)."
    )
    run_p.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    run_p.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the auto-launched Textual dashboard (default: on when stdout is a TTY).",
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

    resume_p = sub.add_parser("resume", help="Inspect a paused run and refuse if state diverged.")
    resume_p.add_argument("run_id", help="Run id under .agent6/runs/.")
    resume_p.add_argument(
        "--force-resume",
        action="store_true",
        help="Resume even if snapshot commit is missing or worktree has diverged.",
    )

    plan_p = sub.add_parser(
        "plan",
        help="Plan management: create, show, revise, or edit a persisted plan.",
    )
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True)

    plan_new = plan_sub.add_parser(
        "new",
        help="Interactive plan-only mode; persists frozen plan into the task graph.",
    )
    plan_new.add_argument("task", help="Task description (in quotes).")
    plan_new.add_argument(
        "--run-id",
        default="",
        help="Explicit run id for the persisted plan (default: generate).",
    )
    plan_new.add_argument(
        "--questions-file",
        default="",
        help=(
            "Path to write open-question JSON stubs to when the critic asks. "
            "Exits with code 4 after writing so you can answer offline."
        ),
    )
    plan_new.add_argument(
        "--answers-file",
        default="",
        help="Path to a previously written questions-file with 'answer' fields filled in.",
    )

    plan_show = plan_sub.add_parser(
        "show",
        help="Print the persisted plan for a run (default: most recent).",
    )
    plan_show.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id under .agent6/runs/ (omit for the most recent run).",
    )

    plan_revise = plan_sub.add_parser(
        "revise",
        help="Apply free-form feedback to a persisted plan, producing a new run.",
    )
    plan_revise.add_argument("feedback", help="Free-form feedback for the planner.")
    plan_revise.add_argument(
        "--run-id",
        default="",
        help="Run id of the plan to revise (default: most recent).",
    )
    plan_revise.add_argument(
        "--questions-file",
        default="",
        help=(
            "Path to write open-question JSON stubs to when the critic asks. "
            "Exits with code 4 after writing so you can answer offline."
        ),
    )
    plan_revise.add_argument(
        "--answers-file",
        default="",
        help="Path to a previously written questions-file with 'answer' fields filled in.",
    )

    plan_edit = plan_sub.add_parser(
        "edit",
        help="Edit a persisted plan as JSON in $EDITOR, producing a new run.",
    )
    plan_edit.add_argument(
        "--run-id",
        default="",
        help="Run id of the plan to edit (default: most recent).",
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

    # Shell tab-completion. argcomplete is a hard dependency; the call is a
    # no-op unless the shell sourced its completion script for this binary
    # (see `agent6 --help` and the README for activation instructions).
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(
            args.config,
            args.task,
            auto_confirm=args.yes,
            run_id=args.run_id,
            tui=not args.no_tui,
        )
    if args.command == "watch":
        return _cmd_watch(args.run_id)
    if args.command == "resume":
        return _cmd_resume(args.run_id, force=args.force_resume)
    if args.command == "plan":
        if args.plan_command == "new":
            return _cmd_plan_new(
                args.config,
                args.task,
                run_id=args.run_id,
                questions_file=args.questions_file,
                answers_file=args.answers_file,
            )
        if args.plan_command == "show":
            return _cmd_plan_show(args.run_id)
        if args.plan_command == "revise":
            return _cmd_plan_revise(
                args.config,
                args.feedback,
                run_id=args.run_id,
                questions_file=args.questions_file,
                answers_file=args.answers_file,
            )
        if args.plan_command == "edit":
            return _cmd_plan_edit(args.config, run_id=args.run_id)
    if args.command == "check-config":
        return _cmd_check_config(args.config, fix=args.fix, assume_yes=args.yes)
    if args.command == "check-sandbox":
        return _cmd_check_sandbox()
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
        return _cmd_init(force=args.force)
    if args.command == "review":
        return _cmd_review(
            args.config,
            base=args.base,
            head=args.head,
            paths=tuple(args.paths),
            model_override=args.model,
        )
    parser.error("unknown command")  # pragma: no cover
    return 2


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


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
    print(f"  workflow.default = {cfg.workflow.default}")
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


def _cmd_run(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path,
    task: str,
    *,
    auto_confirm: bool,
    run_id: str = "",
    tui: bool = True,
) -> int:
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
    print(
        f"[agent6] sandbox profile: {selected_profile} "
        f"(requested={cfg.sandbox.profile}, userns={'yes' if env.userns_supported else 'no'})",
        file=sys.stderr,
    )

    # Apply Landlock to the agent process itself. Refuse-if-available-but-failed:
    # if the kernel supports Landlock at all and our apply call raises anything
    # other than LandlockNotSupportedError, that's a real failure and we abort.
    # Silent degradation is exactly the security-theater anti-pattern we reject.
    #
    # Exception: under the 'strict' sandbox profile, every subprocess runs inside
    # its own user+mount+pid+net namespace with its own pivot_root, Landlock, and
    # seccomp policy — a strictly stronger confinement than parent-process
    # Landlock would provide. Applying Landlock to the agent ALSO breaks the jail
    # on recent kernels (≥ ABI 7), because the kernel correctly blocks mount(2)
    # and pivot_root(2) inside Landlocked processes to prevent the obvious
    # confinement-escape via remounting. So we only apply parent Landlock for the
    # 'hardened' profile, where the jail shares the host mount/pid/net ns and
    # parent-process FS confinement is the load-bearing layer.
    if env.kernel.supports_landlock_fs and selected_profile == "hardened":
        try:
            cwd = Path.cwd().resolve()
            tmp = Path("/tmp")  # noqa: S108 — intentional sandbox allowlist entry
            # /dev/{null,zero,urandom,random,tty} are needed by Python stdlib
            # subprocess (devnull), ssl/random, tty detection. Listing them
            # individually rather than allowing all of /dev keeps the surface
            # minimal — we deny /dev/mem, /dev/kmem, /dev/sda*, etc. by default.
            dev_files: tuple[Path, ...] = tuple(
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
            # /run is needed for systemd-resolved's stub resolver
            # (/etc/resolv.conf → /run/systemd/resolve/stub-resolv.conf).
            # Without read access here, glibc's NSS DNS lookup fails with
            # "Temporary failure in name resolution" before the connect()
            # ever happens. Allow-read only.
            run_paths: tuple[Path, ...] = tuple(p for p in (Path("/run"),) if p.exists())
            # /proc is needed for the jail child to write its own uid_map /
            # gid_map / setgroups when entering a user namespace, and for
            # general libc bookkeeping (/proc/self/maps, /proc/sys/...).
            # Landlock applied above us would otherwise EACCES those writes
            # before unshare(2) ever runs.
            proc_paths: tuple[Path, ...] = (Path("/proc"),)
            apply_agent_landlock(
                read_paths=(
                    cwd,
                    Path.home(),
                    Path("/usr"),
                    Path("/etc"),
                    tmp,
                    *dev_files,
                    *run_paths,
                    *proc_paths,
                ),
                write_paths=(cwd, tmp, *dev_files, *proc_paths),
                tcp_connect_ports=(443,),
            )
        except LandlockNotSupportedError:
            print(
                "WARNING: Landlock not supported by this kernel; agent process is "
                "not FS/network-confined.",
                file=sys.stderr,
            )
        except OSError as exc:
            print(
                "REFUSING: kernel reports Landlock support but applying it to the "
                f"agent process failed: {exc}. This would silently weaken the "
                "agent's own confinement; aborting.",
                file=sys.stderr,
            )
            return 2

    err = _check_provider_env_vars(cfg)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    try:
        verify_git_identity(
            Path.cwd(),
            CommitIdentity(
                name=cfg.git.commit.name,
                email=cfg.git.commit.email,
                coauthor=cfg.git.commit.coauthor,
            ),
        )
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    effective_run_id = run_id or new_friendly_id()
    layout = RunLayout(root=Path.cwd(), run_id=effective_run_id)
    _ensure_agent6_gitignored(
        Path.cwd(),
        identity=CommitIdentity(
            name=cfg.git.commit.name,
            email=cfg.git.commit.email,
            coauthor=cfg.git.commit.coauthor,
        ),
    )
    layout.ensure()
    transcript_sink = TranscriptSink(layout.transcripts_dir)
    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )
    events = EventSink(layout.logs_path)
    user_inputs = UserInputSink(layout.user_inputs_path)

    try:
        planner = _instrument(
            cfg,
            "planner",
            _build_role_provider(cfg, "planner", transcript_sink=transcript_sink, budget=budget),
            events=events,
            budget=budget,
        )
        worker = _instrument(
            cfg,
            "worker",
            _build_role_provider(cfg, "worker", transcript_sink=transcript_sink, budget=budget),
            events=events,
            budget=budget,
        )
        reviewer = _instrument(
            cfg,
            "reviewer",
            _build_role_provider(cfg, "reviewer", transcript_sink=transcript_sink, budget=budget),
            events=events,
            budget=budget,
        )
        critic = _instrument(
            cfg,
            "critic",
            _build_role_provider(cfg, "critic", transcript_sink=transcript_sink, budget=budget),
            events=events,
            budget=budget,
        )
        # Worker escalation: on the worker's retry attempt, route to a stronger
        # model. We reuse the planner role's (provider, model) — planner is
        # typically opus-class, which is the right escalation curve from a
        # sonnet-class primary worker. Same connection class, separate event
        # role label so cost is attributed independently.
        rm_planner = cfg.models.planner
        escalation_inner = _build_role_provider(
            cfg, "planner", transcript_sink=transcript_sink, budget=budget
        )
        worker_escalation: Provider = _InstrumentedProvider(
            inner=escalation_inner,
            role="worker_escalation",
            model=rm_planner.model,
            provider_name=rm_planner.provider,
            events=events,
            budget=budget,
        )
        # Triage: reuse the summarizer's (provider, model) — both want a
        # cheap haiku-class model and decoupling them isn't worth a new
        # config field today. Instrumented under its own role label.
        rm_sum = cfg.models.summarizer
        triage_inner = _build_role_provider(
            cfg, "summarizer", transcript_sink=transcript_sink, budget=budget
        )
        triage_provider: Provider = _InstrumentedProvider(
            inner=triage_inner,
            role="triage",
            model=rm_sum.model,
            provider_name=rm_sum.provider,
            events=events,
            budget=budget,
        )
    except ProviderError as exc:
        print(f"ERROR: provider init failed: {exc}", file=sys.stderr)
        return 2

    approver = _make_tui_approver(
        layout, events, fallback=_default_stdin_approver, user_inputs=user_inputs
    )
    dispatcher = ToolDispatcher(
        root=Path.cwd(),
        config=cfg,
        sandbox_profile=selected_profile,
        approver=approver,
        events=events,
    )
    sock_path = layout.run_dir / "curator.sock"
    curator_proc = spawn_curator(Path.cwd(), effective_run_id, sock_path)
    print(f"[agent6] run id: {effective_run_id}", file=sys.stderr)
    tui_proc = _maybe_spawn_tui(layout, enabled=tui)
    steer_state = _install_steer_sigint(events)
    try:
        with GraphClient(sock_path) as graph_client:
            wf = ImplementWorkflow(
                root=Path.cwd(),
                config=cfg,
                planner=planner,
                worker=worker,
                reviewer=reviewer,
                critic=critic,
                worker_escalation=worker_escalation,
                triage=triage_provider,
                dispatcher=dispatcher,
                confirm_plan=_make_confirm(auto_confirm, user_inputs=user_inputs),
                graph_client=graph_client,
                events=events,
                user_inputs=user_inputs,
                steer_requested=steer_state.requested,
                steer_clear=steer_state.clear,
                steer_prompt=steer_state.prompt,
            )
            try:
                result = wf.run(task)
            except KeyboardInterrupt:
                print("\n[agent6] aborted by operator (double Ctrl-C)", file=sys.stderr)
                events.emit("run.aborted", reason="double_sigint")
                return 130
            except BudgetExceeded as exc:
                print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
                print(budget.format_summary(), file=sys.stderr)
                return 3
            except WorkflowError as exc:
                print(f"WORKFLOW ERROR: {exc}", file=sys.stderr)
                print(budget.format_summary(), file=sys.stderr)
                return 1
    finally:
        curator_proc.terminate()
        try:
            curator_proc.wait(timeout=3)
        except Exception:
            curator_proc.kill()
            curator_proc.wait()
        if tui_proc is not None:
            try:
                tui_proc.terminate()
                tui_proc.wait(timeout=3)
            except Exception:
                tui_proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    tui_proc.wait(timeout=3)
        steer_state.restore()

    print(f"\nBranch: {result.branch}")
    for s in result.steps:
        print(f"  [{s.status}] {s.title}  {s.commit_sha[:12]}")
    print()
    print(budget.format_summary())
    return 0 if result.all_passed else 1


def _cmd_watch(run_id: str) -> int:
    """Read-only TUI viewer for a run directory."""
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
    try:
        from agent6.ui.tui import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    return run_tui(target)


def _cmd_resume(run_id: str, *, force: bool) -> int:
    """Inspect a paused run; refuse to resume if state diverged unsafely.

    The full continuation of in-flight work is not yet implemented; this
    subcommand currently surfaces the resume diff so the operator (or the
    forthcoming alignment guard) can decide whether it is safe to proceed.
    """
    runs_dir = Path.cwd() / ".agent6" / "runs"
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_id = resolved
    layout = RunLayout(root=Path.cwd(), run_id=run_id)
    if not layout.run_dir.is_dir():
        print(f"ERROR: no such run dir: {layout.run_dir}", file=sys.stderr)
        return 2
    curator = GraphCurator(layout)
    diff = curator.compute_resume_diff(run_id, Path.cwd())
    print(f"Run: {run_id}")
    print(f"  snapshot head: {diff.snapshot_head}")
    print(f"  current head:  {diff.current_head}")
    if diff.committed_delta.files:
        print(f"  committed delta: {len(diff.committed_delta.files)} file(s)")
        for f in diff.committed_delta.files:
            print(f"    {f}")
    if diff.uncommitted_diff:
        print(f"  uncommitted diverged: {len(diff.uncommitted_diff)} file(s)")
        for u in diff.uncommitted_diff:
            print(f"    {u.path}  ({u.note})")
    if diff.snapshot_missing:
        print(f"\nGUARD: {diff.guard_summary}", file=sys.stderr)
        if not force:
            print("REFUSING to resume. Re-run with --force-resume to override.", file=sys.stderr)
            return 1
    print("\nResume scaffolding ready; in-flight continuation is not yet implemented.")
    return 0


def _cmd_plan_new(  # noqa: PLR0911, PLR0915
    config_path: Path,
    task: str,
    *,
    run_id: str,
    questions_file: str,
    answers_file: str,
) -> int:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    err = _check_provider_env_vars(cfg)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    qa = _resolve_offline_qa(questions_file, answers_file)
    if isinstance(qa, int):
        return qa
    offline_answers, questions_path = qa

    effective_run_id = run_id or new_friendly_id()
    layout = RunLayout(root=Path.cwd(), run_id=effective_run_id)
    _ensure_agent6_gitignored(
        Path.cwd(),
        identity=CommitIdentity(
            name=cfg.git.commit.name,
            email=cfg.git.commit.email,
            coauthor=cfg.git.commit.coauthor,
        ),
    )
    layout.ensure()
    transcript_sink = TranscriptSink(layout.transcripts_dir)
    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )

    try:
        planner = _build_role_provider(
            cfg, "planner", transcript_sink=transcript_sink, budget=budget
        )
        critic = _build_role_provider(cfg, "critic", transcript_sink=transcript_sink, budget=budget)
    except ProviderError as exc:
        print(f"ERROR: provider init failed: {exc}", file=sys.stderr)
        return 2

    sock_path = layout.run_dir / "curator.sock"
    curator_proc = spawn_curator(Path.cwd(), effective_run_id, sock_path)
    print(f"[agent6] plan run id: {effective_run_id}", file=sys.stderr)
    try:
        with GraphClient(sock_path) as graph_client:
            wf = PlanModeWorkflow(
                root=Path.cwd(),
                repo=load_repo_summary(Path.cwd()),
                critic=critic,
                planner=planner,
                graph_client=graph_client,
                run_id=effective_run_id,
                offline_answers=offline_answers,
                questions_file=questions_path,
                user_inputs=UserInputSink(layout.user_inputs_path),
            )
            try:
                result = wf.run(task)
            except BudgetExceeded as exc:
                print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
                print(budget.format_summary(), file=sys.stderr)
                return 3
            except PlanModeQuestionsPending as exc:
                print(
                    f"\nOPEN QUESTIONS: {len(exc.questions)} written to {exc.path}\n"
                    f"Edit the 'answer' fields and re-run with "
                    f"--run-id {effective_run_id} --answers-file {exc.path}",
                    file=sys.stderr,
                )
                return 4
            except PlanModeError as exc:
                print(f"PLAN MODE: {exc}", file=sys.stderr)
                return 1
    finally:
        curator_proc.terminate()
        try:
            curator_proc.wait(timeout=3)
        except Exception:
            curator_proc.kill()
            curator_proc.wait()

    print(f"\nPlan persisted under run id: {effective_run_id}")
    print(f"Root node: {result.root_node_id}")
    print(f"Steps   : {len(result.step_node_ids)}")
    print(f"Inspect later with: agent6 plan show {effective_run_id}")
    print(f'Execute with: agent6 run --run-id {effective_run_id} "<task>"')
    print("(execution-from-saved-plan is not yet wired; the graph is on disk.)")
    return 0


def _cmd_plan_show(run_id: str) -> int:
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
            (p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(
                f"ERROR: no runs with a manifest.json under {runs_dir}",
                file=sys.stderr,
            )
            return 2
        target_id = candidates[0].name
        print(f"[agent6] showing most recent plan: {target_id}", file=sys.stderr)
    layout = RunLayout(root=Path.cwd(), run_id=target_id)
    try:
        manifest = read_manifest(layout)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if manifest.plan is None:
        print(
            f"ERROR: run {target_id} has no persisted plan (kind={manifest.kind})",
            file=sys.stderr,
        )
        return 2
    print(f"Run id      : {manifest.run_id}")
    print(f"Kind        : {manifest.kind}")
    print(f"Created at  : {manifest.created_at}")
    if manifest.parent_run_id:
        print(f"Derived from: {manifest.parent_run_id}")
    print(f"Task        : {manifest.task}")
    if manifest.refined_task:
        print(f"Refined task: {manifest.refined_task}")
    print()
    print(format_plan(manifest.plan))
    return 0


def _resolve_existing_plan_run(run_id: str) -> tuple[str, RunLayout] | int:
    """Pick a run id and return (id, layout), or an int exit code on error."""

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
            (p for p in runs_dir.iterdir() if p.is_dir() and (p / "manifest.json").is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"ERROR: no runs with a manifest.json under {runs_dir}", file=sys.stderr)
            return 2
        target_id = candidates[0].name
        print(f"[agent6] using most recent plan: {target_id}", file=sys.stderr)
    return target_id, RunLayout(root=Path.cwd(), run_id=target_id)


def _resolve_offline_qa(
    questions_file: str, answers_file: str
) -> tuple[tuple[str, ...], Path | None] | int:
    """Parse the two CLI flags into (answers, questions_path).

    Returns an int exit code on error. ``answers`` is empty when no
    answers file is given. ``questions_path`` is ``None`` when no
    questions-file flag is given.
    """

    answers: tuple[str, ...] = ()
    if answers_file:
        path = Path(answers_file)
        if not path.is_file():
            print(f"ERROR: --answers-file not found: {path}", file=sys.stderr)
            return 2
        try:
            answers = read_answers_file(path)
        except (PlanModeError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    questions_path = Path(questions_file) if questions_file else None
    return answers, questions_path


def _run_plan_workflow_action(
    cfg: Config,
    *,
    new_id: str,
    parent_run_id: str,
    label: str,
    action: Callable[[PlanModeWorkflow], None],
    offline_answers: tuple[str, ...] = (),
    questions_file: Path | None = None,
) -> int:
    """Spin up providers + curator + workflow, run ``action``, tear down.

    Returns a CLI exit code. ``action`` should call one of
    ``PlanModeWorkflow.run*`` and raise on cancellation.
    """

    new_layout = RunLayout(root=Path.cwd(), run_id=new_id)
    new_layout.ensure()
    transcript_sink = TranscriptSink(new_layout.transcripts_dir)
    budget = BudgetTracker(
        max_input_tokens=cfg.budget.max_input_tokens,
        max_output_tokens=cfg.budget.max_output_tokens,
    )
    try:
        planner = _build_role_provider(
            cfg, "planner", transcript_sink=transcript_sink, budget=budget
        )
        critic = _build_role_provider(cfg, "critic", transcript_sink=transcript_sink, budget=budget)
    except ProviderError as exc:
        print(f"ERROR: provider init failed: {exc}", file=sys.stderr)
        return 2

    sock_path = new_layout.run_dir / "curator.sock"
    curator_proc = spawn_curator(Path.cwd(), new_id, sock_path)
    print(f"[agent6] {label}: {new_id}", file=sys.stderr)
    try:
        with GraphClient(sock_path) as graph_client:
            wf = PlanModeWorkflow(
                root=Path.cwd(),
                repo=load_repo_summary(Path.cwd()),
                critic=critic,
                planner=planner,
                graph_client=graph_client,
                run_id=new_id,
                parent_run_id=parent_run_id,
                offline_answers=offline_answers,
                questions_file=questions_file,
                user_inputs=UserInputSink(new_layout.user_inputs_path),
            )
            try:
                action(wf)
            except BudgetExceeded as exc:
                print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
                print(budget.format_summary(), file=sys.stderr)
                return 3
            except PlanModeQuestionsPending as exc:
                print(
                    f"\nOPEN QUESTIONS: {len(exc.questions)} written to {exc.path}\n"
                    f"Edit the 'answer' fields and re-run with --answers-file {exc.path}",
                    file=sys.stderr,
                )
                return 4
            except PlanModeError as exc:
                print(f"PLAN MODE: {exc}", file=sys.stderr)
                return 1
    finally:
        curator_proc.terminate()
        try:
            curator_proc.wait(timeout=3)
        except Exception:
            curator_proc.kill()
            curator_proc.wait()
    return 0


def _cmd_plan_revise(  # noqa: PLR0911
    config_path: Path,
    feedback: str,
    *,
    run_id: str,
    questions_file: str,
    answers_file: str,
) -> int:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    err = _check_provider_env_vars(cfg)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    qa = _resolve_offline_qa(questions_file, answers_file)
    if isinstance(qa, int):
        return qa
    offline_answers, questions_path = qa

    resolved = _resolve_existing_plan_run(run_id)
    if isinstance(resolved, int):
        return resolved
    old_id, old_layout = resolved
    try:
        manifest = read_manifest(old_layout)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if manifest.plan is None:
        print(f"ERROR: run {old_id} has no persisted plan", file=sys.stderr)
        return 2

    new_id = new_friendly_id()
    plan = manifest.plan

    def _action(wf: PlanModeWorkflow) -> None:
        wf.run_revision(
            previous_plan=plan,
            previous_task=manifest.task,
            previous_refined_task=manifest.refined_task,
            initial_feedback=feedback,
        )

    rc = _run_plan_workflow_action(
        cfg,
        new_id=new_id,
        parent_run_id=old_id,
        label=f"revising {old_id} ->",
        action=_action,
        offline_answers=offline_answers,
        questions_file=questions_path,
    )
    if rc == 0:
        print(f"\nRevised plan persisted under run id: {new_id}")
        print(f"  parent: {old_id}")
        print(f"Inspect with: agent6 plan show {new_id}")
    return rc


def _cmd_plan_edit(config_path: Path, *, run_id: str) -> int:  # noqa: PLR0911
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    resolved = _resolve_existing_plan_run(run_id)
    if isinstance(resolved, int):
        return resolved
    old_id, old_layout = resolved
    try:
        manifest = read_manifest(old_layout)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if manifest.plan is None:
        print(f"ERROR: run {old_id} has no persisted plan", file=sys.stderr)
        return 2

    edited = _edit_plan_in_editor(manifest.plan)
    if isinstance(edited, int):
        return edited
    if edited == manifest.plan:
        print("[agent6] plan unchanged; no new run created.", file=sys.stderr)
        return 0

    new_id = new_friendly_id()

    def _action(wf: PlanModeWorkflow) -> None:
        wf.run_edit(
            edited_plan=edited,
            previous_task=manifest.task,
            previous_refined_task=manifest.refined_task,
        )

    rc = _run_plan_workflow_action(
        cfg,
        new_id=new_id,
        parent_run_id=old_id,
        label=f"editing {old_id} ->",
        action=_action,
    )
    if rc == 0:
        print(f"\nEdited plan persisted under run id: {new_id}")
        print(f"  parent: {old_id}")
        print(f"Inspect with: agent6 plan show {new_id}")
    return rc


def _edit_plan_in_editor(plan: Plan) -> Plan | int:
    """Round-trip ``plan`` through ``$EDITOR``. Returns the parsed Plan or exit code.

    Security review note: spawns ``$EDITOR`` (user-controlled config) on
    a tempfile containing JSON serialized from an existing in-memory
    Plan object. The argv is fixed-shape (``[editor, tempfile]``) — no
    shell expansion, no LLM-controlled args at the moment of the call.
    The plan itself was originally produced by an LLM, but it has
    already passed pydantic validation and is treated as trusted text
    by the time it reaches the editor.
    """

    import tempfile  # noqa: PLC0415 - localised to this command

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        print(
            "ERROR: $EDITOR (or $VISUAL) is not set; cannot open the plan for editing.",
            file=sys.stderr,
        )
        return 2
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".plan.json", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(plan.model_dump_json(indent=2))
        tf.write("\n")
        tmp_path = Path(tf.name)
    try:
        proc = subprocess.run([editor, str(tmp_path)], check=False)
        if proc.returncode != 0:
            print(
                f"ERROR: editor exited non-zero ({proc.returncode}); aborting.",
                file=sys.stderr,
            )
            return 1
        try:
            return Plan.model_validate_json(tmp_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: edited plan failed schema validation: {exc}", file=sys.stderr)
            return 1
    finally:
        tmp_path.unlink(missing_ok=True)


def _make_confirm(auto: bool, user_inputs: UserInputSink | None = None) -> Callable[[Plan], bool]:
    def confirm(_plan: Plan) -> bool:
        if auto:
            if user_inputs is not None:
                user_inputs.record(
                    kind="plan_approval",
                    prompt="Proceed with plan?",
                    answer="yes",
                    source="auto_confirm",
                )
            return True
        try:
            raw = input("\nProceed with plan? [y/N] ").strip()
        except EOFError:
            raw = ""
        approved = raw.lower() in {"y", "yes"}
        if user_inputs is not None:
            user_inputs.record(
                kind="plan_approval",
                prompt="Proceed with plan?",
                answer=raw or "",
                source="stdin",
                approved=approved,
            )
        return approved

    return confirm


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


def _cmd_init(*, force: bool) -> int:
    return init_workspace(Path.cwd(), force=force)


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
