# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run` and `agent6 resume` plus their shared execution scaffolding."""

from __future__ import annotations

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
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Literal

from agent6 import __version__
from agent6.budget import BudgetTracker
from agent6.cli._ask import (
    run_ask_repl as _run_ask_repl,
)
from agent6.cli._ask import (
    save_ask_transcript as _save_ask_transcript,
)
from agent6.cli._common import (
    _BudgetOverrides,
    _check_provider_keys,
    _explicit_usd_flag_error,
    _SandboxOverrides,
    _start_mcp_manager_if_enabled,
    _state_dir,
    detect_env,
)
from agent6.cli._merge import execute_merge
from agent6.cli._repl import build_repl_hook as _build_repl_hook
from agent6.cli._steer import (
    make_steer_state as _make_steer_state,
)
from agent6.cli._steer import (
    select_revised_prompt as _select_revised_prompt,
)
from agent6.cli._steer import (
    tty_prompt as _tty_prompt,
)
from agent6.cli.egress import (
    _check_network_profile,
    _maybe_apply_agent_landlock,
    _maybe_start_egress,
    _stop_egress,
    _warn_if_unsandboxed,
    resolve_strict_egress_viability,
)
from agent6.cli.plan_watch import _most_recent_run_id
from agent6.cli.providers import (
    _build_critic_provider,
    _build_prompt_reviser_provider,
    _build_role_provider,
    _build_summariser_provider,
    _InstrumentedProvider,
    _role_temperature,
    build_review_seats,
    resolve_compaction_thresholds,
    review_panel_configured,
)
from agent6.config import (
    Config,
    ConfigError,
    NotifyConfig,
    RoleName,
)
from agent6.config_layer import (
    load_effective,
)
from agent6.detect import select_profile
from agent6.events import EventSink
from agent6.frontend.approval import (
    clear_pending_answers,
    clear_worker_pid,
    frontend_is_live,
    read_answer,
    read_question_answer,
    write_worker_pid,
)
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_exists,
    create_branch,
    delete_branch_if_merged,
    is_ancestor,
    is_git_repo,
    restore_stash,
    set_repo_hook_policy,
    stash_all,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.graph.client import CuratorClientError, GraphClient, spawn_curator
from agent6.graph.storage import RunLayout
from agent6.paths import (
    chown_to_real_user,
)
from agent6.portable import lock_exclusive, unlock
from agent6.pricing import lookup_price
from agent6.providers import (
    Provider,
    TranscriptSink,
)
from agent6.run_id import RunIdError, new_friendly_id, resolve_run_id
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import SandboxProfile
from agent6.verify_infer import VERIFY_INFER_SYSTEM_PROMPT, infer_verify_command
from agent6.workflows._run_state import load_resume_snapshot
from agent6.workflows.loop import ResumeError, RunResult, Workflow

# Distinct exit code for a budget-exhausted run so automation can tell "raise
# the cap and `agent6 resume`" apart from a genuine failure. Documented in
# CONFIG.md ([budget]); a budget-stopped run is resumable from its snapshot.
_EXIT_BUDGET_EXHAUSTED = 3

# Default USD ceiling for `agent6 ask` when no budget is configured, so an
# exploratory question can't quietly run up a bill.
_ASK_DEFAULT_MAX_USD = 0.50


def _run_exit_code(result: RunResult) -> int:
    """Map a finished run to its process exit code (0 ok / 3 budget / 1 else)."""
    if result.completed:
        return 0
    if result.reason == "budget_exhausted":
        return _EXIT_BUDGET_EXHAUSTED
    return 1


def _eprint(msg: str) -> None:
    """Loop logger that writes to stderr (used for `ask`, whose stdout is the
    answer and must stay clean for piping)."""
    print(msg, file=sys.stderr)


def _acquire_single_writer(run_dir: Path) -> int | None:
    """Take a non-blocking exclusive lock on ``<run-dir>/worker.lock``.

    One run's shared state (``loop_state.json``, ``checkpoints/``, the curator
    DAG, the run branch) has exactly one authoritative writer. A second
    ``agent6 run``/``resume``/``fork`` targeting the SAME run dir would spawn a
    second curator whose independent in-memory cache silently clobbers the
    first's parent->child links (a lost update), and would interleave commits on
    the run branch. This is the run-level analogue of ``machine_lock``.

    Returns the held fd on success (the caller keeps the process alive to hold
    it, and passes it to ``_release_single_writer`` at teardown), or ``None``
    when another live process holds it (the caller refuses). A crashed writer
    leaves no lock -- flock releases on process death -- so resume-after-crash is
    never blocked by a stale lock.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_dir / "worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd, blocking=False)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_single_writer(fd: int | None) -> None:
    """Release + close a lock fd from ``_acquire_single_writer`` (no-op on None).

    Explicit close matters: the fd is a raw int (``os.open``), so it does not
    self-close on GC. A leaked fd would keep the flock held and wrongly refuse a
    later same-dir run in the same process (tests, embedding)."""
    if fd is None:
        return
    with contextlib.suppress(OSError):
        unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


_SINGLE_WRITER_BUSY = (
    "REFUSING: run {rid!r} is already being driven by another agent6 process "
    "(its worker.lock is held). Concurrent run/resume of the same run would "
    "corrupt its state (a second curator clobbers the task graph, and commits "
    "interleave on the run branch). Wait for that process to finish; a crashed "
    "one releases the lock automatically."
)


def _warn_if_usd_unenforceable(cfg: Config) -> None:
    """Warn at startup when ``best_effort_usd_limit`` is set but a configured
    role model has no published price. The USD ceiling sums per-model estimated
    cost, and an unpriced model contributes $0, so ANY unpriced role that spends
    tokens (the worker, but also a distinct reviewer/critic/planner) makes the
    ceiling silently under-count and spend is bounded only by the token
    ceilings. We deliberately do NOT guess a price or terminate early (a wrong
    guess could kill a run mid-task); we just make the gap visible. Anthropic
    publishes no pricing, so this fires for Claude in any role."""
    usd = cfg.budget.best_effort_usd_limit
    if usd <= 0:
        return
    unpriced = sorted(
        {rm.model for rm in cfg.models.configured().values() if lookup_price(rm.model) is None}
    )
    if unpriced:
        print(
            f"[agent6] WARNING: best_effort_usd_limit=${usd:g} cannot be enforced "
            f"for {', '.join(repr(m) for m in unpriced)} (no published price); their "
            f"spend is invisible to the dollar ceiling, so it is bounded only by "
            f"max_input_tokens={cfg.budget.max_input_tokens:,} / "
            f"max_output_tokens={cfg.budget.max_output_tokens:,}. Set explicit "
            f"token ceilings if you need a precise dollar bound.",
            file=sys.stderr,
        )


def _warn_if_prompt_override_incomplete(cfg: Config) -> None:
    """Warn when a custom ``prompt.system_prompt_file`` omits the core tool
    contracts the worker needs: ``finish_run`` is the only clean exit, and an
    edit primitive (``apply_edit``/``apply_patch``) is needed to do work. The
    override is advanced + operator-owned, so we don't block -- just flag the
    likely-broken case loudly and point at ``agent6 prompt show``."""
    path = cfg.prompt.system_prompt_file
    if not path:
        return
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return  # config validation already enforces existence; nothing to add
    missing = [t for t in ("finish_run",) if t not in text]
    if "apply_edit" not in text and "apply_patch" not in text:
        missing.append("apply_edit/apply_patch")
    if missing:
        # Name every capability that is actually absent, not just one of them, so
        # a prompt missing both finish_run AND an edit primitive reads correctly.
        actions = []
        if "finish_run" in missing:
            actions.append("terminate")
        if "apply_edit/apply_patch" in missing:
            actions.append("make edits")
        print(
            f"[agent6] WARNING: custom system_prompt_file ({path}) does not mention "
            f"{', '.join(missing)}; the worker may not know how to "
            f"{' or '.join(actions)}. The override "
            "replaces the built-in run-mode base -- you own preserving the tool "
            "contracts. Inspect the assembled prompt with `agent6 prompt show`.",
            file=sys.stderr,
        )


def _require_git_repo(cwd: Path) -> bool:
    """Print a friendly error and return False when *cwd* is not a git repo.

    A clean early exit instead of the misleading "Git identity not configured"
    error (when there's no global identity) or an ugly failure deeper in the
    run. agent6 needs git to branch, commit per step, and let the user
    review/revert what the agent did.
    """
    if is_git_repo(cwd):
        return True
    print(
        f"ERROR: {cwd} is not a git repository.\n"
        "agent6 needs git here to create a run branch, commit each step, and let"
        " you review or revert what the agent did.\n"
        "  Fix: run `agent6 init` (it offers to set up git for you), or\n"
        '       `git init && git add -A && git commit -m "initial commit"`.',
        file=sys.stderr,
    )
    return False


def _default_stdin_approver(prompt: str) -> bool:
    """Plain-terminal fallback for tool approval (no live TUI, or its answer
    timed out). Routed via /dev/tty so the prompt stays visible when a TUI has
    redirected the std streams to its console log; plain stdin without one."""
    ans = _tty_prompt(f"{prompt} [y/N]: ")
    if ans is None:
        return False
    return ans.strip().lower() in {"y", "yes"}


def _build_approver(run_dir: Path, events: EventSink) -> Callable[[str], bool]:
    """Build the `run_command` approver, bridged to a live TUI when present.

    Emits an `approval.prompt` event; if a TUI is live (it wrote `frontend.pid`) the
    answer comes from its Allow/Deny modal via the file bridge
    (`approvals/<id>.answer`), otherwise -- or if the TUI dies / times out -- it
    falls back to the stdin `[y/N]` prompt. Emits `approval.answer` either way.
    This is what actually wires the watch/auto-spawn TUI to run_command approval
    (previously the modal's answer was written but never read)."""
    counter = {"n": 0}

    def approve(prompt: str) -> bool:
        counter["n"] += 1
        prompt_id = f"approval-{counter['n']}"
        events.emit("approval.prompt", id=prompt_id, prompt=prompt)
        approved: bool | None = None
        source = "stdin"
        if frontend_is_live(run_dir):
            approved = read_answer(run_dir, prompt_id)
            if approved is not None:
                source = "tui"
        if approved is None:
            approved = _default_stdin_approver(prompt)
        events.emit("approval.answer", id=prompt_id, approved=approved, source=source)
        return approved

    return approve


def _confirm_unconfined_autorun(selected_profile: SandboxProfile, cfg: Config) -> bool:
    """The one genuinely dangerous combination: the sandbox is OFF and
    run_command is auto-approved, so the agent can run any command on the host
    with no confinement and no prompt. Get one explicit consent at startup when
    interactive; proceed with a loud warning when not (the explicit opt-outs
    are already the consent, and machines/CI must not block). Not a per-command
    guard -- once unconfined, guarding individual commands would be theatre.

    Returns True to proceed, False to abort.
    """
    if selected_profile != "none" or cfg.sandbox.run_commands != "yes":
        return True
    print(
        "[agent6] DANGER: the sandbox is DISABLED and run_command is"
        " AUTO-APPROVED. The agent can run ANY command on this host with no"
        " confinement and no prompt.",
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        print("[agent6] proceeding (non-interactive).", file=sys.stderr)
        return True
    answer = _tty_prompt("Continue? [y/N]: ")
    return (answer or "").strip().lower() in {"y", "yes"}


def _warn_if_headless_ask(cfg: Config, *, tui_enabled: bool) -> None:
    """Warn when run_commands='ask' but no approver is reachable.

    A headless run (no TUI, no controlling TTY) has nothing to answer the
    Allow/Deny prompt, so every ``run_command`` is auto-denied. Surface that up
    front instead of letting the agent hit confusing mid-run "denied" errors
    (observed dogfooding a headless run). run_verify_command is unaffected.
    """
    if cfg.sandbox.run_commands == "ask" and not tui_enabled and not sys.stdin.isatty():
        print(
            "[agent6] NOTE: sandbox.run_commands='ask' but this run is headless (no TUI,"
            " no TTY), so run_command calls will be auto-denied. Pass --auto-approve"
            " (or set sandbox.run_commands='yes') to auto-approve, or 'no' to withhold"
            " run_command, for headless/CI runs.",
            file=sys.stderr,
        )


def _build_questioner(run_dir: Path, events: EventSink) -> Callable[[str, tuple[str, ...]], str]:
    """Build the `ask_user` questioner, bridged to a live TUI when present.

    Emits a `question.prompt` event; if a TUI is live the answer comes from its
    question modal via `questions/<id>.answer`, otherwise (or if the TUI dies /
    times out) it falls back to a numbered stdin prompt. A headless run (no TUI,
    no TTY) gets an empty answer rather than hanging. Emits `question.answer`."""
    counter = {"n": 0}

    def ask(question: str, options: tuple[str, ...]) -> str:
        counter["n"] += 1
        question_id = f"question-{counter['n']}"
        events.emit("question.prompt", id=question_id, question=question, options=list(options))
        answer: str | None = None
        source = "stdin"
        if frontend_is_live(run_dir):
            answer = read_question_answer(run_dir, question_id)
            if answer is not None:
                source = "tui"
        if answer is None:
            answer = _default_stdin_questioner(question, options)
        events.emit("question.answer", id=question_id, answer=answer, source=source)
        return answer

    return ask


def _default_stdin_questioner(question: str, options: tuple[str, ...]) -> str:
    """Numbered terminal prompt via /dev/tty (visible under a TUI's stream
    redirect); headless (no controlling terminal) returns "" so a run never
    hangs or consumes piped stdin meant for something else."""
    lines = [question, *(f"  {i}) {opt}" for i, opt in enumerate(options, start=1))]
    ans = _tty_prompt("\n".join(lines) + "\n> ", fall_back_to_stdin=False)
    if ans is None:
        return ""
    ans = ans.strip()
    if ans.isdigit() and 1 <= int(ans) <= len(options):
        return options[int(ans) - 1]
    return ans


def _tui_available() -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("textual") is not None


def _should_spawn_tui(*, tui: bool, interactive: bool, mode: str) -> bool:
    """Whether `agent6 run`/`resume` opens the dashboard TUI.

    Headless by default (a scrolling CLI event stream); `--tui` opts into the
    full-screen dashboard. It needs the `tui` extra and a real TTY, is for `run`
    mode only (`plan`/`ask` stay text), and is mutually exclusive with `-i` (the
    stdin REPL). When `--tui` is asked for but cannot run, warn and stay
    headless rather than fail the run."""
    if not tui:
        return False
    if interactive or mode != "run":
        print("[agent6] --tui is not available here; continuing in CLI mode.", file=sys.stderr)
        return False
    if not sys.stdout.isatty():
        print("[agent6] --tui needs a TTY; continuing in CLI mode.", file=sys.stderr)
        return False
    if not _tui_available():
        print(
            "[agent6] --tui needs 'textual' (part of the base install; this environment"
            " is missing it); continuing in CLI mode.",
            file=sys.stderr,
        )
        return False
    return True


def _stream_modes(*, tui_enabled: bool) -> tuple[bool, bool]:
    """Return ``(stream_text, console_stream)`` for the worker provider.

    ``stream_text`` makes the provider stream and emit ``role.text_delta`` /
    ``role.thinking_delta`` events, which the dashboard renders as the model's
    live reasoning + answer. ``console_stream`` additionally echoes those deltas
    to stderr.

    Streaming is on for an interactive stderr TTY (so a plain `agent6 ask`/`plan`
    shows live output) or when forced:
    - ``AGENT6_FORCE_STREAM=1``: bench/CI -- emit AND echo (the Kimi/OpenRouter
      gateway corrupts the non-streaming body with SSE heartbeats).
    - ``AGENT6_STREAM_TO_LOG=1``: set by the `agent6 tui` hub when it spawns a run
      detached and then watches it on the dashboard. Emit the delta EVENTS only,
      with NO console echo -- otherwise a long headless run pours its whole
      reasoning into the hub's discarded stderr temp file.
    """
    stream_to_log = os.environ.get("AGENT6_STREAM_TO_LOG") == "1"
    stream_text = (
        sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1" or stream_to_log
    )
    # Echo to stderr only when there is a console to read it: not while the TUI
    # owns the terminal, and not for a hub-watched headless run (dashboard-only).
    console_stream = stream_text and not tui_enabled and not stream_to_log
    return stream_text, console_stream


@contextlib.contextmanager
def _tui_session(run_dir: Path, *, enabled: bool) -> Generator[None]:
    """Run the dashboard TUI as a co-process that owns the terminal.

    While it is up, this process's own console chatter is redirected to
    `<run_dir>/tui_console.log` so it doesn't fight the TUI for the terminal;
    progress still flows through `logs.jsonl`, which the TUI tails, and approvals
    go through the file bridge. The TUI quits itself on the `run.end` event; we
    reap it on the way out (terminating if it lingers). A spawn failure degrades
    gracefully to a normal (TUI-less) run rather than aborting."""
    if not enabled:
        yield
        return
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent6.tui", "--watch", str(run_dir), "--exit-on-end"]
        )
    except OSError as exc:
        print(f"[agent6] could not start TUI ({exc}); continuing without it.", file=sys.stderr)
        yield
        return
    orig_out, orig_err = sys.stdout, sys.stderr
    log_fh = (run_dir / "tui_console.log").open("w", encoding="utf-8")
    sys.stdout = log_fh
    sys.stderr = log_fh
    try:
        yield
    finally:
        # The TUI closes itself on the run.end event. If the run ended without
        # one (a crash), nudge it with SIGINT first -- textual restores the
        # terminal cleanly -- and only hard-terminate as a last resort. Keep our
        # own output redirected until it's gone so nothing scribbles its screen.
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=3)
        sys.stdout, sys.stderr = orig_out, orig_err
        with contextlib.suppress(Exception):
            log_fh.close()


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


def _manifest_model_brief(rm: Any) -> dict[str, str] | None:
    """``{provider, model}`` for a resolved role, or None when unset."""
    if rm is None:
        return None
    return {"provider": rm.provider, "model": rm.model}


def _write_run_manifest(
    layout: RunLayout,
    *,
    run_id: str,
    user_task: str,
    base_sha: str,
    base_branch: str,
    run_branch: str | None,
    cfg: Config,
    mode: str = "run",
    effective_profile: str = "",
    parent_run_id: str | None = None,
    forked_from_turn: int | None = None,
    forked_from_sha: str | None = None,
) -> None:
    """Write the canonical manifest.json for a run.

    This is the only thing that reads/writes ``layout.manifest_path``.
    Format is JSON for the same reason logs.jsonl is JSON: trivially
    grep-able from a shell and easy to consume from any language. The
    on-disk shape is *liquid* until 1.0 - bump ``version`` only when
    the new shape genuinely improves a downstream consumer.

    ``parent_run_id`` / ``forked_from_turn`` / ``forked_from_sha`` are set only
    for a run created by ``agent6 fork``; they record the lineage (source run +
    the turn forked from + the workspace sha at that turn). A non-forked run
    leaves them out.
    """
    manifest: dict[str, Any] = {
        "version": 2,
        "agent6_version": __version__,
        "run_id": run_id,
        "mode": mode,  # run | plan (ask runs live under asks/, not here)
        "start_ts": _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="microseconds"),
        "user_task": user_task[:4000],
        "base_sha": base_sha,
        "base_branch": base_branch,
        "run_branch": run_branch,
        "models": {
            "worker": _manifest_model_brief(cfg.models.resolve("worker")),
            "reviewer": _manifest_model_brief(cfg.models.resolve("reviewer")),
        },
        "workflow": {
            "critic": cfg.review.trigger,
            "revise_prompt": cfg.prompt.revise_prompt,
            # The profile the run actually used (--profile flag or top-level
            # `profile`), so `agent6 resume` re-applies the same strategy.
            "profile": effective_profile,
        },
    }
    if parent_run_id is not None:
        manifest["parent_run_id"] = parent_run_id
        manifest["forked_from_turn"] = forked_from_turn
        manifest["forked_from_sha"] = forked_from_sha
    # tmp+replace: the TUI hub and `runs show` poll this file on live runs, and
    # a bare write_text lets them read a truncated JSON mid-rewrite.
    tmp = layout.manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(layout.manifest_path)


def _infer_verify_if_unset(
    cfg: Config,
    cwd: Path,
    *,
    mode: str,
    events: EventSink,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
) -> Config:
    """When ``workflow.verify_command`` is unset for a run/plan, infer one and
    inject it IN-MEMORY (never persisted -- runs do not mutate config).

    Layered cheapest-first (AGENTS.md -> repo signals -> a cheap reviewer-role
    LLM call); see ``agent6.verify_infer``. Emits ``loop.verify_inferred`` and
    prints what was picked + that it is per-run. If nothing can be inferred the
    run proceeds GATELESS (no verify gate; the loop commits each editing step).
    """
    if mode not in ("run", "plan") or cfg.workflow.verify_command:
        return cfg
    agents_md = ""
    agents_path = cwd / "AGENTS.md"
    if agents_path.is_file():
        try:
            agents_md = agents_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            agents_md = ""

    def _llm_call(context: str) -> str:
        inner = _build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
        )
        rm = cfg.models.resolve("reviewer")
        provider = _InstrumentedProvider(
            inner=inner,
            role="verify_inferer",
            model=rm.model if rm else "",
            provider_name=rm.provider if rm else "",
            events=events,
            budget=budget,
        )
        resp = provider.call(
            system=VERIFY_INFER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
            tools=[],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.text or ""

    inferred = infer_verify_command(cwd, agents_md, llm_call=_llm_call)
    if inferred is None:
        events.emit("loop.verify_inferred", command=[], source="none")
        if mode == "run":
            print(
                "[agent6] no verify_command set and none could be inferred; running"
                " gateless\n         (per-step commits, no green gate). Set"
                " workflow.verify_command to gate commits on a passing verify.",
                file=sys.stderr,
            )
        return cfg
    events.emit("loop.verify_inferred", command=list(inferred.argv), source=inferred.source)
    print(
        f"[agent6] verify_command not set; inferred from {inferred.source}:"
        f" {' '.join(inferred.argv)}\n         (this run only — set"
        " workflow.verify_command in your per-repo config to pin it)",
        file=sys.stderr,
    )
    return cfg.with_inferred_verify(inferred.argv)


def _cmd_run(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    task: str,
    *,
    run_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    mode: Literal["run", "plan", "ask"] = "run",
    budget_overrides: _BudgetOverrides | None = None,
    sandbox_overrides: _SandboxOverrides | None = None,
    profile: str = "",
) -> int:
    """Single-loop agent: one provider, one LLM driving via tool
    calls over the fixed tool surface, deterministic harness (jail +
    budget + verify timeout + DAG curator for persistence/resume).
    Sole ``agent6 run`` path.

    When ``mode="plan"`` the same harness drives a planning
    pass instead of an execution pass: planning system prompt,
    edit-tools filtered out, ``finish_planning`` instead of
    ``finish_run``, no auto-commit. The plan markdown lands at
    ``<run-dir>/plan.md`` and is consumed by ``agent6 run --from-plan``.
    The ``planner`` model role drives plan mode (falls back to ``worker``).
    """
    try:
        cfg = load_effective(Path.cwd(), config_path, profile=profile).config
        set_repo_hook_policy(cfg.git.run_repo_hooks)
        if budget_overrides is not None:
            cfg = budget_overrides.apply(cfg)
        if sandbox_overrides is not None:
            cfg = sandbox_overrides.apply(cfg)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    # Surface the not-a-git-repo wall up front. run/plan need git; ask is
    # read-only and may run outside a repo. Without this, a user in a scratch
    # non-git dir clears the provider, model, and key walls serially only to
    # discover at the end that they also need git. Mirrors the resume path,
    # which already checks git before require_runnable.
    if mode != "ask" and not _require_git_repo(Path.cwd()):
        return 2
    role: RoleName = "planner" if mode == "plan" else "worker"
    try:
        cfg.require_runnable(role)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    # Resolve @path references in the task string before the
    # workflow ever sees it. Lets the user write "fix the bug in @src/x.py
    # described in @notes.md" and have those files inlined verbatim.
    task = _expand_task_file_refs(task, Path.cwd())

    env = detect_env()
    try:
        selected_profile = select_profile(cfg.sandbox.profile, env)
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    _warn_if_unsandboxed(selected_profile)
    if not _confirm_unconfined_autorun(selected_profile, cfg):
        print("[agent6] aborted.", file=sys.stderr)
        return 1

    net_err = _check_network_profile(cfg, selected_profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return 2
    # strict can be selected because the jail launcher has userns, yet this
    # process can't create one for the egress broker (surgical AppArmor profile).
    # Downgrade auto->hardened, or refuse an explicit strict, with guidance.
    selected_profile, egress_err = resolve_strict_egress_viability(cfg, selected_profile)
    if egress_err is not None:
        print(egress_err, file=sys.stderr)
        return 2

    missing = _check_provider_keys(cfg)
    usd_err = _explicit_usd_flag_error(budget_overrides.max_usd if budget_overrides else None, cfg)
    if usd_err is not None:
        print(f"REFUSING: {usd_err}", file=sys.stderr)
        return 2
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2

    # Git pre-flight (verify identity).
    # The auto-commit-on-verify-pass behaviour requires a clean working tree,
    # so the same git assumptions apply. Skipping these left first-time runs
    # crashing on dirty-tree or missing-identity errors deep into a paid run.
    cwd = Path.cwd()
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        coauthor=cfg.git.commit.coauthor,
    )
    # ask is read-only and may run outside a git repo (e.g. agent6 self-help),
    # so it skips the commit-oriented git pre-flight entirely.
    base_sha = ""
    base_branch = ""
    pre_status = None  # set below for run/plan; stays None for read-only ask
    if mode != "ask":
        # The not-a-git-repo guard already ran up front, before require_runnable.
        try:
            verify_git_identity(cwd, identity)
        except GitError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        # Capture base sha + branch BEFORE we (optionally) cut a run branch
        # so `agent6 runs diff <run-id>` knows where the run started.
        try:
            pre_status = git_status(cwd)
        except GitError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        base_sha = pre_status.head_sha
        base_branch = pre_status.branch

    # Layout: standard run-dir scaffolding for transcripts + logs. ask sessions
    # live under the per-repo state dir (asks subdir) to stay separate from real runs.
    effective_run_id = run_id or new_friendly_id()
    state_dir = _state_dir(cwd)
    layout = RunLayout(
        state_dir=state_dir,
        run_id=effective_run_id,
        subdir="asks" if mode == "ask" else "runs",
    )
    layout.ensure()
    # One authoritative writer per run dir. Acquire BEFORE touching any shared
    # run state (clearing answers, the worker pid, the curator) so a second
    # process refuses cleanly instead of clobbering the live run.
    worker_lock_fd = _acquire_single_writer(layout.run_dir)
    if worker_lock_fd is None:
        print(_SINGLE_WRITER_BUSY.format(rid=effective_run_id), file=sys.stderr)
        return 2
    # Drop stale approve/ask/steer answers + frontend.pid from a prior session (the
    # id counters reset on resume, so an old answer must not be read instead of
    # re-prompting; a stale frontend.pid would otherwise stall the answer-poll).
    clear_pending_answers(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    # Enforce the dirty-tree policy BEFORE cutting the run branch, so the
    # branch is cut from a clean tree and the agent's per-step auto-commits
    # (`git add -A`) never swallow the user's pre-existing uncommitted work.
    # Only `run` makes commits; `plan`/`ask` are read-only (matching the
    # branch_per_run guard below).
    # Track an auto-stash so the run-end finalizer can restore or at least report
    # it; otherwise the user's stashed pre-run work is silently left behind.
    stashed = False
    base_branch = pre_status.branch if pre_status is not None else ""
    if mode == "run" and pre_status is not None and not pre_status.is_clean:
        if cfg.git.auto_stash:
            try:
                stash_all(cwd, f"agent6 auto-stash before run {effective_run_id}")
                stashed = True
            except GitError as exc:
                print(f"ERROR: could not auto-stash before run: {exc}", file=sys.stderr)
                clear_worker_pid(layout.run_dir)
                _release_single_writer(worker_lock_fd)
                return 2
        elif cfg.git.require_clean_worktree:
            print(
                "REFUSING: working tree is not clean. Commit, stash, or discard "
                "your changes, set [git].auto_stash=true, or set "
                "[git].require_clean_worktree=false to override.",
                file=sys.stderr,
            )
            clear_worker_pid(layout.run_dir)
            _release_single_writer(worker_lock_fd)
            return 2

    egress_broker = None
    egress_sock_dir = None
    run_branch: str | None = None
    try:
        # Cut a fresh branch named after the run id so it is 1:1 with the run
        # (find it from any run id, `agent6 runs diff <id>`, or just delete the
        # branch to discard everything the agent did). The name is the unique
        # run id, never a timestamp+task-slug that collides into a pile of
        # near-duplicate `agent6/<ts>-<same-task>` branches on re-runs. Only real
        # `run` mode branches: `plan`/`ask` make no commits, so a branch for them
        # is pure litter. create_branch is idempotent (reuses an existing branch).
        if cfg.git.branch_per_run and mode == "run":
            run_branch = f"agent6/{effective_run_id}"
            try:
                create_branch(cwd, run_branch)
            except GitError as exc:
                print(f"ERROR: could not cut run branch {run_branch}: {exc}", file=sys.stderr)
                return 2

        # Write the run manifest. This is the canonical record of where the
        # run started (base_sha + base_branch), which model+provider drove
        # it, and the user_task it was given. `agent6 runs diff <run-id>` and
        # any future tooling that wants to reproduce a run reads from here.
        _write_run_manifest(
            layout,
            run_id=effective_run_id,
            user_task=task,
            base_sha=base_sha,
            base_branch=base_branch,
            run_branch=run_branch,
            cfg=cfg,
            mode=mode,
            effective_profile=profile or cfg.profile,
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

        # ask gets a small default USD ceiling so an exploratory question can't run
        # away; an explicit [budget].best_effort_usd_limit or --max-usd overrides it.
        usd_limit = cfg.budget.best_effort_usd_limit
        ask_max_usd = usd_limit or (_ASK_DEFAULT_MAX_USD if mode == "ask" else 0.0)
        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
            max_usd=ask_max_usd,
        )

        # Workflow uses ONE provider for everything (the worker role, or the
        # planner role in plan mode). No critic/triage/planner/reviewer/escalation
        # cascade inside the loop.
        worker_inner = _build_role_provider(
            cfg, role, transcript_sink=transcript_sink, budget=budget
        )
        rm_worker = cfg.models.resolve(role)
        assert rm_worker is not None  # require_runnable validated this
        _warn_if_usd_unenforceable(cfg)
        _warn_if_prompt_override_incomplete(cfg)
        # Enable SSE streaming when stderr is a TTY (covers TUI
        # and interactive shell use). Bench/CI runs pipe stderr, so they
        # stay on the audited non-streaming code path UNLESS the operator
        # sets AGENT6_FORCE_STREAM=1, the Kimi/OpenRouter bench needs
        # streaming on because the gateway emits SSE keep-alive comment
        # heartbeats during long requests, which corrupt the non-streaming
        # response body (resp.json() blows up with JSONDecodeError).
        tui_enabled = _should_spawn_tui(tui=tui, interactive=interactive, mode=mode)
        _warn_if_headless_ask(cfg, tui_enabled=tui_enabled)
        # The interactive revision prompt reads the terminal; with the TUI owning
        # it the prompt would land invisibly in the console log and contend for
        # stdin. Skip revision for this run instead.
        effective_revise_prompt = cfg.prompt.revise_prompt
        if effective_revise_prompt == "interactive" and tui_enabled:
            print(
                "[agent6] prompt.revise_prompt='interactive' needs the terminal; the TUI"
                " owns it. Skipping prompt revision for this run.",
                file=sys.stderr,
            )
            effective_revise_prompt = "off"
        stream_text, console_stream = _stream_modes(tui_enabled=tui_enabled)
        provider: Provider = _InstrumentedProvider(
            inner=worker_inner,
            role=role,
            model=rm_worker.model,
            provider_name=rm_worker.provider,
            events=events,
            budget=budget,
            stream_text=stream_text,
            console_stream=console_stream,
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
        # The grounded review panel runs at the critic trigger WHEN explicitly
        # configured (any review_* key); otherwise critic!=off keeps the legacy single
        # critic, so a pre-panel before_finish/periodic config still gates as before.
        review_seats = (
            build_review_seats(
                cfg,
                transcript_sink=transcript_sink,
                budget=budget,
                n=cfg.review.panel_size,
                personas=cfg.review.personas,
            )
            if cfg.review.trigger != "off" and review_panel_configured(cfg)
            else []
        )

        # Verify is optional: if unset, infer one for this run (AGENTS.md -> repo
        # signals -> a cheap LLM call) and inject it in-memory. Never persisted.
        cfg = _infer_verify_if_unset(
            cfg, cwd, mode=mode, events=events, transcript_sink=transcript_sink, budget=budget
        )

        # AF_UNIX paths have a 108-char limit (Linux sun_path), which
        # bench setups with long BENCH_ROOT (and any future overlay-mount
        # paths) blew through. Bind the socket under a short /tmp dir and
        # leave a symlink under run_dir for observability. Cleaned up in
        # the finally block. See bench/improvement_plan.md audit cross-cutting.
        sock_path = layout.run_dir / "curator.sock"  # rebound to the /tmp socket inside the try

        # Steering (mid-run Ctrl-C -> a stdin prompt) needs the terminal; skip it
        # when the TUI owns it (then default Ctrl-C aborts cleanly). Double-Ctrl-C
        # within 2s still raises KeyboardInterrupt for the hard-abort path below.
        steer_state = _make_steer_state(events, layout.run_dir)

        result = None
        interrupted = False
        dispatcher: ToolDispatcher | None = None
        # Spawned inside the try so the finally below always tears them down even
        # if a spawn itself fails (otherwise curator/MCP procs + the /tmp socket
        # dir leak past the only cleanup path).
        curator_proc: subprocess.Popen[bytes] | None = None
        sock_tmpdir: Path | None = None
        mcp_manager = None
        try:
            # Spawn the curator + connect a GraphClient so the agent
            # has access to the DAG-as-tool surface.
            sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
            sock_path = sock_tmpdir / "curator.sock"
            sock_link = layout.run_dir / "curator.sock"
            with contextlib.suppress(FileNotFoundError):
                sock_link.unlink()
            sock_link.symlink_to(sock_path)
            curator_proc = spawn_curator(
                state_dir, effective_run_id, sock_path, subdir=layout.subdir
            )
            print(f"[agent6] run id: {effective_run_id}", file=sys.stderr)

            # Spawn any configured MCP servers BEFORE the workflow
            # starts so their tools are visible from iteration 1. The manager
            # owns its subprocesses; we close it in the finally block.
            mcp_manager = _start_mcp_manager_if_enabled(cfg)

            with GraphClient(sock_path) as graph_client:
                dispatcher = ToolDispatcher(
                    root=cwd,
                    config=cfg,
                    sandbox_profile=selected_profile,
                    approver=_build_approver(layout.run_dir, events),
                    questioner=_build_questioner(layout.run_dir, events),
                    events=events,
                    graph_client=graph_client,
                    run_root_node_id=None,  # Workflow seeds the root + calls set_run_root_node_id
                    mcp_manager=mcp_manager,
                    mode=mode,
                )
                compact_drop, compact_summarise = resolve_compaction_thresholds(
                    cfg, rm_worker, log=_eprint if mode == "ask" else print
                )
                wf = Workflow(
                    root=cwd,
                    config=cfg,
                    provider=provider,
                    dispatcher=dispatcher,
                    logger=_eprint if mode == "ask" else print,
                    events=events,
                    graph_client=graph_client,
                    steer_requested=steer_state.requested,
                    steer_clear=steer_state.clear,
                    steer_prompt=steer_state.prompt,
                    budget=budget,
                    # `agent6 ask` (under asks/) is not resumable -- `agent6 resume`
                    # only looks under runs/ -- so don't write an orphan snapshot.
                    resume_state_path=(
                        None if mode == "ask" else layout.run_dir / "loop_state.json"
                    ),
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
                    critic_mode=cfg.review.trigger,
                    critic_period=cfg.review.period,
                    review_seats=review_seats,
                    review_decision=cfg.review.decision,
                    review_quorum=cfg.review.quorum,
                    review_max_total_rejections=cfg.review.max_total_rejections,
                    review_budget_fraction=cfg.review.budget_fraction,
                    review_concurrency=cfg.review.concurrency,
                    base_sha=base_sha,
                    prompt_reviser_provider=prompt_reviser_provider,
                    revise_prompt=effective_revise_prompt,
                    temperature=_role_temperature(cfg, role),
                    critic_temperature=_role_temperature(cfg, "reviewer"),
                    prompt_reviser_temperature=_role_temperature(cfg, "reviewer"),
                    prompt_revision_selector=(
                        _select_revised_prompt if effective_revise_prompt == "interactive" else None
                    ),
                    summariser_provider=summariser_provider,
                    compact_drop_at_chars=compact_drop,
                    compact_summarise_at_chars=compact_summarise,
                    context_summary_max_tokens=cfg.context.summary_max_tokens,
                )
                try:
                    with _tui_session(layout.run_dir, enabled=tui_enabled):
                        if mode == "ask" and interactive:
                            result = _run_ask_repl(wf, budget, layout, first_question=task)
                        else:
                            result = wf.run(task)
                except KeyboardInterrupt:
                    interrupted = True
                    print("\n[agent6] run interrupted", file=sys.stderr)
        except CuratorClientError as exc:
            print(f"ERROR: curator failed to start: {exc}", file=sys.stderr)
            return 1
        finally:
            if curator_proc is not None:
                curator_proc.terminate()
                try:
                    curator_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    curator_proc.kill()
            steer_state.restore()
            # Clean up the /tmp socket dir + symlink under run_dir.
            with contextlib.suppress(FileNotFoundError):
                (layout.run_dir / "curator.sock").unlink()
            if sock_tmpdir is not None:
                shutil.rmtree(sock_tmpdir, ignore_errors=True)
            if dispatcher is not None:
                dispatcher.close()
            if mcp_manager is not None:
                mcp_manager.close()
            if not interrupted and result is not None and result.completed and cfg.git.auto_merge:
                _finalize_auto_merge(cwd, layout=layout, cfg=cfg)
            # Never leave root-owned run state in the user's repo (sudo case).
            chown_to_real_user(state_dir)

        if interrupted:
            return 130
        if result is None:
            return 1

        if mode == "ask":
            # The answer IS result.summary (kept whole in ask mode). stdout gets
            # just the answer (clean for piping); cost + saved-path go to stderr.
            # The REPL already printed + saved each turn, so only the one-shot path
            # prints/saves here.
            if not interactive:
                print(result.summary)
                _save_ask_transcript(layout, question=task, answer=result.summary)
                print(
                    f"\n[agent6] answer saved to {layout.run_dir / 'transcript.md'}",
                    file=sys.stderr,
                )
            print(budget.format_summary(), file=sys.stderr)
            return 0 if result.completed else 1

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
        return _run_exit_code(result)
    finally:
        # Single owner of worker.pid, egress-broker, and auto-stash
        # finalization. Refusal returns, Ctrl-C during verify inference, and
        # setup-window crashes used to skip these, leaving a stale pid, a
        # leaked broker process, and the user's stashed work silently hidden.
        clear_worker_pid(layout.run_dir)
        _stop_egress(egress_broker, egress_sock_dir)
        if stashed:
            _finalize_auto_stash(
                cwd,
                base_branch=base_branch,
                run_branch=run_branch,
                auto_pop=cfg.git.auto_stash_pop,
            )
        _release_single_writer(worker_lock_fd)


def _finalize_auto_merge(cwd: Path, *, layout: RunLayout, cfg: Config) -> None:
    """After a successful run, merge the run branch into its base using
    git.merge_strategy (git.auto_merge). Reads the run context from the manifest, so
    run + resume share it. Ends on the base branch (the pre-run branch) with a clean
    tree. Non-fatal and best-effort: on conflict or error the run branch is left
    intact and the message says how to merge by hand. No-op when branch_per_run was
    off."""
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    run_branch = manifest.get("run_branch")
    base_branch = manifest.get("base_branch")
    if not run_branch or not base_branch:
        return  # branch_per_run was off: the work already landed on the base branch
    run_branch, base_branch = str(run_branch), str(base_branch)
    try:
        st = git_status(cwd)
    except GitError:
        return
    if not st.is_clean:
        print(
            f"[agent6] auto_merge skipped (worktree not clean); merge by hand:\n"
            f"    git checkout {base_branch} && git merge {run_branch}",
            file=sys.stderr,
        )
        return
    identity = CommitIdentity(
        name=cfg.git.commit.name, email=cfg.git.commit.email, coauthor=cfg.git.commit.coauthor
    )
    try:
        verify_git_identity(cwd, identity)
    except GitError as exc:
        print(f"[agent6] auto_merge skipped: {exc}", file=sys.stderr)
        return
    outcome = execute_merge(
        cwd,
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=base_branch,
        base_sha=str(manifest.get("base_sha") or ""),
        strategy=cfg.git.merge_strategy,
        message=None,
        cfg=cfg,
        identity=identity,
        original="",  # stay on the base branch, where the work now lives
    )
    if outcome.status == "merged":
        print(
            f"[agent6] auto_merged {run_branch} into {base_branch} "
            f"({cfg.git.merge_strategy}) -> {outcome.merged_sha[:12]}",
            file=sys.stderr,
        )
        if cfg.git.auto_prune:
            if delete_branch_if_merged(cwd, run_branch):
                print(f"[agent6] auto_pruned {run_branch}", file=sys.stderr)
            else:
                print(
                    f"[agent6] auto_prune kept {run_branch} (squash-merged, unreachable; "
                    f"remove with: git branch -D {run_branch})",
                    file=sys.stderr,
                )
    elif outcome.status == "conflict":
        print(
            f"[agent6] auto_merge into {base_branch} hit conflicts "
            f"({', '.join(outcome.conflicts)}); left a clean tree on {base_branch} with the run "
            f"branch {run_branch} intact. Merge by hand:\n    git merge {run_branch}",
            file=sys.stderr,
        )
    else:
        print(f"[agent6] auto_merge failed: {outcome.error}", file=sys.stderr)


def _finalize_auto_stash(
    cwd: Path, *, base_branch: str, run_branch: str | None, auto_pop: bool
) -> None:
    """Restore or report the pre-run auto-stash so the user's work is never left in a
    hidden stash. With auto_pop off, print how to pop it. With auto_pop on, pop it
    onto the base branch when that is safe (clean worktree, conflict-free apply);
    otherwise leave the stash with a message. Never reset --hard (refused)."""
    recover = f"git checkout {base_branch} && git stash pop" if run_branch else "git stash pop"
    if not auto_pop:
        print(
            f"[agent6] pre-run changes are stashed; restore them with: {recover}", file=sys.stderr
        )
        return
    try:
        st = git_status(cwd)
    except GitError:
        st = None
    if st is None or not st.is_clean:
        print(
            f"[agent6] pre-run changes left stashed (worktree not clean); restore with: {recover}",
            file=sys.stderr,
        )
        return
    if run_branch and st.branch == run_branch:
        if not branch_exists(cwd, base_branch):
            print(
                f"[agent6] base branch {base_branch} no longer exists; pre-run changes left "
                f"stashed (recover with: git stash pop)",
                file=sys.stderr,
            )
            return
        try:
            create_branch(cwd, base_branch)  # checks out the existing base branch
        except GitError as exc:
            print(
                f"[agent6] could not switch to {base_branch} to restore the stash ({exc}); "
                f"restore with: {recover}",
                file=sys.stderr,
            )
            return
    if restore_stash(cwd):
        print(f"[agent6] restored your pre-run changes onto {base_branch}", file=sys.stderr)
    else:
        print(
            "[agent6] restoring your pre-run changes hit a conflict; resolve the markers"
            " (your stash is preserved at stash@{0})",
            file=sys.stderr,
        )


def _fire_notify_hook(
    notify: NotifyConfig,
    *,
    run_id: str,
    run_dir: Path,
    ok: bool,
    reason: str,
) -> None:
    """Run the operator-configured post-completion hook.

    The argv comes from `[notify].on_complete` in your config, operator-
    controlled, not LLM-controlled, so it does not go through the jail.
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


def _ensure_on_run_branch(cwd: Path, layout: RunLayout) -> str | None:
    """Check out the run's branch if HEAD isn't already on it.

    The loop's per-step commits land on whatever branch HEAD points at, so a
    resume must be on the run's branch. ``_cmd_run`` checks it out up front, but
    two paths reach resume off the run branch: ``agent6 fork`` cuts
    ``agent6/<id>`` additively (never switching to it), and an operator may have
    moved branches since the original run. Either way, without this the work
    silently lands on the operator's current branch and the run branch stays
    empty (so ``runs diff`` shows nothing).

    Reads ``run_branch`` from the manifest. Returns None when there's nothing to
    do (no branch recorded, or already on it) or after a clean checkout; returns
    an error string when a switch is needed but the working tree is dirty.
    """
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    run_branch = manifest.get("run_branch")
    try:
        st = git_status(cwd)
    except GitError:
        st = None
    # Nothing to do: branch_per_run was off (no run_branch), git unreadable, or
    # already on the run branch. Commits then land on HEAD as before.
    if not run_branch or st is None or st.branch == run_branch:
        return None
    # Only MODIFIED tracked files block the switch; untracked files are carried
    # across a checkout fine (and a rare untracked-vs-target collision is caught
    # by the create_branch error below), so don't refuse on those.
    if st.modified_count > 0:
        return (
            f"ERROR: resume needs to switch to this run's branch {run_branch!r}, but the "
            "working tree has uncommitted changes to tracked files. Commit or stash them "
            f"(or run `git checkout {run_branch}` yourself), then resume."
        )
    try:
        create_branch(cwd, run_branch)  # idempotent: checks out the existing branch
    except GitError as exc:
        return f"ERROR: could not switch to run branch {run_branch!r}: {exc}"
    return None


def snapshot_head_mismatch(snapshot_path: Path, repo_root: Path) -> tuple[str, str] | None:
    """(snapshot head, current head) when the workspace HEAD DIVERGED from the
    run's last snapshot, else None.

    Divergence, not mere movement: the run's own per-step commits advance HEAD
    forward from the snapshot between snapshot writes (a turn commits, then a
    critic/metric call runs before the next snapshot), so a kill in that window
    leaves HEAD ahead of the recorded head_sha on the SAME line. That must
    resume cleanly. Only refuse when HEAD is not a descendant of the snapshot
    head -- an operator commit on another line, a rebase, a reset, or a
    snapshot commit that git-gc made unreachable -- i.e. the model would resume
    against code that changed under it. Working-tree (uncommitted) divergence
    is not checked; only committed history.

    Best-effort: the snapshot records head_sha as "" when git was unreadable at
    write time (skip), a corrupt snapshot file is left for the loud
    resume-snapshot load to report (skip), and a non-repo raises nothing here
    (the caller's _require_git_repo already ran).
    """
    snap_head = ""
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            snap_head = str(loaded.get("head_sha") or "")
    if not snap_head:
        return None
    try:
        current_head = git_status(repo_root).head_sha
    except GitError:
        return None
    if not current_head or current_head == snap_head:
        return None
    if is_ancestor(repo_root, snap_head, current_head):
        # HEAD moved forward from the snapshot on the same line (the run's own
        # commits): not divergence.
        return None
    return (snap_head, current_head)


def _cmd_resume(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    run_id: str,
    *,
    force: bool,
    tui: bool = False,
    budget_overrides: _BudgetOverrides | None = None,
    sandbox_overrides: _SandboxOverrides | None = None,
    profile: str = "",
) -> int:
    """Resume a paused/crashed run from its snapshot.

    Mirrors ``_cmd_run`` setup but uses the existing run id, refuses
    if no ``loop_state.json`` snapshot exists, and calls ``wf.resume()``
    instead of ``wf.run(task)``. A safety check refuses when the
    workspace HEAD DIVERGED from the snapshot (a rebase/reset/commit on
    another line); plain forward movement on the same line resumes
    cleanly. ``--force-resume`` overrides the refusal.

    NOTE: token budget on resume is a FRESH ceiling, not a continuation
    of the prior run's accounting. Each ``agent6 resume`` invocation
    starts at 0 tokens against ``[budget].max_input_tokens`` /
    ``max_output_tokens``. This is by design - the budget is a per-
    invocation runaway-cost circuit breaker.
    """
    cwd = Path.cwd()
    state_dir = _state_dir(cwd)
    runs_dir = state_dir / "runs"
    if not run_id:
        # "resume my last run" -- the common recovery case, matching `runs *`.
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}; nothing to resume.", file=sys.stderr)
            return 2
        run_id = latest
        print(f"[agent6] resuming most recent run: {run_id}", file=sys.stderr)
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_id = resolved
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    if not layout.run_dir.is_dir():
        print(f"ERROR: no such run dir: {layout.run_dir}", file=sys.stderr)
        return 2
    # One authoritative writer per run dir (see _acquire_single_writer). Refuse a
    # second resume of a still-live run before touching any shared state.
    worker_lock_fd = _acquire_single_writer(layout.run_dir)
    if worker_lock_fd is None:
        print(_SINGLE_WRITER_BUSY.format(rid=run_id), file=sys.stderr)
        return 2
    # Drop a prior session's stale answer files + frontend.pid (the id counters reset
    # on resume, an old answer must not be read instead of re-prompting).
    clear_pending_answers(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    egress_broker = None
    egress_sock_dir = None
    try:
        snapshot_path = layout.run_dir / "loop_state.json"
        if not snapshot_path.is_file():
            print(
                f"ERROR: no resume snapshot at {snapshot_path}; nothing to resume.",
                file=sys.stderr,
            )
            return 2

        # Friendly no-repo guard BEFORE any git-touching resume-diff (which would
        # otherwise print zeroed-out heads first, then the real error).
        if not _require_git_repo(cwd):
            return 2

        # Get onto the run's branch before diffing or committing. A fork cuts
        # agent6/<id> without checking it out; do it here so the loop's commits land
        # on the run branch and the resume-diff below references the right HEAD.
        branch_err = _ensure_on_run_branch(cwd, layout)
        if branch_err is not None:
            print(branch_err, file=sys.stderr)
            return 2

        # Safety check: refuse when the workspace HEAD DIVERGED from the run's last
        # snapshot (a rebase, reset, or a commit on another line would leave the
        # model reasoning about code that changed under it). Plain forward movement
        # on the same line -- the run's own per-step commits -- resumes cleanly. The
        # snapshot records head_sha best-effort ("" when git was unreadable at write
        # time); skip the check then, and let the loud snapshot load below handle a
        # corrupt file.
        mismatch = snapshot_head_mismatch(snapshot_path, cwd)
        if mismatch is not None:
            snap_head, current_head = mismatch
            print(
                "GUARD: the workspace HEAD diverged from this run's last snapshot.",
                file=sys.stderr,
            )
            print(f"  snapshot head: {snap_head}", file=sys.stderr)
            print(f"  current head:  {current_head}", file=sys.stderr)
            if not force:
                print(
                    "REFUSING to resume. Re-run with --force-resume to override.",
                    file=sys.stderr,
                )
                return 1

        # The original run's manifest drives resume: `mode` (a plan run resumes
        # read-only with the plan tools, never as a write run), `profile` (resume
        # has no --profile flag), and `base_sha` (the review-panel diff base).
        manifest: dict[str, Any] = {}
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        mode: Literal["run", "plan"] = "plan" if manifest.get("mode") == "plan" else "run"
        workflow_section = manifest.get("workflow")
        manifest_profile = (
            str(workflow_section.get("profile") or "") if isinstance(workflow_section, dict) else ""
        )
        resume_base_sha = str(manifest.get("base_sha") or "")
        try:
            cfg = load_effective(
                Path.cwd(), config_path, profile=profile or manifest_profile
            ).config
            set_repo_hook_policy(cfg.git.run_repo_hooks)
            if budget_overrides is not None:
                cfg = budget_overrides.apply(cfg)
            if sandbox_overrides is not None:
                cfg = sandbox_overrides.apply(cfg)
            cfg.require_runnable("worker")
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2

        env = detect_env()
        try:
            selected_profile = select_profile(cfg.sandbox.profile, env)
        except RuntimeError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 2
        _warn_if_unsandboxed(selected_profile)
        if not _confirm_unconfined_autorun(selected_profile, cfg):
            print("[agent6] aborted.", file=sys.stderr)
            return 1

        net_err = _check_network_profile(cfg, selected_profile)
        if net_err is not None:
            print(f"REFUSING: {net_err}", file=sys.stderr)
            return 2
        # strict can be selected because the jail launcher has userns, yet this
        # process can't create one for the egress broker (surgical AppArmor profile).
        # Downgrade auto->hardened, or refuse an explicit strict, with guidance.
        selected_profile, egress_err = resolve_strict_egress_viability(cfg, selected_profile)
        if egress_err is not None:
            print(egress_err, file=sys.stderr)
            return 2

        missing = _check_provider_keys(cfg)
        usd_err = _explicit_usd_flag_error(
            budget_overrides.max_usd if budget_overrides else None, cfg
        )
        if usd_err is not None:
            print(f"REFUSING: {usd_err}", file=sys.stderr)
            return 2
        if missing is not None:
            print(missing, file=sys.stderr)
            return 2

        identity = CommitIdentity(
            name=cfg.git.commit.name,
            email=cfg.git.commit.email,
            coauthor=cfg.git.commit.coauthor,
        )
        # (no-repo guard already ran above, before the resume head guard)
        try:
            verify_git_identity(cwd, identity)
        except GitError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

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
            # The egress broker is already running; the outer finally tears it
            # down (and its socket dir) so this refusal does not leak it.
            return 2

        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
            max_usd=cfg.budget.best_effort_usd_limit,
        )

        worker_inner = _build_role_provider(
            cfg, "worker", transcript_sink=transcript_sink, budget=budget
        )
        rm_worker = cfg.models.resolve("worker")
        assert rm_worker is not None  # require_runnable validated this
        _warn_if_usd_unenforceable(cfg)
        _warn_if_prompt_override_incomplete(cfg)
        tui_enabled = _should_spawn_tui(tui=tui, interactive=False, mode=mode)
        _warn_if_headless_ask(cfg, tui_enabled=tui_enabled)
        stream_text, console_stream = _stream_modes(tui_enabled=tui_enabled)
        provider: Provider = _InstrumentedProvider(
            inner=worker_inner,
            role="worker",
            model=rm_worker.model,
            provider_name=rm_worker.provider,
            events=events,
            budget=budget,
            stream_text=stream_text,
            console_stream=console_stream,
        )

        critic_provider = _build_critic_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        summariser_provider = _build_summariser_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        review_seats = (
            build_review_seats(
                cfg,
                transcript_sink=transcript_sink,
                budget=budget,
                n=cfg.review.panel_size,
                personas=cfg.review.personas,
            )
            if cfg.review.trigger != "off" and review_panel_configured(cfg)
            else []
        )
        # Resume reuses the verify command the ORIGINAL run resolved (stored in the
        # snapshot), so the tool list, prompt, and commit branch stay consistent with
        # the frozen system prompt -- never re-inferring (which could flip and
        # diverge). Fall back to re-inference only for a pre-field snapshot, and only
        # when the operator hasn't since pinned a command in config.
        if not cfg.workflow.verify_command:
            snap_verify: tuple[str, ...] | None = None
            if snapshot_path.is_file():
                try:
                    snap_verify = load_resume_snapshot(snapshot_path).verify_command
                except (ValueError, OSError, KeyError):
                    snap_verify = None
            if snap_verify is None:  # older snapshot: re-infer as the original did
                cfg = _infer_verify_if_unset(
                    cfg,
                    cwd,
                    mode=mode,
                    events=events,
                    transcript_sink=transcript_sink,
                    budget=budget,
                )
            elif snap_verify:  # () means the original run was gateless: stay gateless
                cfg = cfg.with_inferred_verify(snap_verify)
                print(
                    f"[agent6] reusing this run's verify command: {' '.join(snap_verify)}",
                    file=sys.stderr,
                )

        sock_path = layout.run_dir / "curator.sock"  # rebound to the /tmp socket inside the try

        steer_state = _make_steer_state(events, layout.run_dir)

        result = None
        interrupted = False
        dispatcher: ToolDispatcher | None = None
        # Spawned inside the try so the finally below always tears them down even
        # if a spawn itself fails (otherwise curator/MCP procs + the /tmp socket
        # dir leak past the only cleanup path).
        curator_proc: subprocess.Popen[bytes] | None = None
        sock_tmpdir: Path | None = None
        mcp_manager = None
        try:
            sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
            sock_path = sock_tmpdir / "curator.sock"
            sock_link = layout.run_dir / "curator.sock"
            with contextlib.suppress(FileNotFoundError):
                sock_link.unlink()
            sock_link.symlink_to(sock_path)
            curator_proc = spawn_curator(state_dir, run_id, sock_path, subdir=layout.subdir)
            print(f"[agent6] resume run id: {run_id}", file=sys.stderr)

            mcp_manager = _start_mcp_manager_if_enabled(cfg)

            with GraphClient(sock_path) as graph_client:
                dispatcher = ToolDispatcher(
                    root=cwd,
                    config=cfg,
                    sandbox_profile=selected_profile,
                    approver=_build_approver(layout.run_dir, events),
                    questioner=_build_questioner(layout.run_dir, events),
                    events=events,
                    graph_client=graph_client,
                    run_root_node_id=None,
                    mcp_manager=mcp_manager,
                    mode=mode,
                )
                compact_drop, compact_summarise = resolve_compaction_thresholds(
                    cfg, rm_worker, log=print
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
                    mode=mode,
                    plan_output_path=(layout.run_dir / "plan.md" if mode == "plan" else None),
                    critic_provider=critic_provider,
                    critic_mode=cfg.review.trigger,
                    critic_period=cfg.review.period,
                    review_seats=review_seats,
                    review_decision=cfg.review.decision,
                    review_quorum=cfg.review.quorum,
                    review_max_total_rejections=cfg.review.max_total_rejections,
                    review_budget_fraction=cfg.review.budget_fraction,
                    review_concurrency=cfg.review.concurrency,
                    base_sha=resume_base_sha,
                    temperature=_role_temperature(cfg, "worker"),
                    critic_temperature=_role_temperature(cfg, "reviewer"),
                    summariser_provider=summariser_provider,
                    compact_drop_at_chars=compact_drop,
                    compact_summarise_at_chars=compact_summarise,
                    context_summary_max_tokens=cfg.context.summary_max_tokens,
                )
                try:
                    with _tui_session(layout.run_dir, enabled=tui_enabled):
                        result = wf.resume()
                except ResumeError as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 1
                except KeyboardInterrupt:
                    interrupted = True
                    print("\n[agent6] resume interrupted", file=sys.stderr)
        except CuratorClientError as exc:
            print(f"ERROR: curator failed to start: {exc}", file=sys.stderr)
            return 1
        finally:
            if curator_proc is not None:
                curator_proc.terminate()
                try:
                    curator_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    curator_proc.kill()
            steer_state.restore()
            with contextlib.suppress(FileNotFoundError):
                (layout.run_dir / "curator.sock").unlink()
            if sock_tmpdir is not None:
                shutil.rmtree(sock_tmpdir, ignore_errors=True)
            if dispatcher is not None:
                dispatcher.close()
            if mcp_manager is not None:
                mcp_manager.close()
            # Egress teardown is owned by the outer finally (a single call).
            # Doing it here too would reap the broker pid, then the auto-merge
            # git subprocesses and the notify hook below could recycle it before
            # the outer close() signalled the pid again.
            if not interrupted and result is not None and result.completed and cfg.git.auto_merge:
                _finalize_auto_merge(cwd, layout=layout, cfg=cfg)
            # Never leave root-owned run state in the user's repo (sudo case).
            chown_to_real_user(state_dir)

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
        return _run_exit_code(result)
    finally:
        # Single owner of worker.pid + egress teardown for every resume exit
        # path; refusals and Ctrl-C during verify inference used to leak both.
        clear_worker_pid(layout.run_dir)
        _stop_egress(egress_broker, egress_sock_dir)
        _release_single_writer(worker_lock_fd)
