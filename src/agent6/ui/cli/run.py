# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run` and `agent6 resume` plus their shared execution scaffolding."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Literal

from agent6.budget import BudgetTracker
from agent6.config import (
    Config,
    ConfigError,
    NotifyConfig,
    RoleName,
)
from agent6.config.layer import (
    load_effective,
)
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_exists,
    create_branch,
    delete_branch_if_merged,
    is_ancestor,
    restore_stash,
    set_repo_hook_policy,
    stash_all,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.graph.client import CuratorClientError, GraphClient, spawn_curator
from agent6.paths import (
    chown_to_real_user,
)
from agent6.providers import (
    Provider,
    TranscriptSink,
)
from agent6.runs.id import RunIdError, new_friendly_id, resolve_run_id
from agent6.runs.layout import RunLayout
from agent6.sandbox.detect import ProfileUnavailableError, select_profile
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.schema import UserQuestion
from agent6.ui.bridge.approval import (
    away_mode,
    clear_away_mode,
    clear_pending_answers,
    clear_worker_pid,
    frontend_is_live,
    read_answer,
    read_question_answers,
    session_allow_set,
    set_away_mode,
    set_session_allow,
    steer_answer_is_abort,
    write_worker_pid,
)
from agent6.ui.bridge.spawn import spawn_detached_resume
from agent6.ui.cli._ask import (
    run_ask_repl as _run_ask_repl,
)
from agent6.ui.cli._ask import (
    save_ask_transcript as _save_ask_transcript,
)
from agent6.ui.cli._common import (
    _BudgetOverrides,
    _check_provider_keys,
    _explicit_usd_flag_error,
    _SandboxOverrides,
    _start_mcp_manager_if_enabled,
    _state_dir,
    detect_env,
)
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._manifest import (
    write_run_manifest as _write_run_manifest,
)
from agent6.ui.cli._merge import execute_merge
from agent6.ui.cli._preflight import (
    confirm_run_on_run_branch as _confirm_run_on_run_branch,
)
from agent6.ui.cli._preflight import (
    confirm_unconfined_autorun as _confirm_unconfined_autorun,
)
from agent6.ui.cli._preflight import (
    infer_verify_if_unset as _infer_verify_if_unset,
)
from agent6.ui.cli._preflight import (
    require_git_repo as _require_git_repo,
)
from agent6.ui.cli._preflight import (
    warn_if_headless_ask as _warn_if_headless_ask,
)
from agent6.ui.cli._preflight import (
    warn_if_prompt_override_incomplete as _warn_if_prompt_override_incomplete,
)
from agent6.ui.cli._preflight import (
    warn_if_usd_unenforceable as _warn_if_usd_unenforceable,
)
from agent6.ui.cli._repl import build_repl_hook as _build_repl_hook
from agent6.ui.cli._single_writer import (
    SINGLE_WRITER_BUSY as _SINGLE_WRITER_BUSY,
)
from agent6.ui.cli._single_writer import (
    acquire_single_writer as _acquire_single_writer,
)
from agent6.ui.cli._single_writer import (
    release_single_writer as _release_single_writer,
)
from agent6.ui.cli._steer import (
    make_steer_state as _make_steer_state,
)
from agent6.ui.cli._steer import (
    select_revised_prompt as _select_revised_prompt,
)
from agent6.ui.cli._steer import (
    tty_message as _tty_message,
)
from agent6.ui.cli._steer import (
    tty_prompt as _tty_prompt,
)
from agent6.ui.cli._task_refs import (
    expand_task_file_refs as _expand_task_file_refs,
)
from agent6.ui.cli.egress import (
    EgressGuard,
    _check_network_profile,
    _maybe_apply_agent_landlock,
    _maybe_start_egress,
    _stop_egress,
    _warn_if_unsandboxed,
    resolve_strict_egress_viability,
)
from agent6.ui.cli.plan_watch import _most_recent_run_id
from agent6.ui.cli.providers import (
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
from agent6.ui.viewmodel import status_word
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


def _loop_logger(mode: str, console_view: ConsoleView | None) -> Callable[[str], None]:
    """The workflow's text logger.

    When the live ConsoleView is rendering the product stream (foreground run),
    notices go THROUGH it (``console_view.notice``) so each clears the spinner
    line first and writes to the same stream under the same lock -- otherwise a
    notice printed to stdout while the stderr spinner is up garbles the line. The
    loop's internal state narration (``LOOP: LOAD_CONTEXT``, ``compaction: …``)
    is pure noise on the glyph stream, so it is suppressed unless
    ``AGENT6_DEBUG=1``; genuine notices (auto-commit, tool errors, critic
    decisions) pass. Headless/`ask` keep the full trace on their own stream (the
    log, not a live stream)."""
    if console_view is None:
        return _eprint if mode == "ask" else print
    debug = os.environ.get("AGENT6_DEBUG") == "1"

    def _filtered(msg: str) -> None:
        stripped = msg.removeprefix("[agent6] ")
        if not debug and ("LOOP:" in msg or stripped.startswith("compaction:")):
            return
        console_view.notice(msg)

    return _filtered


def _print_run_end(
    result: RunResult, *, layout: RunLayout, budget: BudgetTracker, console_stream: bool
) -> None:
    """One composed end-of-run block: outcome, summary, cost, and the next step.

    Replaces the old `result: completed=True reason=... iterations=...` repr line
    plus a re-print of the summary. When the live ConsoleView already rendered the
    `● done <summary>` terminator (console_stream), this omits the summary and
    just adds what the stream lacks: cost and the branch / next-step footer."""
    word, reason = status_word(finished=True, all_passed=result.completed, end_reason=result.reason)
    if not console_stream:
        # Headless: no ConsoleView ran, so this block is the only end output.
        headline = word if not reason else f"{word} · {reason.replace('_', ' ')}"
        print(f"\n{headline}")
        if result.summary:
            print(f"  {result.summary}")
    print()
    print(budget.format_summary())
    run_branch = ""
    with contextlib.suppress(OSError, ValueError):
        run_branch = json.loads(layout.manifest_path.read_text(encoding="utf-8")).get(
            "run_branch", ""
        )
    if result.completed and run_branch:
        print(f"\nchanges are on {run_branch}")
        print(f"  merge with:  agent6 runs merge {layout.run_id}")
        print(f"  inspect:     agent6 runs diff {layout.run_id}")
    elif not result.completed:
        print(f"\nresume with:  agent6 resume {layout.run_id}")


def _eprint(msg: str) -> None:
    """Loop logger that writes to stderr (used for `ask`, whose stdout is the
    answer and must stay clean for piping)."""
    print(msg, file=sys.stderr)


def _default_stdin_approver(prompt: str) -> str:
    """Plain-terminal fallback for tool approval (no live TUI, or its answer
    timed out). Returns "yes", "no", or "session" (allow all for the rest of this
    run). Routed via /dev/tty so the prompt stays visible when a TUI has redirected
    the std streams to its console log; plain stdin without one."""
    ans = _tty_prompt(f"{prompt} [y/N/a]  (a = allow all this session): ")
    if ans is None:
        return "no"
    ans = ans.strip().lower()
    if ans in {"a", "all", "always", "session"}:
        return "session"
    return "yes" if ans in {"y", "yes"} else "no"


def _prompt_detach_away_mode(run_dir: Path) -> None:
    """On detach with run_commands=ask, ask how approvals/questions should be handled
    while nothing is watching, and record it for the background run. Non-interactive
    (no tty) -> deny, the safe default."""
    if not sys.stdin.isatty():
        set_away_mode(run_dir, "deny")
        return
    print(
        "[agent6] Detaching with run_commands=ask -- nothing will be watching to approve.",
        file=sys.stderr,
    )
    ans = _tty_prompt(
        "  While away: [a]pprove all / [d]eny all / [w]ait for a reattached front-end? [d]: ",
        fall_back_to_stdin=False,
    )
    choice = (ans or "").strip().lower()
    if choice in {"a", "approve"}:
        set_session_allow(run_dir)  # reuse the session-allow marker
        print("  -> approving every run_command.", file=sys.stderr)
    elif choice in {"w", "wait"}:
        set_away_mode(run_dir, "wait")
        print("  -> waiting; reattach (agent6 watch / the TUI) to approve.", file=sys.stderr)
    else:
        set_away_mode(run_dir, "deny")
        print("  -> denying run_commands until you reattach.", file=sys.stderr)


def _wait_for_reply(run_dir: Path, read_once: Callable[[], object | None]) -> object | None:
    """Detach 'wait' mode: block until a reattached front-end supplies an answer
    (``read_once`` returns non-None) or stops the run. ``read_once`` polls a live
    front-end for up to its own (short) timeout; between calls, when no front-end is
    attached yet, we poll for one. A reattached front-end's Stop lands as a steer
    abort, which breaks the wait so the run can end. Returns the reply, or None on
    stop."""
    while True:
        if steer_answer_is_abort(run_dir):
            return None
        if frontend_is_live(run_dir):
            reply = read_once()
            if reply is not None:
                return reply
        else:
            time.sleep(1.0)  # no front-end yet; poll for one to reattach


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
        # Already granted for the session (this run + its resumes) -> auto-pass.
        if session_allow_set(run_dir):
            events.emit("approval.answer", id=prompt_id, approved=True, source="session")
            return True
        # A detached run chose how to handle approvals while away (deny / wait).
        away = away_mode(run_dir)
        if away == "deny":
            events.emit("approval.answer", id=prompt_id, approved=False, source="away-deny")
            return False
        events.emit("approval.prompt", id=prompt_id, prompt=prompt)
        if away == "wait":  # block until a reattached front-end answers (or stops)
            reply = _wait_for_reply(
                run_dir, lambda: read_answer(run_dir, prompt_id, timeout_s=20.0, dead_grace_s=8.0)
            )
            approved = bool(reply)
            events.emit("approval.answer", id=prompt_id, approved=approved, source="away-wait")
            return approved
        approved: bool | None = None
        source = "stdin"
        if frontend_is_live(run_dir):
            # A front-end that chose "allow session" set the marker itself, so this
            # answer is just yes; the check above auto-passes every later prompt.
            approved = read_answer(run_dir, prompt_id)
            if approved is not None:
                source = "tui"
        if approved is None:
            answer = _default_stdin_approver(prompt)
            if answer == "session":
                set_session_allow(run_dir)
            approved = answer != "no"
        events.emit("approval.answer", id=prompt_id, approved=approved, source=source)
        return approved

    return approve


def _spawn_detached(guard: EgressGuard, cwd: Path, run_id: str) -> str:
    """Spawn the detached background resume for this run.

    Under network isolation a direct spawn inherits the empty namespace and the
    resume's provider egress is dead on arrival, so it goes through the
    pre-forked host spawner; without isolation the direct spawn is fine."""
    if guard.detach_spawner is not None:
        return guard.detach_spawner.spawn_resume(cwd, run_id)
    return spawn_detached_resume(cwd, run_id)


def _build_questioner(
    run_dir: Path, events: EventSink
) -> Callable[[tuple[UserQuestion, ...]], tuple[str, ...]]:
    """Build the `ask_user` questioner, bridged to a live TUI when present.

    Emits a `question.prompt` event; if a TUI is live the answer comes from its
    question modal via `questions/<id>.answer`, otherwise (or if the TUI dies /
    times out) it falls back to a numbered stdin prompt. A headless run (no TUI,
    no TTY) gets an empty answer rather than hanging. Emits `question.answer`."""
    counter = {"n": 0}

    def ask(questions: tuple[UserQuestion, ...]) -> tuple[str, ...]:
        counter["n"] += 1
        question_id = f"question-{counter['n']}"
        events.emit(
            "question.prompt",
            id=question_id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
        )
        answers: tuple[str, ...] | None = None
        source = "stdin"
        if away_mode(run_dir) == "wait":
            # Detached 'wait': block until a reattached front-end answers (or stops).
            reply = _wait_for_reply(
                run_dir,
                lambda: read_question_answers(
                    run_dir, question_id, timeout_s=20.0, dead_grace_s=8.0
                ),
            )
            answers = reply if isinstance(reply, tuple) else tuple("" for _ in questions)
            source = "away-wait"
        elif frontend_is_live(run_dir):
            answers = read_question_answers(run_dir, question_id)
            if answers is not None:
                source = "tui"
        if answers is None:
            stdin_answers = _default_stdin_questioner(questions)
            if stdin_answers is None:
                # No front-end and no controlling terminal: nobody saw the
                # question. Answer empty so the run never hangs, and say so
                # where a watcher will see it instead of failing silently.
                answers = tuple("" for _ in questions)
                source = "headless-default"
                _tty_message(
                    "[agent6] ask_user: no front-end attached and no terminal;"
                    " returning empty answers\n"
                )
            else:
                answers = stdin_answers
        events.emit("question.answer", id=question_id, answers=list(answers), source=source)
        return answers

    return ask


def _ask_one_stdin(q: UserQuestion, prefix: str = "") -> str | None:
    """Prompt one question on /dev/tty; a digit picks an option, else free text.
    None means no terminal (headless)."""
    lines = [
        f"{prefix}{q.question}",
        *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1)),
    ]
    ans = _tty_prompt("\n".join(lines) + "\n> ", fall_back_to_stdin=False)
    if ans is None:
        return None
    ans = ans.strip()
    if ans.isdigit() and 1 <= int(ans) <= len(q.options):
        return q.options[int(ans) - 1]
    return ans


def _default_stdin_questioner(questions: tuple[UserQuestion, ...]) -> tuple[str, ...] | None:
    """Ask each question on /dev/tty (visible under a TUI's stream redirect). For a
    series, print a summary afterwards and let the operator revise any answer (type
    its number) before submitting (blank). Returns None without a controlling
    terminal (headless) so the caller can answer empty -- never hanging or eating
    piped stdin -- and say so."""
    answers: list[str] = []
    multi = len(questions) > 1
    for i, q in enumerate(questions, start=1):
        prefix = f"[{i}/{len(questions)}] " if multi else ""
        ans = _ask_one_stdin(q, prefix)
        if ans is None:
            return None  # no tty: never block
        answers.append(ans)
    while multi:  # review + revise loop; blank submits
        summary = "\n".join(
            f"  {n}) {q.question} -> {a or '(empty)'}"
            for n, (q, a) in enumerate(zip(questions, answers, strict=True), start=1)
        )
        pick = _tty_prompt(
            f"Review:\n{summary}\nEnter to submit, or a number to change that answer: ",
            fall_back_to_stdin=False,
        )
        if pick is None or not pick.strip():
            break
        if pick.strip().isdigit() and 1 <= int(pick.strip()) <= len(questions):
            j = int(pick.strip()) - 1
            revised = _ask_one_stdin(questions[j])
            if revised is not None:
                answers[j] = revised
    return tuple(answers)


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
    ``role.thinking_delta`` events, which every live view renders as the model's
    reasoning + answer. ``console_stream`` additionally subscribes a
    ``ConsoleView`` to the EventSink, rendering the live conversation -- reasoning,
    text, and every tool call with its result -- to stderr.

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
            [sys.executable, "-m", "agent6.ui.tui", "--watch", str(run_dir), "--exit-on-end"]
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


def _cmd_run(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    task: str,
    *,
    run_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    decompose: bool = False,
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
        if decompose:  # --decompose: plan-first for this run (overrides config)
            cfg = cfg.model_copy(
                update={"prompt": cfg.prompt.model_copy(update={"decompose": True})}
            )
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
    except ProfileUnavailableError as exc:
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
        # Starting a run while checked out on ANOTHER run's branch (agent6/<id>) is
        # usually a slip -- the operator forgot to merge or switch back -- so the new
        # run would pile on top of an unmerged one. Confirm; they may instead intend
        # to continue that line with a fresh session, in which case proceed.
        if (
            mode == "run"
            and base_branch.startswith("agent6/")
            and not _confirm_run_on_run_branch(base_branch)
        ):
            print(
                "[agent6] aborted. Merge (agent6 runs merge) or switch branches first,"
                " then re-run.",
                file=sys.stderr,
            )
            return 2

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
    if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
        clear_away_mode(layout.run_dir)
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

    egress_guard = EgressGuard()
    console_view: ConsoleView | None = None
    run_branch: str | None = None
    detach_requested = False
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

        egress_guard, egress_err = _maybe_start_egress(
            cfg, selected_profile, with_detach_spawner=True
        )
        if egress_err is not None:
            print(f"REFUSING: {egress_err}", file=sys.stderr)
            return 2
        if egress_guard.broker is not None:
            print(
                f"[agent6] provider-only egress: confined to host network "
                f"namespace via broker pid {egress_guard.broker.pid}",
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
        if console_stream:
            console_view = ConsoleView(sys.stderr)
            events.subscribe(console_view)
        provider: Provider = _InstrumentedProvider(
            inner=worker_inner,
            role=role,
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
                loop_log = _loop_logger(mode, console_view)
                compact_drop, compact_summarise = resolve_compaction_thresholds(
                    cfg, rm_worker, log=loop_log
                )
                wf = Workflow(
                    root=cwd,
                    config=cfg,
                    provider=provider,
                    dispatcher=dispatcher,
                    logger=loop_log,
                    events=events,
                    graph_client=graph_client,
                    steer_requested=steer_state.requested,
                    steer_clear=steer_state.clear,
                    steer_prompt=steer_state.prompt,
                    should_abort=steer_state.abort_pending,
                    should_interrupt=steer_state.requested,
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
                    # The loop was cut mid-step, so it never emitted run.end; do it
                    # here so an attached watcher/TUI stops instead of hanging.
                    events.emit("run.end", reason="interrupted", all_passed=False)
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

        if result.reason == "detached":
            # Keep going in the background: the outer finally releases this run's
            # worker lock, then spawns a detached `resume` that picks it up.
            detach_requested = True
            print(f"\n[agent6] detached: {layout.run_id} continues in the background.")
            print(f"          reattach:  agent6 watch {layout.run_id}")
            return 0

        _print_run_end(result, layout=layout, budget=budget, console_stream=console_stream)
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
        if console_view is not None:
            console_view.close()  # stop the heartbeat thread, clear any spinner line
        clear_worker_pid(layout.run_dir)
        _stop_egress(egress_guard)
        if stashed:
            _finalize_auto_stash(
                cwd,
                base_branch=base_branch,
                run_branch=run_branch,
                auto_pop=cfg.git.auto_stash_pop,
            )
        _release_single_writer(worker_lock_fd)
        if detach_requested:
            # Ask how to handle approvals while away BEFORE spawning, so the marker is
            # set when the background run reads it. The worker lock is released now, so
            # the detached `resume` acquires it.
            if cfg.sandbox.run_commands == "ask" and not session_allow_set(layout.run_dir):
                _prompt_detach_away_mode(layout.run_dir)
            err = _spawn_detached(egress_guard, cwd, layout.run_id)
            if err:
                print(f"[agent6] {err}", file=sys.stderr)
        if egress_guard.detach_spawner is not None:
            egress_guard.detach_spawner.close()


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
    if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
        clear_away_mode(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    egress_guard = EgressGuard()
    console_view: ConsoleView | None = None
    detach_requested = False
    cfg: Config | None = None  # bound below; the finally reads it (detach away-mode)
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
        except ProfileUnavailableError as exc:
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

        egress_guard, egress_err = _maybe_start_egress(
            cfg, selected_profile, with_detach_spawner=True
        )
        if egress_err is not None:
            print(f"REFUSING: {egress_err}", file=sys.stderr)
            return 2
        if egress_guard.broker is not None:
            print(
                f"[agent6] provider-only egress: confined to host network "
                f"namespace via broker pid {egress_guard.broker.pid}",
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
        if console_stream:
            console_view = ConsoleView(sys.stderr)
            events.subscribe(console_view)
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
                loop_log = _loop_logger(mode, console_view)
                compact_drop, compact_summarise = resolve_compaction_thresholds(
                    cfg, rm_worker, log=loop_log
                )
                wf = Workflow(
                    root=cwd,
                    config=cfg,
                    provider=provider,
                    dispatcher=dispatcher,
                    logger=loop_log,
                    events=events,
                    graph_client=graph_client,
                    steer_requested=steer_state.requested,
                    steer_clear=steer_state.clear,
                    steer_prompt=steer_state.prompt,
                    should_abort=steer_state.abort_pending,
                    should_interrupt=steer_state.requested,
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
                    events.emit("run.end", reason="interrupted", all_passed=False)
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

        if result.reason == "detached":
            detach_requested = True
            print(f"\n[agent6] detached: {layout.run_id} continues in the background.")
            print(f"          reattach:  agent6 watch {layout.run_id}")
            return 0

        _print_run_end(result, layout=layout, budget=budget, console_stream=console_stream)
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
        if console_view is not None:
            console_view.close()  # stop the heartbeat thread, clear any spinner line
        clear_worker_pid(layout.run_dir)
        _stop_egress(egress_guard)
        _release_single_writer(worker_lock_fd)
        if detach_requested and cfg is not None:
            if cfg.sandbox.run_commands == "ask" and not session_allow_set(layout.run_dir):
                _prompt_detach_away_mode(layout.run_dir)
            err = _spawn_detached(egress_guard, cwd, layout.run_id)
            if err:
                print(f"[agent6] {err}", file=sys.stderr)
        if egress_guard.detach_spawner is not None:
            egress_guard.detach_spawner.close()
