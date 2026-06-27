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
    _start_mcp_manager_if_enabled,
    _state_dir,
    detect_env,
)
from agent6.cli._repl import build_repl_hook as _build_repl_hook
from agent6.cli._steer import (
    make_steer_state as _make_steer_state,
)
from agent6.cli._steer import (
    select_revised_prompt as _select_revised_prompt,
)
from agent6.cli.egress import (
    _check_network_profile,
    _maybe_apply_agent_landlock,
    _maybe_start_egress,
    _stop_egress,
    _warn_if_unsandboxed,
)
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
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    create_branch,
    is_git_repo,
    set_repo_hook_policy,
    stash_all,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.graph.client import CuratorClientError, GraphClient, spawn_curator
from agent6.graph.curator import GraphCurator
from agent6.graph.storage import RunLayout
from agent6.paths import (
    chown_to_real_user,
)
from agent6.pricing import lookup_price
from agent6.providers import (
    Provider,
    TranscriptSink,
)
from agent6.run_id import RunIdError, new_friendly_id, resolve_run_id
from agent6.tools.dispatch import ToolDispatcher
from agent6.ui.approval import (
    clear_pending_answers,
    clear_worker_pid,
    read_answer,
    read_question_answer,
    tui_is_live,
    write_worker_pid,
)
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


def _warn_if_usd_unenforceable(cfg: Config, worker_model: str) -> None:
    """Warn at startup when ``best_effort_usd_limit`` is set but the worker
    model has no published price, so the USD ceiling silently can't be enforced
    and spend is bounded only by the token ceilings. We deliberately do NOT
    guess a price or terminate early (a wrong guess could kill a run mid-task);
    we just make the gap visible so the operator can set token ceilings if they
    need a precise dollar bound. Anthropic publishes no pricing, so this fires
    for Claude workers."""
    usd = cfg.budget.best_effort_usd_limit
    if usd > 0 and lookup_price(worker_model) is None:
        print(
            f"[agent6] WARNING: best_effort_usd_limit=${usd:g} cannot be enforced "
            f"for {worker_model!r} (no published price); spend is bounded only by "
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
    """Plain TTY fallback for tool approval (used when no TUI is live)."""
    try:
        ans = input(f"{prompt} [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in {"y", "yes"}


def _build_approver(run_dir: Path, events: EventSink) -> Callable[[str], bool]:
    """Build the `run_command` approver, bridged to a live TUI when present.

    Emits an `approval.prompt` event; if a TUI is live (it wrote `tui.pid`) the
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
        if tui_is_live(run_dir):
            approved = read_answer(run_dir, prompt_id)
            if approved is not None:
                source = "tui"
        if approved is None:
            approved = _default_stdin_approver(prompt)
        events.emit("approval.answer", id=prompt_id, approved=approved, source=source)
        return approved

    return approve


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
        if tui_is_live(run_dir):
            answer = read_question_answer(run_dir, question_id)
            if answer is not None:
                source = "tui"
        if answer is None:
            answer = _default_stdin_questioner(question, options)
        events.emit("question.answer", id=question_id, answer=answer, source=source)
        return answer

    return ask


def _default_stdin_questioner(question: str, options: tuple[str, ...]) -> str:
    """Numbered stdin prompt; headless (no TTY / EOF) returns "" so a run never
    hangs waiting on an operator that isn't there."""
    if not sys.stdin.isatty():
        return ""
    lines = [question, *(f"  {i}) {opt}" for i, opt in enumerate(options, start=1))]
    try:
        ans = input("\n".join(lines) + "\n> ").strip()
    except EOFError:
        return ""
    if ans.isdigit() and 1 <= int(ans) <= len(options):
        return options[int(ans) - 1]
    return ans


def _tui_available() -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("textual") is not None


def _should_spawn_tui(*, no_tui: bool, interactive: bool, mode: str) -> bool:
    """Whether `agent6 run`/`resume` auto-spawns the dashboard TUI.

    Default yes when the `tui` extra is installed and stdout is a real TTY.
    `--no-tui` opts out; `-i` (the stdin REPL) is mutually exclusive with the
    full-screen TUI, so it wins; `plan` stays a plain text pass."""
    return (
        not no_tui
        and not interactive
        and mode == "run"
        and sys.stdout.isatty()
        and _tui_available()
    )


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
            [sys.executable, "-m", "agent6.ui", "--watch", str(run_dir), "--exit-on-end"]
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
    layout.manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


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
    no_tui: bool = False,
    mode: Literal["run", "plan", "ask"] = "run",
    budget_overrides: _BudgetOverrides | None = None,
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
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
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

    net_err = _check_network_profile(cfg, selected_profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
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
        if not _require_git_repo(cwd):
            return 2
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
    # Drop stale approve/ask/steer answers + tui.pid from a prior session (the
    # id counters reset on resume, so an old answer must not be read instead of
    # re-prompting; a stale tui.pid would otherwise stall the answer-poll).
    clear_pending_answers(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    # Enforce the dirty-tree policy BEFORE cutting the run branch, so the
    # branch is cut from a clean tree and the agent's per-step auto-commits
    # (`git add -A`) never swallow the user's pre-existing uncommitted work.
    # Only `run` makes commits; `plan`/`ask` are read-only (matching the
    # branch_per_run guard below).
    if mode == "run" and pre_status is not None and not pre_status.is_clean:
        if cfg.git.auto_stash:
            try:
                stash_all(cwd, f"agent6 auto-stash before run {effective_run_id}")
            except GitError as exc:
                print(f"ERROR: could not auto-stash before run: {exc}", file=sys.stderr)
                return 2
        elif cfg.git.require_clean_worktree:
            print(
                "REFUSING: working tree is not clean. Commit, stash, or discard "
                "your changes, set [git].auto_stash=true, or set "
                "[git].require_clean_worktree=false to override.",
                file=sys.stderr,
            )
            return 2

    # Cut a fresh branch named after the run id so it is 1:1 with the run
    # (find it from any run id, `agent6 runs diff <id>`, or just delete the
    # branch to discard everything the agent did). The name is the unique
    # run id, never a timestamp+task-slug that collides into a pile of
    # near-duplicate `agent6/<ts>-<same-task>` branches on re-runs. Only real
    # `run` mode branches: `plan`/`ask` make no commits, so a branch for them
    # is pure litter. create_branch is idempotent (reuses an existing branch).
    run_branch: str | None = None
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
        # The egress broker is already running at this point; tear it down so
        # the refusal doesn't leak the broker process + its socket dir.
        _stop_egress(egress_broker, egress_sock_dir)
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
    worker_inner = _build_role_provider(cfg, role, transcript_sink=transcript_sink, budget=budget)
    rm_worker = cfg.models.resolve(role)
    assert rm_worker is not None  # require_runnable validated this
    _warn_if_usd_unenforceable(cfg, rm_worker.model)
    _warn_if_prompt_override_incomplete(cfg)
    # Enable SSE streaming when stderr is a TTY (covers TUI
    # and interactive shell use). Bench/CI runs pipe stderr, so they
    # stay on the audited non-streaming code path UNLESS the operator
    # sets AGENT6_FORCE_STREAM=1, the Kimi/OpenRouter bench needs
    # streaming on because the gateway emits SSE keep-alive comment
    # heartbeats during long requests, which corrupt the non-streaming
    # response body (resp.json() blows up with JSONDecodeError).
    stream_text = sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1"
    tui_enabled = _should_spawn_tui(no_tui=no_tui, interactive=interactive, mode=mode)
    # Echo the model's reasoning + answer to stderr live whenever the TUI is
    # NOT owning the terminal (plan / ask / machine create / --no-tui). With the
    # TUI up it renders the same deltas from the event stream, so console echo
    # would just fight it for the terminal.
    console_stream = stream_text and not tui_enabled
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
        curator_proc = spawn_curator(state_dir, effective_run_id, sock_path, subdir=layout.subdir)
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
                resume_state_path=(None if mode == "ask" else layout.run_dir / "loop_state.json"),
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
                revise_prompt=cfg.prompt.revise_prompt,
                temperature=_role_temperature(cfg, role),
                critic_temperature=_role_temperature(cfg, "reviewer"),
                prompt_reviser_temperature=_role_temperature(cfg, "reviewer"),
                prompt_revision_selector=(
                    _select_revised_prompt if cfg.prompt.revise_prompt == "interactive" else None
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
        clear_worker_pid(layout.run_dir)
        # Clean up the /tmp socket dir + symlink under run_dir.
        with contextlib.suppress(FileNotFoundError):
            (layout.run_dir / "curator.sock").unlink()
        if sock_tmpdir is not None:
            shutil.rmtree(sock_tmpdir, ignore_errors=True)
        if dispatcher is not None:
            dispatcher.close()
        if mcp_manager is not None:
            mcp_manager.close()
        _stop_egress(egress_broker, egress_sock_dir)
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
            print(f"\n[agent6] answer saved to {layout.run_dir / 'transcript.md'}", file=sys.stderr)
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


def _cmd_resume(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    run_id: str,
    *,
    force: bool,
    no_tui: bool = False,
    budget_overrides: _BudgetOverrides | None = None,
    profile: str = "",
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
    cwd = Path.cwd()
    state_dir = _state_dir(cwd)
    runs_dir = state_dir / "runs"
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
    # Drop a prior session's stale answer files + tui.pid (the id counters reset
    # on resume, an old answer must not be read instead of re-prompting).
    clear_pending_answers(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

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

    # `agent6 resume` has no --profile flag, so re-apply the profile the original
    # run used (persisted in the manifest), unless one is passed explicitly.
    manifest_profile = ""
    with contextlib.suppress(OSError, ValueError, KeyError, AttributeError):
        manifest_profile = (
            json.loads(layout.manifest_path.read_text(encoding="utf-8"))
            .get("workflow", {})
            .get("profile", "")
        )
    try:
        cfg = load_effective(Path.cwd(), config_path, profile=profile or manifest_profile).config
        set_repo_hook_policy(cfg.git.run_repo_hooks)
        if budget_overrides is not None:
            cfg = budget_overrides.apply(cfg)
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

    net_err = _check_network_profile(cfg, selected_profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return 2

    missing = _check_provider_keys(cfg)
    usd_err = _explicit_usd_flag_error(budget_overrides.max_usd if budget_overrides else None, cfg)
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
    # (no-repo guard already ran above, before compute_resume_diff)
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
        # The egress broker is already running; tear it down so the refusal
        # doesn't leak the broker process + its socket dir.
        _stop_egress(egress_broker, egress_sock_dir)
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
    _warn_if_usd_unenforceable(cfg, rm_worker.model)
    _warn_if_prompt_override_incomplete(cfg)
    # Streaming gated on stderr TTY (matches _cmd_run);
    # AGENT6_FORCE_STREAM=1 forces it on for bench/CI.
    stream_text = sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1"
    tui_enabled = _should_spawn_tui(no_tui=no_tui, interactive=False, mode="run")
    console_stream = stream_text and not tui_enabled
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
    resume_base_sha = ""
    try:
        resume_base_sha = json.loads(layout.manifest_path.read_text(encoding="utf-8")).get(
            "base_sha", ""
        )
    except (OSError, ValueError):
        resume_base_sha = ""

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
                cfg, cwd, mode="run", events=events, transcript_sink=transcript_sink, budget=budget
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
        clear_worker_pid(layout.run_dir)
        with contextlib.suppress(FileNotFoundError):
            (layout.run_dir / "curator.sock").unlink()
        if sock_tmpdir is not None:
            shutil.rmtree(sock_tmpdir, ignore_errors=True)
        if dispatcher is not None:
            dispatcher.close()
        if mcp_manager is not None:
            mcp_manager.close()
        _stop_egress(egress_broker, egress_sock_dir)
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
