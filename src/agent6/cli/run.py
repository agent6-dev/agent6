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
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent6 import __version__
from agent6.budget import BudgetTracker
from agent6.cli._common import (
    _agent6_dir,
    _BudgetOverrides,
    _check_provider_keys,
    _ensure_agent6_gitignored,
    _explicit_usd_flag_error,
    _runs_dir,
    _start_mcp_manager_if_enabled,
    detect_env,
)
from agent6.cli.egress import (
    _check_network_profile,
    _maybe_apply_agent_landlock,
    _maybe_start_egress,
    _stop_egress,
    _warn_if_unsandboxed,
)
from agent6.cli.misc_cmds import _cmd_diff
from agent6.cli.plan_watch import _event_epoch, _format_plain_event, _most_recent_run_id
from agent6.cli.providers import (
    _build_critic_provider,
    _build_prompt_reviser_provider,
    _build_role_provider,
    _build_summariser_provider,
    _InstrumentedProvider,
    _role_temperature,
)
from agent6.config import (
    Config,
    ConfigError,
    NotifyConfig,
    RoleName,
)
from agent6.config_layer import (
    load_effective,
    repo_config_path_for,
)
from agent6.detect import select_profile
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    create_branch,
    is_git_repo,
    revert_head,
    set_repo_hook_policy,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.graph.client import GraphClient, spawn_curator
from agent6.graph.curator import GraphCurator
from agent6.graph.storage import RunLayout
from agent6.init import init_workspace
from agent6.paths import (
    chown_to_real_user,
)
from agent6.providers import (
    Provider,
    TranscriptSink,
)
from agent6.run_id import RunIdError, new_friendly_id, resolve_run_id
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.mcp_client import MCPManager
from agent6.ui.approval import (
    clear_pending_answers,
    clear_steer_answer,
    read_answer,
    read_question_answer,
    read_steer_answer,
    tui_is_live,
)
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


def _ask_question_snippet(transcript: str) -> str:
    """First non-tag line of the `## Question` section of an ask transcript."""
    lines = transcript.splitlines()
    try:
        start = lines.index("## Question") + 1
    except ValueError:
        return "(no question)"
    for line in lines[start:]:
        s = line.strip()
        if s == "## Answer":
            break
        if s and not s.startswith("<"):  # skip blank lines + digest/file tags
            return s
    return "(question)"


def _cmd_ask_list() -> int:
    """`agent6 ask --list`: enumerate saved asks under .agent6/asks/."""
    asks_dir = _agent6_dir(Path.cwd()) / "asks"
    if not asks_dir.is_dir():
        print("No asks yet (.agent6/asks/ does not exist).")
        return 0
    dirs = sorted(
        (d for d in asks_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not dirs:
        print("No asks yet.")
        return 0
    for d in dirs:
        tp = d / "transcript.md"
        snippet = (
            _ask_question_snippet(tp.read_text(encoding="utf-8", errors="replace"))
            if tp.is_file()
            else "(no transcript)"
        )
        print(f"{d.name}  {snippet[:90]}")
    return 0


def _summarize_run_log(logs_path: Path) -> str:
    """Compact prose summary of a run's logs.jsonl: outcome + event counts +
    recent notable events. Used to seed `agent6 ask --run`."""
    if not logs_path.is_file():
        return "(no logs.jsonl for this run)"
    events: list[dict[str, Any]] = []
    for line in logs_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    if not events:
        return "(empty log)"
    counts: dict[str, int] = {}
    for e in events:
        counts[str(e.get("type", ""))] = counts.get(str(e.get("type", "")), 0) + 1
    out: list[str] = []
    end = next((e for e in reversed(events) if e.get("type") == "run.end"), None)
    if end is not None:
        out.append(f"Ended: reason={end.get('reason')!r} iterations={end.get('iterations')}")
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    out.append("Event counts: " + ", ".join(f"{t}={n}" for t, n in top))
    notable_types = {"tool.call", "verify.end", "run.end", "loop.auto_commit", "loop.metric.sample"}
    notable = [e for e in events if e.get("type") in notable_types][-15:]
    if notable:
        out.append("Recent notable events:")
        out.extend(f"  - {_fmt_run_event(e)}" for e in notable)
    return "\n".join(out)


def _fmt_run_event(e: dict[str, Any]) -> str:
    """One-line summary of a logs.jsonl event for the ask `--run` digest."""
    t = str(e.get("type", ""))
    if t == "tool.call":
        return f"tool.call {e.get('name', '')} {str(e.get('args', ''))[:80]}".rstrip()
    if t == "verify.end":
        return f"verify.end exit={e.get('exit_code')}"
    if t == "run.end":
        return f"run.end reason={e.get('reason')}"
    if t == "loop.metric.sample":
        return f"loop.metric.sample score={e.get('score')}"
    return t


def _build_ask_run_digest(cwd: Path, run_id: str, *, latest: bool) -> str | None:
    """Markdown digest of a prior run to seed an `ask`, or None (after printing
    an error) when the run can't be resolved."""
    runs_dir = _runs_dir(cwd)
    if not runs_dir.is_dir():
        print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
        return None
    if latest:
        target = _most_recent_run_id(runs_dir)
        if target is None:
            print(f"ERROR: --continue: no runs under {runs_dir}", file=sys.stderr)
            return None
    else:
        try:
            target = resolve_run_id(runs_dir, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return None
    layout = RunLayout(state_dir=_agent6_dir(cwd), run_id=target)
    if not layout.manifest_path.is_file():
        print(f"ERROR: run {target} has no manifest.json", file=sys.stderr)
        return None
    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read manifest for {target}: {exc}", file=sys.stderr)
        return None
    base_sha = str(manifest.get("base_sha") or "")
    run_branch = manifest.get("run_branch")
    head_ref = str(run_branch) if run_branch else "HEAD"
    diff = ""
    if base_sha:
        # operator-controlled argv, no LLM input (same as `agent6 diff`).
        proc = subprocess.run(
            ["git", "diff", f"{base_sha}..{head_ref}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        diff = proc.stdout
    cap = 8000
    diff_excerpt = diff[:cap]
    if len(diff) > cap:
        diff_excerpt += "\n... (diff truncated; read more with git)"
    try:
        rel = layout.run_dir.relative_to(cwd)
    except ValueError:
        rel = layout.run_dir
    return (
        f'<prior-run id="{target}">\n'
        f"This question is about a PRIOR agent6 run. Its files live under `{rel}/` --"
        f" read `{rel}/logs.jsonl` or `{rel}/transcripts/` with `read_file` if you"
        f" need more than this digest.\n\n"
        f"## Run task\n{manifest.get('user_task', '')}\n\n"
        f"## Outcome / key events\n{_summarize_run_log(layout.logs_path)}\n\n"
        f"## Diff base_sha..{head_ref} (truncated)\n```diff\n{diff_excerpt}\n```\n"
        f"</prior-run>"
    )


def _seed_files(cwd: Path, files: list[str]) -> str:
    """Wrap explicit --file seeds for an `ask` (a non-fatal, capped read)."""
    parts: list[str] = []
    for f in files:
        try:
            content = (cwd / f).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"WARNING: --file {f}: {exc}", file=sys.stderr)
            continue
        cap = 64 * 1024
        if len(content) > cap:
            content = content[:cap] + "\n... (truncated)"
        parts.append(f'<file path="{f}">\n{content}\n</file>')
    return "\n".join(parts)


def _save_ask_transcript(layout: RunLayout, *, question: str, answer: str) -> None:
    """Write the human-readable `ask` transcript (question + markdown answer)."""
    out = layout.run_dir / "transcript.md"
    out.write_text(
        f"# agent6 ask\n\n## Question\n\n{question}\n\n## Answer\n\n{answer}\n",
        encoding="utf-8",
    )


def _save_ask_repl_transcript(layout: RunLayout, conversation: list[tuple[str, str]]) -> None:
    """Write the cumulative transcript for an interactive ask session."""
    parts = ["# agent6 ask (interactive)\n"]
    for i, (q, a) in enumerate(conversation, 1):
        parts.append(f"## Q{i}\n\n{q}\n\n## A{i}\n\n{a}\n")
    (layout.run_dir / "transcript.md").write_text("\n".join(parts), encoding="utf-8")


def _run_ask_repl(
    wf: Workflow, budget: BudgetTracker, layout: RunLayout, *, first_question: str
) -> RunResult:
    """Interactive multi-turn ask. Each follow-up re-enters the loop with the
    prior Q&A carried as context, reusing the one provider/jail/budget setup.
    The agent re-reads what it needs per turn (prompt-cached); the conversation
    text is what gives continuity."""
    print(
        "[agent6] ask REPL — follow-up, /cost, /reset, or /quit (Ctrl-D exits).",
        file=sys.stderr,
    )
    conversation: list[tuple[str, str]] = []
    pending = first_question.strip()
    result: RunResult | None = None
    while True:
        if pending:
            question = pending
            pending = ""
        else:
            try:
                question = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
        if not question:
            continue
        if question in ("/quit", "/q", "/exit"):
            break
        if question == "/cost":
            print(budget.format_summary(), file=sys.stderr)
            continue
        if question == "/reset":
            conversation = []
            print("[agent6] conversation reset.", file=sys.stderr)
            continue
        if conversation:
            ctx = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in conversation)
            augmented = (
                f"<conversation-so-far>\n{ctx}\n</conversation-so-far>\n\nFollow-up: {question}"
            )
        else:
            augmented = question
        result = wf.run(augmented)
        print(result.summary)
        conversation.append((question, result.summary))
        _save_ask_repl_transcript(layout, conversation)
        if budget.is_exhausted():
            print("[agent6] budget exhausted; ending the REPL.", file=sys.stderr)
            break
    if result is None:
        return RunResult(
            completed=True, reason="ask_repl_empty", summary="", iterations=0, tool_calls=0
        )
    return result


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
    "                              (re)write .agent6/config.toml + AGENTS.md scaffolds\n"
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
    """REPL /watch: snapshot the last n events from this run's logs.jsonl.

    Intentionally NOT a live tail - the REPL is between turns of the
    agent loop; a tail would block the next iteration. Operators who
    want continuous tail use ``agent6 watch --plain`` in another shell.
    """
    if not run_id:
        print("[agent6] /watch: no run id available", file=sys.stderr)
        return
    events_path = _runs_dir(root) / run_id / "logs.jsonl"
    if not events_path.is_file():
        print(f"[agent6] /watch: no logs.jsonl at {events_path}", file=sys.stderr)
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
            if isinstance(obj0, dict):
                run_start_ts = _event_epoch(obj0.get("ts"))
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
            "[agent6] /mcp: no MCP servers configured (set [mcp] in your config)",
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
        rc = init_workspace(
            root, force=False, profile="py", repo_config_target=repo_config_path_for(root)
        )
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


def _tty_message(text: str) -> None:
    """Print to the controlling terminal directly, bypassing any stdout/stderr
    redirection (the TUI redirects the run's std streams to a log file)."""
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:  # noqa: PTH123
            tty.write(text)
            tty.flush()
            return
    except OSError:
        with contextlib.suppress(Exception):
            print(text, file=sys.stderr, flush=True)


def _tty_prompt(text: str) -> str | None:
    """Prompt on the controlling terminal directly (see ``_tty_message``).
    Falls back to stdin when /dev/tty is unavailable."""
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as tty:  # noqa: PTH123
            tty.write(text)
            tty.flush()
            line = tty.readline()
            return line.rstrip("\n") if line else None
    except OSError:
        try:
            return input(text)
        except (EOFError, KeyboardInterrupt):
            return None


def _install_steer_sigint(events: EventSink, run_dir: Path) -> _SteerState:
    """Install a SIGINT handler that asks the workflow to steer.

    * 1st SIGINT — set the "steer requested" flag and emit
      ``run.steer_requested``. The workflow notices at its next safe boundary
      (between steps) and prompts: through a TUI modal when the TUI is live,
      otherwise on the controlling terminal (``/dev/tty``, so the prompt is
      visible even when the TUI has redirected the run's std streams to a log).
    * 2nd SIGINT within 2 s — raise KeyboardInterrupt to abort the run.

    Returns callables for the workflow plus a ``restore`` hook to put the
    previous handler back when the run is done.
    """
    state: dict[str, Any] = {"requested": False, "last_ts": 0.0}
    window_s = 2.0

    def _handler(_signum: int, _frame: Any) -> None:
        now = time.monotonic()
        if state["requested"]:
            # Second Ctrl-C aborts: within the 2s window always, or any time a
            # steer is still pending in TUI mode (no terminal feedback there to
            # re-arm against). Otherwise just refresh the timestamp — crucially
            # WITHOUT re-clearing the answer file, which the TUI may have just
            # written (re-clearing it would strand read_steer_answer for 600s).
            if (now - state["last_ts"]) < window_s or tui_is_live(run_dir):
                raise KeyboardInterrupt
            state["last_ts"] = now
            return
        state["requested"] = True
        state["last_ts"] = now
        clear_steer_answer(run_dir)
        events.emit("run.steer_requested", source="sigint")
        # With the TUI up, the steer prompt is a modal — don't scribble on the
        # terminal it owns. Otherwise tell the user a prompt is coming.
        if not tui_is_live(run_dir):
            _tty_message(
                "\n[agent6] steer requested — finishing current step, then will"
                " prompt. Press Ctrl-C again to abort.\n"
            )

    previous = signal.signal(signal.SIGINT, _handler)

    def requested() -> bool:
        return bool(state["requested"])

    def clear() -> None:
        state["requested"] = False
        clear_steer_answer(run_dir)

    def prompt() -> str | None:
        # TUI live: the user answers a modal; read its file-bridge result.
        if tui_is_live(run_dir):
            return read_steer_answer(run_dir)
        return _tty_prompt("[agent6] steer (blank=continue, 'abort'=stop, else=instruction): ")

    def restore() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, previous)

    return _SteerState(requested=requested, clear=clear, prompt=prompt, restore=restore)


# Used when there is no controlling terminal at all (fully non-interactive):
# no SIGINT handler, default Ctrl-C behaviour.
_NULL_STEER = _SteerState(
    requested=lambda: False,
    clear=lambda: None,
    prompt=lambda: None,
    restore=lambda: None,
)


def _make_steer_state(events: EventSink, run_dir: Path) -> _SteerState:
    """Install the steer SIGINT handler when a controlling terminal exists
    (covers run/plan/ask with or without the TUI); else a no-op."""
    try:
        with open("/dev/tty", encoding="utf-8"):  # noqa: PTH123
            pass
    except OSError:
        return _NULL_STEER
    return _install_steer_sigint(events, run_dir)


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
            "critic": cfg.workflow.critic,
            "revise_prompt": cfg.workflow.revise_prompt,
        },
    }
    layout.manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _cmd_run(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    task: str,
    *,
    run_id: str = "",
    interactive: bool = False,
    no_tui: bool = False,
    mode: Literal["run", "plan", "ask"] = "run",
    budget_overrides: _BudgetOverrides | None = None,
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
    The ``planner`` model role drives plan mode (falls back to ``worker``).
    """
    try:
        cfg = load_effective(Path.cwd(), config_path).config
        set_repo_hook_policy(cfg.git.run_repo_hooks)
        if budget_overrides is not None:
            cfg = budget_overrides.apply(cfg)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    role: RoleName = "planner" if mode == "plan" else "worker"
    try:
        cfg.require_runnable(role, need_verify=(mode == "run"))
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
    # ask is read-only and may run outside a git repo (e.g. agent6 self-help),
    # so it skips the commit-oriented git pre-flight entirely.
    base_sha = ""
    base_branch = ""
    if mode != "ask":
        if not _require_git_repo(cwd):
            return 2
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

    # Layout: standard run-dir scaffolding for transcripts + logs. ask sessions
    # live under .agent6/asks/ to stay separate from real runs.
    effective_run_id = run_id or new_friendly_id()
    agent6_dir = _agent6_dir(cwd)
    layout = RunLayout(
        state_dir=agent6_dir,
        run_id=effective_run_id,
        subdir="asks" if mode == "ask" else "runs",
    )
    if mode != "ask":
        _ensure_agent6_gitignored(cwd, agent6_dir=agent6_dir, identity=identity)
    layout.ensure()
    # Drop stale approve/ask/steer answers + tui.pid from a prior session (the
    # id counters reset on resume, so an old answer must not be read instead of
    # re-prompting; a stale tui.pid would otherwise stall the answer-poll).
    clear_pending_answers(layout.run_dir)

    # Cut a fresh branch named after the run id so it is 1:1 with the run
    # (find it from any run id, `agent6 diff <id>`, or just delete the
    # branch to discard everything the agent did). The name is the unique
    # run id — never a timestamp+task-slug that collides into a pile of
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
        mode=mode,
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
    worker_inner = _build_role_provider(cfg, role, transcript_sink=transcript_sink, budget=budget)
    rm_worker = cfg.models.resolve(role)
    assert rm_worker is not None  # require_runnable validated this
    # Enable SSE streaming when stderr is a TTY (covers TUI
    # and interactive shell use). Bench/CI runs pipe stderr, so they
    # stay on the audited non-streaming code path UNLESS the operator
    # sets AGENT6_FORCE_STREAM=1 — the Kimi/OpenRouter bench needs
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
    curator_proc = spawn_curator(agent6_dir, effective_run_id, sock_path)
    print(f"[agent6] run id: {effective_run_id}", file=sys.stderr)

    # Spawn any configured MCP servers BEFORE the workflow
    # starts so their tools are visible from iteration 1. The manager
    # owns its subprocesses; we close it in the finally block.
    mcp_manager = _start_mcp_manager_if_enabled(cfg)

    # Steering (mid-run Ctrl-C -> a stdin prompt) needs the terminal; skip it
    # when the TUI owns it (then default Ctrl-C aborts cleanly). Double-Ctrl-C
    # within 2s still raises KeyboardInterrupt for the hard-abort path below.
    steer_state = _make_steer_state(events, layout.run_dir)

    result = None
    interrupted = False
    dispatcher: ToolDispatcher | None = None
    try:
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
                temperature=_role_temperature(cfg, role),
                critic_temperature=_role_temperature(cfg, "reviewer"),
                prompt_reviser_temperature=_role_temperature(cfg, "reviewer"),
                prompt_revision_selector=(
                    _select_revised_prompt if cfg.workflow.revise_prompt == "interactive" else None
                ),
                summariser_provider=summariser_provider,
                compact_drop_at_chars=cfg.workflow.compact_drop_at_chars,
                compact_summarise_at_chars=cfg.workflow.compact_summarise_at_chars,
                context_summary_max_tokens=cfg.workflow.context_summary_max_tokens,
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
        # Never leave root-owned run state in the user's repo (sudo case).
        chown_to_real_user(agent6_dir)

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

    The argv comes from `[notify].on_complete` in your config — operator-
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


def _cmd_resume(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    run_id: str,
    *,
    force: bool,
    no_tui: bool = False,
    budget_overrides: _BudgetOverrides | None = None,
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
    agent6_dir = _agent6_dir(cwd)
    runs_dir = agent6_dir / "runs"
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_id = resolved
    layout = RunLayout(state_dir=agent6_dir, run_id=run_id)
    if not layout.run_dir.is_dir():
        print(f"ERROR: no such run dir: {layout.run_dir}", file=sys.stderr)
        return 2
    # Drop a prior session's stale answer files + tui.pid (the id counters reset
    # on resume — an old answer must not be read instead of re-prompting).
    clear_pending_answers(layout.run_dir)

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
        cfg = load_effective(Path.cwd(), config_path).config
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
    _ensure_agent6_gitignored(cwd, agent6_dir=agent6_dir, identity=identity)

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
        max_usd=cfg.budget.best_effort_usd_limit,
    )

    worker_inner = _build_role_provider(
        cfg, "worker", transcript_sink=transcript_sink, budget=budget
    )
    rm_worker = cfg.models.resolve("worker")
    assert rm_worker is not None  # require_runnable validated this
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

    sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
    sock_path = sock_tmpdir / "curator.sock"
    sock_link = layout.run_dir / "curator.sock"
    with contextlib.suppress(FileNotFoundError):
        sock_link.unlink()
    sock_link.symlink_to(sock_path)
    curator_proc = spawn_curator(agent6_dir, run_id, sock_path)
    print(f"[agent6] resume run id: {run_id}", file=sys.stderr)

    mcp_manager = _start_mcp_manager_if_enabled(cfg)

    steer_state = _make_steer_state(events, layout.run_dir)

    result = None
    interrupted = False
    dispatcher: ToolDispatcher | None = None
    try:
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
                temperature=_role_temperature(cfg, "worker"),
                critic_temperature=_role_temperature(cfg, "reviewer"),
                summariser_provider=summariser_provider,
                compact_drop_at_chars=cfg.workflow.compact_drop_at_chars,
                compact_summarise_at_chars=cfg.workflow.compact_summarise_at_chars,
                context_summary_max_tokens=cfg.workflow.context_summary_max_tokens,
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
        # Never leave root-owned run state in the user's repo (sudo case).
        chown_to_real_user(agent6_dir)

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
