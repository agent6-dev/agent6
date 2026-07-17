# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan`/`agent6 attach` and run-id resolution helpers."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from agent6.runs.id import RunIdError, resolve_run_id
from agent6.runs.ipc import (
    clear_frontend_pid,
    read_worker_pid,
    set_session_allow,
    worker_is_alive,
    write_answer,
    write_frontend_pid,
    write_question_answers,
)
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.tools.schema import UserQuestion
from agent6.ui.cli._common import _runs_dir, _state_dir, resolve_or_newest_layout
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._interact import default_stdin_approver, default_stdin_questioner
from agent6.viewmodel import run_mtime, tail_events
from agent6.viewmodel.format import format_compare, format_cost


def event_epoch(value: object) -> float | None:
    """Parse an event ``ts`` to epoch seconds, or None if unparseable.

    EventSink writes ``ts`` as an ISO-8601 string (``datetime.isoformat``),
    so the elapsed-time anchor must parse that, not only bare numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _resolve_plan_run_id(run_id: str) -> str | None:
    """Resolve a (possibly prefix) run-id under the per-repo run-state dir.

    Prints an error and returns None on failure. Used by ``run --from-plan``,
    ``plan show``, and ``plan edit``. An empty *run_id* resolves the most recent
    planning run, matching the omit-for-latest convention of the runs commands.
    """
    runs_dir = _runs_dir(Path.cwd())
    if not run_id:
        latest = _most_recent_plan_run_id(runs_dir)
        if latest is None:
            print("ERROR: no planning runs yet (start one with `agent6 plan`).", file=sys.stderr)
            return None
        run_id = latest
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
    plan = _runs_dir(Path.cwd()) / resolved / "plan.md"
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
    plan = _runs_dir(Path.cwd()) / resolved / "plan.md"
    editor = os.environ.get("EDITOR", "vi")
    try:
        result = subprocess.run([editor, str(plan)], check=False)
    except OSError as exc:
        print(f"ERROR: failed to spawn editor {editor!r}: {exc}", file=sys.stderr)
        return 1
    return result.returncode


def _most_recent_plan_run_id(runs_dir: Path) -> str | None:
    """Most recently active run dir that holds a ``plan.md`` (a plan run).

    Used by bare `agent6 run` (no task) to offer the latest plan for execution.
    """
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and (p / "plan.md").is_file()),
        key=run_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else None


def _cmd_watch(run_id: str, *, tui: bool = False, since: int = 0, raw: bool = False) -> int:
    """Read-only live view of a run directory.

    Default follows the run's conversation (the same render as ``agent6 run``).
    ``--raw`` is the no-deps event-line tail; ``--tui`` the full-screen dashboard.
    """
    cwd = Path.cwd()
    # An explicit id resolves across every run-style bucket (runs/asks/machine-
    # drafts): a listed ask or a `machine create` draft is watchable by id too.
    # Empty most-recent spans every bucket, so a bare `attach` after an `ask`
    # finds it.
    try:
        layout = resolve_or_newest_layout(cwd, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if layout is None:
        print("ERROR: no runs found for this cwd.", file=sys.stderr)
        return 2
    target = layout.run_dir
    if not run_id:
        print(f"[agent6] attached to most recent run: {target.name}", file=sys.stderr)
    if not target.is_dir():
        print(f"ERROR: no such run dir: {target}", file=sys.stderr)
        return 2
    if not tui:
        return _cmd_watch_plain(target, since=since) if raw else _watch_transcript(target)
    try:
        from agent6.ui.tui.app import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "HINT: drop --tui for a no-deps text tail of logs.jsonl.",
            file=sys.stderr,
        )
        return 3
    return run_tui(target)


def _resolve_run_dir(repo_root: Path, run_id: str) -> Path | None:
    """Resolve a run id (or the most-recent run when empty) to its run dir.

    An explicit id resolves across every run-style bucket (runs/, asks/,
    machine-drafts/): anything `agent6 runs` lists must also be inspectable
    by id. The empty (most-recent) case also spans every bucket, so a bare
    `attach` right after an `ask` finds that ask."""
    try:
        layout = resolve_or_newest_layout(repo_root, run_id)
    except RunIdError:
        return None
    return layout.run_dir if layout is not None else None


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _scan_run_events(events_path: Path) -> dict[str, object]:
    """Single pass over logs.jsonl. Returns start_ep, last_ep, last_type,
    iteration, end_reason, and the latest `budget.update` token/cost totals.
    Tolerant of torn/short lines."""
    out: dict[str, object] = {
        "start_ep": None,
        "last_ep": None,
        "last_type": None,
        "iteration": None,
        "end_reason": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "usd_partial": None,
    }
    if not events_path.is_file():
        return out
    with contextlib.suppress(OSError):
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            ep = event_epoch(e.get("ts"))
            etype = e.get("type")
            if etype == "run.start" and out["start_ep"] is None:
                out["start_ep"] = ep
            if isinstance(e.get("iteration"), int):
                out["iteration"] = e["iteration"]
            if etype == "run.end":
                out["end_reason"] = e.get("reason")
            if etype == "budget.update":
                # The authoritative running totals, emitted after each provider
                # call (providers.py). `loop.budget` (emitted BEFORE the call)
                # lags one call and reads 0 on iteration 1, so `runs show` used
                # to under-report; every other cost consumer (watch --json,
                # runs list) reads budget.update, so this makes them agree.
                out["input_tokens"] = e.get("input_total")
                out["output_tokens"] = e.get("output_total")
                out["cost_usd"] = e.get("usd_total")
                out["usd_partial"] = e.get("usd_partial")
            if ep is not None:
                out["last_ep"] = ep
            if isinstance(etype, str):
                out["last_type"] = etype
    return out


def _print_fork_lineage(manifest: dict[str, object]) -> None:
    """Print the fork-lineage line for a run created by `agent6 fork` (no-op
    otherwise)."""
    parent = manifest.get("parent_run_id")
    if not (isinstance(parent, str) and parent):
        return
    sha = manifest.get("forked_from_sha")
    sha_note = f" ({sha[:12]})" if isinstance(sha, str) and sha else ""
    print(f"forked from: {parent}@turn {manifest.get('forked_from_turn')}{sha_note}")


def _print_parallel_compare(manifest: dict[str, object]) -> None:
    """Print the fan-out compare outcome for a lane (no-op for a non-lane run):
    where it placed, whether it won, judged or mechanical, and the judge's
    rationale when there is one."""
    formatted = format_compare(manifest.get("compare"))
    if formatted is None:
        return
    headline, rationale = formatted
    print(f"compare:    {headline}")
    if rationale:
        print(f"  judge: {rationale}")


def _cmd_status(run_id: str, *, as_json: bool = False) -> int:  # noqa: PLR0915
    """One-shot liveness + progress summary for a run, then exit (no follower).

    Answers "is this run still alive, and what is it doing?" from the run dir
    alone: the worker.pid (probed with signal 0, so liveness is known even while
    the worker is blocked in a long provider call that emits no events) plus the
    last event, current iteration, and elapsed time from logs.jsonl. For a quick
    or scripted check; `agent6 attach` is the live follower.
    """
    target = _resolve_run_dir(Path.cwd(), run_id)
    if target is None or not target.is_dir():
        state = _state_dir(Path.cwd())
        print(
            f"ERROR: no run found ({run_id or 'latest'}) under {state}/(runs|asks|machine-drafts)",
            file=sys.stderr,
        )
        return 2

    manifest: dict[str, object] = {}
    with contextlib.suppress(ManifestError):
        manifest = read_manifest(target)

    ev = _scan_run_events(target / "logs.jsonl")
    start_ep = ev["start_ep"]
    last_ep = ev["last_ep"]
    last_type = ev["last_type"]
    iteration = ev["iteration"]
    end_reason = ev["end_reason"]

    pid = read_worker_pid(target)
    alive = worker_is_alive(target)
    now = time.time()
    last_age = (now - last_ep) if isinstance(last_ep, (int, float)) else None
    elapsed = (
        (last_ep - start_ep)
        if isinstance(last_ep, (int, float)) and isinstance(start_ep, (int, float))
        else None
    )

    if end_reason is not None:
        # A diagnostic view: show the precise raw reason (steer_abort, provider_error,
        # verify_settled, ...) rather than a categorized label. The user-facing status
        # (dashboard/hub) uses the friendly run_status_label, which also has all_passed.
        state = f"finished ({end_reason})"
    elif alive and last_age is not None and last_age > 120:
        state = "running (long step, likely a provider call)"
    elif alive:
        state = "running"
    elif last_ep is None:
        state = "unknown (no events yet)"
    else:
        state = "stopped (no worker, no run.end: likely crashed or killed)"

    models = manifest.get("models")
    worker = models.get("worker") if isinstance(models, dict) else None
    model = (worker.get("model") if isinstance(worker, dict) else worker) or "?"

    if as_json:
        print(
            json.dumps(
                {
                    "run_id": target.name,
                    "mode": manifest.get("mode"),
                    "model": model,
                    "state": state,
                    "alive": alive,
                    "pid": pid,
                    "iteration": iteration,
                    "last_event": last_type,
                    "last_event_age_s": round(last_age, 1) if last_age is not None else None,
                    "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
                    "reason": end_reason,
                    "input_tokens": ev["input_tokens"],
                    "output_tokens": ev["output_tokens"],
                    "cost_usd": ev["cost_usd"],
                    "parent_run_id": manifest.get("parent_run_id"),
                    "forked_from_turn": manifest.get("forked_from_turn"),
                    "forked_from_sha": manifest.get("forked_from_sha"),
                    "compare": manifest.get("compare"),
                }
            )
        )
        return 0

    pid_note = ""
    if alive:
        pid_note = f"  (worker pid {pid} alive)"
    elif pid is not None and end_reason is None:
        pid_note = f"  (worker pid {pid} not running)"
    print(f"run:        {target.name}  (mode={manifest.get('mode', '?')})")
    _print_fork_lineage(manifest)
    _print_parallel_compare(manifest)
    print(f"model:      {model}")
    print(f"state:      {state}{pid_note}")
    print(f"iteration:  {iteration if iteration is not None else '-'}")
    print(
        f"last event: {last_type or '-'}"
        f"{f'  ({_fmt_dur(last_age)} ago)' if last_age is not None else ''}"
    )
    print(f"elapsed:    {_fmt_dur(elapsed)}")
    if ev["input_tokens"] is not None or ev["cost_usd"] is not None:
        cost = ev["cost_usd"]
        cost_s = (
            f"  cost {format_cost(cost, partial=bool(ev.get('usd_partial')))}"
            if isinstance(cost, (int, float))
            else ""
        )
        print(f"usage:      in={ev['input_tokens'] or 0} out={ev['output_tokens'] or 0}{cost_s}")
    _print_task_tree(target)
    return 0


def _print_task_tree(run_dir: Path) -> None:
    """Show the run's task DAG when it decomposed into subtasks. Makes the plan
    visible for a headless run (no TUI #plan pane), the decompose case the user
    could not see. A single root (no decomposition) is not worth the block."""
    from agent6.graph.storage import load_graph  # noqa: PLC0415
    from agent6.runs.layout import RunLayout  # noqa: PLC0415
    from agent6.ui.cli._task_tree import task_tree_lines  # noqa: PLC0415

    with contextlib.suppress(Exception):
        layout = RunLayout(state_dir=_state_dir(Path.cwd()), run_id=run_dir.name)
        nodes = load_graph(layout)
        if len(nodes) <= 1:
            return
        lines = task_tree_lines(nodes, show_commit=True)
        if lines:
            print("\nplan:")
            for line in lines:
                print(f"  {line}")


def _cmd_tui() -> int:
    """The TUI hub (`agent6 tui`): browse runs and start new work. Loops between
    the home screen and the run view (the conversation; Ctrl+D toggles the
    dashboard), opening a run watches it, then returns here on close."""
    try:
        from agent6.ui.tui.app import (  # noqa: PLC0415 - lazy: textual optional
            QUIT_HUB_CODE,
            run_tui,
        )
        from agent6.ui.tui.home import run_home  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("HINT: the TUI needs 'textual' (part of the base install).", file=sys.stderr)
        return 3
    cwd = Path.cwd()
    agent6_dir = _runs_dir(cwd).parent
    while True:
        run_dir = run_home(agent6_dir, cwd)
        if run_dir is None:
            return 0
        # Esc in the dashboard returns here (reopen home); q quits the hub.
        if run_tui(run_dir, from_hub=True) == QUIT_HUB_CODE:
            return 0


def format_plain_event(line: str, *, run_start_ts: float | None) -> str:
    """Pretty-print one logs.jsonl line as `<elapsed> <type> key=val ...`.

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
    ts = event_epoch(obj.get("ts"))
    event = obj.get("event") or obj.get("type") or "?"
    if ts is not None and run_start_ts is not None:
        elapsed = max(0.0, ts - run_start_ts)
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


class _CliFrontEnd:
    """Makes an interactive ``agent6 attach`` a real run FRONT-END, not just a
    reader. When the streamed log surfaces an unanswered ``run_command`` approval
    or ``ask_user`` question, it prompts on the controlling terminal with the SAME
    CLI prompts a foreground run uses and writes the answer back over the file
    bridge -- so watching a detached run is "as if you never detached". The
    caller registers ``frontend.pid`` so the worker's approver bridges to it (a
    live front-end always wins over the detach away-mode).

    Prompt ids are deterministic counters, and the log replays from the start on
    attach, so ``_answered`` (ids with an answer seen) and ``_handled`` (ids WE
    prompted for) gate re-prompting a historical or already-answered prompt."""

    def __init__(self, run_dir: Path, view: ConsoleView) -> None:
        self._run_dir = run_dir
        self._view = view
        self._answered: set[str] = set()
        self._handled: set[str] = set()

    def open_prompts_at_attach(self, events_path: Path) -> list[tuple[str, str, object]]:
        """Pre-scan the existing log: seed ``_answered`` and return the prompts
        that are open right now (emitted, not answered) so a run already waiting
        at an approval when you attach is handled at once."""
        open_prompts: dict[str, tuple[str, str, object]] = {}
        for ev in tail_events(events_path, follow=False):
            etype = str(ev.get("type", ""))
            pid = str(ev.get("id", ""))
            if etype == "approval.prompt":
                open_prompts[pid] = ("approval", pid, ev.get("prompt", ""))
            elif etype == "question.prompt":
                open_prompts[pid] = ("question", pid, ev.get("questions", []))
            elif etype in ("approval.answer", "question.answer"):
                self._answered.add(pid)
                open_prompts.pop(pid, None)
        return list(open_prompts.values())

    def handle(self, kind: str, prompt_id: str, content: object) -> None:
        """Prompt on the terminal (spinner paused) and write the answer over the
        bridge. Marks the id handled so the follow-loop replay won't re-ask it."""
        if kind == "approval":
            with self._view.pause():
                answer = default_stdin_approver(str(content))
            if answer == "session":
                set_session_allow(self._run_dir)
            write_answer(self._run_dir, prompt_id, approved=answer != "no")
        else:
            questions = tuple(
                UserQuestion(
                    question=str(q.get("question", "")),
                    options=tuple(str(o) for o in q.get("options", [])),
                )
                for q in (content if isinstance(content, list) else [])
            )
            with self._view.pause():
                answers = default_stdin_questioner(questions)
            write_question_answers(
                self._run_dir,
                prompt_id,
                answers if answers is not None else tuple("" for _ in questions),
            )
        self._handled.add(prompt_id)

    def react(self, event: dict[str, object]) -> None:
        """Live follow: answer a NEW unanswered prompt; a historical/answered one
        (id in ``_answered``/``_handled``) is skipped on the replay."""
        etype = str(event.get("type", ""))
        pid = str(event.get("id", ""))
        if etype in ("approval.answer", "question.answer"):
            self._answered.add(pid)
            return
        if pid in self._handled or pid in self._answered:
            return
        if etype == "approval.prompt":
            self.handle("approval", pid, event.get("prompt", ""))
        elif etype == "question.prompt":
            self.handle("question", pid, event.get("questions", []))


def _watch_transcript(target: Path) -> int:
    """Follow a run's conversation live and, on an interactive terminal, ATTACH
    to it as a front-end: fold ``logs.jsonl`` through the same ``ConsoleView`` as
    ``agent6 run`` and, when the run asks for a ``run_command`` approval or an
    ``ask_user`` answer, prompt on the terminal exactly as the foreground run
    would (see ``_CliFrontEnd``). Piped/redirected (no tty) stays a pure reader.
    Renders from the start, tails until the run ends, then returns; Ctrl-C exits.
    A detach emits no run.end, so watching a detached run follows it to its end."""
    events_path = target / "logs.jsonl"
    if not events_path.is_file():
        print(f"ERROR: no logs.jsonl in {target}", file=sys.stderr)
        return 2
    view = ConsoleView(sys.stdout)
    # Interactive (both streams a tty): become an answering front-end. Piped: read-only.
    front_end: _CliFrontEnd | None = None
    if sys.stdin.isatty() and sys.stdout.isatty():
        front_end = _CliFrontEnd(target, view)
        write_frontend_pid(target, os.getpid())
        print(
            f"[agent6] attached to {target.name}: approvals and questions prompt here."
            " Ctrl-C to detach.",
            file=sys.stderr,
        )
    else:
        print(f"[agent6] following {target.name}. Ctrl-C to exit.", file=sys.stderr)
    try:
        if front_end is not None:
            for kind, prompt_id, content in front_end.open_prompts_at_attach(events_path):
                front_end.handle(kind, prompt_id, content)  # a prompt already pending at attach
        for event in tail_events(events_path, follow=True, stop_when_finished=True):
            view.feed(event)
            if front_end is not None:
                front_end.react(event)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
    finally:
        view.close()  # stop the heartbeat thread, clear any spinner line
        if front_end is not None:
            clear_frontend_pid(target)
    return 0


def _line_is_run_end(raw: bytes | str) -> bool:
    """True if a logs.jsonl line is a ``run.end`` event."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and obj.get("type") == "run.end"


def _run_has_ended(events_path: Path) -> bool:
    """True if the run's last logged event is ``run.end`` (finished, nothing to
    follow). A resume appends events after a run.end, so only the LAST line
    counts."""
    try:
        with events_path.open("rb") as fh:
            last = b""
            for last in fh:  # noqa: B007 - keep the final line
                pass
    except OSError:
        return False
    return bool(last) and _line_is_run_end(last)


def _cmd_watch_plain(target: Path, *, since: int) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    """Tail ``logs.jsonl`` line-by-line with no extra deps.

    Polls the file with 0.25s sleeps; rotates when the inode changes.
    Pretty-prints each event with the type and key fields. Returns 0 on
    EOF (run dir gone) or KeyboardInterrupt.
    """
    events_path = target / "logs.jsonl"
    if not events_path.is_file():
        print(f"ERROR: no logs.jsonl in {target}", file=sys.stderr)
        return 2

    # Read the first event for the elapsed-time anchor. Binary readline: a
    # torn-mid-UTF-8 first line must not crash the watch before it starts.
    run_start_ts: float | None = None
    try:
        with events_path.open("rb") as fh:
            first = fh.readline()
        if first:
            obj0 = json.loads(first.decode("utf-8"))
            if isinstance(obj0, dict):
                run_start_ts = event_epoch(obj0.get("ts"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        run_start_ts = None

    print(
        f"[agent6] tailing {events_path}. Ctrl-C to exit.",
        file=sys.stderr,
    )

    # Binary reads throughout: the writer flushes long lines in several
    # syscalls, so a read can hit EOF mid multibyte UTF-8 sequence and a
    # text-mode readline would raise UnicodeDecodeError. Complete lines are
    # decoded (errors="replace"); a partial tail stays buffered until its
    # newline arrives.
    try:
        fh = events_path.open("rb")
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
            for raw in lines[-since:]:
                line = raw.decode("utf-8", errors="replace")
                # flush: piped/redirected output must not lose the replay to the
                # block buffer when the run is idle/finished (nothing else flushes).
                print(format_plain_event(line, run_start_ts=run_start_ts), flush=True)
            if lines and _line_is_run_end(lines[-1]):
                return 0  # already finished: replayed, nothing to follow
        else:
            # A finished run has no new events to follow; seeking to end would hang.
            if _run_has_ended(events_path):
                print("[agent6] run already finished.", file=sys.stderr)
                return 0
            # Seek to end; only show new events going forward.
            fh.seek(0, 2)
        try:
            current_ino = events_path.stat().st_ino
        except OSError:
            current_ino = -1
        pending = b""
        while True:
            chunk = fh.readline()
            if chunk:
                pending += chunk
                if not pending.endswith(b"\n"):
                    continue  # partial line at EOF; the rest arrives next read
                line = pending.decode("utf-8", errors="replace")
                pending = b""
                print(format_plain_event(line, run_start_ts=run_start_ts), flush=True)
                if _line_is_run_end(line):
                    return 0  # run ended: stop, like the default follower
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
                    fh = events_path.open("rb")
                except OSError:
                    time.sleep(0.5)
                    continue
                current_ino = new_ino
                pending = b""
                continue
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            fh.close()
