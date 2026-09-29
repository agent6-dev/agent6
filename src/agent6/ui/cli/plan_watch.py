# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan` and `agent6 attach`, and the run-id resolution they share."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agent6.errors import read_operator_file
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.ipc import (
    ANSWERED_ELSEWHERE,
    register_frontend,
    unregister_frontend,
    worker_is_alive,
    write_answer,
    write_question_answers,
)
from agent6.sessions.layout import LOGS_NAME
from agent6.tools.schema import UserQuestion
from agent6.ui.cli._common import (
    _plans_dir,
    editor_argv,
    error,
    print_nothing_yet,
    resolve_or_newest_layout,
)
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._interact import default_stdin_approver, default_stdin_questioner
from agent6.viewmodel import (
    StatusFacts,
    event_epoch,
    scan_session_log,
    session_is_live,
    session_mtime,
    session_policy,
    status_for_session_dir,
    tail_events,
)
from agent6.viewmodel.events import SESSION_START_EVENTS
from agent6.viewmodel.format import dead_run_note, status_label


def _resolve_plan_session_id(session_id: str) -> str | None:
    """Resolve a plan id or prefix under the per-repo state dir.

    An empty id resolves the most recent plan, as the sessions commands do.

    Returns:
        The id, or None after printing an error.
    """
    plans_dir = _plans_dir(Path.cwd())
    if not session_id:
        latest = _most_recent_plan_session_id(plans_dir)
        if latest is None:
            error("no plans yet (start one with `agent6 plan`).")
            return None
        session_id = latest
    state = state_dir(Path.cwd())
    try:
        resolved = resolve_session(state, session_id, buckets=("plans",)).session_id
    except SessionIdError as exc:
        # An existing run or ask is named as such.
        if exc.no_match:
            try:
                other = resolve_session(state, session_id)
            except SessionIdError as other_exc:
                exc = other_exc
            else:
                error(f"{other.session_id} is a session under {other.subdir}/, not a plan")
                return None
        error(f"{exc}")
        return None
    plan = plans_dir / resolved / "plan.md"
    if not plan.is_file():
        error(f"{resolved} has no plan.md (was it created with `agent6 plan`?)")
        return None
    return resolved


def _cmd_plan_show(session_id: str) -> int:
    """Print a planning run's plan.md.

    Returns:
        The exit code; 2 when the plan cannot be resolved.
    """
    resolved = _resolve_plan_session_id(session_id)
    if resolved is None:
        return 2
    sys.stdout.write(read_operator_file(_plans_dir(Path.cwd()) / resolved / "plan.md"))
    return 0


def _cmd_plan_edit(session_id: str) -> int:
    """Open a planning run's plan.md in $EDITOR (default: vi).

    The argv is the operator's editor name and the resolved plan path, never LLM input, so
    a direct subprocess is allowed.

    Returns:
        The editor's exit code; 2 when the plan cannot be resolved.
    """
    resolved = _resolve_plan_session_id(session_id)
    if resolved is None:
        return 2
    plan = _plans_dir(Path.cwd()) / resolved / "plan.md"
    argv = editor_argv()
    if argv is None:
        return 1
    try:
        result = subprocess.run([*argv, str(plan)], check=False)
    except OSError as exc:
        error(f"failed to spawn editor {argv[0]!r}: {exc}")
        return 1
    if result.returncode != 0:
        error(f"editor {argv[0]!r} exited {result.returncode}")
        return 1
    return 0


def _most_recent_plan_session_id(plans_dir: Path) -> str | None:
    """Return the most recently active plan dir that holds a `plan.md`.

    A bare `agent6 run` offers it for execution.
    """
    if not plans_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in plans_dir.iterdir() if p.is_dir() and (p / "plan.md").is_file()),
        key=session_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else None


def _cmd_watch(
    session_id: str,
    *,
    tui: bool = False,
    since: int = 0,
    raw: bool = False,
    config_path: Path | None = None,
) -> int:
    """Follow a run directory read-only.

    The default follows the run's conversation, the same render as `agent6 run`; `--raw`
    follows the event log, one line per event; `--tui` opens the full-screen dashboard.

    Args:
        session_id: The session, or "" for the newest.
        tui: Open the dashboard.
        since: Replay this many event lines first; `--raw` only.
        raw: Tail the log lines.
        config_path: The `--config` file, if any.

    Returns:
        The exit code; 2 when the session cannot be resolved, 3 when the TUI cannot load.
    """
    cwd = Path.cwd()
    # Every run-style bucket, by id and for the newest, so a bare `attach` after an `ask` finds it.
    try:
        layout = resolve_or_newest_layout(cwd, session_id)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if layout is None:
        print_nothing_yet()
        return 2
    target = layout.session_dir
    if not session_id:
        print(f"[agent6] attached to most recent run: {target.name}", file=sys.stderr)
    if not target.is_dir():
        error(f"no such run dir: {target}")
        return 2
    if not tui:
        return _cmd_watch_plain(target, since=since) if raw else _watch_transcript(target)
    try:
        from agent6.ui.tui.app import run_tui  # noqa: PLC0415  # textual is optional
    except ImportError as e:
        error(f"{e}")
        print(
            "HINT: drop --tui for the conversation view, or pass --raw for the line tail.",
            file=sys.stderr,
        )
        return 3
    run_tui(target, config_path=config_path)
    return 0


def _cmd_tui(config_path: Path | None = None) -> int:
    """Run the TUI hub (`agent6 tui`): browse runs and start new work.

    Loops between the home screen and the run view; opening a run watches it, then returns
    here on close.

    Returns:
        The exit code; 3 when textual is not installed.
    """
    try:
        from agent6.ui.tui.app import (  # noqa: PLC0415  # textual is optional
            run_tui,
        )
        from agent6.ui.tui.home import run_home  # noqa: PLC0415
    except ImportError as e:
        error(f"{e}")
        print("HINT: the TUI needs 'textual' (part of the base install).", file=sys.stderr)
        return 3
    cwd = Path.cwd()
    # Every bucket lookup goes through `bucket_dir`, which appends `sessions/` itself.
    agent6_dir = state_dir(cwd)
    session_dir: Path | None = None
    while True:
        # Esc in a run view reopens home, Run this plan hands the new run back, Ctrl+Q quits.
        session_dir = session_dir or run_home(agent6_dir, cwd, config_path)
        if session_dir is None:
            return 0
        result = run_tui(session_dir, from_hub=True, config_path=config_path)
        if result.quit_hub:
            return 0
        session_dir = result.open_next


def format_plain_event(line: str, *, session_start_ts: float | None) -> str:
    """Return one logs.jsonl line as `<elapsed> <type> key=val ...`.

    Falls back to the raw line on a parse error, so a corrupt event does not abort the tail.

    Args:
        line: The raw line.
        session_start_ts: The earliest event's wall-clock time so far, for the elapsed column.
    """
    raw = line.rstrip("\n")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(obj, dict):
        return raw
    ts = event_epoch(obj.get("ts"))
    event = str(obj.get("event") or obj.get("type") or "?")
    if ts is not None and session_start_ts is not None:
        elapsed = max(0.0, ts - session_start_ts)
        ts_str = f"+{elapsed:7.1f}s"
    else:
        ts_str = "        "
    skip = {"ts", "event", "type", "session_id"}
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


def _standing(event: dict[str, object]) -> bool:
    """Return whether an `approval.prompt` offers a session-wide answer.

    Absent means it does; only the scopeless gates journal False.
    """
    return bool(event.get("standing", True))


class _CliFrontEnd:
    """The answering front-end an interactive `agent6 attach` becomes.

    When the streamed log surfaces an unanswered `run_command` approval or `ask_user`
    question, it prompts on the controlling terminal with the CLI prompts a foreground run
    uses and writes the answer back over the file bridge. The caller registers a
    `frontends/` claim, so the worker's approver bridges here: a live front-end wins over
    the detach away-mode. Prompt ids are deterministic counters and the log replays from
    the start on attach, so the answered and handled id sets gate re-prompting.
    """

    def __init__(self, session_dir: Path, view: ConsoleView) -> None:
        """Bind the session dir and the console view; nothing is prompted yet."""
        self._session_dir = session_dir
        self._view = view
        self._answered: set[str] = set()
        self._handled: set[str] = set()
        # How many events the pre-scan decided; the append-only log hands them to `react` first.
        self._replayed: int = 0

    def open_prompts_at_attach(self, events_path: Path) -> list[dict[str, object]]:
        """Pre-scan the existing log and return the prompts open right now.

        Seeds the answered set, so a run already waiting at an approval when you attach is
        handled at once.

        Args:
            events_path: The run's logs.jsonl.

        Returns:
            The prompt events emitted but not answered.
        """
        open_prompts: dict[str, dict[str, object]] = {}
        scanned = 0
        for ev in tail_events(events_path, follow=False):
            scanned += 1
            etype = str(ev.get("type", ""))
            pid = str(ev.get("id", ""))
            if etype in SESSION_START_EVENTS:
                self._new_session()
                open_prompts.clear()
            if etype in ("approval.prompt", "question.prompt"):
                open_prompts[pid] = ev
            elif etype in ("approval.answer", "question.answer"):
                self._answered.add(pid)
                open_prompts.pop(pid, None)
        self._replayed = scanned
        return list(open_prompts.values())

    def handle(self, event: dict[str, object]) -> None:
        """Prompt on the terminal for an open prompt event and write the answer over the bridge.

        The spinner is paused for the prompt, and the id is marked handled so the follow-loop
        replay does not re-ask it. An approval's journaled `standing` rides along: False is a
        gate with no scope to grant (`fetch`), so the prompt offers no session choice.
        """
        prompt_id = str(event.get("id", ""))
        if event.get("type") == "approval.prompt":
            with self._view.pause():
                answer = default_stdin_approver(
                    str(event.get("prompt", "")), standing=_standing(event)
                )
            if not write_answer(self._session_dir, prompt_id, answer or "no"):
                self._view.notice(f"[agent6] {ANSWERED_ELSEWHERE}")
        else:
            raw_questions = event.get("questions", [])
            questions = tuple(
                UserQuestion(
                    question=str(q.get("question", "")),
                    options=tuple(str(o) for o in q.get("options", [])),
                )
                for q in (raw_questions if isinstance(raw_questions, list) else [])
            )
            with self._view.pause():
                answers = default_stdin_questioner(questions)
            written = write_question_answers(
                self._session_dir,
                prompt_id,
                answers if answers is not None else tuple("" for _ in questions),
            )
            if not written:
                self._view.notice(f"[agent6] {ANSWERED_ELSEWHERE}")
        self._handled.add(prompt_id)

    def _new_session(self) -> None:
        """Forget the prior execution's prompt ids.

        A session boundary restarts the counters at approval-1 and question-1.
        """
        self._answered.clear()
        self._handled.clear()

    def react(self, event: dict[str, object]) -> None:
        """Answer a new unanswered prompt on the live follow; a replayed one is skipped."""
        etype = str(event.get("type", ""))
        pid = str(event.get("id", ""))
        if self._replayed > 0:
            # Inside the pre-scan's window: keep the bookkeeping in step, never prompt.
            self._replayed -= 1
            if etype in SESSION_START_EVENTS:
                self._new_session()
            elif etype in ("approval.answer", "question.answer"):
                self._answered.add(pid)
            return
        if etype in SESSION_START_EVENTS:
            self._new_session()
            return
        if etype in ("approval.answer", "question.answer"):
            self._answered.add(pid)
            return
        if pid in self._handled or pid in self._answered:
            return
        if etype in ("approval.prompt", "question.prompt"):
            self.handle(event)


def _print_crashed_line(target: Path) -> None:
    """Print the crashed-or-killed line for a run no session end settled."""
    print(
        f"[agent6] {target.name}: {status_label('stale', dead_run_note('stale', '')[0])};"
        f" see `agent6 sessions show {target.name}`.",
        file=sys.stderr,
    )


def _render_over_session(target: Path, events_path: Path, *, finished: bool) -> int:
    """Render the log of a session no worker is driving, then say how it ended.

    Nothing more will be appended and no answer would be read, so there is no front-end and
    no re-asked prompt.

    Args:
        target: The session dir.
        events_path: Its logs.jsonl.
        finished: The run ended cleanly and already said its outcome, so it must not read
            as crashed.

    Returns:
        The exit code, 0.
    """
    view = ConsoleView(sys.stdout, policy=lambda: session_policy(target).line())
    try:
        for event in tail_events(events_path, follow=False):
            view.feed(event)
        if not finished:
            # No session.end settled the call the worker died on.
            view.settle_dead("the run died")
    finally:
        view.close()
    if not finished:
        _print_crashed_line(target)
        return 1
    return 0


def _install_front_end(target: Path, view: ConsoleView) -> _CliFrontEnd | None:
    """Attach as the answering front-end on an interactive terminal.

    Returns:
        The front-end, or None when a stream is piped or redirected: then attach stays a
        pure reader.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            f"[agent6] following {target.name}. Ctrl-C exits; agent6 stop {target.name} stops it.",
            file=sys.stderr,
        )
        return None
    front_end = _CliFrontEnd(target, view)
    register_frontend(target, os.getpid())
    print(
        f"[agent6] attached to {target.name}: approvals and questions prompt here."
        f" Ctrl-C detaches; agent6 stop {target.name} stops it.",
        file=sys.stderr,
    )
    return front_end


def _watch_transcript(target: Path) -> int:
    """Follow a run's conversation live and, on a terminal, attach as its front-end.

    Folds `logs.jsonl` through the same `ConsoleView` as `agent6 run`; when the run asks
    for a `run_command` approval or an `ask_user` answer, prompts exactly as the foreground
    run would. Renders from the start, tails until the run ends, then returns; Ctrl-C exits.
    A detach emits no session end, so a detached run is followed to its end.

    Returns:
        The exit code; 1 when the worker died mid-call, 2 when the run has no log yet.
    """
    events_path = target / LOGS_NAME
    if not events_path.is_file():
        # Not an error: a parked submission, a `fork --no-run` or a launching run has no log yet.
        word, reason = status_for_session_dir(target, StatusFacts())
        print(f"{target.name}: {status_label(word, reason)}")
        if word == "starting":
            # A live worker mid-preflight is running, not resumable; it has no log to follow yet.
            print("it is starting; run this again in a moment to follow it.")
        else:
            print(f"start it with: agent6 resume {target.name}")
        return 0

    # Liveness as every surface answers it; whether the run ended is a separate fact.
    if not session_is_live(target):
        scan = scan_session_log(events_path)
        return _render_over_session(target, events_path, finished=scan.finished)

    def worker_dead() -> bool:
        """Return whether the worker is gone, from its pid alone."""
        # Per poll, O(1): once following, the worker pid is the liveness evidence.
        return not worker_is_alive(target)

    view = ConsoleView(sys.stdout, policy=lambda: session_policy(target).line())
    front_end = _install_front_end(target, view)
    interrupted = False
    try:
        if front_end is not None:
            for event in front_end.open_prompts_at_attach(events_path):
                front_end.handle(event)
        for event in tail_events(
            events_path, follow=True, stop_when_finished=True, should_stop=worker_dead
        ):
            view.feed(event)
            if front_end is not None:
                front_end.react(event)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[agent6] watch: stopped.", file=sys.stderr)
    finally:
        view.close()  # stop the heartbeat thread, clear any spinner line
        if front_end is not None:
            unregister_frontend(target, os.getpid())  # our claim only
    if not interrupted and not scan_session_log(events_path).finished:
        # No session.end settled the call the worker died on.
        view.settle_dead("the run died")
        _print_crashed_line(target)
        return 1
    return 0


def _line_is_session_end(raw: bytes | str) -> bool:
    """Return whether a logs.jsonl line is a `session.end` event, so the follower stops there."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and obj.get("type") == "session.end"


def _cmd_watch_plain(target: Path, *, since: int) -> int:  # noqa: C901, PLR0911, PLR0912, PLR0915  # one branch per event kind the watch prints
    """Tail `logs.jsonl` line by line with no extra deps.

    Polls the file with 0.25s sleeps; rotates when the inode changes.

    Args:
        target: The session dir.
        since: Replay this many lines first.

    Returns:
        0 at `session.end` or on Ctrl-C, 1 when the worker or the log dies first.
    """
    events_path = target / LOGS_NAME
    if not events_path.is_file():
        error(f"no logs.jsonl in {target}")
        return 2

    # The first event is the elapsed-time anchor; a torn first line must not crash the watch.
    session_start_ts: float | None = None
    try:
        with events_path.open("rb") as fh:
            first = fh.readline()
        if first:
            obj0 = json.loads(first.decode("utf-8"))
            if isinstance(obj0, dict):
                session_start_ts = event_epoch(obj0.get("ts"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        session_start_ts = None

    print(
        f"[agent6] tailing {events_path}. Ctrl-C to exit.",
        file=sys.stderr,
    )

    # Binary reads: a read can hit EOF mid UTF-8 sequence, so a partial tail stays buffered.
    try:
        fh = events_path.open("rb")
    except OSError as exc:
        error(f"cannot open {events_path}: {exc}")
        return 2

    try:
        pending = b""
        if since > 0:
            # Replay the last `since` lines before following.
            try:
                lines = fh.readlines()
            except OSError as exc:
                error(f"read failed: {exc}")
                return 2
            if lines and not lines[-1].endswith(b"\n"):
                pending = lines.pop()
            for raw in lines[-since:]:
                line = raw.decode("utf-8", errors="replace")
                # Piped output must not lose the replay to the block buffer while the run is idle.
                print(format_plain_event(line, session_start_ts=session_start_ts), flush=True)
            if lines and scan_session_log(events_path).finished:
                return 0  # already finished: replayed, nothing to follow
        else:
            # A finished run has no new events to follow; seeking to end would hang.
            if scan_session_log(events_path).finished:
                print("[agent6] run already finished.", file=sys.stderr)
                return 0
            # Seek to end; only show new events going forward.
            fh.seek(0, 2)
        try:
            current_ino = events_path.stat().st_ino
        except OSError:
            current_ino = -1
        while True:
            chunk = fh.readline()
            if chunk:
                pending += chunk
                if not pending.endswith(b"\n"):
                    continue  # partial line at EOF; the rest arrives next read
                line = pending.decode("utf-8", errors="replace")
                pending = b""
                print(format_plain_event(line, session_start_ts=session_start_ts), flush=True)
                if _line_is_session_end(line):
                    return 0  # run ended: stop, like the default follower
                continue
            # A vanished log is EOF only once the worker is dead: a live run recreates the path.
            try:
                new_ino = events_path.stat().st_ino
            except OSError:
                if worker_is_alive(target):
                    time.sleep(0.5)
                    continue
                print(f"[agent6] {events_path} is gone; stopping.", file=sys.stderr)
                return 1
            if new_ino != current_ino:
                with contextlib.suppress(OSError):
                    fh.close()
                try:
                    fh = events_path.open("rb")
                except OSError:
                    if worker_is_alive(target):
                        time.sleep(0.5)
                        continue
                    print(f"[agent6] {events_path} is gone; stopping.", file=sys.stderr)
                    return 1
                current_ino = new_ino
                pending = b""
                continue
            if not worker_is_alive(target):
                _print_crashed_line(target)
                return 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            fh.close()
