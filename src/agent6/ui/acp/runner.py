# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run one agent6 run per ACP prompt.

The protocol owns stdout, so the reporter writes to stderr. The run id is minted
before the run starts, so `session/cancel` has something to address, and once per
ACP session: later prompts resume the same run with their text as its first
steering instruction. `run_task` reads the process cwd, so runs are serialised on
the connection and a second prompt waits.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any, BinaryIO

from agent6 import errors, kinds
from agent6 import paths as agent6_paths
from agent6.app import _setup, finalize, frontend, resume
from agent6.app import reporter as app_reporter
from agent6.app import run as app_run
from agent6.sessions import id, ipc
from agent6.ui import spawn
from agent6.ui.acp import frontend as acp_frontend
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as acp_session
from agent6.ui.acp import updates
from agent6.viewmodel import listing, transcript
from agent6.viewmodel import tail as viewmodel_tail

# Bounds the join on a tail wedged on a filesystem; `_stop` ends it one read pass after the run.
DRAIN_S = 5.0
# How often a queued turn checks for its own cancel while another turn holds the run lock.
QUEUE_POLL_S = 0.1


def _stderr(message: str) -> None:
    """Print to stderr, the editor's agent log."""
    print(message, file=sys.stderr)


@dataclasses.dataclass
class ProseOrder:
    """Queue the lifecycle's lines for the journal tail to emit in order.

    The tail projects the journal a poll behind the run thread, so a line sent as
    said would land before the turn's last tool calls. Each line is stamped with
    the journal's size when said, and emitted once the tail has read past it.

    Attributes:
        server: The connection.
        acp_session_id: The conversation the lines address.
        logs_path: The journal.
    """

    server: acp_server.ACPServer
    acp_session_id: str
    logs_path: pathlib.Path
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _pending: list[tuple[int, str]] = dataclasses.field(default_factory=list)

    def say(self, text: str) -> None:
        """Queue a line, stamped with the journal's size now."""
        with self._lock:
            self._pending.append((viewmodel_tail.journal_size(self.logs_path), text))

    def flush(self, consumed: int | None) -> None:
        """Emit every line stamped at or before the consumed journal size; all for None."""
        with self._lock:
            due = [t for stamp, t in self._pending if consumed is None or stamp <= consumed]
            self._pending = [
                (s, t) for s, t in self._pending if consumed is not None and s > consumed
            ]
        for text in due:
            # A note or warning already carries the marker `message_update` adds.
            self.server.notify_raw(
                updates.message_update(self.acp_session_id, text.removeprefix("[agent6] "))
            )


def forwarding_reporter(
    server: acp_server.ACPServer,
    acp_session_id: str,
    said: list[str],
    *,
    order: ProseOrder | None = None,
) -> app_reporter.Reporter:
    """Build the lifecycle's reporter: stderr, and the same line to the editor as agent6's prose.

    No journal event carries a refusal's reason, so the reporter is the one route
    for it; the cost receipt goes to stderr only, since the fold's done item
    carries the cost.

    Args:
        server: The connection.
        acp_session_id: The conversation the lines address.
        said: Collects the forwarded lines, so a silent end is told from an explained one.
        order: Emits the lines in journal order while a tail projects the journal.

    Returns:
        The reporter.
    """

    def _say(message: str) -> None:
        _stderr(message)
        text = message.strip()
        if not text:
            return
        said.append(text)
        if order is not None:
            order.say(text)
        else:
            server.notify_raw(
                updates.message_update(acp_session_id, text.removeprefix("[agent6] "))
            )

    return app_reporter.Reporter(out=_say, err=_say, receipt=_stderr)


def option_kind(text: str, standing: bool | None) -> str:
    """Return ACP's button kind from who asked, never from the option text.

    A model-written option named "allow" must not be advertised as `allow_always`.

    Args:
        text: The option.
        standing: True for an approval the editor may remember, False for one it
            must not, None for a question the model wrote.
    """
    if standing is None:
        return "allow_once"
    if text == "deny":
        return "reject_once"
    return "allow_always" if standing else "allow_once"


def stop_reason(code: int, *, end_reason: str = "") -> acp_session.StopReason:
    """Return ACP's stop reason for the lifecycle's exit code.

    A deliberate finish is `end_turn` even over a red gate or stranded edits, which
    are already on the wire; `refusal` is ACP's one word for a run that could not
    complete.

    Args:
        code: The exit code.
        end_reason: The journal's end reason for this turn, or "".
    """
    if code == 130:
        return "cancelled"
    if end_reason == "max_iterations":
        return "max_turn_requests"
    return (
        "end_turn"
        if code in (0, finalize.EXIT_VERIFY_FAILED, finalize.EXIT_NO_COMMIT_LANDED)
        else "refusal"
    )


def _selected(answer: dict[str, Any], options: tuple[str, ...]) -> str | None:
    """Return the option the editor chose, or None for no usable answer.

    The answer must be an offered index, or an unknown string could become an
    "allow" by prefix.
    """
    outcome = answer.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
        return None
    chosen = outcome.get("optionId")
    if not isinstance(chosen, str):
        return None
    offered = {str(index): option for index, option in enumerate(options)}
    return offered.get(chosen)


class Announced:
    """Register one turn's number and the tool calls the editor has been told about.

    A permission request waits here until the call it names is announced, so the
    editor never hears of a call first through its approval. The wait ends on
    liveness, never a clock: the tail closes the register when it stops reading.

    Attributes:
        turn: The turn's number.
    """

    def __init__(self, turn: int) -> None:
        self.turn = turn
        self._ids: set[str] = set()
        self._closed = False
        self._changed = threading.Condition()

    def __contains__(self, tool_call_id: str) -> bool:
        with self._changed:
            return tool_call_id in self._ids

    def add(self, tool_call_id: str) -> None:
        """Record a call as announced and wake every waiter."""
        with self._changed:
            self._ids.add(tool_call_id)
            self._changed.notify_all()

    def close(self) -> None:
        """Close the register once the tail stops reading, ending every wait."""
        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def wait_for(self, tool_call_id: str, *, abandoned: Callable[[], bool]) -> None:
        """Wait until the call is announced, the register closes or the turn is abandoned."""
        with self._changed:
            while tool_call_id not in self._ids and not self._closed and not abandoned():
                self._changed.wait(0.5)  # abandonment is polled; add and close wake at once


def _result_paths(event: dict[str, Any]) -> tuple[str, ...]:
    """Return the paths a tool.result journaled, for the editor's follow-along."""
    raw = event.get("paths")
    if not isinstance(raw, list):
        return ()
    return tuple(path for path in raw if isinstance(path, str) and path.strip())


@dataclasses.dataclass
class RunBridge:
    """Run prompts for one ACP connection.

    Attributes:
        server: The connection.
        config_path: The `--config FILE` overlay every session load threads.
    """

    server: acp_server.ACPServer
    config_path: pathlib.Path | None = None
    # One run at a time: the chdir in `_run` is process-global.
    _runs: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _running: acp_session.Session | None = None  # the turn holding `_runs`, named to a queued one
    _asks: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _asked: int = 0

    def __post_init__(self) -> None:
        if self.config_path is not None:
            self.config_path = self.config_path.resolve()

    def sessions(self) -> acp_session.Sessions:
        """Return the session table wired to this bridge."""
        return acp_session.Sessions(run=self.run, state_dir_for=agent6_paths.state_dir)

    def ask(
        self,
        session: acp_session.Session,
        announced: Announced,
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        """Put one approval or question to the editor as a permission request.

        ACP v1 has no free-form question, so a question's options are its buttons,
        and one with no options is answered "said nothing" at once. The request's
        `toolCall` carries the whole prompt as the title; a prompt gating a call
        names that call once announced, and one gating none announces and closes
        an entity of its own.

        Args:
            session: The session.
            announced: The turn's register of announced calls.
            prompt: The text.
            options: The choices, in order.
            standing: Whether "always" may be offered; None for a question.
            call_id: The dispatcher's stamp on the gated call, or None.
            until: Polled while the answer is pending; True ends the wait.

        Returns:
            The chosen option, or None for no answer.
        """
        if not options:
            return None
        if call_id is not None:
            gated = updates.wire_call_id(session.session_id, announced.turn, str(call_id))
            announced.wait_for(gated, abandoned=lambda: session.cancelled)
            if session.cancelled or gated not in announced:
                return None
            tool_call: dict[str, Any] = {
                "toolCallId": gated,
                "title": updates.printable(prompt),
                "status": "pending",
            }
        else:
            with self._asks:
                self._asked += 1
                gated = f"ask-{session.acp_id}-{self._asked}"
            tool_call = {
                "toolCallId": gated,
                "title": updates.printable(prompt),
                "kind": "other",
                "status": "pending",
            }

        def _done() -> bool:
            return session.cancelled or (until is not None and until())

        answer = self.server.request(
            "session/request_permission",
            {
                "sessionId": session.acp_id,
                "toolCall": tool_call,
                "options": [
                    {
                        # An index, not the text: the text can be model-written.
                        "optionId": str(index),
                        "name": updates.printable(text),
                        "kind": option_kind(text, standing),
                    }
                    for index, text in enumerate(options)
                ],
            },
            timeout_s=acp_frontend.PERMISSION_TIMEOUT_S,
            until=_done,
        )
        chosen = _selected(answer, options)
        if call_id is None:
            self.server.notify_raw(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session.acp_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": gated,
                            "status": "completed" if chosen else "failed",
                        },
                    },
                }
            )
        return chosen

    def _frontend(
        self, session: acp_session.Session, announced: Announced
    ) -> frontend.SessionFrontend:
        """Return the front-end for one turn."""
        return acp_frontend.acp_frontend(
            ask=lambda prompt, options, standing, call_id, until=None: self.ask(
                session, announced, prompt, options, standing, call_id, until
            ),
            # Before `initialize` lands nothing is known about the client, so it can do nothing.
            capabilities=self.server.client_capabilities or frontend.FrontendCapabilities(),
            agent6_exe=spawn.agent6_exe,
            spawn_detached_resume=lambda cwd, sid, flags: spawn.spawn_detached_resume(
                cwd, sid, config_path=self.config_path, flags=flags
            ),
        )

    def _resumable(self, session: acp_session.Session) -> bool:
        """Return whether the prior turn left a resume snapshot for this prompt to continue."""
        if not session.session_id:
            return False
        layout = session.layout(agent6_paths.state_dir(session.cwd))
        return (layout.session_dir / "loop_state.json").is_file()

    def _recorded(self, session: acp_session.Session) -> bool:
        """Return whether the session's run id names a recorded run, which starts nothing again."""
        if not session.session_id:
            return False
        return session.layout(agent6_paths.state_dir(session.cwd)).manifest_path.exists()

    def _cancelled_unstarted(self, session: acp_session.Session) -> acp_session.StopReason:
        """Clear the cancel's marker, which would otherwise stop the next turn at its first step.

        Returns:
            The stop reason "cancelled".
        """
        ipc.clear_stop_request(session.layout(agent6_paths.state_dir(session.cwd)).session_dir)
        return "cancelled"

    def had_journal(self, session: acp_session.Session) -> bool:
        """Return whether the turn got far enough to write a journal."""
        if not session.session_id:
            return False
        return session.layout(agent6_paths.state_dir(session.cwd)).logs_path.exists()

    def run(self, session: acp_session.Session, text: str) -> acp_session.StopReason:
        """Run one prompt as a turn, queued behind another session's turn.

        Returns:
            ACP's stop reason.
        """
        started_at = time.time()  # a cancel written from here on is this turn's to honor
        # The id is minted before the queue, so a cancel during the wait has a run to address.
        try:
            resuming = self._resumable(session)
            if not resuming and self._recorded(session):
                # A recorded run with no resume point is refused both ways; a new run says so.
                self.server.notify_raw(
                    updates.message_update(
                        session.acp_id,
                        f"[agent6] run {session.session_id} left no resume point;"
                        " this prompt starts a new run",
                    )
                )
                session.session_id = ""
            if not session.session_id:
                session.session_id = id.unused_session_id(
                    agent6_paths.state_dir(session.cwd), kinds.session_bucket(acp_session.ACP_MODE)
                )
        except Exception as exc:  # an unreadable config raises here, before any journal
            self._could_not_finish(session, exc)
            return "refusal"
        if not self._runs.acquire(blocking=False):
            holder = self._running
            who = f"session {holder.acp_id}" if holder is not None else "another session"
            self.server.notify_raw(
                updates.message_update(
                    session.acp_id,
                    f"waiting: {who} is running a turn; this prompt starts when it ends",
                )
            )
            while not self._runs.acquire(timeout=QUEUE_POLL_S):
                if session.cancelled:
                    return self._cancelled_unstarted(session)
        try:
            self._running = session
            if session.cancelled:
                return self._cancelled_unstarted(session)
            try:
                return self._run(session, text, resuming=resuming, started_at=started_at)
            except Exception as exc:
                self._could_not_finish(session, exc)
                return "refusal"
        finally:
            self._running = None
            self._runs.release()

    def _could_not_finish(self, session: acp_session.Session, exc: Exception) -> None:
        """Tell the editor why a run died before the turn returns its stop reason.

        A non-operator fault's traceback goes to stderr.
        """
        what = "could not start" if not self.had_journal(session) else "failed"
        self.server.notify_raw(updates.message_update(session.acp_id, f"the run {what}: {exc}"))
        if not isinstance(exc, errors.OperatorError):
            _stderr(f"[agent6] run {session.session_id}: {traceback.format_exc()}")

    def _run(
        self, session: acp_session.Session, text: str, *, resuming: bool, started_at: float
    ) -> acp_session.StopReason:
        """Run the turn under the run lock, with a tail projecting its journal.

        Returns:
            ACP's stop reason.
        """
        layout = session.layout(agent6_paths.state_dir(session.cwd))
        os.chdir(session.cwd)
        session.turn += 1
        journal_before = viewmodel_tail.journal_size(layout.logs_path)

        said: list[str] = []
        announced = Announced(turn=session.turn)
        ended, drained = threading.Event(), threading.Event()

        def _stop() -> bool:
            """Return whether the tail stops: one pass after the run, so its last lines land."""
            if not ended.is_set():
                return False
            if drained.is_set():
                return True
            drained.set()
            return False

        order = ProseOrder(self.server, session.acp_id, layout.logs_path)
        start_at = journal_before if resuming else None
        tail = threading.Thread(
            target=self._stream,
            args=(session, layout.logs_path, _stop, start_at, announced, order),
            name=f"acp-tail-{session.acp_id}",
            daemon=True,
        )
        tail.start()
        reporter = forwarding_reporter(self.server, session.acp_id, said, order=order)
        try:
            if resuming:
                # The prompt is the resumed run's first steering instruction.
                code = resume.resume_task(
                    self.config_path,
                    session.session_id,
                    frontend=self._frontend(session, announced),
                    force=False,
                    started_at=started_at,
                    steer=text,
                    reporter=reporter,
                )
            else:
                effective = _setup.load_session_config(
                    session.cwd, self.config_path, mode=acp_session.ACP_MODE
                )
                code = app_run.run_task(
                    effective.config,
                    text,
                    frontend=self._frontend(session, announced),
                    started_at=started_at,
                    session_id=session.session_id,
                    explicit_leaves=effective.explicit_leaves,
                    reporter=reporter,
                )
        finally:
            ended.set()
            tail.join(timeout=DRAIN_S)
            order.flush(None)  # a tail that outlived the drain still owes these
        if code != 0 and not said and not self.had_journal(session):
            self.server.notify_raw(
                updates.message_update(session.acp_id, f"the run stopped (exit {code})")
            )
        # Only a journal that grew and ended this execution carries a reason of this turn.
        grown = viewmodel_tail.journal_size(layout.logs_path) > journal_before
        scan = listing.scan_session_log(layout.logs_path)
        end_reason = scan.end_reason if grown and scan.finished else ""
        return stop_reason(code, end_reason=end_reason)

    def _stream(  # noqa: PLR0912
        self,
        session: acp_session.Session,
        logs_path: pathlib.Path,
        stop: Callable[[], bool],
        journal_before: int | None,
        announced: Announced,
        order: ProseOrder | None = None,
    ) -> None:
        """Project the run's journal into `session/update` as it is written.

        The lifecycle's own lines take their place between events; the ending also
        goes to stderr, since the lifecycle prints none of its own.

        Args:
            session: The session.
            logs_path: The journal.
            stop: Polled by the tail; True ends it.
            journal_before: Where to start on a resumed run, whose earlier turns the
                editor already rendered; None replays the whole journal.
            announced: The turn's register of announced calls.
            order: The lifecycle's queued lines.
        """
        fold = transcript.TranscriptFold()
        streamed: set[str] = set()
        consumed = [0]

        def _at(position: int) -> None:
            consumed[0] = position

        try:
            for event in viewmodel_tail.tail_events(
                logs_path,
                stop_when_finished=True,
                should_stop=stop,
                start_at=journal_before,
                on_position=_at,
            ):
                event_type = str(event.get("type", ""))
                paths = _result_paths(event) if event_type == "tool.result" else ()
                if event_type == "role.call":
                    streamed.clear()
                is_delta = event_type in ("role.thinking_delta", "role.text_delta")
                side_delta = is_delta and kinds.is_side_role(str(event.get("role", "")))
                items = [] if side_delta else fold.feed(event)
                if is_delta and not side_delta:
                    kind = "thinking" if event_type == "role.thinking_delta" else "text"
                    streamed.add(kind)
                    for body in updates.updates_for(
                        transcript.TranscriptItem(kind, body=str(event.get("text", ""))),
                        acp_session_id=session.acp_id,
                        streamed=True,
                    ):
                        self.server.notify_raw(body)
                for item in items:
                    if item.kind in streamed:
                        continue
                    wire_id = (
                        updates.tool_call_id(item, session.session_id, announced.turn)
                        if item.kind == "tool"
                        else ""
                    )
                    for body in updates.updates_for(
                        item,
                        acp_session_id=session.acp_id,
                        wire_id=wire_id,
                        announced=wire_id in announced,
                        cwd=session.cwd,
                        paths=paths,
                    ):
                        self.server.notify_raw(body)
                    if item.kind == "tool":
                        announced.add(wire_id)
                    elif item.kind == "done":
                        _stderr(updates.ending(item))
                if event_type == "role.result":
                    streamed.clear()
                if order is not None:
                    order.flush(consumed[0])
        finally:
            announced.close()
            for item in fold.settle_open_calls("the run died"):
                wire_id = updates.tool_call_id(item, session.session_id, announced.turn)
                for body in updates.updates_for(
                    item,
                    acp_session_id=session.acp_id,
                    wire_id=wire_id,
                    announced=wire_id in announced,
                    cwd=session.cwd,
                ):
                    self.server.notify_raw(body)
            if order is not None:
                order.flush(None)


def serve_acp(
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    *,
    config_path: pathlib.Path | None = None,
) -> int:
    """Speak ACP on the process's stdio until the editor closes it.

    Args:
        stdin: The client's requests; None is the process's stdin.
        stdout: The replies; None is the process's stdout.
        config_path: The `--config FILE` overlay, or None.

    Returns:
        The exit code, 0.
    """
    server = acp_server.ACPServer(
        stdin=stdin if stdin is not None else sys.stdin.buffer,
        stdout=stdout if stdout is not None else sys.stdout.buffer,
    )
    server.sessions = RunBridge(server=server, config_path=config_path).sessions()
    server.serve()
    return 0
