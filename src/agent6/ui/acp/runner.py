# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run one agent6 run per ACP prompt.

The protocol owns stdout, so the reporter writes to stderr. The run id is minted
before the run starts, so `session/cancel` has something to address, and once per
ACP session: later prompts resume the same run with their text as its first
steering instruction. Each execution receives its session cwd explicitly.
Runs are serialised on the connection and a second prompt waits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import sys
import threading
import time
import traceback
from collections.abc import Callable, Generator
from typing import Any, BinaryIO

from agent6 import directive, errors, kinds
from agent6 import paths as agent6_paths
from agent6.app import finalize, frontend, resume
from agent6.app import reporter as app_reporter
from agent6.app import run as app_run
from agent6.app.session import setup as session_setup
from agent6.sessions import id, ipc
from agent6.sessions import layout as sessions_layout
from agent6.ui import keymap, operator_images, spawn
from agent6.ui.acp import frontend as acp_frontend
from agent6.ui.acp import model, options, rpc, stream, updates
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as acp_session
from agent6.viewmodel import tail as viewmodel_tail
from agent6.viewmodel.session import log as session_log

# Bounds the join on a tail wedged on a filesystem; `_stop` ends it one read pass after the run.
DRAIN_S = 5.0
# How often a queued prompt checks for its own cancel while another execution holds the run lock.
QUEUE_POLL_S = 0.1


def _stderr(message: str) -> None:
    """Print to stderr, the editor's agent log."""
    print(message, file=sys.stderr)


def forwarding_reporter(
    server: acp_server.ACPServer,
    acp_session_id: str,
    said: list[str],
    *,
    order: stream.ProseOrder | None = None,
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
            server.update(acp_session_id, updates.notice(text.removeprefix("[agent6] ")))

    return app_reporter.Reporter(out=_say, err=_say, receipt=_stderr)


# What each answer of a scoped approval grants, as ACP's option kinds name it.
_STANDING_KINDS = {
    entry.label: ("allow" if entry.grants else "reject")
    + ("_always" if entry.standing else "_once")
    for entry in keymap.APPROVAL_ANSWERS
}


def option_kind(text: str, standing: bool | None) -> str:
    """Return ACP's button kind from who asked and what the answer grants.

    A model-written option named "allow" must not be advertised as `allow_always`, and an
    `_always` kind names only an answer that writes a session grant or deny.

    Args:
        text: The option.
        standing: True for a scoped approval, whose "allow all" and "deny all" the session
            remembers; False for one nothing may remember; None for a question the model
            wrote.

    Returns:
        `allow_once`, `allow_always`, `reject_once` or `reject_always`; every option of a
        question is `allow_once`.
    """
    if standing is None:
        return "allow_once"
    if not standing:
        return "reject_once" if text == "deny" else "allow_once"
    return _STANDING_KINDS.get(text, "allow_once")


def stop_reason(code: int, *, stop_reason: str = "") -> acp_session.StopReason:
    """Return ACP's stop reason for the lifecycle's exit code.

    Args:
        code: The exit code.
        stop_reason: The journal's end reason for this execution, or "".

    Returns:
        `cancelled` for an operator stop (exit 130) and `max_turn_requests` for a run that
        hit its turn cap. A deliberate finish is `end_turn` even over a red gate or
        stranded edits: those are already on the wire.

    Raises:
        RpcError: The run failed or never started. ACP answers the prompt with an error, and
            keeps `refusal` for a model that declines the prompt.
    """
    if code == 130:
        return "cancelled"
    if stop_reason == "max_turns":
        return "max_turn_requests"
    if code in (0, finalize.EXIT_VERIFY_FAILED, finalize.EXIT_NO_COMMIT):
        return "end_turn"
    why = f": {stop_reason}" if stop_reason else ""
    raise rpc.RpcError(rpc.INTERNAL_ERROR, f"the run stopped{why} (exit {code})")


class Announced:
    """Register one execution's id and the tool calls the editor has been told about.

    A permission request waits here until the call it names is announced, so the
    editor never hears of a call first through its approval. The wait ends on
    liveness, never a clock: the tail closes the register when it stops reading.
    Calls are registered by the dispatcher's stamp; the tail learns the execution's
    id from its start row, before any call.

    Attributes:
        execution_id: The execution's id; "" until the tail reads its start row.
    """

    def __init__(self, execution_id: str = "") -> None:
        self.execution_id = execution_id
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
        """Wait until the call is announced, the register closes or the execution is abandoned."""
        with self._changed:
            while tool_call_id not in self._ids and not self._closed and not abandoned():
                self._changed.wait(0.5)  # abandonment is polled; add and close wake at once


@dataclasses.dataclass
class RunBridge:
    """Run prompts for one ACP connection.

    Attributes:
        server: The connection.
        config_paths: The `--config FILE` layers every session load threads.
    """

    server: acp_server.ACPServer
    config_paths: tuple[pathlib.Path, ...] = ()
    _runs: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _running: acp_session.Session | None = None  # under `_runs`; named to a queued prompt

    def __post_init__(self) -> None:
        self.config_paths = tuple(path.resolve() for path in self.config_paths)

    def sessions(self) -> acp_session.Sessions:
        """Return the session table wired to this bridge."""
        return acp_session.Sessions(
            run=self.run, state_dir_for=agent6_paths.state_dir, build_options=self.config_options
        )

    def config_options(self, session: acp_session.Session) -> list[dict[str, Any]]:
        """Return the session's complete config options: what its next prompt runs under."""
        recorded = (
            session.layout(agent6_paths.state_dir(session.cwd)).session_dir
            if session.session_id
            else None
        )
        return options.config_options(
            session.cwd,
            self.config_paths,
            recorded,
            mode=acp_session.ACP_MODE,
            model=session.model,
            effort=session.effort,
        )

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

        ACP has no free-form question here, so a question's options are its buttons, and
        one with no options is answered "said nothing" at once. The request carries the
        whole prompt as its title; a prompt gating a call names that call once announced.

        Args:
            session: The session.
            announced: The execution's register of announced calls.
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
        gated = ""
        if call_id is not None:
            announced.wait_for(str(call_id), abandoned=lambda: session.cancelled)
            if session.cancelled or str(call_id) not in announced:
                return None
            gated = updates.wire_call_id(session.session_id, announced.execution_id, str(call_id))

        def _done() -> bool:
            return session.cancelled or (until is not None and until())

        request = model.PermissionRequest(
            updates.printable(prompt),
            gated,
            tuple(
                model.PermissionOption(updates.printable(text), option_kind(text, standing))
                for text in options
            ),
        )
        self.server.update(session.acp_id, model.StateUpdate("requires_action"))
        chosen = self.server.request_permission(
            session.acp_id, request, timeout_s=acp_frontend.PERMISSION_TIMEOUT_S, until=_done
        )
        if not session.cancelled:  # a cancelled execution reports `idle` as it ends
            self.server.update(session.acp_id, model.StateUpdate("running"))
        return None if chosen is None else options[chosen]

    def _frontend(
        self, session: acp_session.Session, announced: Announced
    ) -> frontend.SessionFrontend:
        """Return the front-end for one turn."""
        return acp_frontend.acp_frontend(
            say=lambda line: self.server.update(session.acp_id, updates.notice(line)),
            ask=lambda prompt, options, standing, call_id, until=None: self.ask(
                session, announced, prompt, options, standing, call_id, until
            ),
            # Before `initialize` lands nothing is known about the client, so it can do nothing.
            capabilities=self.server.client_capabilities or frontend.FrontendCapabilities(),
            agent6_exe=spawn.agent6_exe,
            spawn_detached_resume=lambda cwd, sid, flags: spawn.spawn_detached_resume(
                cwd, sid, config_paths=self.config_paths, flags=flags
            ),
        )

    def _resumable(self, session: acp_session.Session) -> bool:
        """Follow undo continuations and return whether the run has a resume snapshot.

        Args:
            session: The thread whose next execution resumes the run.

        Returns:
            Whether the current run has a snapshot.

        Raises:
            OperatorError: Undo continuations cycle or have no resume point.
        """
        seen: set[str] = set()
        while session.session_id:
            if session.session_id in seen:
                raise errors.OperatorError("the undo continuations form a cycle")
            seen.add(session.session_id)
            layout = session.layout(agent6_paths.state_dir(session.cwd))
            scan = session_log.scan_session_log(layout.logs_path)
            if not (scan.finished and scan.stop_reason == "undone" and scan.undone_to):
                resuming = (layout.session_dir / "loop_state.json").is_file()
                if len(seen) > 1 and not resuming:
                    raise errors.OperatorError(
                        f"undo continuation {session.session_id} has no resume point"
                    )
                return resuming
            session.session_id = scan.undone_to
        return False

    def _recorded(self, session: acp_session.Session) -> bool:
        """Return whether the session's run id names a recorded run, which starts nothing again."""
        if not session.session_id:
            return False
        return session.layout(agent6_paths.state_dir(session.cwd)).manifest_path.exists()

    def _cancelled_unstarted(self, session: acp_session.Session) -> acp_session.StopReason:
        """Clear a queued cancel's requests so the next execution starts without them.

        Args:
            session: The cancelled session.

        Returns:
            The stop reason "cancelled".
        """
        session_dir = session.layout(agent6_paths.state_dir(session.cwd)).session_dir
        ipc.clear_stop_request(session_dir)
        ipc.clear_steer_answer(session_dir)
        ipc.clear_steer_request(session_dir)
        return "cancelled"

    def had_journal(self, session: acp_session.Session) -> bool:
        """Return whether the turn got far enough to write a journal."""
        if not session.session_id:
            return False
        return session.layout(agent6_paths.state_dir(session.cwd)).logs_path.exists()

    def run(self, session: acp_session.Session, text: str) -> acp_session.StopReason:
        """Run one prompt as one execution of the session's run, queued behind another session's.

        Args:
            session: The session the prompt came on.
            text: The prompt: a new run's task, or the resumed run's first steering
                instruction.

        Returns:
            ACP's stop reason.

        Raises:
            RpcError: The run failed, or agent6 refused the prompt.
        """
        started_at = time.time()  # this execution honors a cancel written from here on
        # The id is minted before the queue, so a cancel during the wait has a run to address.
        with self._telling_faults(session):  # an unreadable config raises here, before any journal
            resuming = self._resumable(session)
            if session.require_resume and not resuming:
                why = "this run has no resume point; Fix cannot start a new run"
                self.server.update(session.acp_id, updates.notice(why))
                raise rpc.RpcError(rpc.INVALID_PARAMS, why)
            if not resuming and self._recorded(session):
                # A recorded run with no resume point is refused both ways; a new run says so.
                self.server.update(
                    session.acp_id,
                    updates.notice(
                        f"run {session.session_id} left no resume point;"
                        " this prompt starts a new run"
                    ),
                )
                session.session_id = ""
            if not session.session_id:
                # The id `session/new` reserved, unless a run took it meanwhile.
                state_dir = agent6_paths.state_dir(session.cwd)
                bucket = kinds.session_bucket(acp_session.ACP_MODE)
                reserved = session.acp_id
                taken = sessions_layout.session_layout(state_dir, reserved) is not None
                session.session_id = id.unused_session_id(state_dir, bucket) if taken else reserved
            session.steerable = True
        if not self._runs.acquire(blocking=False):
            holder = self._running
            who = f"session {holder.acp_id}" if holder is not None else "another session"
            self.server.update(
                session.acp_id,
                updates.notice(
                    f"waiting: {who} is running a prompt; this prompt starts when it ends"
                ),
            )
            while not self._runs.acquire(timeout=QUEUE_POLL_S):
                if session.cancelled:
                    return self._cancelled_unstarted(session)
        try:
            self._running = session
            if session.cancelled:
                return self._cancelled_unstarted(session)
            with self._telling_faults(session):
                # Terminal verbs have no ACP meaning; only session/cancel stops the execution.
                if resuming and text.strip().lower() in ("abort", "exit", "detach", "/undo"):
                    text = f"Task: {text}"
                return self._run(session, text, resuming=resuming, started_at=started_at)
        finally:
            self._running = None
            self._runs.release()

    @contextlib.contextmanager
    def _telling_faults(self, session: acp_session.Session) -> Generator[None]:
        """Turn a fault in the block into the error that answers the prompt.

        The editor is told why as prose too, and a non-operator fault's traceback goes to stderr.

        Args:
            session: The session whose run the block starts.

        Yields:
            Nothing.

        Raises:
            RpcError: The block raised; an error of its own passes through.
        """
        try:
            yield
        except rpc.RpcError:
            raise
        except Exception as exc:
            what = "could not start" if not self.had_journal(session) else "failed"
            message = f"the run {what}: {exc}"
            self.server.update(session.acp_id, updates.notice(message))
            if not isinstance(exc, errors.OperatorError):
                _stderr(f"[agent6] run {session.session_id}: {traceback.format_exc()}")
            raise rpc.RpcError(rpc.INTERNAL_ERROR, message) from exc

    def _run(
        self, session: acp_session.Session, text: str, *, resuming: bool, started_at: float
    ) -> acp_session.StopReason:
        """Run the execution under the run lock, with a tail projecting its journal.

        Args:
            session: The session the prompt came on.
            text: The prompt: a new run's task, or the resumed run's first steering
                instruction.
            resuming: Resume the run from its snapshot, the tail skipping the journal the
                editor already rendered; False starts a new run.
            started_at: When the prompt arrived; a cancel written since is this execution's to
                honor.

        Returns:
            ACP's stop reason.

        Raises:
            RpcError: The run failed, or agent6 refused the prompt.
        """
        layout = session.layout(agent6_paths.state_dir(session.cwd))
        text = self._record_launch(session, layout.session_dir, text, resuming=resuming)
        journal_before = viewmodel_tail.journal_size(layout.logs_path)

        said: list[str] = []
        announced = Announced()
        ended, drained = threading.Event(), threading.Event()

        def _stop() -> bool:
            """Return whether the tail stops: one pass after the run, so its last lines land."""
            if not ended.is_set():
                return False
            if drained.is_set():
                return True
            drained.set()
            return False

        order = stream.ProseOrder(self.server, session.acp_id, layout.logs_path)
        start_at = journal_before if resuming else None
        banked = (
            session_log.scan_session_log(layout.logs_path).budget.session.usd if resuming else 0.0
        )
        info = updates.RunInfo(banked_usd=banked)
        tail = threading.Thread(
            target=stream.project,
            args=(self.server, session, layout.logs_path, _stop, start_at, announced, order),
            kwargs={"info": info, "prompt": text, "objects_dir": layout.objects_dir},
            name=f"acp-tail-{session.acp_id}",
            daemon=True,
        )
        tail.start()
        reporter = forwarding_reporter(self.server, session.acp_id, said, order=order)
        try:
            if resuming:
                # The prompt is the resumed run's first steering instruction.
                code = resume.resume_task(
                    self.config_paths,
                    session.session_id,
                    frontend=self._frontend(session, announced),
                    cwd=session.cwd,
                    force=False,
                    started_at=started_at,
                    steer=text,
                    model=session.model,
                    effort=session.effort,
                    reporter=reporter,
                )
            else:
                effective = session_setup.load_session_config(
                    session.cwd,
                    self.config_paths,
                    mode=acp_session.ACP_MODE,
                    model=session.model or None,
                    effort=session.effort,
                )
                try:
                    opening = directive.parse_directive(text)
                except directive.DirectiveError as exc:
                    reporter.error(str(exc))
                    raise rpc.RpcError(rpc.INVALID_PARAMS, str(exc)) from exc
                code = app_run.run_task(
                    effective.config,
                    text,
                    opening=tuple(opening or ()),
                    frontend=self._frontend(session, announced),
                    cwd=session.cwd,
                    started_at=started_at,
                    session_id=session.session_id,
                    model=session.model or None,
                    config_paths=self.config_paths,
                    explicit_leaves=effective.explicit_leaves,
                    effective=effective,
                    reporter=reporter,
                )
        finally:
            ended.set()
            tail.join(timeout=DRAIN_S)
            order.flush(None)  # a tail that outlived the drain still owes these
        current = self.config_options(session)
        if current != session.config_options:
            # The execution recorded its route or cached a provider's listing: agent6's change.
            session.config_options = current
            self.server.update(session.acp_id, model.ConfigOptionUpdate(current))
        if code != 0 and not said and not self.had_journal(session):
            self.server.update(session.acp_id, updates.notice(f"the run stopped (exit {code})"))
        # Only a journal that grew and ended in this execution carries its end reason.
        grown = viewmodel_tail.journal_size(layout.logs_path) > journal_before
        if grown and session.turn is not None:
            session.turn.inserted()  # a tail that never drained read no start row
        scan = session_log.scan_session_log(layout.logs_path)
        reason = scan.stop_reason if grown and scan.finished else ""
        return stop_reason(code, stop_reason=reason)

    def _record_launch(
        self, session: acp_session.Session, session_dir: pathlib.Path, text: str, *, resuming: bool
    ) -> str:
        """Record the prompt in the run's own directory before the execution starts.

        The execution finds the record by its words and journals the prompt's images and
        its message id with them.

        Args:
            session: The session the prompt came on.
            session_dir: The run's directory, reserved here for a new run.
            text: The prompt.
            resuming: The prompt is a follow-up rather than the task.

        Returns:
            The words the execution runs: the prompt, then a label per image.
        """
        message_id = session.turn.message_id if session.turn is not None else ""
        if not session.images and not message_id:
            return text
        message = operator_images.accept(session_dir, text, session.images)
        named = message.model_copy(update={"message_id": message_id})
        ipc.write_launch_input(session_dir, named, follow_up=resuming)
        return message.text


def serve_acp(
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    *,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> int:
    """Speak ACP on the process's stdio until the editor closes it.

    Args:
        stdin: The client's requests; None is the process's stdin.
        stdout: The replies; None is the process's stdout.
        config_paths: The `--config FILE` layers in command-line order.

    Returns:
        The exit code, 0.
    """
    server = acp_server.ACPServer(
        stdin=stdin if stdin is not None else sys.stdin.buffer,
        stdout=stdout if stdout is not None else sys.stdout.buffer,
    )
    server.sessions = RunBridge(server=server, config_paths=config_paths).sessions()
    server.serve()
    return 0
