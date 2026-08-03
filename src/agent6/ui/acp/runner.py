# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One ACP prompt becomes one agent6 run.

Three things here are load-bearing.

The protocol OWNS stdout, so the run's reporter writes to stderr. One status
line on stdout desynchronises the stream irrecoverably, and no editor recovers
from it.

The run id is minted HERE, before the run starts, so `session/cancel` has
something to address. Letting the lifecycle mint its own left the session with
no handle: the cancel reported success while the run continued to completion,
spending budget and making commits.

`run_task` reads the process cwd, so a run in a session's directory has to
chdir there -- which is process-global. Runs are therefore serialised on the
connection: a second prompt waits rather than running in the wrong repository.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from agent6.app.reporter import Reporter
from agent6.app.run import FrontendCapabilities, run_task
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.runs.id import new_friendly_id
from agent6.runs.layout import RunLayout
from agent6.ui.acp.frontend import acp_frontend
from agent6.ui.acp.server import ACPServer
from agent6.ui.acp.session import Session, Sessions, StopReason
from agent6.ui.acp.updates import message_update, updates_for
from agent6.ui.spawn import agent6_exe, spawn_detached_resume
from agent6.viewmodel.tail import tail_events
from agent6.viewmodel.transcript import TranscriptFold

# How long a permission request waits for the editor. An operator who has
# walked away must not hold a run forever, and the seam already reads silence
# as the cautious answer: an approval becomes a denial, a question becomes no
# answer at all.
PERMISSION_TIMEOUT_S = 300.0
# A safety net on joining the streaming tail, not the normal path: `_stop`
# ends it one read pass after the run returns. This bounds a tail wedged on a
# filesystem that is not answering.
DRAIN_S = 5.0


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


# Everything the lifecycle would print goes to stderr, where an editor shows it
# as agent log output. stdout is the wire.
STDERR_REPORTER = Reporter(out=_stderr, err=_stderr)


def option_kind(text: str) -> str:
    """ACP's button kinds, from the option text the seam offered.

    "allow" is the one an editor may REMEMBER. "allow once" is the fetch tool's
    off-list host, where remembering would silently cover a different host.
    Anything else is one answer among several (a `UserQuestion`), which is not
    a permission at all -- `allow_once` is the only kind that does not invite
    the editor to reuse it.
    """
    if text == "deny":
        return "reject_once"
    if text == "allow":
        return "allow_always"
    return "allow_once"


def stop_reason(code: int) -> StopReason:
    """ACP's vocabulary, from the lifecycle's exit code.

    ACP has no "the run failed": a red gate, a provider error and a budget stop
    are all `refusal`. The DETAIL is already on the wire -- `updates_for` sends
    how the run ended as a message -- so the stop reason only has to be one the
    editor will not drop the turn over.
    """
    if code == 130:
        return "cancelled"
    return "end_turn" if code == 0 else "refusal"


@dataclass
class RunBridge:
    """Runs prompts for one ACP connection."""

    server: ACPServer
    # One at a time: the chdir in `_run` is process-global, and a run in the
    # wrong directory commits to the wrong repository.
    _runs: threading.Lock = field(default_factory=threading.Lock)
    _asks: threading.Lock = field(default_factory=threading.Lock)
    _asked: int = 0

    def sessions(self) -> Sessions:
        return Sessions(run=self.run, state_dir_for=resolved_state_dir)

    def ask(self, session: Session, prompt: str, options: tuple[str, ...]) -> str | None:
        """Put one approval or question to the editor.

        ACP v1 has no method for a free-form question, so a `UserQuestion` goes
        out as a permission request too -- its options ARE the answers. The
        editor renders buttons either way, which is what the seam needs.
        """
        with self._asks:
            self._asked += 1
            tool_call_id = f"ask-{session.id}-{self._asked}"
        answer = self.server.request(
            "session/request_permission",
            {
                "sessionId": session.id,
                "toolCall": {
                    "toolCallId": tool_call_id,
                    "title": prompt,
                    "kind": "other",
                    "status": "pending",
                },
                "options": [
                    {"optionId": text, "name": text, "kind": option_kind(text)} for text in options
                ],
            },
            timeout_s=PERMISSION_TIMEOUT_S,
        )
        outcome = answer.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
            return None  # cancelled, timed out, or an answer we cannot read
        chosen = outcome.get("optionId")
        # Only an option we offered. An editor that echoes something else is
        # not choosing, and an unknown string could become an "allow" by prefix.
        return chosen if isinstance(chosen, str) and chosen in options else None

    def run(self, session: Session, text: str) -> StopReason:
        with self._runs:
            try:
                return self._run(session, text)
            except Exception as exc:
                # A run that dies before it has a journal has no other way to
                # say so, and the turn still ends with a stop reason. A broken
                # config is the ordinary case: the CLI prints it, and here the
                # editor would have seen a turn end with no words at all.
                self.server.notify_raw(
                    message_update(session.id, f"the run could not start: {exc}")
                )
                return "refusal"

    def _run(self, session: Session, text: str) -> StopReason:
        effective = load_effective(session.cwd)
        session.run_id = new_friendly_id()
        layout = RunLayout(state_dir=resolved_state_dir(session.cwd), run_id=session.run_id)
        os.chdir(session.cwd)

        ended, drained = threading.Event(), threading.Event()

        def _stop() -> bool:
            """Stop the tail one read pass after the run returns.

            `tail_events` checks this at the TOP of each poll, so answering
            False once lets the journal's last lines still reach the editor.
            Stopping immediately dropped them; waiting for the run's own
            `run.end` taxed every turn that ends without one (a config error,
            an early refusal) with the full drain timeout.
            """
            if not ended.is_set():
                return False
            if drained.is_set():
                return True
            drained.set()
            return False

        tail = threading.Thread(
            target=self._stream,
            args=(session, layout.logs_path, _stop),
            name=f"acp-tail-{session.id}",
            daemon=True,
        )
        tail.start()
        try:
            code = run_task(
                effective.config,
                text,
                frontend=acp_frontend(
                    ask=lambda prompt, options: self.ask(session, prompt, options),
                    # `initialize` has not landed if this is None, and nothing
                    # is known about the client; the cautious answer is that it
                    # can do nothing.
                    capabilities=self.server.client_capabilities or FrontendCapabilities(),
                    agent6_exe=agent6_exe,
                    spawn_detached_resume=spawn_detached_resume,
                ),
                run_id=session.run_id,
                explicit_leaves=frozenset(effective.sources),
                reporter=STDERR_REPORTER,
            )
        finally:
            ended.set()
            tail.join(timeout=DRAIN_S)
        return stop_reason(code)

    def _stream(self, session: Session, logs_path: Path, stop: Callable[[], bool]) -> None:
        """Project the run's journal into `session/update` as it is written."""
        fold = TranscriptFold()
        for event in tail_events(logs_path, stop_when_finished=True, should_stop=stop):
            for item in fold.feed(event):
                for body in updates_for(item, session_id=session.id):
                    self.server.notify_raw(body)


def serve_acp(stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Speak ACP on this process's stdio until the editor closes it."""
    server = ACPServer(
        stdin=stdin if stdin is not None else sys.stdin.buffer,
        stdout=stdout if stdout is not None else sys.stdout.buffer,
    )
    server.sessions = RunBridge(server=server).sessions()
    server.serve()
    return 0
