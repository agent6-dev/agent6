# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Turns: new, prompt, cancel.

The load-bearing property: a prompt runs on a worker, so the read loop is free
to receive the cancel that ACP requires to work DURING a turn.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent6.ui.acp.server import ACPServer
from agent6.ui.acp.session import Session, Sessions, prompt_text


def _ends(_session: Session, _text: str) -> str:
    return "end_turn"


def _sessions(run: Any) -> Sessions:
    return Sessions(run=run, state_dir_for=lambda cwd: cwd / ".state")


def _drive(payload: bytes, sessions: Sessions) -> list[dict[str, Any]]:
    out = io.BytesIO()
    ACPServer(stdin=io.BytesIO(payload), stdout=out, sessions=sessions).serve()
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def _msg(req_id: int, method: str, **params: Any) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}).encode()


def test_a_new_session_needs_an_absolute_cwd() -> None:
    """The spec makes every path absolute; a relative one would resolve
    against whatever directory the editor happened to launch us in."""
    sessions = _sessions(_ends)
    (reply,) = _drive(_msg(1, "session/new", cwd="relative/path") + b"\n", sessions)
    assert "absolute" in reply["error"]["message"]


def test_a_prompt_runs_and_answers_with_its_stop_reason() -> None:
    ran: list[str] = []

    def _run(_session: Session, text: str) -> str:
        ran.append(text)
        return "end_turn"

    sessions = _sessions(_run)
    session = Session(id="s1", cwd=Path("/repo"))
    sessions._by_id["s1"] = session  # pyright: ignore[reportPrivateUsage]
    payload = _msg(2, "session/prompt", sessionId="s1", prompt=[{"type": "text", "text": "fix it"}])
    replies = _drive(payload + b"\n", sessions)
    if session.thread is not None:
        session.thread.join(timeout=5)
    assert ran == ["fix it"]
    assert replies == [] or replies[0]["result"]["stopReason"] == "end_turn"


def test_the_read_loop_stays_free_while_a_turn_runs() -> None:
    """THE property. Answering a prompt inline would block reading for the
    whole run, so the cancel an editor sends would arrive only after the thing
    it meant to stop had already finished."""
    started = threading.Event()
    release = threading.Event()
    cancelled_during: list[bool] = []

    def _run(session: Session, _text: str) -> str:
        started.set()
        release.wait(timeout=5)
        cancelled_during.append(session.cancelled)
        return "end_turn"

    sessions = _sessions(_run)
    session = Session(id="s1", cwd=Path("/repo"))
    sessions._by_id["s1"] = session  # pyright: ignore[reportPrivateUsage]

    out = io.BytesIO()
    reader, writer = _pipe()
    server = ACPServer(stdin=reader, stdout=out, sessions=sessions)
    loop = threading.Thread(target=server.serve, daemon=True)
    loop.start()
    try:
        writer.write(
            _msg(2, "session/prompt", sessionId="s1", prompt=[{"type": "text", "text": "go"}])
            + b"\n"
        )
        writer.flush()
        assert started.wait(timeout=5), "the turn never started"
        # The turn is still running. If the loop were blocked, this never lands.
        writer.write(_msg(3, "session/cancel", sessionId="s1") + b"\n")
        writer.flush()
        for _ in range(100):
            if session.cancelled:
                break
            time.sleep(0.02)
        assert session.cancelled, "the cancel could not reach a running turn"
    finally:
        release.set()
        writer.close()
        loop.join(timeout=5)
    assert cancelled_during == [True]


def test_a_turn_cancelled_while_it_runs_reports_itself_as_cancelled() -> None:
    """The operator's own act, not a failure."""
    started = threading.Event()
    release = threading.Event()

    def _slow(_session: Session, _text: str) -> str:
        started.set()
        release.wait(timeout=5)
        return "end_turn"

    sessions = _sessions(_slow)
    session = Session(id="s1", cwd=Path("/repo"))
    seen: list[str] = []
    sessions.start_turn(session, "go", finish=seen.append)
    assert started.wait(timeout=5)
    sessions.cancel(session)
    release.set()
    if session.thread is not None:
        session.thread.join(timeout=5)
    assert seen == ["cancelled"]


def test_a_stale_cancel_does_not_kill_the_next_turn() -> None:
    """The flag belongs to the turn it cancelled. Carrying it forward would
    make the following prompt end before it began."""
    sessions = _sessions(_ends)
    session = Session(id="s1", cwd=Path("/repo"), cancelled=True)
    seen: list[str] = []
    sessions.start_turn(session, "go", finish=seen.append)
    if session.thread is not None:
        session.thread.join(timeout=5)
    assert seen == ["end_turn"]


def test_a_run_that_dies_still_ends_the_turn() -> None:
    """An editor waiting forever on a prompt is worse than a refusal."""

    def _boom(_session: Session, _text: str) -> str:
        raise RuntimeError("the provider fell over")

    sessions = _sessions(_boom)
    session = Session(id="s1", cwd=Path("/repo"))
    seen: list[str] = []
    sessions.start_turn(session, "go", finish=seen.append)
    if session.thread is not None:
        session.thread.join(timeout=5)
    assert seen == ["refusal"]


def test_a_second_turn_on_a_busy_session_is_refused() -> None:
    """Two runs in one workspace is what the repo lock exists to prevent."""
    release = threading.Event()

    def _slow(_session: Session, _text: str) -> str:
        release.wait(timeout=5)
        return "end_turn"

    sessions = _sessions(_slow)
    session = Session(id="s1", cwd=Path("/repo"))
    try:
        sessions.start_turn(session, "one", finish=lambda _r: None)
        with pytest.raises(Exception, match="already has a turn"):
            sessions.start_turn(session, "two", finish=lambda _r: None)
    finally:
        release.set()
        if session.thread is not None:
            session.thread.join(timeout=5)


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "a\n\nb"),
        ([{"type": "image", "data": "x"}, {"type": "text", "text": "a"}], "a"),
    ],
)
def test_only_text_blocks_become_the_task(blocks: list[Any], expected: str) -> None:
    """A placeholder for an image is something the model would try to read."""
    assert prompt_text({"prompt": blocks}) == expected


@pytest.mark.parametrize("blocks", [[], [{"type": "image", "data": "x"}], "not a list"])
def test_a_prompt_with_no_text_is_refused(blocks: Any) -> None:
    with pytest.raises(Exception, match="prompt"):
        prompt_text({"prompt": blocks})


def _pipe() -> tuple[Any, Any]:
    import os

    read_fd, write_fd = os.pipe()
    return os.fdopen(read_fd, "rb", buffering=0), os.fdopen(write_fd, "wb", buffering=0)


def test_the_next_prompt_is_not_refused_by_the_reply_it_just_read() -> None:
    """`thread.is_alive()` is still true while `finish` runs -- and `finish` IS
    the reply -- so a conforming editor that writes its next prompt the instant
    it reads the answer was refused at random."""
    sessions = _sessions(_ends)
    session = Session(id="s1", cwd=Path("/repo"))
    seen: list[bool] = []

    def _finish(_reason: str) -> None:
        seen.append(session.is_running())

    sessions.start_turn(session, "one", finish=_finish)
    if session.thread is not None:
        session.thread.join(timeout=5)
    assert seen == [False], "the turn was still 'in flight' as it answered"


def test_eof_lets_a_live_turn_reach_a_boundary() -> None:
    """Closing the editor is the ordinary way EOF arrives. A daemon worker torn
    down mid-git holds the repo and worker locks and the run-dir pid."""
    started = threading.Event()
    finished = threading.Event()

    def _slow(_session: Session, _text: str) -> str:
        started.set()
        time.sleep(0.2)
        finished.set()
        return "end_turn"

    sessions = _sessions(_slow)
    session = Session(id="s1", cwd=Path("/repo"))
    sessions._by_id["s1"] = session  # pyright: ignore[reportPrivateUsage]

    out = io.BytesIO()
    reader, writer = _pipe()
    server = ACPServer(stdin=reader, stdout=out, sessions=sessions)
    loop = threading.Thread(target=server.serve, daemon=True)
    loop.start()
    writer.write(
        _msg(2, "session/prompt", sessionId="s1", prompt=[{"type": "text", "text": "go"}]) + b"\n"
    )
    writer.flush()
    assert started.wait(timeout=5)
    writer.close()  # the editor goes away mid-turn
    loop.join(timeout=10)
    assert finished.is_set(), "the turn was cut off instead of reaching a boundary"


def test_a_prompt_sent_as_a_notification_is_refused() -> None:
    """A turn's whole point is the stopReason it answers with; replying with a
    null id is not valid JSON-RPC."""
    sessions = _sessions(_ends)
    sessions._by_id["s1"] = Session(id="s1", cwd=Path("/repo"))  # pyright: ignore[reportPrivateUsage]
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "session/prompt",
            "params": {"sessionId": "s1", "prompt": [{"type": "text", "text": "go"}]},
        }
    ).encode()
    assert _drive(payload + b"\n", sessions) == []


def test_a_cancel_for_an_unknown_session_says_so() -> None:
    """It is a notification, so an error reply is dropped with zero bytes
    written -- the stop button does nothing and reports nothing."""
    sessions = _sessions(_ends)
    replies = _drive(_msg(0, "session/cancel", sessionId="typo") + b"\n", sessions)
    assert replies, "the editor was told nothing at all"
    assert "cancel" in replies[0]["params"]["update"]["content"]["text"]
