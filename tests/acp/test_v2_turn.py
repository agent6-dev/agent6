# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The prompt turn as a v2 editor hears it: an answer at insertion, then the session's state."""

from __future__ import annotations

import io
import json
import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any, override

import pytest

from agent6.app import run, stop
from agent6.app.session import setup as session_setup
from agent6.ui.acp import model, runner
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as acp_session
from tests.acp import schema_check, test_runner, test_session_lifecycle

Frame = dict[str, Any]


class _Wire(test_runner._Wire):
    """A live v2 connection."""

    @override
    def new_session(self, cwd: pathlib.Path) -> str:
        info = {"name": "editor", "version": "0"}
        self.send(id=1, method="initialize", params={"protocolVersion": 2, "info": info})
        assert self.recv()["result"]["protocolVersion"] == 2
        self.send(id=2, method="session/new", params={"cwd": str(cwd)})
        return str(self.recv()["result"]["sessionId"])

    def collect(self, done: Callable[[Frame], bool]) -> list[Frame]:
        """Read frames until one satisfies `done`, each update checked against the schema."""
        got: list[Frame] = []
        while not got or not done(got[-1]):
            got.append(self.recv())
            if got[-1].get("method") == "session/update":
                found = schema_check.v2_errors("UpdateSessionNotification", got[-1]["params"])
                assert not found, f"{found}\n{got[-1]}"
        return got


def _states(frames: list[Frame], session_id: str) -> list[tuple[str, str | None]]:
    """Return the session's state updates as (state, stop reason)."""
    return [
        (update["state"], update.get("stopReason"))
        for frame in frames
        if frame.get("method") == "session/update"
        and frame["params"]["sessionId"] == session_id
        and (update := frame["params"]["update"])["sessionUpdate"] == "state_update"
    ]


def _idle(session_id: str) -> Callable[[Frame], bool]:
    def done(frame: Frame) -> bool:
        return _states([frame], session_id)[-1:] not in ([], [("running", None)])

    return done


@pytest.fixture(autouse=True)
def _stubbed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(stop, "stop_session", test_runner._ignore)
    monkeypatch.setattr(session_setup, "load_session_config", test_runner._loaded)
    monkeypatch.chdir(tmp_path)


def test_a_prompt_that_cannot_start_answers_with_its_error_and_reports_idle(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure before the message is in answers the request; the state still closes the work."""

    def _refusing(*_a: object, **_k: object) -> int:
        raise RuntimeError("no provider")

    monkeypatch.setattr(run, "run_task", _refusing)
    wire = _Wire()
    try:
        session = wire.new_session(test_runner._repo(tmp_path))
        wire.prompt(session, "do the thing")
        frames = wire.collect(lambda f: f.get("id") == 3)
        assert frames[-1]["error"]["code"] == -32603
        assert "no provider" in frames[-1]["error"]["message"]
        assert not [f for f in frames if "user_message" in json.dumps(f)], "nothing was inserted"
        assert _states(frames, session) == [("running", None), ("idle", "error")]
        failure = next(
            f["params"]["update"]
            for f in frames
            if _states([f], session)[-1:] == [("idle", "error")]
        )
        assert failure["error"]["message"] == frames[-1]["error"]["message"]
    finally:
        wire.close()


def test_a_prompt_cancelled_before_its_message_is_added_answers_request_cancelled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt held behind another session's run is dropped by its cancel with -32800."""
    started, release = threading.Event(), threading.Event()

    def _blocking(*_a: object, **_k: object) -> int:
        started.set()
        release.wait(timeout=10.0)
        return 0

    monkeypatch.setattr(run, "run_task", _blocking)
    wire = _Wire()
    try:
        first = wire.new_session(test_runner._repo(tmp_path))
        wire.send(id=9, method="session/new", params={"cwd": str(tmp_path)})
        second = str(wire.recv()["result"]["sessionId"])
        wire.prompt(first, "the long one", req_id=3)
        assert started.wait(timeout=5.0)
        wire.prompt(second, "the held one", req_id=4)
        wire.send(method="session/cancel", params={"sessionId": second})
        frames = wire.collect(lambda f: f.get("id") == 4)
        assert frames[-1]["error"]["code"] == -32800
        assert not schema_check.v2_errors("Error", frames[-1]["error"])
        assert _states(frames, second) == [("running", None), ("idle", "cancelled")]
    finally:
        release.set()
        wire.close()


def test_a_close_reports_the_cancelled_work_before_it_answers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 orders a close: the work's last state, then the close's reply, then silence."""
    started, release = threading.Event(), threading.Event()

    def _until_cancelled(*_a: object, **_k: object) -> int:
        started.set()
        release.wait(timeout=10.0)
        return 130

    monkeypatch.setattr(run, "run_task", _until_cancelled)
    wire = _Wire()
    try:
        session = wire.new_session(test_runner._repo(tmp_path))
        wire.prompt(session, "do the thing")
        assert started.wait(timeout=5.0)
        wire.send(id=5, method="session/close", params={"sessionId": session})
        time.sleep(0.3)  # the close holds its reply while the run is still stopping
        release.set()
        frames = wire.collect(lambda f: f.get("id") == 5)
        assert frames[-1]["result"] == {}
        assert _states(frames, session)[-1] == ("idle", "cancelled")
        assert [f.get("id") for f in frames if "id" in f] == [3, 5]
        wire.send(id=6, method="session/close", params={"sessionId": session})
        assert wire.collect(lambda f: f.get("id") == 6)[-1]["result"] == {}
    finally:
        release.set()
        wire.close()


def test_a_command_answered_at_once_is_a_turn_of_its_own(tmp_path: pathlib.Path) -> None:
    """Its message is in, the session runs while it is answered, and it ends `end_turn`."""
    cwd = test_session_lifecycle._repo(tmp_path)
    saved = test_session_lifecycle._record(cwd, "quiet-run-AAAAAA", "task", mtime=time.time())
    wire = _Wire()
    try:
        info = {"name": "editor", "version": "0"}
        wire.send(id=1, method="initialize", params={"protocolVersion": 2, "info": info})
        wire.recv()
        resume = {"sessionId": saved.name, "cwd": str(cwd)}
        wire.send(id=2, method="session/resume", params=resume)
        wire.collect(lambda f: f.get("id") == 2)
        wire.prompt(saved.name, "/findings")
        frames = wire.collect(_idle(saved.name))
        reply = next(f for f in frames if f.get("id") == 3)
        assert not schema_check.v2_errors("PromptResponse", reply["result"])
        bodies = [f["params"]["update"] for f in frames if f.get("method") == "session/update"]
        assert [body["sessionUpdate"] for body in bodies] == [
            "user_message",
            "state_update",
            "agent_message",
            "state_update",
        ]
        assert bodies[0]["messageId"] == reply["result"]["messageId"]
        assert bodies[0]["content"] == [{"type": "text", "text": "/findings"}]
        assert bodies[2]["content"][0]["text"] == "[agent6] No recorded findings.\n"
        assert _states(frames, saved.name) == [("running", None), ("idle", "end_turn")]
    finally:
        wire.close()


def test_a_command_answered_during_a_run_leaves_the_runs_state_alone() -> None:
    """An `idle` for the command would end work that is still running."""
    sent: list[model.Update] = []
    ended: list[acp_session.TurnEnd] = []
    current = acp_session.Session(acp_id="s", cwd=pathlib.Path("/x"), execution_live=True)
    from agent6.ui.acp import review, updates

    turn = acp_session.Turn(ended.append, notify=sent.append, message_id="m1")
    review._answered(current, turn, "/answer q1 yes", [updates.notice("answered q1: yes")])
    assert not [update for update in sent if isinstance(update, model.StateUpdate)]
    assert ended == ["end_turn"]
    assert sent[0] == model.Message("user", "m1", "/answer q1 yes", prompt=True)


def test_a_permission_wait_reports_requires_action_then_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.BytesIO()
    server = acp_server.ACPServer(stdin=io.BytesIO(), stdout=out, protocol_version=2)
    bridge = runner.RunBridge(server=server)

    def _answer(_method: str, _params: Frame, **_kw: object) -> Frame:
        server.notify_raw({"asked": True})
        return {"outcome": {"outcome": "selected", "optionId": "0"}}

    monkeypatch.setattr(server, "request", _answer)
    current = acp_session.Session(acp_id="s", cwd=pathlib.Path("/x"), session_id="run-x")
    chosen = bridge.ask(current, runner.Announced(), "Go on?", ("yes", "no"), None, None)
    assert chosen == "yes"
    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [f.get("params", {}).get("update", f) for f in frames] == [
        {"sessionUpdate": "state_update", "state": "requires_action"},
        {"asked": True},
        {"sessionUpdate": "state_update", "state": "running"},
    ]
