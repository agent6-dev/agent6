from __future__ import annotations

import io
import json
import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import paths
from agent6.app import resume, session_setup
from agent6.harness import loop_conversation, loop_state, operator_bridge
from agent6.ui.acp import runner, server, session
from tests.acp import test_runner, test_session_lifecycle


@pytest.mark.parametrize("followup", [False, True])
def test_composer_refusal_has_no_cli_remedy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, followup: bool
) -> None:
    """An editor gets the refusal, not a CLI flag or advice to use a nonexistent composer."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_setup, "load_session_config", test_runner._loaded)
    current = session.Session(acp_id="run-AAAAAA", cwd=tmp_path)
    if followup:
        current.session_id = current.acp_id
        saved = current.layout(paths.state_dir(tmp_path))
        saved.ensure()
        (saved.session_dir / "loop_state.json").write_text("{}")
    wire = io.BytesIO()
    bridge = runner.RunBridge(server=server.ACPServer(io.BytesIO(), wire))
    assert bridge.run(current, "/compact keep the plan") == "refusal"
    texts = [
        json.loads(line)["params"]["update"]["content"]["text"]
        for line in wire.getvalue().splitlines()
    ]
    assert texts == ["[agent6] ERROR: /compact is a composer command, not an instruction\n"]


@pytest.mark.parametrize("text", ["Abort", "EXIT", "detach", "/UNDO", "  abort  "])
def test_followup_verbs_reach_the_model_as_task_text(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """ACP prompts cannot abort, exit, detach or undo the execution through steer parsing."""
    monkeypatch.chdir(tmp_path)
    current = session.Session(acp_id="run-AAAAAA", cwd=tmp_path, session_id="run-AAAAAA")
    saved = current.layout(paths.state_dir(tmp_path))
    saved.ensure()
    (saved.session_dir / "loop_state.json").write_text("{}")
    conversation = loop_conversation.Conversation()
    events: list[str] = []

    def emit(event: str, **_kwargs: object) -> None:
        events.append(event)

    def record(*_args: object) -> None:
        pass

    def drive(*_args: object, **kwargs: Any) -> int:
        steering = operator_bridge.Steering(
            bridge=operator_bridge.OperatorBridge(
                steer_requested=lambda: True, steer_prompt=lambda: kwargs["steer"]
            ),
            parallel=mock.Mock(),
            dispatcher=mock.Mock(),
            record_decision=record,
            log=lambda _s: None,
            emit=emit,
        )
        assert (
            steering.handle(
                conversation, 1, loop_state.LoopState(original_task="task", tool_calls=0)
            )
            is None
        )
        return 0

    monkeypatch.setattr(resume, "resume_task", drive)
    bridge = runner.RunBridge(server=server.ACPServer(io.BytesIO(), io.BytesIO()))
    assert bridge.run(current, text) == "end_turn"
    assert events == ["loop.steer.requested", "loop.steer.injected"]
    assert text.strip() in json.dumps(conversation.to_wire())


def test_prompt_follows_an_undo_continuation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undo from another surface rebinds the thread before its next resume."""
    monkeypatch.chdir(tmp_path)
    cwd = test_session_lifecycle._repo(tmp_path)
    parent = test_session_lifecycle._record(cwd, "run-AAAAAA", "task", mtime=1)
    child = test_session_lifecycle._record(cwd, "run-BBBBBB", "task", mtime=2)
    with (parent / "logs.jsonl").open("a") as log:
        log.write(json.dumps({"type": "session.undone", "new_session_id": child.name}) + "\n")
        log.write(json.dumps({"type": "session.end", "reason": "undone"}) + "\n")
    current = session.Session(acp_id=parent.name, cwd=cwd, session_id=parent.name)
    resumed: list[str] = []

    def drive(_config: object, session_id: str, **_kwargs: Any) -> int:
        resumed.append(session_id)
        return 0

    monkeypatch.setattr(resume, "resume_task", drive)
    bridge = runner.RunBridge(server=server.ACPServer(io.BytesIO(), io.BytesIO()))
    assert bridge.run(current, "continue with the edited message") == "end_turn"
    assert resumed == [child.name]
    assert current.session_id == child.name
    assert current.acp_id == parent.name
