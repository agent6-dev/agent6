from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6.config import Config
from agent6.harness import guards, loop, loop_conversation, loop_state, run_chain
from agent6.tools import dispatch, results
from tests.unit.turn_context import turn_context


def _dispatcher(tmp_path: pathlib.Path, checkin: float) -> dispatch.ToolDispatcher:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "allow"},
            "harness": {"command_checkin_s": checkin},
        }
    )
    return dispatch.ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
    )


@pytest.mark.parametrize(
    ("action", "neutral"),
    [
        ("completed", False),
        ("failed", False),
        ("roster", False),
        ("stopped", False),
        ("exited_during_read", False),
        ("live_poll", True),
        ("handback", True),
        ("detached", True),
    ],
)
def test_settled_accounting_follows_the_actual_command_result(
    tmp_path: pathlib.Path, action: str, neutral: bool
) -> None:
    """A real live sibling does not shelter unrelated commands or ended reads from the guard."""
    d = _dispatcher(tmp_path, checkin=0.1 if action == "handback" else 30.0)
    wf = loop.Harness(
        chain=run_chain.RunChain(tmp_path / "repo"),
        config=Config(),
        provider=mock.MagicMock(),
        dispatcher=d,
        logger=lambda _line: None,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_pass()
    state.settled = guards.SettledGuard(tree="tree", idle=1)
    turn = loop_state.TurnState(
        number=3, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    try:
        sibling = d.dispatch("run_command", {"argv": ["/bin/sleep", "60"], "background": True})
        assert isinstance(sibling, results.ExecResult) and sibling.background_id
        name = "run_command"
        raw: dict[str, Any] = {"argv": ["/bin/true"]}
        if action == "failed":
            raw = {"argv": ["/bin/false"]}
        elif action == "roster":
            name, raw = "read_background", {}
        elif action == "live_poll":
            name, raw = "read_background", {"id": sibling.background_id, "wait_s": 0}
        elif action in ("handback", "detached"):
            raw = {"argv": ["/bin/sleep", "60"], "background": action == "detached"}
        elif action in ("stopped", "exited_during_read"):
            selected = d.dispatch("run_command", {"argv": ["/bin/sleep", "1"], "background": True})
            assert isinstance(selected, results.ExecResult)
            if action == "stopped":
                d.dispatch("stop_background", {"id": selected.background_id})
            name, raw = "read_background", {"id": selected.background_id, "wait_s": 5}
        outcome = d.dispatch(name, raw)
        wf._note_tool_effects(state, turn, name, outcome, raw)  # pyright: ignore[reportPrivateUsage]
        guards.verify_settled(turn, state, turn_context(tree_sha=lambda: "tree"))
        assert state.settled.idle == (1 if neutral else 2)
    finally:
        d.close()
    roster = d.dispatch("read_background", {}).to_wire()["shells"]
    assert roster and all("running" not in line for line in roster)
