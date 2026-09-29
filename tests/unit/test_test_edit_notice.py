# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The test-edits-only flip notice, decided by a git tree diff between the red and the green.

A gate that goes red, then green while only test files changed, gets a notice naming them
and an event that counts it; a run_command edit is seen like an apply_edit. In the
SWE-rebench autopsy, 7 of 8 broke-P2P executions greened a red gate by editing the test.
"""

from __future__ import annotations

import pathlib
import subprocess as sp
from typing import Any
from unittest import mock

from agent6.config import Config
from agent6.harness import _chain, _conversation, _loop_state, _nudges, loop
from agent6.tools import results

NOTICE_HEAD = "[harness verify] The gate was red at the last verify and green at this one;"
EVENT = "loop.test_only_green.notice"


def test_is_test_path_conventions() -> None:
    assert _nudges.is_test_path("tests/test_a.py")
    assert _nudges.is_test_path("pkg/a_test.py")
    assert _nudges.is_test_path("pkg/tests/conftest.py")
    assert not _nudges.is_test_path("pkg/testing_utils.py")
    assert not _nudges.is_test_path("src/app.py")


def test_the_notice_states_the_world_and_caps_its_list() -> None:
    """World state only, the `[harness verify]` prefix, no advice; twelve paths then a count."""
    text = _nudges.test_only_green_notice([f"tests/test_{i:02d}.py" for i in range(15)])
    assert text.startswith(NOTICE_HEAD)
    assert "tests/test_00.py, " in text
    assert text.endswith("tests/test_11.py (+3 more).")
    assert "tests/test_12.py" not in text
    assert "Confirm" not in text and "usually wrong" not in text
    assert _nudges.test_only_green_notice(["tests/test_a.py"]).endswith(
        " every file changed in between is a test file: tests/test_a.py."
    )


def _repo(root: pathlib.Path) -> pathlib.Path:
    sp.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    for rel in ("src/mod.py", "tests/test_mod.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x = 1\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=root, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)
    return root


Emitted = list[tuple[str, dict[str, Any]]]


def _wf(root: pathlib.Path) -> tuple[loop.Harness, Emitted]:
    wf = loop.Harness(
        chain=_chain.RunChain(root),
        config=Config.model_validate({"harness": {"verify_command": ["true"]}}),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        mode="run",
    )
    emitted: Emitted = []

    def _capture(event_type: str, **fields: Any) -> None:
        emitted.append((event_type, fields))

    wf.events = mock.MagicMock(emit=_capture)
    return wf, emitted


def _exec(rc: int) -> results.ExecResult:
    return results.ExecResult(
        returncode=rc, stdout="", stderr="x", duration_s=0.1, exec_failed=False
    )


def _turn(iteration: int) -> _loop_state.TurnState:
    return _loop_state.TurnState(
        iteration=iteration, resp=mock.MagicMock(), assistant=mock.MagicMock()
    )


def _edit(
    wf: loop.Harness, state: _loop_state.LoopState, turn: _loop_state.TurnState, rel: str
) -> None:
    """Apply an edit through the real path: the file changes on disk, then the loop notes it."""
    (wf.chain.root / rel).write_text("y = 2\n", encoding="utf-8")
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "apply_edit", results.EditResult(applied=("replace",), path=rel), {"path": rel}
    )


def _command(
    wf: loop.Harness,
    state: _loop_state.LoopState,
    turn: _loop_state.TurnState,
    *,
    writes: tuple[str, ...] = (),
) -> None:
    """A run_command edit is on disk when the loop asks git whether the tree moved."""
    before = wf._tree_before_command("run_command")  # pyright: ignore[reportPrivateUsage]
    for rel in writes:
        (wf.chain.root / rel).write_text("z = 3\n", encoding="utf-8")
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_command", _exec(0), {"command": "ls"}, tree_before=before
    )


def _verify(
    wf: loop.Harness, state: _loop_state.LoopState, turn: _loop_state.TurnState, rc: int
) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state, turn, "run_verify_command", _exec(rc), {}
    )


def _notices(turn: _loop_state.TurnState) -> list[str]:
    return [n.text for n in turn.tool_results if isinstance(n, _conversation.Notice)]


def _fired(emitted: Emitted) -> list[dict[str, Any]]:
    return [fields for name, fields in emitted if name == EVENT]


def test_a_source_edit_then_red_then_a_test_edit_then_green_is_noticed(
    tmp_path: pathlib.Path,
) -> None:
    """The tree diff between the red and the green names exactly the test file that changed.

    The breaking edit and the red land in one turn, a read-only command follows, then only a
    test file changes before the green.
    """
    wf, emitted = _wf(_repo(tmp_path))
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(1)
    _edit(wf, state, turn, "src/mod.py")
    _verify(wf, state, turn, 1)
    turn2 = _turn(2)
    _command(wf, state, turn2)
    _edit(wf, state, turn2, "tests/test_mod.py")
    turn3 = _turn(3)
    _verify(wf, state, turn3, 0)
    assert _notices(turn3) == [
        f"{NOTICE_HEAD} every file changed in between is a test file: tests/test_mod.py."
    ]
    assert _fired(emitted) == [{"iteration": 3, "paths": ["tests/test_mod.py"]}]


def test_a_command_that_wrote_a_source_file_withholds_the_notice(tmp_path: pathlib.Path) -> None:
    """A source file changed by run_command in the window makes the claim false, so none is made."""
    wf, emitted = _wf(_repo(tmp_path))
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(1)
    _verify(wf, state, turn, 1)
    turn2 = _turn(2)
    _command(wf, state, turn2, writes=("src/mod.py",))
    _edit(wf, state, turn2, "tests/test_mod.py")
    turn3 = _turn(3)
    _verify(wf, state, turn3, 0)
    assert _notices(turn3) == []
    assert _fired(emitted) == []


def test_source_and_test_edits_withhold_the_notice(tmp_path: pathlib.Path) -> None:
    wf, emitted = _wf(_repo(tmp_path))
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(1)
    _verify(wf, state, turn, 1)
    turn2 = _turn(2)
    _edit(wf, state, turn2, "src/mod.py")
    _edit(wf, state, turn2, "tests/test_mod.py")
    turn3 = _turn(3)
    _verify(wf, state, turn3, 0)
    assert _notices(turn3) == []
    assert _fired(emitted) == []


def test_a_green_after_no_edits_has_no_notice(tmp_path: pathlib.Path) -> None:
    wf, emitted = _wf(_repo(tmp_path))
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(1)
    _verify(wf, state, turn, 1)
    turn2 = _turn(2)
    _verify(wf, state, turn2, 0)
    assert _notices(turn2) == []
    assert _fired(emitted) == []


def test_the_notice_is_withheld_when_git_cannot_say(tmp_path: pathlib.Path) -> None:
    """With no repository under the root nothing is claimed, whatever the edit tools reported."""
    (tmp_path / "tests").mkdir()
    wf, emitted = _wf(tmp_path)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _turn(1)
    _verify(wf, state, turn, 1)
    turn2 = _turn(2)
    _edit(wf, state, turn2, "tests/test_mod.py")
    turn3 = _turn(3)
    _verify(wf, state, turn3, 0)
    assert _notices(turn3) == []
    assert _fired(emitted) == []
