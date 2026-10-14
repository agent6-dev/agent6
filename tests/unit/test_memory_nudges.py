# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The cross-run memory write advisory.

The one-shot advisory at the first red-to-green verify flip. It fires only in run mode with a
memory store wired and `harness.memory_advisory` on, and only while nothing has been recorded
(bench/longhorizon FINDINGS #2). A finish after such a recovery is honoured at once.
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

from agent6.config import Config
from agent6.harness import (
    _chain,
    _conversation,
    _finish_gates,
    _guards,
    _loop_state,
    _nudges,
    _verify_verdict,
    loop,
)
from agent6.tools import results


def _wf(**kw: Any) -> loop.Harness:
    kw.setdefault("state_dir", pathlib.Path("/tmp/state"))
    kw.setdefault("config", Config.model_validate({}))
    return loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        **kw,
    )


def _state(**kw: Any) -> _loop_state.LoopState:
    return _loop_state.LoopState(original_task="t", tool_calls=0, **kw)


def _turn(iteration: int = 1, **kw: Any) -> _loop_state.TurnState:
    return _loop_state.TurnState(
        iteration=iteration,
        resp=mock.MagicMock(),
        assistant=_conversation.AssistantTurn((), ()),
        **kw,
    )


def _verify(
    wf: loop.Harness, state: _loop_state.LoopState, turn: _loop_state.TurnState, rc: int
) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        "run_verify_command",
        results.ExecResult(returncode=rc, stdout="", stderr="", duration_s=0.0, exec_failed=False),
        {},
    )


def _notice_texts(turn: _loop_state.TurnState) -> list[str]:
    return [item.text for item in turn.tool_results if isinstance(item, _conversation.Notice)]


def _flip(
    wf: loop.Harness, state: _loop_state.LoopState, turn: _loop_state.TurnState
) -> str | None:
    """The flip advisory's text for this turn, as the advisor answers it."""
    ctx = wf._turn_context(state, iteration=turn.iteration, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    nudge = _guards.memory_flip(turn, state, ctx)
    return None if nudge is None else nudge.text


def test_flip_advisory_fires_once_at_first_red_green_flip() -> None:
    wf = _wf()
    state = _state()
    fail = _turn(1)
    _verify(wf, state, fail, rc=1)
    assert state.verify.ever_failed is True
    assert fail.verify_flipped_green is False

    flip = _turn(2)
    _verify(wf, state, flip, rc=0)
    assert flip.verify_flipped_green is True
    assert _flip(wf, state, flip) == _nudges.MEMORY_FLIP_NUDGE
    assert state.memory.flip_nudged is True

    # A second recovery does not re-nudge.
    again = _turn(3)
    _verify(wf, state, again, rc=1)
    _verify(wf, state, again, rc=0)
    assert again.verify_flipped_green is True
    assert _flip(wf, state, again) is None


def test_flip_advisory_needs_a_prior_red_verify() -> None:
    wf = _wf()
    state = _state()
    green = _turn(1)
    _verify(wf, state, green, rc=0)
    assert green.verify_flipped_green is False
    assert _flip(wf, state, green) is None


def test_flip_advisory_suppressed_without_store_write_or_run_mode() -> None:
    for wf, state_kw in (
        (_wf(state_dir=None), {}),
        (_wf(), {"memory": _guards.MemoryState(written=True)}),
        (_wf(mode="ask"), {}),
    ):
        state = _state(
            **state_kw, verify=_verify_verdict.VerifyVerdict(last_ok=False, ever_failed=True)
        )
        flip = _turn(2)
        _verify(wf, state, flip, rc=0)
        assert _flip(wf, state, flip) is None


def test_memory_dir_edit_marks_memory_written() -> None:
    """An edit under the memory dir is a memory write.

    The advisory goes quiet and none of the workspace-edit bookkeeping applies (the verify gate's
    tree is untouched).
    """
    wf = _wf()
    state = _state()
    turn = _turn(1)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        "apply_edit",
        # EditResult spells memory paths store-relative; the INPUT path is
        # what identifies a memory write.
        results.EditResult(applied=("create",), path="new-fact.md"),
        {"path": "/tmp/state/memory/new-fact.md", "edits": []},
    )
    assert state.memory.written is True
    assert state.ever_edited is False
    assert turn.edited is False


def test_memory_dir_patch_without_a_path_marks_memory_written() -> None:
    """A memory-dir patch without a `path` marks memory written.

    `apply_patch`'s normal shape carries no `path` (the targets are the headers); counted as a
    workspace edit, a memory-dir patch withdraws a green verify, leaves `memory_written` False (the
    advisory then fires at a model that just wrote memory) and sets `ever_edited`. The targets come
    from the patch; a patch over the store and the workspace together is workspace work.
    """

    def note(patch: str) -> tuple[_loop_state.LoopState, _loop_state.TurnState]:
        wf = _wf()
        state = _state()
        turn = _turn(1)
        wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
            state,
            turn,
            "apply_patch",
            results.PatchResult(path="new-fact.md", bytes_written=2),
            {"patch": patch},
        )
        return state, turn

    memory = "--- /dev/null\n+++ /tmp/state/memory/new-fact.md\n@@ -0,0 +1 @@\n+x\n"
    state, turn = note(memory)
    assert (state.memory.written, state.ever_edited, turn.edited) == (True, False, False)
    v4a = "*** Begin Patch\n*** Add File: /tmp/state/memory/other.md\n+x\n*** End Patch\n"
    state, turn = note(v4a)
    assert (state.memory.written, state.ever_edited, turn.edited) == (True, False, False)
    workspace = "--- a/src/code.py\n+++ b/src/code.py\n@@ -1 +1 @@\n-a\n+b\n"
    state, turn = note(workspace)
    assert (state.memory.written, state.ever_edited, turn.edited) == (False, True, True)
    both = "diff --git a/src/code.py b/src/code.py\n" + workspace + "diff --git a/x b/x\n" + memory
    state, turn = note(both)
    assert (state.memory.written, state.ever_edited, turn.edited) == (False, True, True)


def test_a_preview_edit_is_no_edit() -> None:
    """A preview edit is no edit.

    `preview=true` writes nothing, so it marks neither a memory write (the advisory would go quiet
    over a dry run of the store) nor a tree edit (a green verify would be withdrawn over a tree
    nothing touched).
    """

    def preview(path: str) -> results.PreviewResult:
        return results.PreviewResult(
            path=path, diff="", hunks=1, bytes_before=1, bytes_after=2, truncated=False
        )

    calls: list[tuple[str, dict[str, object]]] = [
        ("apply_edit", {"path": "/tmp/state/memory/f.md", "edits": [], "preview": True}),
        ("apply_edit", {"path": "src/code.py", "edits": [], "preview": True}),
        ("apply_patch", {"patch": "--- a/src/code.py\n+++ b/src/code.py\n@@ -1 +1 @@\n-a\n+b\n"}),
    ]
    for name, tool_input in calls:
        wf, state, turn = _wf(), _state(), _turn(1)
        wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
            state, turn, name, preview(str(tool_input.get("path", "src/code.py"))), tool_input
        )
        assert (state.memory.written, state.ever_edited, turn.edited) == (False, False, False)
        assert turn.edit_since_verify_pass is False


def test_workspace_edit_does_not_mark_memory_written() -> None:
    wf = _wf()
    state = _state()
    turn = _turn(1)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        "apply_edit",
        results.EditResult(applied=("replace",), path="src/code.py"),
        {"path": "src/code.py", "edits": []},
    )
    assert state.memory.written is False
    assert state.ever_edited is True


def test_a_finish_after_a_recovery_is_honoured_at_once() -> None:
    """A recovery with nothing recorded draws the flip advisory and its finish is not deferred."""
    wf = _wf()
    state = _state()
    _verify(wf, state, _turn(1), rc=1)
    flip = _turn(2, finish=_finish_gates.FinishCall("finish_session", "done"))
    _verify(wf, state, flip, rc=0)
    ctx = wf._turn_context(state, iteration=2, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    wf._turn_advisors(state, flip, ctx)  # pyright: ignore[reportPrivateUsage]
    wf._turn_finish_gates(state, flip, ctx)  # pyright: ignore[reportPrivateUsage]
    assert flip.finish is not None
    assert _notice_texts(flip) == [_nudges.MEMORY_FLIP_NUDGE]


def test_the_flip_advisory_follows_the_config_key() -> None:
    """`harness.memory_advisory = false` sends no advisory at a recovery; the default sends it."""
    assert Config.model_validate({}).harness.memory_advisory is True
    for harness, sent in (({}, [_nudges.MEMORY_FLIP_NUDGE]), ({"memory_advisory": False}, [])):
        wf = _wf(config=Config.model_validate({"harness": harness}))
        state = _state()
        _verify(wf, state, _turn(1), rc=1)
        flip = _turn(2)
        _verify(wf, state, flip, rc=0)
        ctx = wf._turn_context(state, iteration=2, execution_start=1)  # pyright: ignore[reportPrivateUsage]
        wf._turn_advisors(state, flip, ctx)  # pyright: ignore[reportPrivateUsage]
        assert _notice_texts(flip) == sent
        assert state.memory.flip_nudged is bool(sent)
