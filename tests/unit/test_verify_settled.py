# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The settled advisor: one nudge, then a soft stop the end gates judge.

A plain run that reached a good state and made no progress; the stop's end is a pass only
when a green verify covers the tree as it stands.
"""

from __future__ import annotations

from unittest import mock

from agent6.harness import _advice, _conversation, _finish_gates, _guards, _loop_state, _nudges
from tests.unit.turn_context import turn_context


def _turn(iteration: int, **kw: bool) -> _loop_state.TurnState:
    turn = _loop_state.TurnState(
        iteration=iteration, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )
    for name, value in kw.items():
        setattr(turn, name, value)
    return turn


def _green_state() -> _loop_state.LoopState:
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_pass()
    return state


def test_idle_turns_draw_one_nudge_then_a_soft_declared_stop() -> None:
    state = _green_state()
    ctx = turn_context(tree_sha=lambda: "tree")
    # The first turn measures the tree (progress); the idle streak starts after it.
    turns = range(1, _nudges.VERIFY_SETTLED_STOP_AFTER + 2)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in turns]
    nudges = [a for a in answers if isinstance(a, _advice.Nudge)]
    stops = [a for a in answers if isinstance(a, _advice.Stop)]
    assert [a.text for a in nudges] == [_nudges.VERIFY_SETTLED_NUDGE]
    assert nudges[0].fields == {"iteration": _nudges.VERIFY_SETTLED_NUDGE_AFTER + 1, "idle": 3}
    assert len(stops) == 1 and answers[-1] is stops[0]
    stop = stops[0]
    assert stop.soft == "verify_settled" and stop.declared == "settled"
    assert (
        stop.log == f"LOOP: verify_settled at iter {_nudges.VERIFY_SETTLED_STOP_AFTER + 1} (idle 6)"
    )
    end = stop.end()
    assert end.reason == "verify_settled" and end.completed and end.verdict == "passed"


def test_progress_restarts_the_streak_and_a_verify_run_is_neutral() -> None:
    state = _green_state()
    ctx = turn_context(tree_sha=lambda: "tree")
    for i in range(1, _nudges.VERIFY_SETTLED_NUDGE_AFTER):
        assert _guards.verify_settled(_turn(i), state, ctx) is None
    assert state.settled.idle == 1
    assert _guards.verify_settled(_turn(3, verify_just_passed=True), state, ctx) is None
    assert state.settled.idle == 1
    assert _guards.verify_settled(_turn(4, edited=True), state, ctx) is None
    assert state.settled.idle == 0
    moved = turn_context(tree_sha=lambda: "another tree")
    state.settled.idle = 2
    assert _guards.verify_settled(_turn(5), state, moved) is None and state.settled.idle == 0


def test_a_finish_call_a_metric_run_or_an_unseeded_run_disarms_it() -> None:
    ready = _guards.SettledGuard(tree="tree", idle=_nudges.VERIFY_SETTLED_STOP_AFTER)
    state = _green_state()
    state.settled = ready
    ctx = turn_context(tree_sha=lambda: "tree")
    finishing = _turn(9)
    finishing.finish = _finish_gates.FinishCall("finish_session", "done")
    assert _guards.verify_settled(finishing, state, ctx) is None
    metric = turn_context(metric=True, tree_sha=lambda: "tree")
    assert _guards.verify_settled(_turn(9), state, metric) is None
    unseeded = _loop_state.LoopState(original_task="t", tool_calls=0)
    assert _guards.verify_settled(_turn(9), unseeded, ctx) is None
    gateless = _loop_state.LoopState(original_task="t", tool_calls=0)
    gateless.settled = _guards.SettledGuard(
        tree="tree", idle=_nudges.VERIFY_SETTLED_STOP_AFTER, gateless_ever_edited=True
    )
    assert isinstance(_guards.verify_settled(_turn(9), gateless, ctx), _advice.Stop)


def test_the_settled_end_says_why_it_is_not_a_pass() -> None:
    ctx = turn_context(tree_green=lambda: False, open_subtasks=lambda: [("t1", "audit")])
    red = _green_state()
    red.verify.note_fail("sig")
    end = _guards.settled_end(red, ctx)
    assert end.reason == "settled" and end.roots is True and end.verdict == "failed"
    assert end.summary == (
        "the worker settled, but the verify gate is still red (1 open task(s): audit)"
    )
    edited = _green_state()
    edited.verify.note_edit()
    assert "never re-verified" in _guards.settled_end(edited, ctx).summary
    never = _loop_state.LoopState(original_task="t", tool_calls=0)
    assert "no verify command existed" in _guards.settled_end(never, ctx).summary
    withheld = turn_context(tree_green=lambda: False, verify_command=lambda: ("pytest",))
    assert "could not run" in _guards.settled_end(never, withheld).summary
    runnable = turn_context(
        tree_green=lambda: False, verify_command=lambda: ("pytest",), gate_present=lambda: True
    )
    assert "the verify never passed" in _guards.settled_end(never, runnable).summary
    passed = _guards.settled_end(_green_state(), turn_context(tree_green=lambda: True))
    assert passed.reason == "verify_settled" and passed.scoped is False
