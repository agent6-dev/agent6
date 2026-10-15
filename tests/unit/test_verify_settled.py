# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The settled advisor: one nudge, then a soft stop the end gates judge.

A plain run that reached a good state and made no progress; the stop's end is
`verify_settled` only when a green verify covers the tree as it stands.
"""

from __future__ import annotations

import dataclasses
import pathlib
from unittest import mock

import pytest

from agent6 import config
from agent6.harness import (
    _advice,
    _chain,
    _conversation,
    _finish_gates,
    _guards,
    _loop_state,
    _nudges,
    _operator,
    loop,
)
from agent6.tools import results
from agent6.viewmodel import _session_status
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
    turns = range(1, 14)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in turns]
    nudges = [a for a in answers if isinstance(a, _advice.Nudge)]
    stops = [a for a in answers if isinstance(a, _advice.Stop)]
    assert [a.text for a in nudges] == [
        "[harness settled] Changes are committed and the last three turns changed"
        " nothing; finish_session ends the run, and at 12 unchanged turns the run ends on its"
        " own."
    ]
    assert nudges[0].fields == {"iteration": _nudges.VERIFY_SETTLED_NUDGE_AFTER + 1, "idle": 3}
    assert len(stops) == 1 and answers[-1] is stops[0]
    stop = stops[0]
    assert stop.soft == "verify_settled" and stop.declared == "settled"
    assert stop.log == "LOOP: verify_settled at iter 13 (idle 12)"
    end = stop.end()
    assert end.reason == "verify_settled" and end.completed and end.verdict == "passed"


def test_a_red_gate_draws_no_settled_nudge_but_the_stop_still_comes() -> None:
    """The nudge says `finish_session ends the run`; over a red gate that finish is refused.

    A gated run under `verify_when = "finish"` arms on an unjudged edit, so an idle model
    was invited to a finish the gate then bounced.
    """
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.settled.gateless_ever_edited = True
    state.verify.note_fail("boom")
    ctx = turn_context(tree_sha=lambda: "tree", gate_present=lambda: True)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, 14)]
    assert not [a for a in answers if isinstance(a, _advice.Nudge)]
    stops = [a for a in answers if isinstance(a, _advice.Stop)]
    assert len(stops) == 1 and answers[-1] is stops[0]
    assert stops[0].end().reason == "settled"
    assert "still red" in stops[0].end().summary
    # Green again: the nudge is back on the next idle streak.
    state.verify.note_pass()
    state.settled.restart()
    answers = [_guards.verify_settled(_turn(i, edited=i == 1), state, ctx) for i in range(1, 6)]
    assert len([a for a in answers if isinstance(a, _advice.Nudge)]) == 1


def test_a_nudge_withheld_over_a_red_gate_is_not_spent() -> None:
    """A gate that goes green again within the idle streak still gets its one nudge.

    A withheld nudge stays unsent, so a gate green again before the stop still draws it.
    """
    state = _green_state()
    state.verify.note_fail("boom")
    ctx = turn_context(tree_sha=lambda: "tree", gate_present=lambda: True)
    red = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, 6)]
    assert not [a for a in red if isinstance(a, _advice.Nudge)]
    state.verify.note_pass()  # a green verify on the same tree, as a flaky test gives
    green = [
        _guards.verify_settled(_turn(i, verify_just_passed=i == 6), state, ctx) for i in range(6, 9)
    ]
    assert len([a for a in green if isinstance(a, _advice.Nudge)]) == 1


def test_a_gate_red_at_start_still_draws_the_settled_nudge() -> None:
    """A red start gate keeps the nudge: a finish over that red stands.

    Withholding it would leave a run beside an unrelated failing test idling to the settled stop.
    """
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_baseline(ok=False)
    # Under `verify_when = "finish"` the model's unjudged edit arms the guard.
    state.verify.note_edit()
    state.settled.gateless_ever_edited = True
    ctx = turn_context(tree_sha=lambda: "tree", gate_present=lambda: True)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, 6)]
    assert len([a for a in answers if isinstance(a, _advice.Nudge)]) == 1


def test_the_stop_waits_twelve_idle_turns_and_the_nudge_says_so() -> None:
    """The default stops at 12 idle turns and the nudge states that number."""
    state = _green_state()
    ctx = turn_context(tree_sha=lambda: "tree")
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, 14)]
    kinds = [type(a) for a in answers]
    # Turn 1 measures the tree; the nudge lands at idle 3 (turn 4), the stop at idle 12 (turn 13).
    assert kinds == [type(None)] * 3 + [_advice.Nudge] + [type(None)] * 8 + [_advice.Stop]
    nudge = answers[3]
    assert isinstance(nudge, _advice.Nudge)
    assert "at 12 unchanged turns" in nudge.text


def _configured_context(
    tmp_path: pathlib.Path, state: _loop_state.LoopState, stop_after: int
) -> _advice.TurnContext:
    """Build the advisor's context through the harness with a configured stop."""
    wf = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=config.Config.model_validate({"harness": {"verify_settled_stop_after": stop_after}}),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _line: None,
        mode="run",
    )
    ctx = wf._turn_context(state, iteration=1, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    return dataclasses.replace(ctx, tree_sha=lambda: "tree")


@pytest.mark.parametrize("stop_after", [1, 3, 6, 18])
@pytest.mark.parametrize("gated", [False, True])
def test_the_configured_threshold_moves_the_settled_stop(
    tmp_path: pathlib.Path, stop_after: int, gated: bool
) -> None:
    """The settled guard uses the configured threshold, not a fixed 12 turns."""
    state = _green_state() if gated else _loop_state.LoopState(original_task="t", tool_calls=0)
    state.settled.gateless_ever_edited = not gated
    ctx = _configured_context(tmp_path, state, stop_after)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, stop_after + 2)]
    assert not any(isinstance(a, _advice.Stop) for a in answers[:-1])
    assert isinstance(answers[-1], _advice.Stop)
    assert state.settled.idle == stop_after


@pytest.mark.parametrize("gated", [False, True])
def test_zero_disables_the_settled_stop_but_keeps_the_nudge(
    tmp_path: pathlib.Path, gated: bool
) -> None:
    """A disabled settled stop still nudges at idle 3 without announcing an end."""
    state = _green_state() if gated else _loop_state.LoopState(original_task="t", tool_calls=0)
    state.settled.gateless_ever_edited = not gated
    ctx = _configured_context(tmp_path, state, 0)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, 102)]
    nudges = [a for a in answers if isinstance(a, _advice.Nudge)]
    assert not any(isinstance(a, _advice.Stop) for a in answers)
    assert len(nudges) == 1 and answers[3] is nudges[0]
    assert nudges[0].fields == {"iteration": 4, "idle": 3}
    assert nudges[0].text == (
        "[harness settled] Changes are committed and the last three turns changed"
        " nothing; finish_session ends the run."
    )


@pytest.mark.parametrize("stop_after", [6, 18])
def test_the_settled_nudge_names_the_configured_threshold(
    tmp_path: pathlib.Path, stop_after: int
) -> None:
    """The model's settled nudge states the configured stop instead of twelve."""
    state = _green_state()
    ctx = _configured_context(tmp_path, state, stop_after)
    answers = [_guards.verify_settled(_turn(i), state, ctx) for i in range(1, 5)]
    nudge = answers[-1]
    assert isinstance(nudge, _advice.Nudge)
    assert f"at {stop_after} unchanged turns" in nudge.text
    assert "twelve" not in nudge.text


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


def test_a_memory_store_write_restarts_the_idle_streak() -> None:
    """The settled nudge followed the memory advisory: the writes it asked for counted as idle."""
    wf = loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=config.Config(),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _line: None,
        state_dir=pathlib.Path("/tmp/state"),
    )
    state = _green_state()
    state.settled = _guards.SettledGuard(tree="tree", idle=_nudges.VERIFY_SETTLED_NUDGE_AFTER - 1)
    turn = _turn(5)
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        turn,
        "apply_edit",
        results.EditResult(applied=("create",), path="fact.md"),
        {"path": "/tmp/state/memory/fact.md", "edits": []},
    )
    assert turn.edited is False
    assert _guards.verify_settled(turn, state, turn_context(tree_sha=lambda: "tree")) is None
    assert state.settled.idle == 0


def test_a_finish_call_a_metric_run_or_an_unseeded_run_disarms_it() -> None:
    ready = _guards.SettledGuard(tree="tree", idle=12)
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
    gateless.settled = _guards.SettledGuard(tree="tree", idle=12, gateless_ever_edited=True)
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
    # A green on the untouched base is the baseline's: the run never passed on its own.
    base_green = _loop_state.LoopState(original_task="t", tool_calls=0)
    base_green.verify.note_baseline(ok=True)
    assert "the only green was the base's" in _guards.settled_end(base_green, runnable).summary
    passed = _guards.settled_end(_green_state(), turn_context(tree_green=lambda: True))
    assert passed.reason == "verify_settled" and passed.scoped is False


def _steered(tmp_path: pathlib.Path, text: str | None) -> _loop_state.LoopState:
    """Return a state one turn short of the settled stop, after the operator's steer landed."""
    wf = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=config.Config(),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _line: None,
        mode="run",
        bridge=_operator.OperatorBridge(
            steer_requested=lambda: text is not None, steer_prompt=lambda: text
        ),
    )
    state = _green_state()
    state.settled = _guards.SettledGuard(tree="tree", idle=11, nudged=True)
    wf.steering.handle(_conversation.Conversation(), 4, state)
    return state


def test_an_operator_steer_restarts_the_idle_streak_and_its_nudge(tmp_path: pathlib.Path) -> None:
    """A steer left the idle streak running, so the stop could cut the work the steer asked for."""
    state = _steered(tmp_path, "run the six test files one at a time")
    assert state.settled.idle == 0 and state.settled.nudged is False
    ctx = turn_context(tree_sha=lambda: "tree")
    assert _guards.verify_settled(_turn(5), state, ctx) is None
    # No steer, or an empty one, leaves the streak as it was.
    for text in (None, "  "):
        untouched = _steered(tmp_path, text)
        assert untouched.settled.idle == 11
        assert untouched.settled.nudged is True


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        ("verify_settled", "settled"),
        ("silent_finish", "no finish call"),
        ("metric_plateau", "metric plateau"),
    ],
)
def test_a_green_harness_end_reads_finished_never_passed(reason: str, detail: str) -> None:
    """A harness-declared end over a green gate must not read passed, even when scoped."""
    for scoped in (False, True):
        assert _session_status.status_word(
            finished=True, all_passed=True, end_reason=reason, scoped=scoped
        ) == ("finished", detail)
