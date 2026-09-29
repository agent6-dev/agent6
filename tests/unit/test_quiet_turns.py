# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The quiet-turn nudges.

An early prose stall on an untouched tree, a prose turn that ends on a question, and an empty turn,
each bounded.
"""

from __future__ import annotations

from agent6.harness import _loop_state, _nudges, _quiet_turns
from agent6.providers import ProviderResponse
from tests.unit.turn_context import turn_context


def _state() -> _loop_state.LoopState:
    return _loop_state.LoopState(original_task="t", tool_calls=0)


def _empty(output_tokens: int = 0, *, starved: bool = False) -> ProviderResponse:
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="length" if starved else "end_turn",
        input_tokens=1,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "thinking", "thinking": "x" * 40}]} if starved else {},
    )


def test_an_early_prose_stall_is_steered_back_a_bounded_number_of_times() -> None:
    state = _state()
    early = turn_context(iteration=2)
    answers = [
        _quiet_turns.silent_no_work(state, early)
        for _ in range(_nudges.SILENT_NO_WORK_PATIENCE + 1)
    ]
    assert [a is not None for a in answers] == [True, True, False]
    first = answers[0]
    assert first is not None and first.text == _nudges.SILENT_NO_WORK_NUDGE
    assert first.event == "loop.silent_no_work.nudge"
    assert first.fields == {"iteration": 2, "nudges_used": 1}
    assert first.log == "  silent finish rejected: no work yet (nudge #1) at iter 2"
    late = turn_context(iteration=_quiet_turns.SILENT_NO_WORK_UNTIL + 1)
    assert _quiet_turns.silent_no_work(_state(), late) is None
    edited = _state()
    edited.ever_edited = True
    assert _quiet_turns.silent_no_work(edited, early) is None
    assert _quiet_turns.silent_no_work(_state(), turn_context(iteration=2, mode="ask")) is None


def test_a_question_in_prose_draws_one_nudge_per_run() -> None:
    state = _state()
    nudge = _quiet_turns.question_in_prose(
        state, turn_context(iteration=5), "Should I keep squash?"
    )
    assert nudge is not None and nudge.text == _nudges.QUESTION_NUDGE
    assert nudge.event == "loop.question_nudge" and nudge.fields == {"iteration": 5}
    assert _quiet_turns.question_in_prose(state, turn_context(), "And this one?") is None
    assert _quiet_turns.question_in_prose(_state(), turn_context(), "Done.") is None
    assert _quiet_turns.question_in_prose(_state(), turn_context(mode="plan"), "Which?") is None


def test_an_empty_turn_is_nudged_up_to_the_cap_with_the_starved_wording() -> None:
    state = _state()
    ctx = turn_context(iteration=3, went_quiet_max_nudges=2)
    first = _quiet_turns.went_quiet(state, ctx, _empty())
    assert first is not None and first.text == _nudges.WENT_QUIET_NUDGE
    assert first.event == "loop.went_quiet.nudge"
    assert first.fields == {"iteration": 3, "nudges_used": 1, "nudges_max": 2, "output_tokens": 0}
    second = _quiet_turns.went_quiet(state, ctx, _empty(900, starved=True))
    assert second is not None and "900 tokens" in second.text
    assert _quiet_turns.went_quiet(state, ctx, _empty()) is None


def test_the_config_knob_sets_the_cap() -> None:
    """`[harness].went_quiet_max_nudges` is the one knob.

    0 ends the run on the first empty turn (no nudge), and the default caps the streak at 4.
    """
    assert (
        _quiet_turns.went_quiet(_state(), turn_context(went_quiet_max_nudges=0), _empty()) is None
    )
    default = _quiet_turns.went_quiet(_state(), turn_context(), _empty())
    assert default is not None and default.fields["nudges_max"] == 4
