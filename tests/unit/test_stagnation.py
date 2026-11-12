# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The stagnation advisor: one notice past the knob with no edit and no verify.

The operator's time is excluded.
"""

from __future__ import annotations

import time
from unittest import mock

from agent6.harness import advice, guards, loop_conversation, loop_state
from agent6.sessions import journal
from tests.unit.turn_context import turn_context


def _state() -> loop_state.LoopState:
    """Two seconds into the run."""
    started = guards.StagnationGuard(started_monotonic=time.monotonic() - 2.0)
    return loop_state.LoopState(original_task="t", tool_calls=0, stagnation=started)


def _turn(turn: int = 4) -> loop_state.TurnState:
    return loop_state.TurnState(
        number=turn, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )


def _ctx(**kw: object) -> advice.TurnContext:
    return turn_context(stagnation_notice_after_s=1.0, **kw)


def test_the_notice_fires_once_and_names_the_gate_only_when_one_can_run() -> None:
    state = _state()
    nudge = guards.stagnation(_turn(), state, _ctx(gate_present=lambda: True))
    assert nudge is not None and "no edit and no verify" in nudge.text
    assert isinstance(nudge.event, journal.LoopStagnationNudged) and nudge.event.turn == 4
    assert state.stagnation.nudged is True
    assert guards.stagnation(_turn(5), state, _ctx(gate_present=lambda: True)) is None

    gateless = guards.stagnation(_turn(), _state(), _ctx())
    assert gateless is not None and "nothing edited yet" in gateless.text
    assert "verify" not in gateless.text


def test_the_start_gates_baseline_is_no_verify_of_the_runs_own() -> None:
    """A green or red baseline kept the notice quiet for the rest of the run."""
    for ok in (True, False):
        state = _state()
        state.verify.note_baseline(ok=ok)
        nudge = guards.stagnation(_turn(), state, _ctx(gate_present=lambda: True))
        assert nudge is not None and "no edit and no verify" in nudge.text


def test_time_blocked_on_the_operator_is_not_the_models() -> None:
    assert guards.stagnation(_turn(), _state(), _ctx(operator_wait_s=lambda: 3600.0)) is None


def test_an_edit_a_verify_another_mode_or_a_zero_knob_keeps_it_quiet() -> None:
    edited = _state()
    edited.ever_edited = True
    assert guards.stagnation(_turn(), edited, _ctx()) is None
    verification = _state()
    verification.verify.note_fail("sig")
    assert guards.stagnation(_turn(), verification, _ctx()) is None
    assert guards.stagnation(_turn(), _state(), _ctx(mode="plan")) is None
    assert guards.stagnation(_turn(), _state(), turn_context()) is None  # the default knob, 300 s
    assert guards.stagnation(_turn(), _state(), turn_context(stagnation_notice_after_s=0)) is None
