# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The metric plateau advisor.

A tie draws the notice while runway remains, counts against the patience in the final budget slice,
then stops; a ceiling stops at once. Its end grounds on the tree.
"""

from __future__ import annotations

from unittest import mock

from agent6.harness import advice, loop_conversation, loop_state, metric_rules
from agent6.sessions import journal
from tests.unit.turn_context import turn_context


def _tie(turn: int) -> loop_state.TurnState:
    turn_state = loop_state.TurnState(
        number=turn, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    turn_state.metric_plateau_finish = "score plateaued at 10"
    return turn_state


def test_a_tie_with_runway_is_a_notice_that_costs_no_patience() -> None:
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    ctx = turn_context(metric=True, budget_remaining=lambda: 0.8)
    for i in range(1, 6):
        answer = metric_rules.metric_plateau(_tie(i), state, ctx)
        assert isinstance(answer, advice.Nudge) and "80% of the budget remains" in answer.text
        assert answer.event == journal.LoopMetricPlateauNudge(
            turn=i, nudges_used=0, budget_remaining=0.8
        )
    assert state.metric.plateau_nudges_used == 0


def test_final_slice_ties_spend_the_patience_then_stop() -> None:
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    ctx = turn_context(metric=True, open_subtasks=lambda: [("t1", "measure")])
    answers = [
        metric_rules.metric_plateau(_tie(i), state, ctx)
        for i in range(1, metric_rules.METRIC_PLATEAU_PATIENCE + 2)
    ]
    notices, stop = answers[:-1], answers[-1]
    assert all(isinstance(a, advice.Nudge) and "unknown" in a.text for a in notices)
    assert [
        a.event.nudges_used
        for a in notices
        if isinstance(a, advice.Nudge) and isinstance(a.event, journal.LoopMetricPlateauNudge)
    ] == [1, 2, 3]
    assert isinstance(stop, advice.Stop) and stop.event is None
    assert stop.soft == "metric_plateau" and stop.declared == "metric_plateau"
    assert stop.log == f"LOOP: metric_plateau at turn {metric_rules.METRIC_PLATEAU_PATIENCE + 1}"
    end = stop.end()
    assert end.reason == "metric_plateau" and end.verdict == "grounded" and end.completed
    assert end.summary == "score plateaued at 10 (1 open task(s): measure)"


def test_a_ceiling_stops_at_once_with_its_event() -> None:
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.metric = metric_rules.MetricGuard(
        history=[metric_rules.MetricSample(label="best", score=27, returncode=0, at_ceiling=True)]
    )
    stop = metric_rules.metric_plateau(
        _tie(4), state, turn_context(metric=True, budget_remaining=lambda: 0.9)
    )
    assert isinstance(stop, advice.Stop)
    assert stop.event == journal.LoopMetricCeilingStop(turn=4)
    assert stop.end().reason == "metric_plateau"


def test_a_turn_without_a_tie_is_quiet() -> None:
    turn = loop_state.TurnState(
        number=1, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    assert metric_rules.metric_plateau(turn, state, turn_context(metric=True)) is None


def test_the_auto_metric_says_it_runs_after_each_committed_edit() -> None:
    """The log line, the docstring and the config text said "after a green verify".

    The harness samples after every step that commits, verified or a checkpoint: under
    `verify_when = "finish"` no per-turn verify exists to wait for.
    """
    from unittest import mock

    from agent6.config import HarnessConfig, MetricConfig
    from agent6.harness import metric_sampler
    from agent6.tools import tool_errors

    lines: list[str] = []
    dispatcher = mock.MagicMock(
        last_verification=None, execution_id="execution", verify_withheld=False
    )
    dispatcher.dispatch.side_effect = tool_errors.ToolError("no score")
    sampler = metric_sampler.MetricSampler(
        settings=MetricConfig(command=("true",), pattern="(\\d+)", goal="minimize"),
        enabled=True,
        dispatcher=dispatcher,
        log=lines.append,
        emit=lambda *_a, **_k: None,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    sampler.auto_feedback(state, turn=3, sha="")
    assert lines == ["LOOP: auto metric after the committed edit at turn 3"]
    told = [
        HarnessConfig.model_fields["metric"].description or "",
        MetricConfig.model_fields["command"].description or "",
        metric_sampler.__doc__ or "",
        metric_sampler.MetricSampler.auto_feedback.__doc__ or "",
    ]
    for text in told:
        assert "committed edit" in text, text
        assert "green verify" not in text and "verify-pass" not in text, text
