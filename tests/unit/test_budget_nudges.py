# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The before-call budget notices.

A planner's one-shot finish directive on a low budget or too many turns, and a plain run's one-shot
wrap-up directive on a low budget, each once per run and only in its mode. The budget line states
what the execution has used of each cap, before each worker call, when it changed.
"""

from __future__ import annotations

import pathlib
from typing import Any, override

from agent6 import budget, directive, kinds
from agent6.harness import guards, loop_conversation, loop_state, nudges, operator_bridge
from agent6.harness.compaction import window as compaction_window
from agent6.prompts import review
from agent6.providers import ProviderResponse
from agent6.sessions import journal
from agent6.tools import results
from tests.unit import test_harness
from tests.unit.turn_context import turn_context

_PLAN_ROUTE = kinds.ModelRoute("chatgpt", "gpt-test")
_RESETS_AT = 4_000_000_000.0


def _state() -> loop_state.LoopState:
    return loop_state.LoopState(original_task="t", tool_calls=0)


def test_the_planner_is_told_once_on_a_low_budget_or_too_many_turns() -> None:
    low = turn_context(mode="plan", budget_remaining=lambda: nudges.PLAN_BUDGET_NUDGE_BELOW)
    state = _state()
    nudge = guards.plan_budget_nudge(state, low)
    assert nudge is not None and nudge.text == nudges.PLAN_BUDGET_NUDGE
    assert nudge.event == journal.NudgePlanFinish(
        turn=1, budget_remaining=nudges.PLAN_BUDGET_NUDGE_BELOW
    )
    assert nudge.log == "LOOP: plan finish-nudge at turn 1 (turns=False, low_budget=True)"
    assert guards.plan_budget_nudge(state, low) is None

    late = turn_context(mode="plan", turn=nudges.PLAN_NUDGE_AFTER_TURNS + 4, execution_start=5)
    nudge = guards.plan_budget_nudge(_state(), late)
    assert nudge is not None and "turns=True, low_budget=False" in nudge.log
    early = turn_context(mode="plan", turn=nudges.PLAN_NUDGE_AFTER_TURNS, execution_start=2)
    assert guards.plan_budget_nudge(_state(), early) is None
    assert guards.plan_budget_nudge(_state(), turn_context(budget_remaining=lambda: 0.1)) is None


def test_a_plain_run_is_told_once_on_a_low_budget_naming_its_gate() -> None:
    low = turn_context(budget_remaining=lambda: nudges.RUN_BUDGET_NUDGE_BELOW, turn=9)
    state = _state()
    nudge = guards.run_budget_nudge(state, low)
    assert nudge is not None and nudge.text == nudges.RUN_BUDGET_NUDGE_GATELESS
    assert nudge.event == journal.NudgeRunBudget(
        turn=9, budget_remaining=nudges.RUN_BUDGET_NUDGE_BELOW
    )
    assert nudge.log == "LOOP: run budget-nudge at turn 9"
    assert guards.run_budget_nudge(state, low) is None

    gated = turn_context(budget_remaining=lambda: 0.1, gate_present=lambda: True)
    nudge = guards.run_budget_nudge(_state(), gated)
    assert nudge is not None and nudge.text == nudges.RUN_BUDGET_NUDGE
    assert guards.run_budget_nudge(_state(), turn_context(budget_remaining=lambda: 0.5)) is None
    assert guards.run_budget_nudge(_state(), turn_context()) is None
    metric = turn_context(
        budget_remaining=lambda: 0.1, metric=directive.parse_metric_spec("maximize -- score")
    )
    assert guards.run_budget_nudge(_state(), metric) is None
    plan = turn_context(budget_remaining=lambda: 0.1, mode="plan")
    assert guards.run_budget_nudge(_state(), plan) is None


def _tracker(
    *, max_usd: float = -1, max_tokens_fallback: int = -1, max_plan_points: float = -1
) -> budget.BudgetTracker:
    return budget.BudgetTracker(
        max_usd=max_usd, max_tokens_fallback=max_tokens_fallback, max_plan_points=max_plan_points
    )


def _record(tracker: budget.BudgetTracker, route: kinds.ModelRoute, **usage: Any) -> None:
    tracker.record(
        route=route,
        input_tokens=500,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        **usage,
    )


def _plan_reading(used_percent: float) -> budget.PlanUsage:
    return budget.PlanUsage.single(used_percent, 300, _RESETS_AT)


def _budget_lines(messages: list[dict[str, Any]]) -> list[str]:
    """The budget lines in one call's messages, oldest first."""
    return [
        block["text"]
        for message in messages
        if message["role"] == "user"
        for block in message["content"]
        if block.get("text", "").startswith(nudges.BUDGET_LINE_PREFIX)
    ]


def _marked_blocks(messages: list[dict[str, Any]]) -> list[str]:
    """What each cache-marked block of one call holds: a result's call id, or a notice's text."""
    return [
        block.get("tool_use_id") or block["text"]
        for message in messages
        if message["role"] == "user"
        for block in message["content"]
        if "cache_control" in block
    ]


class _PlanMeteredWorker:
    """A scripted worker whose every reply carries the account's next plan reading."""

    def __init__(self, tracker: budget.BudgetTracker, readings: list[float]) -> None:
        self._tracker = tracker
        self._readings = readings
        self.systems: list[str] = []
        self.lines: list[list[str]] = []
        self.marked: list[list[str]] = []
        self.opened: list[str] = []

    def call(self, **kwargs: Any) -> ProviderResponse:
        self.systems.append(kwargs["system"])
        self.lines.append(_budget_lines(kwargs["messages"]))
        self.marked.append(_marked_blocks(kwargs["messages"]))
        self.opened.append(kwargs["messages"][0]["content"][0]["text"])
        turn = len(self.lines)
        if turn > len(self._readings):
            return test_harness._tool_resp("finish_session", {"summary": "done"}, tool_id="done")
        _record(self._tracker, _PLAN_ROUTE, plan_usage=_plan_reading(self._readings[turn - 1]))
        return test_harness._tool_resp("list_dir", {"path": f"d{turn}"}, tool_id=f"list-{turn}")


class _Dispatcher(test_harness._StubDispatcher):
    def dispatch(
        self, name: str, raw_input: dict[str, Any], *, origin: object = None
    ) -> results.ToolResult:
        del name, raw_input, origin
        return results.RawResult({"ok": True})


def _drive(
    tmp_path: pathlib.Path,
    tracker: budget.BudgetTracker,
    readings: list[float],
    *,
    ends: str = "finish_session",
    **kw: Any,
) -> _PlanMeteredWorker:
    worker = _PlanMeteredWorker(tracker, readings)
    kw.setdefault("dispatcher", _Dispatcher())
    wf = test_harness._wf(root=tmp_path, provider=worker, budget=tracker, **kw)
    task = [{"role": "user", "content": [{"type": "text", "text": "TASK:\nfix"}]}]
    result = wf._drive_loop(
        system="SYSTEM",
        conversation=loop_conversation.Conversation.from_wire(task),
        tool_calls=0,
        start_turn=1,
        root_task_id=None,
        original_task="fix",
    )
    assert result.reason == ends
    return worker


def test_each_call_reads_the_plan_points_the_execution_has_moved(tmp_path: pathlib.Path) -> None:
    """A run capped at 95 plan points was told to stop at 55 and could not read the meter.

    Each call's messages state the movement the tracker holds, the figure `sessions show`
    prints; the system prompt stays the same bytes, and an unchanged reading adds no line.
    """
    tracker = _tracker(max_plan_points=95)
    tracker.record_plan_preflight(_PLAN_ROUTE.provider, _plan_reading(10))
    worker = _drive(tmp_path, tracker, [18, 64, 65, 65])

    def line(moved: int) -> str:
        return f"[harness budget] this execution: plan moved={moved}/95pt"

    assert worker.lines == [
        [line(0)],
        [line(0), line(8)],
        [line(0), line(8), line(54)],
        [line(0), line(8), line(54), line(55)],
        [line(0), line(8), line(54), line(55)],  # the fifth reading moved nothing
    ]
    assert worker.systems == ["SYSTEM"] * 5
    assert tracker.snapshot().plan_observed_points == 55


def test_the_budget_line_takes_no_cache_mark_from_the_turns_results(
    tmp_path: pathlib.Path,
) -> None:
    """A roll onto the line would drop the previous call's mark, the prefix that call cached."""
    worker = _drive(tmp_path, _tracker(), [18, 64], max_turns=7)
    assert worker.marked == [
        ["TASK:\nfix"],
        ["TASK:\nfix", "list-1"],
        ["list-1", "list-2"],
    ]


def test_a_restart_that_replaced_the_budget_line_shows_it_again(tmp_path: pathlib.Path) -> None:
    """A plan reading can hold for many turns; a summary in between would leave no line to read."""
    tracker = _tracker(max_plan_points=95)
    tracker.record_plan_preflight(_PLAN_ROUTE.provider, _plan_reading(10))
    armed = [False]

    class Summarizer:
        def call(self, **kwargs: Any) -> ProviderResponse:
            del kwargs
            return test_harness._resp("PROGRESS: listed two directories.")

    class ArmingDispatcher(_Dispatcher):
        @override
        def dispatch(
            self, name: str, raw_input: dict[str, Any], *, origin: object = None
        ) -> results.ToolResult:
            armed[0] = raw_input == {"path": "d2"}  # the second turn's call asks for a restart
            return super().dispatch(name, raw_input, origin=origin)

    def disarm() -> None:
        armed[0] = False

    worker = _drive(
        tmp_path,
        tracker,
        [10, 10, 10],
        dispatcher=ArmingDispatcher(),
        compaction=compaction_window.CompactionSettings(
            summarizer="helper",
            helper=Summarizer(),  # type: ignore[arg-type]
            helper_context_tokens=200_000,
            keep_recent_turns=0,
        ),
        bridge=operator_bridge.OperatorBridge(
            compact_requested=lambda: "" if armed[0] else None, compact_clear=disarm
        ),
    )
    line = "[harness budget] this execution: plan moved=0/95pt"
    assert worker.lines == [[line], [line], [line], [line]]
    # The third call's history starts at the restart; its one line follows the kept turn.
    assert "PROGRESS: listed two directories." in worker.opened[2]
    assert worker.opened[1].startswith("TASK:")


def test_an_execution_with_no_cap_reads_no_budget_line(tmp_path: pathlib.Path) -> None:
    """An unlimited ledger has no cap to read against, so the conversation gains nothing."""
    tracker = _tracker()
    tracker.record_plan_preflight(_PLAN_ROUTE.provider, _plan_reading(10))
    worker = _drive(tmp_path, tracker, [18, 64])
    assert worker.lines == [[], [], []]


def test_a_turn_cap_is_read_each_turn_as_the_cap_counts_it(tmp_path: pathlib.Path) -> None:
    """The turn meter counts from the execution's first turn, as `max_turns` does."""
    worker = _drive(tmp_path, _tracker(), [18, 64], max_turns=7)
    assert [lines[-1] for lines in worker.lines] == [
        f"[harness budget] this execution: turn={turn}/7" for turn in (1, 2, 3)
    ]


def test_a_reviewer_reads_its_reply_count_and_no_budget_line(tmp_path: pathlib.Path) -> None:
    """A reviewer reads `reply i of n` on every call; a turn meter beside it would say it twice."""
    contract = kinds.ConversationContract(system="s", opening="o", max_turns=3)
    worker = _drive(
        tmp_path,
        _tracker(max_usd=10.0),
        [18, 64],
        ends="max_turns",
        max_turns=7,
        mode="reviewer",
        contract=contract,
    )
    assert worker.lines == [[], [], []]
    assert worker.marked[-1][-1] == review.reply_notice(3, 3)


def test_the_budget_line_reads_each_capped_ledger_as_its_cap_counts_it() -> None:
    """Dollars and fallback tokens count running children; an unpriced call marks the dollars."""
    tracker = _tracker(max_usd=10.0, max_tokens_fallback=2000, max_plan_points=95)

    def line() -> str:
        return nudges.budget_line(tracker.snapshot(), turn=3, max_turns=20)

    start = "[harness budget] this execution: plan moved=0/95pt; "
    assert line() == start + "metered=$0/$10.00; fallback tokens=0/2000; turn=3/20"
    _record(tracker, kinds.ModelRoute("api", "priced"), cost_usd=0.42)
    assert line() == start + "metered=$0.42/$10.00; fallback tokens=0/2000; turn=3/20"
    _record(tracker, kinds.ModelRoute("local", "unpriced"))
    assert line() == start + "metered=~$0.42/$10.00; fallback tokens=600/2000; turn=3/20"
    tracker.note_running("child", budget.UsageSettlement(usd=0.25, usd_partial=False, tokens=100))
    assert line() == start + "metered=~$0.67/$10.00; fallback tokens=700/2000; turn=3/20"


def test_the_budget_line_omits_an_unlimited_and_a_refused_ledger() -> None:
    """A cap of 0 refuses its ledger's first call, so that ledger never holds a reading."""
    refused = _tracker(max_usd=0, max_tokens_fallback=0, max_plan_points=0)
    assert nudges.budget_line(refused.snapshot(), turn=1, max_turns=-1) == ""
    assert nudges.budget_line(_tracker().snapshot(), turn=1, max_turns=-1) == ""
    assert nudges.budget_line(None, turn=1, max_turns=-1) == ""
    assert nudges.budget_line(None, turn=2, max_turns=5) == (
        "[harness budget] this execution: turn=2/5"
    )


def test_the_newest_budget_line_is_read_from_the_history() -> None:
    """A line a restart replaced is shown again: the history is what the model can read."""
    conv = loop_conversation.Conversation()
    conv.notice("TASK")
    assert conv.latest_notice(nudges.BUDGET_LINE_PREFIX) is None
    conv.notice(nudges.BUDGET_LINE_PREFIX + "turn=1/5")
    conv.notice(nudges.RUN_BUDGET_NUDGE)
    conv.notice(nudges.BUDGET_LINE_PREFIX + "turn=2/5")
    assert conv.latest_notice(nudges.BUDGET_LINE_PREFIX) == nudges.BUDGET_LINE_PREFIX + "turn=2/5"
    conv.restart("summary")
    assert conv.latest_notice(nudges.BUDGET_LINE_PREFIX) is None


def test_the_queue_reports_only_changed_counts_and_its_drain() -> None:
    """Only unfinished children and the operator's waiting tasks count."""
    from unittest import mock

    graph = mock.MagicMock()
    nodes = {"root": test_harness._tn("root", created_by="user")}
    graph.nodes.return_value = nodes
    wf = test_harness._wf(curator=graph, mode="run")
    conversation = loop_conversation.Conversation()
    conversation.notice("Task")
    ctx = turn_context(mode="run")

    def lines() -> list[str]:
        wf._show_run_state(conversation, ctx)
        return [
            block["text"]
            for message in conversation.to_wire()
            for block in message["content"]
            if block.get("text", "").startswith("[harness queue]")
        ]

    assert lines() == []
    children = mock.MagicMock()
    children.running.return_value = 2
    wf.__dict__["parallel"] = children
    nodes["q"] = test_harness._tn("q", parent_id="root", created_by="user")
    nodes["own"] = test_harness._tn("own", parent_id="root", created_by="worker")
    first = "[harness queue] running children=2; queued tasks=1"
    assert lines() == [first]
    assert lines() == [first]
    nodes["q"] = nodes["q"].model_copy(update={"status": "in_progress"})
    children.running.return_value = 0
    empty = "[harness queue] running children=0; queued tasks=0"
    assert lines() == [first, empty]
    assert lines() == [first, empty]
