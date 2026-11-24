from __future__ import annotations

import copy
import dataclasses
import pathlib
from typing import Any

import pytest

from agent6 import budget, providers
from agent6.config import harness_config
from agent6.harness import loop_conversation, loop_state
from agent6.harness.compaction import driver, strategy, window
from agent6.providers import provider_types, responses_codec
from agent6.sessions import event_log
from agent6.sessions import layout as sessions_layout
from tests.unit import test_harness


class Writer:
    def __init__(self, reply: provider_types.ProviderResponse | Exception) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def call(self, **kwargs: Any) -> provider_types.ProviderResponse:
        self.calls.append(copy.deepcopy(kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def response(text: str = "summary", **kwargs: Any) -> provider_types.ProviderResponse:
    return (
        provider_types.ProviderResponse(
            text=text,
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=20,
            cache_write_tokens=0,
        )
        if not kwargs
        else dataclasses.replace(response(text), **kwargs)
    )


def history() -> loop_conversation.Conversation:
    conv = loop_conversation.Conversation.from_wire(
        [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "EARLY " + "a" * 80000}]},
            {"role": "user", "content": [{"type": "text", "text": "operator direction"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read",
                        "name": "read_file",
                        "input": {"path": "file.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "read", "content": "TOOL OUTPUT " * 1000}
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "NEWEST " + "z" * 2000}]},
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        ]
    )
    conv.roll_cache_marks()
    return conv


def context(
    worker: providers.Provider,
    helper: providers.Provider,
    events: list[Any],
    *,
    conversation: loop_conversation.Conversation | None = None,
    tools: list[provider_types.ToolDefinition] | None = None,
    helper_context_tokens: int = 200000,
    task: str = "task",
    pins: tuple[str, ...] = (),
    **settings: Any,
) -> strategy.Context:
    harness = test_harness._wf(
        provider=worker,
        compaction=window.CompactionSettings(
            helper=helper,
            helper_context_tokens=helper_context_tokens,
            keep_recent_chars=0,
            drop_at_chars=1000,
            summarize_at_chars=2000,
            **settings,
        ),
    )
    turn = harness.compaction_context(
        conversation or history(),
        loop_state.LoopState(task, 1, system="system", pins=list(pins)),
        tools or [],
    )
    return dataclasses.replace(turn, emit=events.append)


def test_worker_is_the_default() -> None:
    assert harness_config.CompactionConfig().summarizer == "worker"


def test_the_cold_helper_reads_the_pending_focus_once() -> None:
    worker, helper = Writer(response()), Writer(response("Focused handoff."))
    events: list[Any] = []
    pending = ["Keep the rollback conditions."]
    turn = dataclasses.replace(
        context(worker, helper, events, cold_start="summarize"),
        compact_requested=lambda: pending[0] if pending else None,
        compact_clear=pending.clear,
    )
    assert driver.Tiered().cold_start(turn) == "summarize"
    assert len(helper.calls) == 1 and not worker.calls
    assert "Keep the rollback conditions." in str(helper.calls[0]["messages"])
    assert not pending and not any(e.TYPE == "compact.refused" for e in events)
    kinds = [e.TYPE for e in events]
    assert kinds.index("compact.requested") < kinds.index("compact.summarize.call")


def test_a_cold_drop_makes_no_model_call_before_the_ordinary_size_check() -> None:
    worker, helper = Writer(response()), Writer(response())
    events: list[Any] = []
    turn = context(worker, helper, events, cold_start="drop", elision_gists=False)
    compactor = driver.Tiered()
    assert compactor.cold_start(turn) == "drop"
    assert not worker.calls and not helper.calls and "EARLY" in str(turn.conversation.to_wire())
    assert compactor.compact(turn) and len(worker.calls) == 1


@pytest.mark.parametrize("offered", [True, False])
def test_restart_names_its_readable_journal(tmp_path: pathlib.Path, offered: bool) -> None:
    layout = sessions_layout.SessionLayout(tmp_path, "calm-owl-AAAAAA")
    harness = test_harness._wf(
        provider=Writer(response("short summary")),
        events=event_log.EventSink(layout.logs_path),
        state_dir=tmp_path,
        compaction=window.CompactionSettings(keep_recent_chars=0, summarize_at_chars=2000),
    )
    conv = history()
    tools = [provider_types.ToolDefinition("read_session", "Read entries", {})] if offered else []
    turn = harness.compaction_context(conv, loop_state.LoopState("task", 1), tools)
    assert driver.Tiered().compact(turn)
    request = str(conv.to_wire())
    assert "short summary" in request and "operator direction" not in request
    assert ("read_session" in request) == offered
    assert (layout.session_id in request) == offered


def test_worker_preserves_request_and_skips_tier_one() -> None:
    worker, helper = Writer(response()), Writer(response())
    events: list[Any] = []
    conv = history()
    before = conv.to_wire()
    tools = [provider_types.ToolDefinition("read_file", "Read a file", {"type": "object"})]
    assert driver.Tiered().compact(context(worker, helper, events, conversation=conv, tools=tools))
    sent = worker.calls[0]
    assert sent["system"] == "system"
    assert sent["tools"] == tools
    assert sent["messages"][:-1] == before
    assert "write the summary now" in sent["messages"][-1]["content"].lower()
    assert "tool call" in sent["messages"][-1]["content"].lower()
    assert helper.calls == []
    assert not any(e.TYPE in ("compact.dropped", "compact.gist.call") for e in events)
    assert events[-1].writer == "worker"


@pytest.mark.parametrize(
    "reply,reason",
    [
        (response(""), "empty summary"),
        (response(stop_reason="max_tokens"), "output cap"),
        (response(stop_reason="refusal"), "refusal"),
        (response(raw={"choices": [{"message": {"refusal": "no"}}]}), "refusal"),
        (response(tool_uses=({"id": "bad", "name": "run_command", "input": {}},)), "tool call"),
        (provider_types.ProviderError("request too big", status_code=400), "request too big"),
    ],
)
def test_unusable_worker_summary_falls_back_once(
    reply: provider_types.ProviderResponse | Exception,
    reason: str,
) -> None:
    worker, helper = Writer(reply), Writer(response("helper summary"))
    events: list[Any] = []
    turn = context(worker, helper, events)
    conv = turn.conversation
    assert driver.Tiered().compact(turn)
    assert len(worker.calls) == len(helper.calls) == 1
    assert events[-1].writer == "helper"
    assert reason in events[-1].fallback_reason
    assert "helper summary" in str(conv.to_wire())


@pytest.mark.parametrize(
    "failure",
    [
        provider_types.ProviderAborted("stop"),
        provider_types.ProviderInterrupted("steer"),
        provider_types.ProviderError("signed out", status_code=401),
        provider_types.ProviderError("forbidden", status_code=403),
        provider_types.ProviderError("credits", status_code=402),
        provider_types.ProviderError("plan usage limit", status_code=429),
        provider_types.ProviderError("sign in", fatal=True),
        budget.BudgetExceededError("spent"),
    ],
)
def test_terminal_failure_never_calls_helper(failure: Exception) -> None:
    worker, helper = Writer(failure), Writer(response())
    turn = context(worker, helper, [])
    conv = turn.conversation
    before = conv.to_wire()
    with pytest.raises(type(failure)):
        driver.Tiered().compact(turn)
    assert helper.calls == []
    assert conv.to_wire() == before


def test_both_rejections_leave_history_unchanged() -> None:
    worker, helper = Writer(response("")), Writer(response(stop_reason="length"))
    turn = context(worker, helper, [])
    conv = turn.conversation
    before = conv.to_wire()
    assert not driver.Tiered().compact(turn)
    assert conv.to_wire() == before
    assert len(worker.calls) == len(helper.calls) == 1


def test_helper_reads_full_text_without_tool_output() -> None:
    worker, helper = Writer(response()), Writer(response())
    turn = context(worker, helper, [], summarizer="helper")
    assert driver.Tiered().compact(turn)
    sent = helper.calls[0]["messages"][0]["content"]
    assert "EARLY " + "a" * 80000 in sent
    assert "NEWEST " + "z" * 2000 in sent
    assert "TOOL OUTPUT" not in sent
    assert worker.calls == []


def test_helper_bounds_input_and_marks_missing_beginning() -> None:
    worker, helper = Writer(response()), Writer(response())
    turn = context(worker, helper, [], summarizer="helper")
    turn = dataclasses.replace(
        turn, settings=dataclasses.replace(turn.settings, helper_context_tokens=40000)
    )
    assert driver.Tiered().compact(turn)
    sent = helper.calls[0]["messages"][0]["content"]
    assert "beginning is missing" in sent
    assert "NEWEST " + "z" * 2000 in sent
    assert "EARLY" not in sent
    assert len(sent.encode()) + len(helper.calls[0]["system"].encode()) <= 40000 - 32768


def test_claude_code_uses_helper_without_advancing_worker() -> None:
    worker, helper = Writer(response()), Writer(response())
    events: list[Any] = []
    turn = context(
        worker, helper, events, endpoint=provider_types.EndpointCompaction(keeps_session=True)
    )
    assert driver.Tiered().compact(turn)
    assert worker.calls == []
    assert len(helper.calls) == 1
    assert "owns its live session" in events[-1].fallback_reason


def test_responses_refusal_is_not_lost_beside_text() -> None:
    parsed = responses_codec.parse_output_items(
        [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "preface"},
                    {"type": "refusal", "refusal": "no"},
                ],
            }
        ],
        usage={},
        stop_reason="end_turn",
        label="Responses",
        route="test/test",
    )
    assert any(block.get("type") == "refusal" for block in parsed.raw["content"])
