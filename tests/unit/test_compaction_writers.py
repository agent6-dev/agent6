from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from agent6 import budget
from agent6.config import harness_config
from agent6.harness import loop_conversation, loop_state
from agent6.harness.compaction import driver, window
from agent6.providers import provider_types, responses_codec


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


def compactor(
    worker: Writer, helper: Writer, events: list[Any], **settings: Any
) -> driver.Compactor:
    return driver.Compactor(
        settings=window.CompactionSettings(
            helper=helper,
            helper_context_tokens=200000,
            keep_recent_chars=0,
            drop_at_chars=1000,
            summarise_at_chars=2000,
            **settings,
        ),
        provider=worker,
        curator=None,
        mode="run",
        dag_available=False,
        decisions=lambda: "",
        verify_fact=lambda _: "",
        compact_requested=lambda: None,
        compact_clear=lambda: None,
        log=lambda _: None,
        emit=events.append,
        emit_graph_snapshot=lambda: None,
    )


def test_worker_is_the_default() -> None:
    assert harness_config.CompactionConfig().summariser == "worker"


def test_worker_preserves_request_and_skips_tier_one() -> None:
    worker, helper = Writer(response()), Writer(response())
    events: list[Any] = []
    comp = compactor(worker, helper, events)
    conv = history()
    before = conv.to_wire()
    tools = [provider_types.ToolDefinition("read_file", "Read a file", {"type": "object"})]
    assert comp.compact(conv, loop_state.LoopState("task", 1, system="system"), tools=tools)
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
    comp = compactor(worker, helper, events)
    conv = history()
    assert comp.compact(conv, loop_state.LoopState("task", 1))
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
    comp = compactor(worker, helper, [])
    conv = history()
    before = conv.to_wire()
    with pytest.raises(type(failure)):
        comp.compact(conv, loop_state.LoopState("task", 1))
    assert helper.calls == []
    assert conv.to_wire() == before


def test_both_rejections_leave_history_unchanged() -> None:
    worker, helper = Writer(response("")), Writer(response(stop_reason="length"))
    comp = compactor(worker, helper, [])
    conv = history()
    before = conv.to_wire()
    assert not comp.compact(conv, loop_state.LoopState("task", 1))
    assert conv.to_wire() == before
    assert len(worker.calls) == len(helper.calls) == 1


def test_helper_reads_full_text_without_tool_output() -> None:
    worker, helper = Writer(response()), Writer(response())
    comp = compactor(worker, helper, [], summariser="helper")
    assert comp.compact(history(), loop_state.LoopState("task", 1))
    sent = helper.calls[0]["messages"][0]["content"]
    assert "EARLY " + "a" * 80000 in sent
    assert "NEWEST " + "z" * 2000 in sent
    assert "TOOL OUTPUT" not in sent
    assert worker.calls == []


def test_helper_bounds_input_and_marks_missing_beginning() -> None:
    worker, helper = Writer(response()), Writer(response())
    comp = compactor(worker, helper, [], summariser="helper")
    comp = dataclasses.replace(
        comp, settings=dataclasses.replace(comp.settings, helper_context_tokens=40000)
    )
    assert comp.compact(history(), loop_state.LoopState("task", 1))
    sent = helper.calls[0]["messages"][0]["content"]
    assert "beginning is missing" in sent
    assert "NEWEST " + "z" * 2000 in sent
    assert "EARLY" not in sent
    assert len(sent.encode()) + len(helper.calls[0]["system"].encode()) <= 40000 - 32768


def test_claude_code_uses_helper_without_advancing_worker() -> None:
    worker, helper = Writer(response()), Writer(response())
    events: list[Any] = []
    comp = compactor(worker, helper, events, worker_backend="claude_code")
    assert comp.compact(history(), loop_state.LoopState("task", 1))
    assert worker.calls == []
    assert len(helper.calls) == 1
    assert "claude_code" in events[-1].fallback_reason


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
