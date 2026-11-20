"""Exercise endpoint adoption, its fallback and the model-dependent strategy choice."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import pytest

from agent6 import config
from agent6.app import role_providers
from agent6.harness import loop_conversation
from agent6.harness.compaction import driver, window
from agent6.providers import anthropic_parse, provider_types
from tests.unit import test_anthropic_compaction as captured
from tests.unit import test_compaction_writers as fixtures
from tests.unit import test_harness


@pytest.mark.parametrize(
    "backend,model,enabled,writer,session",
    [
        ("anthropic", "claude-haiku-5-5", True, True, False),
        ("anthropic", "claude-haiku-5-5", False, False, False),
        ("anthropic", "claude-haiku-4-5", True, False, False),
        ("claude_code", "m", False, False, True),
        ("openai", "m", False, False, False),
        ("chatgpt", "m", False, False, False),
        ("responses", "m", False, False, False),
    ],
)
def test_route_facts_select_the_strategy(
    backend: str, model: str, enabled: bool, writer: bool, session: bool
) -> None:
    entry: dict[str, Any] = {"backend": backend}
    if backend == "anthropic":
        entry["server_compaction"] = enabled
    if backend == "responses":
        entry["base_url"] = "https://example.invalid"
    cfg = config.Config.model_validate({"providers": {"route": entry}})
    facts = role_providers.endpoint_compaction(cfg, config.RoleModel(provider="route", model=model))
    assert facts == provider_types.EndpointCompaction(writer, writer, session)
    harness = test_harness._wf(compaction=window.CompactionSettings(endpoint=facts))
    assert isinstance(harness.compactor, driver.Endpoint if writer else driver.Tiered)


def test_size_summary_uses_all_messages_then_repeats_task_and_pins() -> None:
    response = anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    worker, helper = fixtures.Writer(response), fixtures.Writer(fixtures.response())
    events: list[Any] = []
    turn = fixtures.context(
        worker, helper, events, summariser="helper", pins=("a pin",), task="exact task"
    )
    turn = dataclasses.replace(turn, compact_requested=lambda: "the tests")
    before = turn.conversation.to_wire()
    assert driver.Endpoint().compact(turn)
    assert worker.calls[0]["messages"] == before
    instruction = worker.calls[0]["compaction"].instructions
    assert all(s in instruction for s in ("the tests", "exact task", "a pin"))
    assert not helper.calls
    assert turn.conversation.to_wire()[0] == {
        "role": "assistant",
        "content": [response.summary_block],
    }
    assert all(text in str(turn.conversation.to_wire()[1:]) for text in ("exact task", "a pin"))
    assert events[-1].writer == "endpoint" and events[-1].reason == "requested"


def test_idle_adoption_keeps_the_waiting_thinking_and_its_result_before_the_notice() -> None:
    sample = captured.SAMPLES["adopt-strict"]["messages"]
    response = anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    worker, helper = fixtures.Writer(response), fixtures.Writer(fixtures.response())
    turn = fixtures.context(worker, helper, [], idle=window.IdleCompaction("p", 1, 0))
    previous = turn.conversation.to_wire()
    pending = turn.conversation.assistant(sample[1]["content"])
    deadline = driver.Endpoint().idle(turn, time.monotonic() - 2)
    assert deadline and deadline.callback()
    assert worker.calls[0]["messages"] == previous
    assert turn.conversation.to_wire() == sample[:2]
    call = pending.tool_uses[0]
    turn.conversation.results(
        [loop_conversation.ToolResultItem(call.id, sample[2]["content"][0]["content"], call)]
    )
    after = turn.conversation.to_wire()
    assert after[:2] == sample[:2]
    assert after[2]["content"][0] == {
        k: v for k, v in sample[2]["content"][0].items() if k != "cache_control"
    }
    assert "task" in str(after[2:]) and not helper.calls
    assert loop_conversation.Conversation.from_wire(after).to_wire() == after


@pytest.mark.parametrize(
    "reason",
    [
        "max_tokens",
        "model_context_window_exceeded",
        "refusal",
        "tool_use",
        "end_turn",
        "compaction",
    ],
)
def test_no_summary_block_records_the_reason_and_tries_only_the_helper(reason: str) -> None:
    worker = fixtures.Writer(fixtures.response("", stop_reason=reason))
    helper = fixtures.Writer(fixtures.response(""))
    events: list[Any] = []
    turn = fixtures.context(worker, helper, events)
    before = turn.conversation.to_wire()
    assert not driver.Endpoint().compact(turn)
    assert len(worker.calls) == len(helper.calls) == 1 and before == turn.conversation.to_wire()
    failure = next(e for e in events if e.TYPE == "compact.summarise.failed")
    assert failure.writer == "endpoint" and reason in failure.error


def test_long_summary_instruction_keeps_the_focus_and_restates_the_whole_task() -> None:
    response = anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    worker = fixtures.Writer(response)
    task = "x" * 16384
    turn = fixtures.context(worker, worker, [], task=task, pins=("y" * 16384,))
    turn = dataclasses.replace(turn, compact_requested=lambda: "important focus")
    assert driver.Endpoint().compact(turn)
    instruction = worker.calls[0]["compaction"].instructions
    assert len(instruction) <= 16384 and "important focus" in instruction
    assert task in str(turn.conversation.to_wire()[-1])


def test_size_repeats_the_successful_worker_request_before_adopting_its_reply() -> None:
    """The size summary repeats the worker's messages without the clearing edit."""
    from agent6.harness import loop_state

    calls: list[dict[str, Any]] = []

    class Worker:
        def call(self, **kw: Any) -> provider_types.ProviderResponse:
            calls.append(kw)
            sample = (
                "compact" if isinstance(kw.get("compaction"), provider_types.Summarise) else "turn3"
            )
            return anthropic_parse.parse_response(captured.SAMPLES[sample]["response"])

    worker = Worker()
    conv = fixtures.history()
    settings = fixtures.context(
        worker, worker, [], endpoint=provider_types.EndpointCompaction(True, True)
    ).settings
    wf = test_harness._wf(provider=worker, compaction=settings)
    state = loop_state.LoopState("original task", 1, system="sys")
    got = wf._turn_provider_call(
        "sys", conv, [], state, turn=2, ctx=wf._turn_context(state, turn=2, execution_start=1)
    )
    assert isinstance(got, provider_types.ProviderResponse)
    assert len(calls) == 2 and isinstance(calls[0]["compaction"], provider_types.ClearToolResults)
    assert calls[0]["compaction"].exclude_tools == ("ask_user",)
    assert isinstance(calls[1]["compaction"], provider_types.Summarise)
    assert all(calls[0][key] == calls[1][key] for key in ("system", "tools", "messages"))
    assistant = conv.assistant(got.raw["content"])
    tu = assistant.tool_uses[0]
    conv.results([loop_conversation.ToolResultItem(tu.id, "result", tu)])
    assert conv.to_wire()[1]["content"] == got.raw["content"]
    assert "original task" in str(conv.to_wire()[3:])


def test_idle_summary_only_covers_the_request_sent_before_parking() -> None:
    worker = fixtures.Writer(
        anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    )
    turn = fixtures.context(worker, worker, [], idle=window.IdleCompaction("p", 1, 0))
    before = turn.conversation.to_wire()
    turn.conversation.assistant([{"type": "text", "text": "Answer before parking."}])
    alarm = driver.Endpoint().idle(turn, time.monotonic())
    turn.conversation.notice("A gate notice before the wait.")
    assert alarm and alarm.callback()
    assert worker.calls[0]["messages"] == before
    assert "Answer before parking." in str(turn.conversation.to_wire()[1])
    assert "A gate notice before the wait." in str(turn.conversation.to_wire()[2])


def test_idle_floor_counts_the_notice_waiting_for_a_tool_result() -> None:
    worker = fixtures.Writer(
        anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    )
    turn = fixtures.context(
        worker, worker, [], task="x" * 12000, idle=window.IdleCompaction("p", 1, 0)
    )
    pending = turn.conversation.assistant(
        captured.SAMPLES["adopt-strict"]["messages"][1]["content"]
    )
    compactor = driver.Endpoint()
    alarm = compactor.idle(turn, time.monotonic())
    assert alarm and alarm.callback()
    use = pending.tool_uses[0]
    turn.conversation.results([loop_conversation.ToolResultItem(use.id, "ok", use)])
    assert not compactor.compact(turn)
    assert len(worker.calls) == 1


def test_quiet_reply_to_a_summarised_request_never_enters_the_next_request() -> None:
    conv = fixtures.history()
    block = captured.SAMPLES["compact"]["response"]["content"][0]
    conv.adopt(block, [], "task and pins", awaiting_reply=True)
    conv.assistant([])
    conv.pop_quiet_assistant()
    assert conv.to_wire() == [
        {"role": "assistant", "content": [block]},
        {"role": "user", "content": [{"type": "text", "text": "task and pins"}]},
    ]


def test_helper_restart_after_an_endpoint_block_reads_the_prior_summary() -> None:
    block = captured.SAMPLES["compact"]["response"]["content"][0]
    worker = fixtures.Writer(fixtures.response("", stop_reason="refusal"))
    helper = fixtures.Writer(fixtures.response("Helper summary."))
    turn = fixtures.context(worker, helper, [], task="verbatim task")
    turn.conversation.adopt(block, [], turn.endpoint_notice())
    turn.conversation.assistant([{"type": "text", "text": "a reply"}])
    turn.conversation.notice("Continue.")
    assert driver.Endpoint().compact(dataclasses.replace(turn, compact_requested=lambda: ""))
    assert len(worker.calls) == len(helper.calls) == 1
    assert block["content"] in helper.calls[0]["messages"][0]["content"]
    after = turn.conversation.to_wire()
    assert after[0]["content"][0]["text"] == "verbatim task" and "Helper summary." in str(after[1])
    assert "compaction" not in after[0]["content"][0]
