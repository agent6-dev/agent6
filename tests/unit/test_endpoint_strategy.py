"""Exercise endpoint adoption, its fallback and the model-dependent strategy choice."""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any

import pytest

from agent6 import config
from agent6.app import role_providers
from agent6.harness import loop_conversation, loop_resume, loop_snapshot, loop_state
from agent6.harness.compaction import driver, window
from agent6.providers import anthropic_parse, provider_types
from tests.unit import test_anthropic_compaction as captured
from tests.unit import test_compaction_writers as fixtures
from tests.unit import test_harness
from tests.unit import test_restart_retention as local


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


def _with_reads(
    conv: loop_conversation.Conversation, count: int, *, name: str = "read_file", size: int = 400
) -> loop_conversation.Conversation:
    """Append tool exchanges, one call and one result each."""
    start = sum(isinstance(m, loop_conversation.AssistantMessage) for m in conv.messages)
    for n in range(start, start + count):
        use = conv.assistant(
            [{"type": "tool_use", "id": f"use{n}", "name": name, "input": {"n": n}}]
        ).tool_uses[0]
        conv.results([loop_conversation.ToolResultItem(use.id, "r" * size, use)])
    return conv


def _reads(count: int, **kw: Any) -> loop_conversation.Conversation:
    conv = loop_conversation.Conversation()
    conv.notice("task", origin="operator")
    return _with_reads(conv, count, **kw)


def test_clearing_moves_in_steps_and_holds_between_them() -> None:
    """Pin the clearing point: a point that moved every request would empty the cache each time."""
    from agent6.harness.compaction import rules

    conv = _reads(5)
    assert rules.clearing(conv, 2000, ()) == ((), None)
    _with_reads(conv, 1)
    # 2,400 characters pass the limit of 2,000: the oldest go until 1,000 or less stay.
    cleared, edit = rules.clearing(conv, 2000, ())
    assert cleared == ("use0", "use1", "use2", "use3")
    assert edit == provider_types.ClearToolResults(2, ("ask_user",))
    _with_reads(conv, 2)
    again, edit = rules.clearing(conv, 2000, cleared)
    assert again == cleared and edit == provider_types.ClearToolResults(4, ("ask_user",))
    _with_reads(conv, 2)
    moved, edit = rules.clearing(conv, 2000, cleared)
    assert moved == (*cleared, "use4", "use5", "use6", "use7")
    assert edit == provider_types.ClearToolResults(2, ("ask_user",))
    # A restart replaces the history, and the point starts again.
    assert rules.clearing(_reads(3), 2000, cleared) == ((), None)


def test_clearing_keeps_operator_answers_and_every_unread_result() -> None:
    """Pin the floor: with one kept tool use, parallel results were cleared before being read."""
    from agent6.harness.compaction import rules

    conv = _with_reads(_with_reads(_reads(3), 1, name="ask_user"), 1)
    uses = conv.assistant(
        [
            {"type": "tool_use", "id": f"p{n}", "name": "read_file", "input": {"n": n}}
            for n in range(3)
        ]
    ).tool_uses
    conv.results([loop_conversation.ToolResultItem(u.id, "r" * 400, u) for u in uses])
    cleared, edit = rules.clearing(conv, 100, ())
    # The operator's answer is not cleared and not counted; the three unread results stay.
    assert cleared == ("use0", "use1", "use2", "use4")
    assert edit == provider_types.ClearToolResults(3, ("ask_user",))


def test_endpoint_strategy_journals_each_clearing_step_once() -> None:
    worker = fixtures.Writer(fixtures.response())
    events: list[Any] = []
    turn = fixtures.context(
        worker,
        worker,
        events,
        conversation=_reads(6),
        endpoint=provider_types.EndpointCompaction(True, True),
    )
    turn = dataclasses.replace(
        turn, settings=dataclasses.replace(turn.settings, drop_at_chars=2000)
    )
    compactor = driver.Endpoint()
    edit = compactor.clearing(turn)
    assert edit == compactor.clearing(turn) == provider_types.ClearToolResults(2, ("ask_user",))
    dropped = [e for e in events if e.TYPE == "compact.dropped"]
    assert [(e.count, list(e.calls)) for e in dropped] == [(4, ["use0", "use1", "use2", "use3"])]
    off = dataclasses.replace(
        turn,
        settings=dataclasses.replace(turn.settings, endpoint=provider_types.EndpointCompaction()),
    )
    assert driver.Endpoint().clearing(off) is None


def test_a_route_without_the_endpoint_reads_an_adopted_summary_as_text() -> None:
    """Pin the cross-route resume: other backends skipped the block and lost the summary."""
    from agent6.harness import loop_state
    from agent6.prompts import compaction as prompts

    block = captured.SAMPLES["compact"]["response"]["content"][0]
    conv = loop_conversation.Conversation()
    conv.adopt(block, [], "task and pins")
    wf = test_harness._wf(provider=fixtures.Writer(fixtures.response()))
    state = loop_state.LoopState("task", 1, system="sys")
    wire = wf._turn_pre_call(
        conversation=conv,
        state=state,
        ctx=wf._turn_context(state, turn=2, execution_start=1),
        tools=[],
    )
    assert isinstance(wire, list) and wire[0]["role"] == "user"
    assert prompts.progress_summary_from_notice(wire[0]["content"][0]["text"]) == block["content"]
    assert "compaction" not in str([b.get("type") for m in wire for b in m["content"]])


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
    conv = _with_reads(fixtures.history(), 4)
    conv.notice("A notice after the last cache mark.")
    settings = fixtures.context(
        worker, worker, [], endpoint=provider_types.EndpointCompaction(True, True)
    ).settings
    wf = test_harness._wf(provider=worker, compaction=settings)
    state = loop_state.LoopState("original task", 1, system="sys")
    got = wf._turn_provider_call(
        "sys", conv, [], state, turn=2, ctx=wf._turn_context(state, turn=2, execution_start=1)
    )
    assert isinstance(got, provider_types.ProviderResponse)
    assert len(calls) == 2
    assert calls[0]["compaction"] == provider_types.ClearToolResults(2, ("ask_user",))
    assert isinstance(calls[1]["compaction"], provider_types.Summarise)
    assert all(calls[0][key] == calls[1][key] for key in ("system", "tools", "messages"))
    # The cache mark moves to the newest user block on this route too.
    assert "cache_control" in calls[0]["messages"][-1]["content"][-1]
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


@pytest.mark.parametrize("source", ["operator", "model", "harness"])
def test_restored_signed_summary_does_not_change_the_task_author(
    source: loop_conversation.TaskSource, tmp_path: pathlib.Path
) -> None:
    """A helper fallback reconstructs the opening without promoting the composed task."""
    snap = tmp_path / "loop_state.json"
    original = "Rendered context.\nAssignment."
    words = "Assignment."
    signed = anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    helper = fixtures.Writer(fixtures.response("Ready."))
    settings = window.CompactionSettings(
        endpoint=provider_types.EndpointCompaction(True, True),
        helper=helper,
        helper_context_tokens=200000,
        summarise_at_chars=10,
    )
    wf = test_harness._wf(
        root=tmp_path,
        provider=fixtures.Writer(signed),
        compaction=settings,
        task_source=source,
        resume_state_path=snap,
    )
    system, conv, _ = wf._open_task(original, words)
    state = loop_state.LoopState(original, 0, system=system, operator_task=words)
    ctx = wf.compaction_context(conv, state, [])
    assert driver.Endpoint().summarise_and_restart(ctx)
    assert conv.to_wire()[0]["content"] == [signed.summary_block]
    _with_reads(conv, 1)
    loop_resume.save_resume_snapshot(wf, state, conv.to_wire(snapshot=True), next_turn=3)

    resumed = test_harness._wf(
        root=tmp_path,
        provider=fixtures.Writer(fixtures.response("")),
        compaction=settings,
        resume_state_path=snap,
        max_turns=1,
    )
    assert resumed.resume().reason == "max_turns"
    assert len(helper.calls) == 1
    conv = loop_conversation.Conversation.from_wire(
        loop_snapshot.load_session_snapshot(snap).messages
    )
    assert conv.to_wire()[0]["content"][0]["text"] == original
    local._reply(conv, "Done.")
    local._restart(conv, keep_recent_chars=500, keep_recent_turns=0)
    notice = local._notice(conv)
    assert original not in notice
    assert ("Operator message:\nAssignment." in notice) is (source == "operator")


def test_rejected_sent_image_stays_in_the_prefix() -> None:
    """Refuse a retry that would replace an image before the last reply."""
    from unittest import mock

    from agent6 import kinds, media
    from agent6.harness import provider_call

    conv = fixtures.history()
    call = conv.assistant(
        [{"type": "tool_use", "name": "read_file", "id": "image", "input": {}}]
    ).tool_uses[0]
    image = kinds.MediaAttachment(b"x", "image/png", "page.png")
    conv.results([loop_conversation.ToolResultItem(call.id, "image", call, image)])
    conv.assistant(captured.SAMPLES["adopt-strict"]["messages"][1]["content"])
    last = conv.messages[-1]
    assert isinstance(last, loop_conversation.AssistantMessage)
    tool = last.tool_uses[0]
    conv.results([loop_conversation.ToolResultItem(tool.id, "ok", tool)])
    before = conv.to_wire()
    failure = provider_types.MediaInputError(
        modality="image", reason="invalid", call_ids=(call.id,)
    )
    send = mock.Mock(side_effect=failure)
    recovery = provider_call.MediaRecovery(media.InputCapabilities(), append_only=True)
    with pytest.raises(provider_types.MediaInputError):
        provider_call.call_with_media_recovery(send, conv, recovery, emit=None, changed=None)
    assert send.call_count == 1 and conv.to_wire() == before


def test_summary_retry_keeps_the_call_id_and_worker_clock() -> None:
    """Keep summary attempts off the worker clock that sets its idle deadline."""
    from unittest import mock

    from agent6 import kinds

    provider = mock.Mock()
    response = anthropic_parse.parse_response(captured.SAMPLES["compact"]["response"])
    provider.call.side_effect = [
        provider_types.ProviderError("try again", status_code=503),
        response,
    ]
    caller = test_harness._caller(provider, retry_count=1)
    caller.counter.request_started = 42
    option = provider_types.Summarise("Keep the task.")
    assert (
        caller.call("s", [], [], 4096, call_context=kinds.CallContext("summary"), compaction=option)
        is response
    )
    assert [c.kwargs["context"].attempt for c in provider.call.call_args_list] == [1, 2]
    assert all(
        c.kwargs["compaction"] is option
        and c.kwargs["context"].call_context.model_call_id == "summary"
        for c in provider.call.call_args_list
    )
    assert caller.counter.request_started == 42


def test_endpoint_fallback_only_sends_the_question_to_the_helper() -> None:
    worker, helper = (
        fixtures.Writer(fixtures.response("")),
        fixtures.Writer(fixtures.response("ok")),
    )
    ctx = fixtures.context(worker, helper, [])
    c = ctx.conversation
    call = c.assistant(
        [
            {
                "type": "tool_use",
                "name": "ask_user",
                "id": "q",
                "input": {"question": "Which parser?"},
            }
        ]
    ).tool_uses[0]
    c.results(
        [
            loop_conversation.ToolResultItem(
                call.id, '{"status":"answered","answers":["private parser reference"]}', call
            )
        ]
    )
    assert driver.Endpoint().summarise_and_restart(ctx)
    transcript = helper.calls[0]["messages"][0]["content"]
    assert "Which parser?" in transcript and "private parser reference" not in transcript
