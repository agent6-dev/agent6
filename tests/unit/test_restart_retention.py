# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Local restart selection at the conversation and provider boundaries."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import directive
from agent6.context import skills
from agent6.harness import loop_conversation as history
from agent6.harness import loop_resume, loop_snapshot, tool_effects
from agent6.harness.compaction import driver, window
from agent6.prompts import compaction
from agent6.providers import (
    anthropic,
    claude_code_wire,
    openai_messages,
    provider_types,
    responses_codec,
)
from agent6.tools import results
from tests.unit import test_compaction_writers as writers
from tests.unit import test_harness


def _reply(conv: history.Conversation, text: str) -> None:
    conv.assistant([{"type": "text", "text": text}])


def _restart(
    conv: history.Conversation, *, idle: bool = False, task: str = "", **settings: int
) -> str:
    helper = writers.Writer(writers.response("Ready."))
    ctx = writers.context(
        helper,
        helper,
        [],
        conversation=conv,
        pins=("Standing rule.",),
        summariser="helper",
        task=task,
    )
    ctx = dataclasses.replace(ctx, settings=dataclasses.replace(ctx.settings, **settings))
    assert driver.Tiered().summarise_and_restart(ctx, idle_gap_s=2 if idle else None)
    return helper.calls[0]["messages"][0]["content"]


def _notice(conv: history.Conversation) -> str:
    return "\n\n".join(b["text"] for b in conv.to_wire()[0]["content"])


@pytest.mark.parametrize(
    "operators,turns,chars,quoted,tail",
    [
        (-1, -1, 80_000, [], ["goal", "A", "one", "B", "two", "C"]),
        (0, 1, 80_000, [], ["two", "C"]),
        (1, 0, 80_000, ["two"], []),
        (-1, 0, 80_000, ["goal", "one", "two"], []),
        (-1, -1, 0, [], []),
        (-1, -1, 11, ["goal", "one"], ["two", "C"]),
        (-1, -1, 12, ["goal"], ["one", "B", "two", "C"]),
        (0, -1, 1, [], ["two", "C"]),
    ],
)
def test_operator_reservations_and_whole_turns_share_one_cap(
    operators: int, turns: int, chars: int, quoted: list[str], tail: list[str]
) -> None:
    conv = history.Conversation()
    for steer, reply in (("goal", "A"), ("one", "B"), ("two", "C")):
        conv.notice(steer, origin="operator")
        _reply(conv, reply)
    _restart(
        conv, keep_recent_chars=chars, keep_recent_turns=turns, keep_operator_messages=operators
    )
    notice = _notice(conv)
    assert "[harness context restart]" in notice
    words = [notice.index(f"Operator message:\n{text}") for text in quoted]
    assert words == sorted(words) and notice.count("Operator message:") == len(quoted)
    assert not words or words[-1] < notice.index("Standing rule.") < notice.index(
        "PROGRESS SUMMARY:"
    )
    assert [m["content"][0]["text"] for m in conv.to_wire()[1:]] == tail


_TASK = "Execute the prepared plan: cache\n\n1. Add CacheV2.\n2. Flip the readers."
_STATED = f"TASK (the goal, verbatim):\n{_TASK}"


def _opened(source: history.TaskSource = "operator") -> history.Conversation:
    """Open a run of `_TASK`, a plan under its typed first line, then take one steer."""
    conv = test_harness._wf(task_source=source)._open_task(_TASK)[1]
    _reply(conv, "A")
    conv.notice("steer", origin="operator")
    _reply(conv, "B")
    return conv


@pytest.mark.parametrize("source", ["operator", "model", "harness"])
def test_a_restart_states_the_composed_task_of_every_source_once(
    source: history.TaskSource,
) -> None:
    """A restart kept the typed first line of a plan, and no task from a parent or a machine."""
    conv = _opened(source)
    _restart(conv, task=_TASK, keep_recent_turns=0, keep_recent_chars=80_000)
    notice = _notice(conv)
    assert notice.count(_TASK) == 1 and notice.count("Operator message:") == 1
    assert notice.index(_STATED) < notice.index("Operator message:\nsteer")

    # A kept opening already states the task, and its words are charged once.
    conv = _opened(source)
    _restart(conv, task=_TASK, keep_recent_chars=sum(map(window.message_chars, conv.messages)))
    assert str(conv.to_wire()).count("Flip the readers.") == 1 and len(conv) == 5
    assert "TASK (the goal" not in _notice(conv) and "Operator message:" not in _notice(conv)


@pytest.mark.parametrize(
    "operators,chars,kept",
    [
        (-1, 80_000, True),
        (2, 80_000, True),
        (1, 80_000, False),
        (0, 80_000, False),
        (-1, len(_TASK) + 5, True),
        (-1, len(_TASK) + 4, False),
    ],
)
def test_the_task_is_the_oldest_message_under_both_caps(
    operators: int, chars: int, kept: bool
) -> None:
    conv = _opened()
    _restart(
        conv,
        task=_TASK,
        keep_operator_messages=operators,
        keep_recent_chars=chars,
        keep_recent_turns=0,
    )
    assert (_STATED in _notice(conv)) is kept
    assert ("Operator message:\nsteer" in _notice(conv)) is (operators != 0)


def test_a_task_the_history_lost_returns_from_the_run() -> None:
    """After an endpoint summary cleared the opening, a local restart stated no task."""
    conv = history.Conversation()
    conv.adopt({"type": "compaction", "content": "Signed."}, [], "Continue from the summary.")
    conv.unsign(compaction.unsigned_summary_notice)
    _reply(conv, "A")
    _restart(conv, task=_TASK, keep_recent_turns=0, keep_recent_chars=len(_TASK) - 1)
    assert _TASK not in str(conv.to_wire())
    conv = history.Conversation.from_wire(conv.to_wire(snapshot=True))
    _reply(conv, "B")
    _restart(conv, task=_TASK, keep_recent_turns=0, keep_recent_chars=80_000)
    assert _notice(conv).count(_STATED) == str(conv.to_wire()).count("Flip the readers.") == 1


def _restart_run(wf: Any, conv: history.Conversation, state: Any, **settings: int) -> str:
    """Restart a conversation with the pins and facts of its own run."""
    helper = writers.Writer(writers.response("Ready."))
    wf.compaction = window.CompactionSettings(
        helper=helper, helper_context_tokens=200000, summariser="helper", **settings
    )
    assert driver.Tiered().summarise_and_restart(wf.compaction_context(conv, state, []))
    return _notice(conv)


def test_a_pinned_steer_is_stated_once_by_the_pinned_block() -> None:
    """A `/pin` was quoted as an operator message and again in the pinned block."""
    wf, state = test_harness._wf(), test_harness._state(original_task="")
    conv = history.Conversation()
    conv.notice("first steer", origin="operator")
    wf.steering.operator_steer(
        conv, 1, state, "/pin Never touch vendor/.", heading="OPERATOR STEERING:"
    )
    _reply(conv, "ok")
    # The pin is neither quoted nor counted: the one reserved message is the steer before it.
    notice = _restart_run(wf, conv, state, keep_recent_turns=0, keep_operator_messages=1)
    assert notice.count("Never touch vendor/.") == 1
    assert "PINNED operator instructions (verbatim):\n1. Never touch vendor/." in notice
    assert "Operator message:\nfirst steer" in notice and notice.count("Operator message:") == 1


def test_a_steered_skill_is_quoted_by_name_without_its_text(tmp_path: pathlib.Path) -> None:
    """A skill steer was quoted as `/release v2`, which names no skill to load again."""
    skill = skills.Skill("release", "Use when releasing.", tmp_path, "SKILL-BODY")
    dispatcher = mock.MagicMock(last_verification=None, execution_id="e", verify_withheld=False)
    dispatcher.resolved_skills.return_value = skills.ResolvedSkills((skill,), (), ())
    wf, state = test_harness._wf(dispatcher=dispatcher), test_harness._state(original_task="")
    conv = history.Conversation()
    wf.steering.operator_steer(conv, 1, state, "/release v2", heading="OPERATOR STEERING:")
    assert "SKILL-BODY" in str(conv.to_wire())
    _reply(conv, "ok")
    notice = _restart_run(wf, conv, state, keep_recent_turns=0)
    assert (
        "Operator message:\nApply the operator-installed skill 'release' for the rest of this"
        " run.\nSkill arguments: v2\n\n"
    ) in notice
    assert "SKILL-BODY" not in notice and "/release" not in notice


def test_a_fact_a_kept_notice_states_is_not_repeated_by_the_restart() -> None:
    """A cold restart stated the verify fact and the metric twice: its notice and the resume's."""
    wf, state = test_harness._wf(), test_harness._state(original_task="")
    state.verify.last_ok = False
    wf.metrics.change(directive.parse_metric_spec("minimize -- old"))
    facts = [wf.verify_fact(state), wf.metrics.notice()]
    conv = history.Conversation()
    conv.notice("steer", origin="operator")
    _reply(conv, "ok")
    notice = _restart_run(wf, conv, state, keep_recent_turns=0)
    assert all(fact in notice for fact in facts)
    _reply(conv, "ok")
    # A resume states both before its first request; the restart keeps those unread notices.
    conv.notice(f"[harness verify] {facts[0]}")
    conv.notice(facts[1])
    notice = _restart_run(wf, conv, state, keep_recent_turns=0)
    whole = "\n".join(b["text"] for m in conv.to_wire() for b in m["content"] if "text" in b)
    assert [whole.count(fact) for fact in facts] == [1, 1]
    assert not any(fact in notice for fact in facts) and "VERIFICATION" not in notice


def test_first_turn_that_does_not_fit_ends_replay() -> None:
    conv = history.Conversation()
    for reply in ("a", "X" * 20, "b"):
        _reply(conv, reply)
    _restart(conv, keep_recent_chars=10)
    assert [m["content"][0]["text"] for m in conv.to_wire()[1:]] == ["b"]


def test_dropped_operator_words_reach_helper_and_reserved_answers_are_plain_text() -> None:
    conv = history.Conversation()
    conv.notice("opening detail", origin="operator")
    for i, status in enumerate(("answered", "answered", "pending")):
        call = conv.assistant(
            [{"type": "tool_use", "name": "ask_user", "id": str(i), "input": {}}]
        ).tool_uses[0]
        result = json.dumps(
            {
                "id": str(i),
                "status": status,
                "answers": [f"line {i}\nexact"] if status == "answered" else None,
            }
        )
        conv.results([history.ToolResultItem(call.id, result, call)])
    _reply(conv, "done")
    prompt = _restart(conv, keep_operator_messages=1, keep_recent_turns=0, keep_recent_chars=200)
    assert "opening detail" in prompt and "line 0\nexact" in prompt
    assert "Operator message:\nline 1\nexact" in _notice(conv)
    assert "opening detail" not in _notice(conv) and "line 0" not in _notice(conv)
    assert len(conv) == 1 and "tool_result" not in str(conv.to_wire())


def test_quotes_survive_a_snapshot_and_age_out_on_a_later_restart() -> None:
    conv = history.Conversation()
    for text in ("Original wording A.", "later"):
        conv.notice(text, origin="operator")
        _reply(conv, "done")
    _restart(conv, keep_operator_messages=2, keep_recent_turns=0, keep_recent_chars=100)
    conv = history.Conversation.from_wire(conv.to_wire(snapshot=True))
    assert conv.initiator == "agent"
    conv.notice("newest", origin="operator")
    _reply(conv, "ok")
    prompt = _restart(conv, keep_operator_messages=2, keep_recent_turns=-1, keep_recent_chars=10000)
    notice = _notice(conv)
    assert "Original wording A." in prompt and "Original wording A." not in str(conv.to_wire())
    assert "Operator message:\nlater" in notice
    assert str(conv.to_wire()).count("[harness context restart]") == 1
    assert conv.to_wire()[1]["content"][0]["text"] == "newest"


def _multi() -> history.Conversation:
    conv = history.Conversation()
    conv.notice("task", origin="operator")
    for i in range(3):
        conv.notice(f"input-{i}", origin="operator")
        blocks: list[dict[str, Any]] = [
            {"type": "thinking", "thinking": f"thought-{i}", "signature": f"sig-{i}"},
            {
                "type": "thinking",
                "thinking": "",
                "reasoning_replay": {"route": "r", "item": {"type": "reasoning", "id": f"rs-{i}"}},
            },
        ]
        blocks += [
            {"type": "tool_use", "name": name, "id": f"{name}-{i}", "input": {"n": i}}
            for name in ("left", "right")
        ]
        calls = conv.assistant(blocks).tool_uses
        conv.results(
            [
                *(history.ToolResultItem(t.id, "result " + t.id, t) for t in calls),
                history.Notice(f"next-{i}"),
            ]
        )
    return conv


@pytest.mark.parametrize("thinking", [-1, 1])
def test_thinking_count_applies_at_an_idle_restart(thinking: int) -> None:
    conv = _multi()
    _restart(
        conv, idle=True, keep_recent_turns=2, keep_thinking_turns=thinking, keep_recent_chars=80000
    )
    wire = str(conv.to_wire())
    assert ("thought-1" in wire) is (thinking == -1)
    assert ("rs-1" in wire) is (thinking == -1)
    assert "sig-2" in wire and "rs-2" in wire and "next-0" in wire
    assert "left-0" not in wire


@pytest.mark.parametrize("encoder", ["anthropic", "chat", "responses", "claude_code"])
def test_restart_replays_a_whole_multi_call_turn_on_each_wire(encoder: str) -> None:
    conv = _multi()
    _restart(conv, keep_operator_messages=0, keep_recent_turns=1, keep_recent_chars=80000)
    wire = conv.to_wire()
    if encoder == "anthropic":
        msgs = anthropic.shape_anthropic_messages(wire)
        reply = next(m["content"] for m in msgs if m["role"] == "assistant")
        assert reply[0]["signature"] == "sig-2"
        assert [b["id"] for b in reply if b["type"] == "tool_use"] == ["left-2", "right-2"]
        assert [
            b["tool_use_id"] for m in msgs for b in m["content"] if b["type"] == "tool_result"
        ] == ["left-2", "right-2"]
    elif encoder == "chat":
        msgs = openai_messages.anthropic_to_openai_messages("system", wire)
        i = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        assert [c["id"] for c in msgs[i]["tool_calls"]] == ["left-2", "right-2"]
        assert [m["tool_call_id"] for m in msgs[i + 1 : i + 3]] == ["left-2", "right-2"]
    elif encoder == "responses":
        items = responses_codec.responses_input(wire, route="r")
        assert [
            (it["type"], it.get("call_id", it.get("id"))) for it in items if "role" not in it
        ] == [
            ("reasoning", "rs-2"),
            ("function_call", "left-2"),
            ("function_call", "right-2"),
            ("function_call_output", "left-2"),
            ("function_call_output", "right-2"),
        ]
    else:
        text = claude_code_wire.render_history(wire)[0]["text"]
        assert "[tool_use left]" in text and "[tool_use right]" in text
        assert (
            "tool_result left\nresult left-2" in text
            and "tool_result right\nresult right-2" in text
        )
        assert "rs-2" not in text and "thought-2" not in text
    assert "next-1" in str(wire) and "input-2" in str(wire)
    assert "left-1" not in str(wire) and "next-0" not in str(wire)


def test_zero_caps_keep_unread_calls_results_and_notices() -> None:
    conv = _multi()
    conv.notice("act on this", origin="operator")
    _restart(conv, keep_operator_messages=0, keep_recent_turns=0, keep_recent_chars=0)
    wire = conv.to_wire()
    assert "left-2" in str(wire) and "right-2" in str(wire)
    assert "left-1" not in str(wire) and "act on this" in str(wire[-1])


def test_quote_contains_the_steer_without_the_harness_heading() -> None:
    conv = history.Conversation()
    conv.notice("task", origin="operator")
    harness = test_harness._wf()
    harness.steering.operator_steer(
        conv, 2, test_harness._state(), "Two\nlines.", heading="Operator steer:"
    )
    _reply(conv, "ok")
    conv = history.Conversation.from_wire(conv.to_wire(snapshot=True))
    _restart(conv, keep_operator_messages=1, keep_recent_turns=0, keep_recent_chars=10)
    assert "Operator message:\nTwo\nlines." in _notice(conv)
    assert "Operator steer:" not in _notice(conv)


def test_budget_charges_only_the_thinking_that_will_be_replayed() -> None:
    conv = history.Conversation()
    conv.assistant([{"type": "thinking", "thinking": "x" * 10000}, {"type": "text", "text": "old"}])
    _reply(conv, "new")
    _restart(conv, keep_thinking_turns=1, keep_recent_chars=6)
    assert [m["content"] for m in conv.to_wire()[1:]] == [
        [{"type": "text", "text": w}] for w in ("old", "new")
    ]


def test_stripping_old_thinking_does_not_replay_an_empty_assistant() -> None:
    conv = history.Conversation()
    conv.notice("first input", origin="operator")
    conv.assistant([{"type": "redacted_thinking", "data": "opaque"}])
    _reply(conv, "new")
    _restart(conv, keep_thinking_turns=1, keep_recent_chars=100)
    assert [m["content"] for m in conv.to_wire()[1:]] == [
        [{"type": "text", "text": w}] for w in ("first input", "new")
    ]


def test_operator_reservation_precedes_the_oversize_turn_exception() -> None:
    conv = history.Conversation()
    words = "o" * 31
    conv.notice(words, origin="operator")
    _reply(conv, "old")
    _reply(conv, "r" * 200)
    _restart(conv, keep_recent_chars=100, keep_operator_messages=1)
    assert f"Operator message:\n{words}" in _notice(conv)
    assert conv.to_wire()[1]["content"][0]["text"] == "r" * 200


@pytest.mark.parametrize("operators", [0, 1])
def test_answer_before_tool_capping_reaches_the_local_restart(operators: int) -> None:
    words = "a" * 61000 + " end words"
    dispatcher = mock.MagicMock()
    dispatcher.dispatch.return_value = results.AnswersResult("q", (words,), asked=("Which?",))
    wf = test_harness._wf(dispatcher=dispatcher)
    conv = history.Conversation()
    conv.notice("task", origin="operator")
    resp = test_harness._tool_resp("ask_user", {"questions": [{"question": "Which?"}]})
    turn = test_harness._turn(resp=resp, assistant=conv.assistant(resp.raw["content"]))
    state = test_harness._state()
    wf._turn_dispatch_tools(state, turn, test_harness._ctx(wf, state))
    conv.results(turn.tool_results)
    assert len(conv.to_wire()[-1]["content"][0]["content"].encode()) <= 60000
    _reply(conv, "done")
    conv = history.Conversation.from_wire(conv.to_wire(snapshot=True))
    prompt = _restart(
        conv, keep_operator_messages=operators, keep_recent_turns=0, keep_recent_chars=80000
    )
    assert words in prompt
    assert (f"Operator message:\n{words}" in _notice(conv)) is bool(operators)


def _questions(*calls: tuple[str, str]) -> provider_types.ProviderResponse:
    return responses_codec.parse_output_items(
        [
            {
                "type": "function_call",
                "name": "ask_user",
                "call_id": call_id,
                "arguments": json.dumps({"questions": [{"question": q}]}),
            }
            for call_id, q in calls
        ],
        usage={},
        stop_reason="end_turn",
        label="Responses",
        route="r",
    )


def _answers(
    first: str, second: str, call_ids: tuple[str, str] = ("q1", "q2")
) -> history.Conversation:
    conv = history.Conversation()
    conv.notice("Opening task.", origin="operator")
    reply = _questions((call_ids[0], "Which file?"), (call_ids[1], "Which test?"))
    calls = conv.assistant(reply.raw["content"]).tool_uses
    items: list[history.ToolResultItem] = []
    for call, words in zip(calls, (first, second), strict=True):
        answer = results.AnswersResult(call.id, (words,))
        items.append(
            tool_effects.bounded_result(call, json.dumps(answer.to_wire()), None, None, answer, 400)
        )
    conv.results(items)
    assert sum(item.unshown_answer is not None for item in items) == 1
    return conv


def test_interrupted_turn_keeps_answers_to_repeated_call_ids_in_order(
    tmp_path: pathlib.Path,
) -> None:
    """A capped answer must not replace a later answer with the same provider call ID."""
    first, second = "A" * 600 + " first", "Second answer."
    path = tmp_path / "loop_state.json"
    wf = test_harness._wf(
        root=tmp_path,
        resume_state_path=path,
        compaction=window.CompactionSettings(tool_result_cap_bytes=400),
    )
    state = test_harness._state()
    conv = history.Conversation()
    conv.notice("Opening task.", origin="operator")
    loop_resume.save_resume_snapshot(wf, state, conv.to_wire(snapshot=True), next_turn=1)
    reply = _questions(("same", "First?"), ("same", "Second?"), ("last", "Third?"))
    turn = test_harness._turn(resp=reply, assistant=conv.assistant(reply.raw["content"]))
    dispatches = [
        results.AnswersResult("q1", (first,)),
        results.AnswersResult("q2", (second,)),
        test_harness._Died,
    ]
    with (
        mock.patch.object(wf.dispatcher, "dispatch", side_effect=dispatches),
        pytest.raises(test_harness._Died),
    ):
        wf._turn_dispatch_tools(state, turn, test_harness._ctx(wf, state))
    assert [
        (r.tool_use_id, r.unshown_answer)
        for r in turn.tool_results
        if isinstance(r, history.ToolResultItem)
    ] == [("same", first), ("same", None)]

    snapshot, interrupted = loop_resume.fold_interrupted_turn(
        wf, loop_snapshot.load_session_snapshot(path)
    )
    assert interrupted == ("ask_user",) and snapshot.tool_calls == 2
    conv = history.Conversation.from_wire(snapshot.messages)
    _reply(conv, "Done.")
    prompt = _restart(conv, keep_recent_turns=0, keep_operator_messages=2, keep_recent_chars=2000)
    quotes = [
        p.removeprefix("Operator message:\n")
        for p in _notice(conv).split("\n\n")
        if p.startswith("Operator message:\n")
    ]
    answers = [
        line.removeprefix("[operator answer] ")
        for line in prompt.splitlines()
        if line.startswith("[operator answer] ")
    ]
    assert (quotes, answers) == ([first, second], [first, second])


def test_endpoint_fallback_omits_the_answer_notice_from_a_kept_turn() -> None:
    # Even words shaped like a prior summary are still a tool answer.
    first = compaction.unsigned_summary_notice("a" * 600 + " capped ending")
    conv = _answers(first, "short answer")
    _restart(conv, keep_recent_turns=1, keep_operator_messages=0, keep_recent_chars=2000)
    conv = history.Conversation.from_wire(conv.to_wire(snapshot=True))
    wire = conv.to_wire()
    assert any(b.get("text") == first for m in wire for b in m["content"])
    assert all(
        "capped ending" not in str(b)
        for m in wire
        for b in m["content"]
        if b["type"] == "tool_result"
    )
    conv.notice("Check the parser.", origin="operator")
    worker, helper = (
        writers.Writer(writers.response("")),
        writers.Writer(writers.response("Ready.")),
    )
    ctx = writers.context(worker, helper, [], conversation=conv)
    assert driver.Endpoint().summarise_and_restart(ctx)
    prompt = helper.calls[0]["messages"][0]["content"]
    assert "Which file?" in prompt and "Check the parser." in prompt
    assert "capped ending" not in prompt and "short answer" not in prompt


@pytest.mark.parametrize("operators", [1, 2])
@pytest.mark.parametrize("after_replay", [False, True])
@pytest.mark.parametrize("call_ids", [("q1", "q2"), ("same", "same")], ids=["distinct", "repeated"])
@pytest.mark.parametrize(
    "answers",
    [("A" * 600 + " first", "second answer"), ("first answer", "B" * 600 + " second")],
    ids=["capped-first", "capped-second"],
)
def test_mixed_answers_are_reserved_and_summarised_in_call_order(
    operators: int, after_replay: bool, call_ids: tuple[str, str], answers: tuple[str, str]
) -> None:
    first, second = answers
    conv = _answers(first, second, call_ids)
    if after_replay:
        _restart(conv, keep_recent_turns=1, keep_operator_messages=0, keep_recent_chars=2000)
        conv = history.Conversation.from_wire(conv.to_wire(snapshot=True))
    _reply(conv, "Done.")
    prompt = _restart(
        conv, keep_recent_turns=0, keep_operator_messages=operators, keep_recent_chars=2000
    )
    quote = "Operator message:\n"
    expected = [first, second] if operators == 2 else [second]
    assert _notice(conv).count(quote) == operators and (
        "\n\n".join(quote + w for w in expected)
    ) in _notice(conv)
    assert prompt.count(first) == prompt.count(second) == 1 and prompt.index(first) < prompt.index(
        second
    )
