# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Local restart selection at the conversation and provider boundaries."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from agent6.harness import loop_conversation as history
from agent6.harness.compaction import driver
from agent6.providers import anthropic, claude_code_wire, openai_messages, responses_codec
from tests.unit import test_compaction_writers as writers


def _reply(conv: history.Conversation, text: str) -> None:
    conv.assistant([{"type": "text", "text": text}])


def _restart(conv: history.Conversation, *, idle: bool = False, **settings: int) -> str:
    helper = writers.Writer(writers.response("Ready."))
    ctx = writers.context(
        helper, helper, [], conversation=conv, pins=("Standing rule.",), summariser="helper"
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
    from tests.unit import test_harness

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
