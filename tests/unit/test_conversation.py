# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Conversation type: wire round-trip fidelity, structural pair safety,
and the rolling cache-mark semantics (ported from the _cache module tests)."""

from __future__ import annotations

from typing import Any

import pytest

from agent6.workflows._conversation import (
    AssistantTurn,
    Conversation,
    Notice,
    ToolResultItem,
    ToolUse,
    UserTurn,
)


def _tool_use_block(tid: str, name: str = "read_file", **inp: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _result(conv: Conversation, *contents: str) -> None:
    """Answer the pending tool_use turn with one result per content string."""
    last = conv.turns[-1]
    assert isinstance(last, AssistantTurn)
    conv.results(
        [
            ToolResultItem(tool_use_id=tu.id, content=c, for_call=tu)
            for tu, c in zip(last.tool_uses, contents, strict=True)
        ]
    )


def _marked(wire: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [
        (mi, bi)
        for mi, m in enumerate(wire)
        for bi, b in enumerate(m["content"])
        if "cache_control" in b
    ]


# --- wire round-trip -------------------------------------------------------


def _rich_conversation() -> Conversation:
    conv = Conversation()
    conv.notice("TASK:\nfix it\n\nBegin.")
    conv.assistant(
        [
            {"type": "thinking", "thinking": "hm", "signature": "sig=="},
            {"type": "text", "text": "reading"},
            _tool_use_block("t1", "read_file", path="a.py"),
            _tool_use_block("t2", "run_verify_command"),
        ]
    )
    last = conv.turns[-1]
    assert isinstance(last, AssistantTurn)
    conv.results(
        [
            ToolResultItem(tool_use_id="t1", content="A" * 40, for_call=last.tool_uses[0]),
            Notice("[harness] interleaved notice"),
            ToolResultItem(
                tool_use_id="t2", content='{"returncode": 1}', for_call=last.tool_uses[1]
            ),
        ]
    )
    conv.notice("OPERATOR STEERING: focus")
    conv.roll_cache_marks()
    return conv


def test_wire_round_trip_is_byte_identical() -> None:
    import json

    wire = _rich_conversation().to_wire()
    again = Conversation.from_wire(wire).to_wire()
    assert json.dumps(again, ensure_ascii=False) == json.dumps(wire, ensure_ascii=False)


def test_from_wire_pairs_results_to_their_calls() -> None:
    conv = Conversation.from_wire(_rich_conversation().to_wire())
    results_turn = conv.turns[2]
    assert isinstance(results_turn, UserTurn)
    items = [it for it in results_turn.items if isinstance(it, ToolResultItem)]
    assert [it.for_call.name for it in items] == ["read_file", "run_verify_command"]
    assert items[0].for_call.input == {"path": "a.py"}
    assert isinstance(results_turn.items[1], Notice)  # the interleaved notice survives


def test_assistant_raw_blocks_pass_through_verbatim() -> None:
    conv = Conversation()
    conv.notice("t")
    exotic = [{"type": "server_tool_use", "weird": {"nested": [1, 2]}}, {"not": "a block"}]
    conv.assistant(exotic)
    assert conv.to_wire()[1]["content"] == exotic
    assert Conversation.from_wire(conv.to_wire()).to_wire()[1]["content"] == exotic


def test_trailing_tool_use_turn_is_accepted() -> None:
    # A crash between the assistant append and its results is transient-legal
    # in memory; from_wire mirrors that (snapshots are never written there).
    conv = Conversation()
    conv.notice("t")
    conv.assistant([_tool_use_block("t1")])
    assert len(Conversation.from_wire(conv.to_wire())) == 2


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([{"role": "user", "content": "a plain string"}], "not a block list"),
        ([{"role": "user", "content": [{"type": "image", "data": "x"}]}], "unsupported type"),
        ([{"role": "user", "content": [{"type": "text", "text": "x", "extra": 1}]}], "text block"),
        ([{"role": "user", "content": [{"type": "text", "text": 7}]}], "text block"),
        ([{"role": "system", "content": []}], "role 'system'"),
        ([{"role": "user", "content": [], "id": 3}], "role/content"),
        (["nope"], "role/content"),
        (
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "x", "cache_control": {"type": "other"}}],
                }
            ],
            "non-ephemeral",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "x"}],
                }
            ],
            "out of order",
        ),
        (
            [
                {"role": "assistant", "content": [_tool_use_block("t1")]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "x"}],
                },
            ],
            "out of order",
        ),
        (
            [
                {"role": "assistant", "content": [_tool_use_block("t1")]},
                {"role": "user", "content": [{"type": "text", "text": "no results"}]},
            ],
            "unanswered",
        ),
        (
            [
                {"role": "assistant", "content": [_tool_use_block("t1"), _tool_use_block("t2")]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}],
                },
            ],
            "do not answer",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": 5}],
                }
            ],
            "tool_result block",
        ),
    ],
)
def test_from_wire_rejects_shapes_the_loop_never_writes(
    messages: list[dict[str, Any]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Conversation.from_wire(messages)


# --- structural pair safety ------------------------------------------------


def test_appends_after_an_unanswered_tool_use_raise() -> None:
    conv = Conversation()
    conv.notice("t")
    conv.assistant([_tool_use_block("t1")])
    with pytest.raises(ValueError, match="unanswered"):
        conv.notice("nudge")
    with pytest.raises(ValueError, match="unanswered"):
        conv.assistant([{"type": "text", "text": "hi"}])
    _result(conv, "ok")  # answering unblocks
    conv.notice("nudge")


def test_results_must_cover_the_call_ids_in_order() -> None:
    conv = Conversation()
    conv.notice("t")
    turn = conv.assistant([_tool_use_block("t1"), _tool_use_block("t2")])
    partial = [ToolResultItem(tool_use_id="t1", content="x", for_call=turn.tool_uses[0])]
    with pytest.raises(ValueError, match="do not answer"):
        conv.results(partial)
    reordered = [
        ToolResultItem(tool_use_id="t2", content="y", for_call=turn.tool_uses[1]),
        ToolResultItem(tool_use_id="t1", content="x", for_call=turn.tool_uses[0]),
    ]
    with pytest.raises(ValueError, match="do not answer"):
        conv.results(reordered)


def test_results_without_a_pending_call_raise() -> None:
    conv = Conversation()
    conv.notice("t")
    orphan = ToolResultItem(tool_use_id="t1", content="x", for_call=ToolUse("t1", "grep", {}))
    with pytest.raises(ValueError, match="do not answer"):
        conv.results([orphan])


def test_pop_quiet_assistant_only_drops_dead_turns() -> None:
    conv = Conversation()
    conv.notice("t")
    conv.assistant([{"type": "thinking", "thinking": "spent the whole budget"}])
    conv.pop_quiet_assistant()
    assert len(conv) == 1  # thinking-only turn dropped
    conv.assistant([{"type": "text", "text": "real answer"}])
    conv.pop_quiet_assistant()
    assert len(conv) == 2  # substantive turn kept
    conv.pop_quiet_assistant()
    assert len(conv) == 2  # non-assistant tails are never popped twice


def test_set_result_content_rewrites_in_place() -> None:
    conv = Conversation()
    conv.notice("t")
    conv.assistant([_tool_use_block("t1", "read_file", path="a.py")])
    _result(conv, "B" * 100)
    conv.set_result_content(2, 0, "<elided>")
    wire = conv.to_wire()
    assert wire[2]["content"][0]["content"] == "<elided>"
    assert wire[2]["content"][0]["tool_use_id"] == "t1"
    turn = conv.turns[2]
    assert not isinstance(turn, AssistantTurn)
    item = turn.items[0]
    assert isinstance(item, ToolResultItem) and item.for_call.name == "read_file"


def test_restart_keeps_the_first_turn_and_its_marks() -> None:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()  # marks the task turn
    conv.assistant([_tool_use_block("t1")])
    _result(conv, "x")
    conv.restart("[restart] summary")
    wire = conv.to_wire()
    assert [m["content"][0].get("text", m["content"][0].get("type")) for m in wire] == [
        "TASK",
        "[restart] summary",
    ]
    assert _marked(wire) == [(0, 0)]  # the kept turn's mark survives; the rest are gone


# --- rolling cache marks (ported from the _cache module tests) --------------


def test_first_roll_marks_the_initial_message() -> None:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()
    wire = conv.to_wire()
    assert _marked(wire) == [(0, 0)]
    assert wire[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_roll_keeps_previous_position_and_marks_new_tail() -> None:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()  # call 1
    conv.assistant([{"type": "text", "text": "thinking"}, _tool_use_block("t1")])
    _result(conv, "ok")
    conv.roll_cache_marks()  # call 2
    assert _marked(conv.to_wire()) == [(0, 0), (2, 0)]


def test_third_roll_unmarks_the_oldest() -> None:
    conv = Conversation()
    conv.notice("TASK")
    for i in range(1, 4):
        conv.roll_cache_marks()
        conv.assistant([_tool_use_block(f"t{i}")])
        _result(conv, "ok")
    conv.roll_cache_marks()
    wire = conv.to_wire()
    assert _marked(wire) == [(4, 0), (6, 0)]  # the two newest user tails
    assert "cache_control" not in wire[0]["content"][0]


def test_nudges_between_calls_do_not_lose_the_previous_position() -> None:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()  # call 1 marks the task
    conv.assistant([_tool_use_block("t1")])
    _result(conv, "ok")
    conv.notice("[focus banner]")
    conv.notice("[budget nudge]")
    conv.roll_cache_marks()  # call 2
    assert _marked(conv.to_wire()) == [(0, 0), (4, 0)]


def test_roll_is_idempotent_without_new_turns() -> None:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()
    conv.assistant([_tool_use_block("t1")])
    _result(conv, "ok")
    conv.roll_cache_marks()
    before = _marked(conv.to_wire())
    conv.roll_cache_marks()  # crash-resume re-issues the same call
    assert _marked(conv.to_wire()) == before


def test_roll_survives_the_wire_round_trip() -> None:
    # Marks persist in snapshots as cache_control keys; a resumed conversation
    # must keep rolling from the same positions.
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()
    conv.assistant([_tool_use_block("t1")])
    _result(conv, "ok")
    conv.roll_cache_marks()
    resumed = Conversation.from_wire(conv.to_wire())
    resumed.roll_cache_marks()  # idempotent on the restored positions
    assert _marked(resumed.to_wire()) == _marked(conv.to_wire())


def test_roll_skips_a_trailing_assistant_turn() -> None:
    # The loop always rolls with a user tail; a (transient) trailing assistant
    # turn is never stamped -- the newest user block takes the mark.
    conv = Conversation()
    conv.notice("TASK")
    conv.assistant([{"type": "thinking", "thinking": "..."}, _tool_use_block("t1")])
    conv.roll_cache_marks()
    assert _marked(conv.to_wire()) == [(0, 0)]


def test_restart_then_roll_starts_a_fresh_pair() -> None:
    conv = Conversation()
    conv.notice("TASK")
    conv.roll_cache_marks()
    conv.assistant([_tool_use_block("t1")])
    _result(conv, "ok")
    conv.roll_cache_marks()
    conv.restart("[context restart] summary")
    conv.roll_cache_marks()
    assert _marked(conv.to_wire()) == [(0, 0), (1, 0)]
