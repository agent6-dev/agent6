# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rolling prompt-cache breakpoints (workflows._cache) and the provider strip."""

from __future__ import annotations

from typing import Any

from agent6.providers.anthropic import strip_cache_control_messages
from agent6.workflows._cache import roll_cache_breakpoints


def _user(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": list(blocks)}


def _assistant(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": list(blocks)}


def _text(t: str = "x") -> dict[str, Any]:
    return {"type": "text", "text": t}


def _tool_result(tid: str = "t1") -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tid, "content": "ok"}


def _tool_use(tid: str = "t1") -> dict[str, Any]:
    return {"type": "tool_use", "id": tid, "name": "x", "input": {}}


def _marked_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        out.extend(b for b in content if isinstance(b, dict) and "cache_control" in b)
    return out


def test_first_call_marks_the_initial_message() -> None:
    messages = [_user(_text("TASK"))]
    roll_cache_breakpoints(messages)
    marked = _marked_blocks(messages)
    assert marked == [messages[0]["content"][0]]
    assert marked[0]["cache_control"] == {"type": "ephemeral"}


def test_roll_keeps_previous_position_and_marks_new_tail() -> None:
    messages = [_user(_text("TASK"))]
    roll_cache_breakpoints(messages)  # call 1
    messages.append(_assistant(_text("thinking"), _tool_use("t1")))
    messages.append(_user(_tool_result("t1")))
    roll_cache_breakpoints(messages)  # call 2
    marked = _marked_blocks(messages)
    assert len(marked) == 2
    assert marked[0] is messages[0]["content"][0]  # previous call's position
    assert marked[1] is messages[2]["content"][-1]  # new tail


def test_third_roll_unmarks_the_oldest() -> None:
    messages = [_user(_text("TASK"))]
    for i in range(1, 4):  # three iterations of assistant + tool_result
        roll_cache_breakpoints(messages)
        messages.append(_assistant(_tool_use(f"t{i}")))
        messages.append(_user(_tool_result(f"t{i}")))
    roll_cache_breakpoints(messages)
    marked = _marked_blocks(messages)
    assert len(marked) == 2
    # The two newest user tails hold the markers; msg0 is unmarked.
    assert "cache_control" not in messages[0]["content"][0]
    assert marked[0] is messages[-3]["content"][-1]
    assert marked[1] is messages[-1]["content"][-1]


def test_nudge_messages_between_calls_do_not_lose_the_previous_position() -> None:
    messages = [_user(_text("TASK"))]
    roll_cache_breakpoints(messages)  # call 1 marks msg0
    messages.append(_assistant(_tool_use("t1")))
    messages.append(_user(_tool_result("t1")))
    messages.append(_user(_text("[focus banner]")))
    messages.append(_user(_text("[budget nudge]")))
    roll_cache_breakpoints(messages)  # call 2
    marked = _marked_blocks(messages)
    assert len(marked) == 2
    assert marked[0] is messages[0]["content"][0]  # call 1's position survives
    assert marked[1] is messages[-1]["content"][0]  # the nudge is the new tail


def test_roll_is_idempotent_without_new_messages() -> None:
    messages = [_user(_text("TASK"))]
    roll_cache_breakpoints(messages)
    messages.append(_assistant(_text("a")))
    messages.append(_user(_tool_result()))
    roll_cache_breakpoints(messages)
    before = [id(b) for b in _marked_blocks(messages)]
    roll_cache_breakpoints(messages)  # crash-resume re-issues the same call
    assert [id(b) for b in _marked_blocks(messages)] == before


def test_marks_only_markable_block_types() -> None:
    # A trailing assistant message (thinking + tool_use only) is never marked;
    # the scan falls back to the newest user block.
    messages = [
        _user(_text("TASK")),
        _assistant({"type": "thinking", "thinking": "..."}, _tool_use()),
    ]
    roll_cache_breakpoints(messages)
    marked = _marked_blocks(messages)
    assert marked == [messages[0]["content"][0]]


def test_string_content_messages_are_skipped() -> None:
    messages = [
        {"role": "user", "content": "plain string"},
        _user(_text("blocks")),
        {"role": "user", "content": "another string"},
    ]
    roll_cache_breakpoints(messages)
    marked = _marked_blocks(messages)
    assert marked == [messages[1]["content"][0]]


def test_compaction_restart_starts_a_fresh_rolling_pair() -> None:
    messages = [_user(_text("TASK"))]
    roll_cache_breakpoints(messages)
    messages.append(_assistant(_text("a")))
    messages.append(_user(_tool_result()))
    roll_cache_breakpoints(messages)
    # tier-2 restart: history replaced by (original, summary)
    messages[:] = [messages[0], _user(_text("[context restart] summary"))]
    roll_cache_breakpoints(messages)
    marked = _marked_blocks(messages)
    assert len(marked) == 2
    assert marked[0] is messages[0]["content"][0]
    assert marked[1] is messages[1]["content"][0]


def test_strip_cache_control_is_copy_on_write() -> None:
    messages = [_user(_text("TASK"))]
    roll_cache_breakpoints(messages)
    stripped = strip_cache_control_messages(messages)
    assert stripped is not messages
    assert "cache_control" not in stripped[0]["content"][0]
    # The original (loop-owned, snapshot-shared) list keeps its marker.
    assert "cache_control" in messages[0]["content"][0]


def test_strip_cache_control_passthrough_when_unmarked() -> None:
    messages = [_user(_text("TASK")), {"role": "user", "content": "plain"}]
    assert strip_cache_control_messages(messages) is messages
