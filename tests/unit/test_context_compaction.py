# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for context compaction (oldest tool_result elision)."""

from __future__ import annotations

from typing import Any

from agent6.workflows._compaction import (
    compact_old_tool_results,
    context_chars,
)


def test_context_chars_counts_text_tool_use_and_tool_results() -> None:
    # tier-2's trigger must see content tier-1 does NOT cap (assistant prose,
    # tool_use inputs), not just tool_result bytes.
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "abcd"},  # 4
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},  # 5
                {"type": "tool_use", "name": "grep", "input": {"q": "x"}},  # len(str(dict))
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "content": "RESULT"}]},  # 6
    ]
    total = context_chars(msgs)
    # 4 + 5 + 6 + len(str({"q": "x"})) -- well above just the 6 tool_result bytes.
    assert total == 4 + 5 + 6 + len(str({"q": "x"}))
    assert total > 6


def _user_msg_with_tool_results(*contents: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": c}
            for i, c in enumerate(contents)
        ],
    }


def test_compact_skips_tool_result_smaller_than_placeholder() -> None:
    # Eliding a tool_result already smaller than the 201-char placeholder would
    # GROW cumulative size, not shrink it. Such blocks must be left intact.
    from agent6.workflows._compaction import (
        ELISION_PLACEHOLDER as PLACEHOLDER,
    )

    tiny = "x" * 50  # < len(placeholder) == 201
    big = "y" * 5000
    # Oldest-first within one message; keep_recent=2 keeps the last two.
    msgs: list[dict[str, Any]] = [_user_msg_with_tool_results(tiny, big, big, big)]
    compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    # The oldest (tiny) block is eligible but must be skipped, not ballooned.
    assert msgs[0]["content"][0]["content"] == tiny
    assert len(PLACEHOLDER) == 201


def test_compact_noop_when_under_threshold() -> None:
    msgs: list[dict[str, Any]] = [_user_msg_with_tool_results("small")]
    elided = compact_old_tool_results(msgs, max_total_bytes=1000)
    assert elided == 0
    assert msgs[0]["content"][0]["content"] == "small"


def test_compact_elides_oldest_when_over_threshold() -> None:
    big = "x" * 1000
    msgs: list[dict[str, Any]] = [
        _user_msg_with_tool_results(big),  # turn 0 - oldest
        _user_msg_with_tool_results(big),  # turn 1
        _user_msg_with_tool_results(big),  # turn 2 - newest
    ]
    elided = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2)
    assert elided == 1
    # Turn 0 (oldest) replaced with marker; turns 1 and 2 kept.
    assert "elided" in msgs[0]["content"][0]["content"]
    assert msgs[1]["content"][0]["content"] == big
    assert msgs[2]["content"][0]["content"] == big


def test_compact_preserves_keep_recent_floor() -> None:
    """Even when over threshold, the newest `keep_recent` entries
    are never elided."""
    big = "x" * 10_000
    msgs: list[dict[str, Any]] = [_user_msg_with_tool_results(big) for _ in range(5)]
    elided = compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    # 3 oldest elided, 2 most recent preserved.
    assert elided == 3
    assert "elided" in msgs[0]["content"][0]["content"]
    assert "elided" in msgs[1]["content"][0]["content"]
    assert "elided" in msgs[2]["content"][0]["content"]
    assert msgs[3]["content"][0]["content"] == big
    assert msgs[4]["content"][0]["content"] == big


def test_compact_idempotent_on_already_elided() -> None:
    """Running compaction twice doesn't double-elide or churn."""
    big = "x" * 1000
    msgs: list[dict[str, Any]] = [_user_msg_with_tool_results(big) for _ in range(4)]
    e1 = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2)
    e2 = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2)
    assert e1 == 2  # oldest 2 elided
    assert e2 == 0  # no further work needed on second pass


def test_compact_skips_non_tool_result_blocks() -> None:
    """Assistant messages with tool_use blocks must not be touched."""
    msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t0", "name": "read_file", "input": {}},
            ],
        },
        _user_msg_with_tool_results("x" * 1000),
        _user_msg_with_tool_results("y" * 1000),
        _user_msg_with_tool_results("z" * 1000),
    ]
    elided = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2)
    # Assistant message untouched.
    assert msgs[0]["content"][0]["type"] == "tool_use"
    # Oldest tool_result elided.
    assert elided == 1


def test_restart_notice_is_dag_aware() -> None:
    """The tier-2 summarise-and-restart notice must point the worker at its
    durable task DAG so cross-compaction task state is recovered."""
    from agent6.workflows._prompts import (
        CONTEXT_RESTART_NOTICE as notice,
    )

    # The real tool is ``list_tasks`` (no ``dag_`` prefix); the notice must
    # name it exactly or the post-compaction recovery call 404s.
    assert "list_tasks" in notice
    assert "dag_list_tasks" not in notice
    assert "DAG" in notice
    # Still tells the worker not to start over.
    assert "Do NOT start over" in notice
