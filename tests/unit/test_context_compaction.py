# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for context compaction (oldest tool_result elision)."""

from __future__ import annotations

from typing import Any

from agent6.workflows.loop import (
    _compact_old_tool_results as compact_old_tool_results,  # pyright: ignore[reportPrivateUsage]
)


def _user_msg_with_tool_results(*contents: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": c}
            for i, c in enumerate(contents)
        ],
    }


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
    from agent6.workflows.loop import (
        _CONTEXT_RESTART_NOTICE as notice,  # pyright: ignore[reportPrivateUsage]
    )

    assert "dag_list_tasks" in notice
    assert "DAG" in notice
    # Still tells the worker not to start over.
    assert "Do NOT start over" in notice
