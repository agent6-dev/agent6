# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for context compaction (oldest tool_result elision)."""

from __future__ import annotations

from typing import Any

from agent6.workflows._compaction import (
    compact_old_tool_results,
    context_chars,
    parse_checkoff,
    strip_checkoff,
)


def test_parse_checkoff_valid_block() -> None:
    text = (
        "Progress summary here.\n\n"
        '```checkoff\n{"completed_ids": ["01A", "01B"], "new_tasks": ["fix the parser", ""]}\n```'
    )
    completed, new_tasks = parse_checkoff(text)
    assert completed == ["01A", "01B"]
    assert new_tasks == ["fix the parser"]  # empty title filtered


def test_parse_checkoff_absent_or_malformed() -> None:
    assert parse_checkoff("no block at all") == ([], [])
    assert parse_checkoff("```checkoff\nnot json\n```") == ([], [])
    assert parse_checkoff('```checkoff\n["not", "a", "dict"]\n```') == ([], [])
    # non-string ids/titles are dropped
    assert parse_checkoff('```checkoff\n{"completed_ids": [1, "ok"], "new_tasks": [2]}\n```') == (
        ["ok"],
        [],
    )


def test_strip_checkoff_removes_block() -> None:
    text = 'the summary\n\n```checkoff\n{"completed_ids": []}\n```'
    assert strip_checkoff(text) == "the summary"
    assert strip_checkoff("no block") == "no block"


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
    # Eliding a tool_result already smaller than the 263-char placeholder would
    # GROW cumulative size, not shrink it. Such blocks must be left intact.
    from agent6.workflows._compaction import (
        ELISION_PLACEHOLDER as PLACEHOLDER,
    )

    tiny = "x" * 50  # < len(placeholder) == 263
    big = "y" * 5000
    # Oldest-first; keep_recent=2 keeps the last two, and the final message is
    # exempt, so the eligible blocks are the two in the first message.
    msgs: list[dict[str, Any]] = [
        _user_msg_with_tool_results(tiny, big),
        _user_msg_with_tool_results(big, big),
    ]
    compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    # The oldest (tiny) block is eligible but must be skipped, not ballooned;
    # its eligible sibling is elided as normal.
    assert msgs[0]["content"][0]["content"] == tiny
    assert "elided" in msgs[0]["content"][1]["content"]
    assert len(PLACEHOLDER) == 263


def test_compact_noop_when_under_threshold() -> None:
    msgs: list[dict[str, Any]] = [_user_msg_with_tool_results("small")]
    stats = compact_old_tool_results(msgs, max_total_bytes=1000)
    assert stats.elided == 0
    assert msgs[0]["content"][0]["content"] == "small"


def test_compact_elides_oldest_when_over_threshold() -> None:
    big = "x" * 1000
    msgs: list[dict[str, Any]] = [
        _user_msg_with_tool_results(big),  # turn 0 - oldest
        _user_msg_with_tool_results(big),  # turn 1
        _user_msg_with_tool_results(big),  # turn 2 - newest
    ]
    stats = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2)
    assert stats.elided == 1
    # Turn 0 (oldest) replaced with marker; turns 1 and 2 kept.
    assert "elided" in msgs[0]["content"][0]["content"]
    assert msgs[1]["content"][0]["content"] == big
    assert msgs[2]["content"][0]["content"] == big


def test_compact_preserves_keep_recent_floor() -> None:
    """Even when over threshold, the newest `keep_recent` entries
    are never elided."""
    big = "x" * 10_000
    msgs: list[dict[str, Any]] = [_user_msg_with_tool_results(big) for _ in range(5)]
    stats = compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    # 3 oldest elided, 2 most recent preserved.
    assert stats.elided == 3
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
    assert e1.elided == 2  # oldest 2 elided
    assert e2.elided == 0  # no further work needed on second pass


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
    stats = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2)
    # Assistant message untouched.
    assert msgs[0]["content"][0]["type"] == "tool_use"
    # Oldest tool_result elided.
    assert stats.elided == 1


def test_compact_never_elides_unseen_results_in_final_message() -> None:
    """Compaction runs at top-of-iteration, BEFORE the provider call that
    delivers the final message's tool_results: the model has never seen them.
    A turn with 3+ large results must not have its oldest same-turn results
    replaced by the "re-call the tool" placeholder (which previously sent the
    model into a paid re-call cycle chasing content it never received)."""
    big = "x" * 10_000
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {}}
                for i in range(3)
            ],
        },
        _user_msg_with_tool_results(big, big, big),
    ]
    stats = compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    assert stats.elided == 0
    assert [c["content"] for c in msgs[2]["content"]] == [big, big, big]


def test_compact_elides_seen_results_but_protects_final_message() -> None:
    # Results the model has already consumed (an assistant turn follows them)
    # stay eligible; only the undelivered final message is exempt.
    big = "x" * 10_000
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        _user_msg_with_tool_results(big, big, big),  # seen: answered below
        {"role": "assistant", "content": [{"type": "text", "text": "on it"}]},
        _user_msg_with_tool_results(big, big, big),  # unseen: awaiting delivery
    ]
    stats = compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    assert stats.elided == 3
    assert all("elided" in c["content"] for c in msgs[1]["content"])
    assert [c["content"] for c in msgs[3]["content"]] == [big, big, big]


def test_compact_never_elides_undelivered_results_behind_a_steer_message() -> None:
    """Undelivered tool_results are not always the final message: an operator
    steer (or a pre-call nudge) appends a trailing user message after them, so
    they sit at index -2. They are still unseen (the delivering provider call
    runs after this compaction), so keying on the final index alone let their
    older same-turn blocks be elided into a paid re-call cycle. The exemption
    tracks 'after the last assistant message', which still holds here."""
    big = "x" * 10_000
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {}}
                for i in range(3)
            ],
        },
        _user_msg_with_tool_results(big, big, big),  # unseen: awaiting delivery
        {"role": "user", "content": [{"type": "text", "text": "steer: focus on the parser"}]},
    ]
    stats = compact_old_tool_results(msgs, max_total_bytes=100, keep_recent=2)
    assert stats.elided == 0
    assert [c["content"] for c in msgs[2]["content"]] == [big, big, big]


def test_restart_notice_is_dag_aware() -> None:
    """The tier-2 summarise-and-restart notice must point the worker at its
    durable task DAG so cross-compaction task state is recovered."""
    from agent6.workflows._prompts import context_restart_notice

    for mode in ("run", "plan"):
        notice = context_restart_notice(mode)
        # The real tool is ``list_tasks`` (no ``dag_`` prefix); the notice must
        # name it exactly or the post-compaction recovery call 404s.
        assert "list_tasks" in notice
        assert "dag_list_tasks" not in notice
        assert "DAG" in notice
        # Still tells the worker not to start over.
        assert "Do NOT start over" in notice
    # ask/machine/agent have no DAG tools: instructing list_tasks there burns a
    # turn on an unknown-tool error, so the DAG paragraph must be absent.
    for mode in ("ask", "machine", "agent"):
        notice = context_restart_notice(mode)
        assert "list_tasks" not in notice
        assert "Do NOT start over" in notice
        assert notice.endswith("PROGRESS SUMMARY:\n")


# --- read-waste reduction: identity placeholders + hot-file protection ------


def _assistant_tool_use(uid: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": uid, "name": name, "input": tool_input}],
    }


def _tool_result_msg(uid: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": uid, "content": content}],
    }


def test_elision_placeholder_names_the_call() -> None:
    from agent6.workflows._compaction import ELISION_PREFIX, elision_placeholder

    p = elision_placeholder("read_file", {"path": "src/x.py", "offset": 10, "limit": 50})
    assert p.startswith(ELISION_PREFIX)
    assert "read_file src/x.py" in p and "offset=10" in p
    g = elision_placeholder("grep", {"pattern": "def foo"})
    assert "grep pattern 'def foo'" in g
    # Unknown pairing (orphan result) falls back to the generic marker.
    from agent6.workflows._compaction import ELISION_PLACEHOLDER

    assert elision_placeholder("", None) == ELISION_PLACEHOLDER
    assert elision_placeholder("read_file", "not-a-dict") == ELISION_PLACEHOLDER
    # A pathological arg is clipped, keeping the placeholder short.
    long = elision_placeholder("read_file", {"path": "x" * 5000})
    assert len(long) < 500


def test_recently_edited_paths_extraction() -> None:
    from agent6.workflows._compaction import recently_edited_paths

    unified = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    v4a = "*** Begin Patch\n*** Update File: pkg/v4a.py\n@@\n-a\n+b\n*** End Patch\n"
    msgs: list[dict[str, Any]] = [
        _assistant_tool_use("e1", "apply_edit", {"path": "edited.py", "edits": []}),
        _assistant_tool_use("e2", "apply_patch", {"path": "explicit.py", "patch": "x"}),
        _assistant_tool_use("e3", "apply_patch", {"path": "", "patch": unified}),
        _assistant_tool_use("e4", "apply_patch", {"patch": v4a}),
        _assistant_tool_use("r1", "read_file", {"path": "only-read.py"}),
    ]
    got = recently_edited_paths(msgs)
    assert got == frozenset({"edited.py", "explicit.py", "pkg/mod.py", "pkg/v4a.py"})
    # The window is per assistant TURN: an edit older than last_turns drops out.
    windowed = recently_edited_paths(
        [
            _assistant_tool_use("e1", "apply_edit", {"path": "old.py", "edits": []}),
            _assistant_tool_use("t2", "read_file", {"path": "a"}),
            _assistant_tool_use("t3", "read_file", {"path": "b"}),
        ],
        last_turns=2,
    )
    assert windowed == frozenset()


def test_compact_elides_protected_reads_last_but_bound_still_holds() -> None:
    def build() -> list[dict[str, Any]]:
        return [
            _assistant_tool_use("r1", "read_file", {"path": "hot.py"}),
            _tool_result_msg("r1", "H" * 1000),
            _assistant_tool_use("r2", "read_file", {"path": "cold.py"}),
            _tool_result_msg("r2", "C" * 1000),
            _assistant_tool_use("r3", "grep", {"pattern": "x"}),
            _tool_result_msg("r3", "G" * 1000),
            _assistant_tool_use("r4", "list_dir", {"path": "."}),
            _tool_result_msg("r4", "L" * 1000),
        ]

    # Budget forces ONE elision: with hot.py protected, the (older) hot read
    # survives and the cold read goes first.
    msgs = build()
    n = compact_old_tool_results(
        msgs, max_total_bytes=3500, keep_recent=2, protect_paths=frozenset({"hot.py"})
    )
    assert n.elided == 1
    assert msgs[1]["content"][0]["content"] == "H" * 1000
    assert "cold.py" in msgs[3]["content"][0]["content"]
    # Tighter budget: protection is a priority, not an exemption; the hot read
    # is elided too and the bound holds.
    msgs2 = build()
    n2 = compact_old_tool_results(
        msgs2, max_total_bytes=2500, keep_recent=2, protect_paths=frozenset({"hot.py"})
    )
    assert n2.elided == 2
    assert "hot.py" in msgs2[1]["content"][0]["content"]


def test_compact_placeholder_carries_tool_identity() -> None:
    msgs = [
        _assistant_tool_use("r1", "read_file", {"path": "src/lib.py"}),
        _tool_result_msg("r1", "X" * 1000),
        _assistant_tool_use("r2", "grep", {"pattern": "q"}),
        _tool_result_msg("r2", "Y" * 1000),
        _assistant_tool_use("r3", "list_dir", {"path": "."}),
        _tool_result_msg("r3", "Z" * 1000),
        _assistant_tool_use("r4", "outline", {"path": "a.py"}),
        _tool_result_msg("r4", "W" * 1000),
    ]
    n = compact_old_tool_results(msgs, max_total_bytes=3000, keep_recent=2)
    assert n.elided >= 1
    elided = msgs[1]["content"][0]["content"]
    assert elided.startswith("<elided by context compaction")
    assert "read_file src/lib.py" in elided
