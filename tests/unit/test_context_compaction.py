# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for context compaction (oldest tool_result elision)."""

from __future__ import annotations

import pathlib
from typing import Any

from agent6.harness import _chain, _compaction, _conversation


def _add_exchange(
    conv: _conversation.Conversation, *calls: tuple[str, dict[str, Any], str]
) -> None:
    """One assistant turn of (tool name, input, result content) calls plus its results turn.

    Ids are unique per conversation position.
    """
    base = len(conv)
    turn = conv.assistant(
        [
            {"type": "tool_use", "id": f"t{base}-{i}", "name": name, "input": tool_input}
            for i, (name, tool_input, _content) in enumerate(calls)
        ]
    )
    conv.results(
        [
            _conversation.ToolResultItem(tool_use_id=tu.id, content=content, for_call=tu)
            for tu, (_name, _input, content) in zip(turn.tool_uses, calls, strict=True)
        ]
    )


def _reads(conv: _conversation.Conversation, *contents: str, name: str = "read_file") -> None:
    """One single-call exchange per content string."""
    for c in contents:
        _add_exchange(conv, (name, {"path": "x.py"}, c))


def _result_contents(conv: _conversation.Conversation) -> list[str]:
    return [
        item.content
        for turn in conv.turns
        if isinstance(turn, _conversation.UserTurn)
        for item in turn.items
        if isinstance(item, _conversation.ToolResultItem)
    ]


def test_parse_checkoff_valid_block() -> None:
    text = (
        "Progress summary here.\n\n"
        '```checkoff\n{"completed_ids": ["01A", "01B"], "new_tasks": ["fix the parser", ""]}\n```'
    )
    completed, new_tasks = _compaction.parse_checkoff(text)
    assert completed == ["01A", "01B"]
    assert new_tasks == ["fix the parser"]  # empty title filtered


def test_parse_checkoff_absent_or_malformed() -> None:
    assert _compaction.parse_checkoff("no block at all") == ([], [])
    assert _compaction.parse_checkoff("```checkoff\nnot json\n```") == ([], [])
    assert _compaction.parse_checkoff('```checkoff\n["not", "a", "dict"]\n```') == ([], [])
    # non-string ids/titles are dropped
    assert _compaction.parse_checkoff(
        '```checkoff\n{"completed_ids": [1, "ok"], "new_tasks": [2]}\n```'
    ) == (
        ["ok"],
        [],
    )


def test_parse_checkoff_present_but_non_list_field_is_total() -> None:
    # A present but non-list value (null, a number, a bool) yields [] instead of a TypeError.
    for bad in ("null", "0", "false", '"a string"'):
        assert _compaction.parse_checkoff(
            f'```checkoff\n{{"completed_ids": {bad}, "new_tasks": {bad}}}\n```'
        ) == (
            [],
            [],
        )


def test_strip_checkoff_removes_block() -> None:
    text = 'the summary\n\n```checkoff\n{"completed_ids": []}\n```'
    assert _compaction.strip_checkoff(text) == "the summary"
    assert _compaction.strip_checkoff("no block") == "no block"


def test_context_chars_counts_text_tool_use_and_tool_results() -> None:
    # The tier-2 trigger sees content tier-1 does not cap: assistant prose and tool_use inputs.
    conv = _conversation.Conversation()
    conv.notice("abcd")  # 4
    turn = conv.assistant(
        [
            {"type": "text", "text": "hello"},  # 5
            {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "x"}},
        ]
    )
    conv.results(
        [
            _conversation.ToolResultItem(
                tool_use_id="t1", content="RESULT", for_call=turn.tool_uses[0]
            )
        ]  # 6
    )
    total = _compaction.context_chars(conv)
    # Every value of the tool_use block counts: id and name go on the wire with the input.
    assert total == 4 + 5 + 6 + len("t1") + len("grep") + len(str({"q": "x"}))
    assert total > 6


def test_compact_skips_tool_result_smaller_than_placeholder() -> None:
    # Eliding a result smaller than the placeholder would grow the size; such blocks stay intact.

    tiny = "x" * 50  # smaller than the placeholder, so eliding it would grow the context
    big = "y" * 5000
    # Oldest first; keep_recent=2 and the exempt final turn leave the first turn's blocks eligible.
    conv = _conversation.Conversation()
    _add_exchange(conv, ("grep", {}, tiny), ("grep", {}, big))
    _add_exchange(conv, ("grep", {}, big), ("grep", {}, big))
    _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    contents = _result_contents(conv)
    # The oldest (tiny) block is skipped, not ballooned; its eligible sibling is elided.
    assert contents[0] == tiny
    assert "elided" in contents[1]
    assert len(tiny) < len(_compaction.ELISION_PLACEHOLDER)  # the premise the skip guards


def test_an_operator_answer_is_never_elided() -> None:
    """An `ask_user` result is never elided.

    The operator's ruling exists nowhere else in the context.
    """
    # Long enough that eliding it frees bytes.
    answer = '{"answers": ["' + "use the v2 table only. " * 300 + '"]}'
    big = "y" * 5000
    conv = _conversation.Conversation()
    _add_exchange(conv, ("ask_user", {"questions": [{"question": "which table?"}]}, answer))
    _add_exchange(conv, ("grep", {}, big), ("grep", {}, big))
    _add_exchange(conv, ("grep", {}, big), ("grep", {}, big))

    _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)

    contents = _result_contents(conv)
    assert contents[0] == answer
    assert any("elided" in c for c in contents), "the rest still compacts"


def test_an_operator_answer_is_never_deduped_either() -> None:
    """An `ask_user` result is never deduped either; the dedup pass runs before elision."""
    answer = '{"answers": ["' + "use the v2 table only. " * 30 + '"]}'
    big = "y" * 1000
    conv = _conversation.Conversation()
    _add_exchange(conv, ("ask_user", {"questions": [{"question": "which table?"}]}, answer))
    _add_exchange(conv, ("grep", {}, big))
    _add_exchange(conv, ("ask_user", {"questions": [{"question": "which table?"}]}, answer))
    _add_exchange(conv, ("grep", {}, big))
    _add_exchange(conv, ("grep", {}, big))

    _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=1)

    contents = _result_contents(conv)
    assert contents[0] == answer, "the older ask_user copy must survive whole, not as a pointer"


def test_tier2_measures_the_request_not_just_the_conversation() -> None:
    """The tier-2 threshold measures the whole request.

    System prompt and tool definitions included.
    """
    from agent6.providers import types

    tools = (
        types.ToolDefinition(
            name="read_file",
            description="Read a file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    )
    system = "s" * 4096

    prefix = _compaction.request_prefix_chars(system, tools)

    assert prefix > len(system), "the tool definitions ride in every request too"
    assert _compaction.request_prefix_chars("", ()) == 0


def test_compact_noop_when_under_threshold() -> None:
    conv = _conversation.Conversation()
    _reads(conv, "small")
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=1000)
    assert len(stats.elided_calls) == 0
    assert _result_contents(conv) == ["small"]


def test_compact_elides_oldest_when_over_threshold() -> None:
    # Distinct payloads: identical ones would be deduplicated before elision.
    a, b, c = "a" * 1000, "b" * 1000, "c" * 1000
    conv = _conversation.Conversation()
    _reads(conv, a, b, c)  # oldest first
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    assert len(stats.elided_calls) == 1
    contents = _result_contents(conv)
    # Oldest replaced with marker; the newer two kept.
    assert "elided" in contents[0]
    assert contents[1] == b
    assert contents[2] == c


def test_compact_preserves_keep_recent_floor() -> None:
    """Even when over threshold, the newest `keep_recent` entries are never elided."""
    bodies = [ch * 10_000 for ch in "abcde"]  # distinct: dedup must not fire
    conv = _conversation.Conversation()
    _reads(conv, *bodies)
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    # 3 oldest elided, 2 most recent preserved.
    assert len(stats.elided_calls) == 3
    contents = _result_contents(conv)
    assert all("elided" in c for c in contents[:3])
    assert contents[3] == bodies[3]
    assert contents[4] == bodies[4]


def test_compact_with_no_recent_floor_still_elides_seen_results() -> None:
    bodies = [ch * 1_000 for ch in "abc"]
    conv = _conversation.Conversation()
    _reads(conv, *bodies)

    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=0)

    assert len(stats.elided_calls) == 2
    contents = _result_contents(conv)
    assert all("elided" in content for content in contents[:2])
    assert contents[2] == bodies[2]  # the final, undelivered result stays whole

    duplicates = _conv_with_repeated_reads("d" * 1_000)
    duplicate_stats = _compaction.compact_old_tool_results(
        duplicates, max_total_bytes=2_500, keep_recent=0
    )
    assert len(duplicate_stats.deduped_calls) == 2


def test_compact_idempotent_on_already_elided() -> None:
    """Running compaction twice doesn't double-elide or churn."""
    bodies = [ch * 1000 for ch in "abcd"]  # distinct: dedup must not fire
    conv = _conversation.Conversation()
    _reads(conv, *bodies)
    e1 = _compaction.compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    e2 = _compaction.compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    assert len(e1.elided_calls) == 2  # oldest 2 elided
    assert len(e2.elided_calls) == 0  # no further work needed on second pass


def test_compact_never_elides_unseen_results_in_final_turn() -> None:
    """Compaction never elides the final turn's results, which the model has not seen yet.

    It runs at top-of-iteration, before the provider call that delivers them.
    """
    big = "x" * 10_000
    conv = _conversation.Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, big)] * 3)
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    assert len(stats.elided_calls) == 0
    assert _result_contents(conv) == [big, big, big]


def test_compact_elides_seen_results_but_protects_final_turn() -> None:
    # Results the model has consumed stay eligible; only the undelivered final turn is exempt.
    seen = [(f"s{i}" * 5_000) for i in range(3)]  # distinct: dedup must not fire
    fresh = [(f"f{i}" * 5_000) for i in range(3)]
    conv = _conversation.Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, c) for c in seen])  # seen: answered below
    _add_exchange(conv, *[("read_file", {}, c) for c in fresh])  # unseen: awaiting delivery
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    assert len(stats.elided_calls) == 3
    contents = _result_contents(conv)
    assert all("elided" in c for c in contents[:3])
    assert contents[3:] == fresh


def test_compact_never_elides_undelivered_results_behind_a_steer_message() -> None:
    """Undelivered results behind a trailing steer turn are exempt too.

    The exemption tracks the last tool_result-bearing turn, not the final index.
    """
    big = "x" * 10_000
    conv = _conversation.Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, big)] * 3)  # unseen: awaiting delivery
    conv.notice("steer: focus on the parser")
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)
    assert len(stats.elided_calls) == 0
    assert _result_contents(conv) == [big, big, big]


def test_compact_can_elide_the_last_result_batch_after_it_was_consumed() -> None:
    bodies = [ch * 10_000 for ch in "abc"]
    conv = _conversation.Conversation()
    conv.notice("task")
    _add_exchange(conv, *[("read_file", {}, body) for body in bodies])
    conv.assistant([{"type": "text", "text": "I have read those results."}])
    conv.notice("[harness] Continue working.")

    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=2)

    assert len(stats.elided_calls) == 1
    contents = _result_contents(conv)
    assert "elided" in contents[0]
    assert contents[1:] == bodies[1:]

    duplicates = _conversation.Conversation()
    duplicates.notice("task")
    _add_exchange(duplicates, *[("read_file", {}, "d" * 10_000)] * 4)
    duplicates.assistant([{"type": "text", "text": "I consumed the duplicate results."}])
    duplicates.notice("[harness] Continue working.")
    duplicate_stats = _compaction.compact_old_tool_results(
        duplicates, max_total_bytes=35_000, keep_recent=2
    )
    assert len(duplicate_stats.deduped_calls) == 2


def test_restart_notice_is_dag_aware() -> None:
    """The tier-2 restart notice points the worker at its durable task DAG."""
    from agent6.prompts import revision

    for mode in ("run", "plan"):
        notice = revision.context_restart_notice(mode)
        # The real tool is `list_tasks`; a `dag_` prefix in the notice would 404 the recovery call.
        assert "list_tasks" in notice
        assert "dag_list_tasks" not in notice
        assert "DAG" in notice
        # `list_tasks` returns tasks and a count; the focus banner carries the cursor.
        assert "cursor" not in notice
        # Still tells the worker not to start over.
        assert "the task continues from it" in notice
    # ask, machine and agent have no DAG tools, so the DAG paragraph is absent.
    for mode in ("ask", "agent"):
        notice = revision.context_restart_notice(mode)
        assert "list_tasks" not in notice
        assert "the task continues from it" in notice
        assert notice.endswith("PROGRESS SUMMARY:\n")


# --- read-waste reduction: identity placeholders + hot-file protection ------


def test_restart_notice_omits_dag_recovery_without_a_curator() -> None:
    """Without a curator the restart notice does not tell the worker to call list_tasks."""
    from agent6.prompts import revision

    notice = revision.context_restart_notice("run", dag_available=False)
    assert "list_tasks" not in notice
    assert "DAG" not in notice
    assert "the task continues from it" in notice


def test_elision_placeholder_names_the_call() -> None:
    p = _compaction.elision_placeholder(
        "read_file", {"path": "src/x.py", "start_line": 10, "limit": 50}
    )
    assert p.startswith(_compaction.ELISION_PREFIX)
    assert "read_file src/x.py" in p and "start_line=10" in p
    g = _compaction.elision_placeholder("find_definition", {"symbol": "foo"})
    assert "find_definition foo" in g
    # Unknown pairing (orphan result) falls back to the generic marker.

    assert _compaction.elision_placeholder("", None) == _compaction.ELISION_PLACEHOLDER
    assert (
        _compaction.elision_placeholder("read_file", "not-a-dict")
        == _compaction.ELISION_PLACEHOLDER
    )
    # A pathological arg is clipped, keeping the placeholder short.
    long = _compaction.elision_placeholder("read_file", {"path": "x" * 5000})
    assert len(long) < 500


def test_recently_edited_paths_extraction() -> None:
    unified = (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        "diff --git a/pkg/other.py b/pkg/other.py\n"
        "--- a/pkg/other.py\n+++ b/pkg/other.py\n@@ -1,1 +1,1 @@\n-c\n+d\n"
    )
    v4a = "*** Begin Patch\n*** Update File: pkg/v4a.py\n@@\n-a\n+b\n*** End Patch\n"
    conv = _conversation.Conversation()
    _add_exchange(conv, ("apply_edit", {"path": "edited.py", "edits": []}, "ok"))
    _add_exchange(conv, ("apply_patch", {"path": "explicit.py", "patch": "x"}, "ok"))
    _add_exchange(conv, ("apply_patch", {"path": "", "patch": unified}, "ok"))
    _add_exchange(conv, ("apply_patch", {"patch": v4a}, "ok"))
    _add_exchange(conv, ("read_file", {"path": "only-read.py"}, "ok"))
    got = _compaction.recently_edited_paths(conv)
    assert got == frozenset(
        {"edited.py", "explicit.py", "pkg/mod.py", "pkg/other.py", "pkg/v4a.py"}
    )
    # The window is per assistant TURN: an edit older than last_turns drops out.
    conv2 = _conversation.Conversation()
    _add_exchange(conv2, ("apply_edit", {"path": "old.py", "edits": []}, "ok"))
    _add_exchange(conv2, ("read_file", {"path": "a"}, "ok"))
    _add_exchange(conv2, ("read_file", {"path": "b"}, "ok"))
    assert _compaction.recently_edited_paths(conv2, last_turns=2) == frozenset()


def test_compact_elides_protected_reads_last_but_bound_still_holds() -> None:
    def build() -> _conversation.Conversation:
        conv = _conversation.Conversation()
        _add_exchange(conv, ("read_file", {"path": "hot.py"}, "H" * 1000))
        _add_exchange(conv, ("read_file", {"path": "cold.py"}, "C" * 1000))
        _add_exchange(conv, ("grep", {"pattern": "x"}, "G" * 1000))
        _add_exchange(conv, ("list_dir", {"path": "."}, "L" * 1000))
        return conv

    # The budget forces one elision: with hot.py protected, the cold read goes first.
    conv = build()
    n = _compaction.compact_old_tool_results(
        conv, max_total_bytes=3500, keep_recent=2, protect_paths=frozenset({"hot.py"})
    )
    assert len(n.elided_calls) == 1
    contents = _result_contents(conv)
    assert contents[0] == "H" * 1000
    assert "cold.py" in contents[1]
    # Tighter budget: protection is a priority, not an exemption; the hot read is elided too.
    conv2 = build()
    n2 = _compaction.compact_old_tool_results(
        conv2, max_total_bytes=2500, keep_recent=2, protect_paths=frozenset({"hot.py"})
    )
    assert len(n2.elided_calls) == 2
    assert "hot.py" in _result_contents(conv2)[0]


def test_compact_placeholder_carries_tool_identity() -> None:
    conv = _conversation.Conversation()
    _add_exchange(conv, ("read_file", {"path": "src/lib.py"}, "X" * 1000))
    _add_exchange(conv, ("grep", {"pattern": "q"}, "Y" * 1000))
    _add_exchange(conv, ("list_dir", {"path": "."}, "Z" * 1000))
    _add_exchange(conv, ("outline", {"path": "a.py"}, "W" * 1000))
    n = _compaction.compact_old_tool_results(conv, max_total_bytes=3000, keep_recent=2)
    assert len(n.elided_calls) >= 1
    elided = _result_contents(conv)[0]
    assert elided.startswith("<elided by context compaction")
    assert "read_file src/lib.py" in elided


def test_call_label_identities() -> None:
    assert _compaction.call_label("read_file", {"path": "src/foo.py"}) == "read_file src/foo.py"
    assert (
        _compaction.call_label("read_file", {"path": "a.py", "start_line": 10, "limit": 40})
        == "read_file a.py (start_line=10, limit=40)"
    )
    assert _compaction.call_label("list_dir", {"path": "src"}) == "list_dir src"
    # Every tool with an identifying argument: a compacted run_command with no command says nothing.
    assert _compaction.call_label("run_command", {"argv": ["pytest", "-x", "tests/t.py"]}) == (
        "run_command pytest -x tests/t.py"
    )
    assert (
        _compaction.call_label("apply_edit", {"path": "src/a.py", "edits": []})
        == "apply_edit src/a.py"
    )
    assert _compaction.call_label("use_skill", {"name": "debugging"}) == "use_skill debugging"
    assert _compaction.call_label("read_background", {"id": "bg1"}) == "read_background bg1"
    assert _compaction.call_label("fetch", {"url": "https://x.test/s"}) == "fetch https://x.test/s"
    assert _compaction.call_label("read_session", {"query": "parser regression"}) == (
        "read_session parser regression"
    )
    # An argv is rendered as a command line, so a quoted pattern stays readable.
    assert _compaction.call_label("run_command", {"argv": ["rg", "-n", "def f"]}) == (
        "run_command rg -n 'def f'"
    )
    # Nothing identifying: the bare name, not an invented hint.
    assert _compaction.call_label("finish_session", {"summary": "done"}) == "finish_session"
    assert _compaction.call_label("find_definition", {"symbol": "Foo"}) == "find_definition Foo"
    assert _compaction.call_label("frobnicate", {"x": 1}) == "frobnicate"
    assert _compaction.call_label("frobnicate", None) == "frobnicate"
    assert _compaction.call_label("", None) == ""
    long_path = "d/" * 80
    assert (
        _compaction.call_label("read_file", {"path": long_path})
        == f"read_file {long_path[:120]}..."
    )


def test_elision_placeholder_unchanged_by_label_refactor() -> None:
    """The elision placeholder's exact bytes are pinned.

    The idempotency walk and the model key on them.
    """
    assert _compaction.elision_placeholder("read_file", {"path": "src/foo.py"}) == (
        "<elided by context compaction: the result of read_file src/foo.py was"
        " replaced with this short marker to keep the loop's cumulative input"
        " bounded. If you still need it, re-read only the part you need"
        " (read_file with a targeted start_line/limit); do not re-issue the"
        " identical call.>"
    )


def test_stats_carry_elided_identities() -> None:
    big = "x" * 1000
    conv = _conversation.Conversation()
    _add_exchange(conv, ("read_file", {"path": "a.py"}, big))
    _add_exchange(conv, ("read_file", {"path": "b.py"}, big))
    _add_exchange(conv, ("read_file", {"path": "c.py"}, big))
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=1500, keep_recent=2)
    assert stats.elided_calls == ("read_file a.py",)
    assert stats.gist_paths == ()
    assert stats.demoted_paths == ()


def test_context_chars_counts_a_reasoning_model_s_thinking() -> None:
    """Context chars count a reasoning model's thinking blocks.

    `Conversation.to_wire` sends an assistant turn's raw blocks verbatim, so the thinking is in
    every later request; the tier-2 trigger has to measure it.
    """
    conv = _conversation.Conversation()
    conv.assistant(
        [
            {"type": "thinking", "thinking": "R" * 10_000, "signature": "sig"},
            {"type": "text", "text": "ok"},
        ]
    )
    total = _compaction.context_chars(conv)
    assert total >= 10_000, f"thinking is on the wire but counted as {total}"


def test_context_chars_counts_an_unknown_block_type() -> None:
    """Context chars count a block type nobody has met yet."""
    conv = _conversation.Conversation()
    conv.assistant([{"type": "something_new", "payload": "P" * 5_000}])
    assert _compaction.context_chars(conv) >= 5_000


def test_recent_tail_start_respects_cap_and_boundaries() -> None:
    conv = _conversation.Conversation()
    conv.notice("task")
    conv.assistant([{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}])
    last = conv.turns[-1]
    conv.results(
        [
            _conversation.ToolResultItem(
                tool_use_id="t1",
                content="x" * 100,
                for_call=last.tool_uses[0],  # type: ignore[union-attr]
            )
        ]
    )
    conv.assistant([{"type": "text", "text": "y" * 100}])
    turns = conv.turns

    # Cap covering only the final text turn: the tail starts there.
    assert _compaction.recent_tail_start(turns, 150) == 3
    # A results-turn start is unsafe (its call was summarised away), so the cap advances past it.
    assert _compaction.recent_tail_start(turns, 210) == 3
    # Cap covering the balanced call+result+text triple keeps all three.
    assert _compaction.recent_tail_start(turns, 100_000) == 1
    # Cap 0 keeps nothing.
    assert _compaction.recent_tail_start(turns, 0) == len(turns)
    # A cap smaller than the newest exchange keeps that exchange: its results may be undelivered.
    assert _compaction.recent_tail_start(turns, 50) == 3


def _conv_with_repeated_reads(payload: str) -> _conversation.Conversation:
    conv = _conversation.Conversation()
    conv.notice("task")
    for tid in ("t1", "t2", "t3"):
        conv.assistant(
            [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "a.py"}}]
        )
        last = conv.turns[-1]
        conv.results(
            [
                _conversation.ToolResultItem(
                    tool_use_id=tid,
                    content=payload,
                    for_call=last.tool_uses[0],  # type: ignore[union-attr]
                )
            ]
        )
    return conv


def test_tier1_dedupes_identical_results_keeping_the_newest() -> None:
    """Tier 1 dedupes identical results, keeping the newest whole; lossless, so no knob."""
    payload = "x" * 1_000
    conv = _conv_with_repeated_reads(payload)
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=1_500, keep_recent=1)
    # Both older copies dedupe; only the newest (t3) keeps the bytes.
    assert len(stats.deduped_calls) == 2
    result_turns = [t for t in conv.turns if isinstance(t, _conversation.UserTurn) and t.items]
    contents = [
        item.content
        for t in result_turns
        for item in t.items
        if isinstance(item, _conversation.ToolResultItem)
    ]
    assert sum(c == payload for c in contents) == 1  # only the newest copy keeps the bytes
    assert any(c.startswith(f"{_compaction.ELISION_PREFIX} (duplicate)") for c in contents)
    assert stats.deduped_calls == ("read_file a.py", "read_file a.py")


def test_a_duplicate_marker_claims_no_copy_the_same_pass_elides() -> None:
    """A duplicate marker never points at a copy the same pass elides."""
    payload = "x" * 4_000
    conv = _conversation.Conversation()
    conv.notice("task")
    calls = [("t1", "read_file", {"path": "a.py"}), ("t2", "read_file", {"path": "a.py"})]
    calls += [(f"c{i}", "run_command", {"command": f"echo {i}"}) for i in range(3)]
    for tid, name, args in calls:
        conv.assistant([{"type": "tool_use", "id": tid, "name": name, "input": args}])
        last = conv.turns[-1]
        assert isinstance(last, _conversation.AssistantTurn)
        conv.results(
            [
                _conversation.ToolResultItem(
                    tool_use_id=tid, content=payload, for_call=last.tool_uses[0]
                )
            ]
        )
    # The duplicated read is older than the kept tail, so its newest copy is eligible too.
    _compaction.compact_old_tool_results(conv, max_total_bytes=9_000, keep_recent=2)

    result_turns = [t for t in conv.turns if isinstance(t, _conversation.UserTurn) and t.items]
    contents = [
        item.content
        for t in result_turns
        for item in t.items
        if isinstance(item, _conversation.ToolResultItem)
    ]
    assert any(c.startswith(f"{_compaction.ELISION_PREFIX} (duplicate)") for c in contents)
    assert payload not in contents[:2], "the deduped read still holds a full copy"
    assert not any("newer result" in c for c in contents)


def test_a_duplicate_marker_never_grows_the_result_it_replaces() -> None:
    """A duplicate marker never grows the result it replaces."""
    payload = "z" * 210  # over _DEDUP_MIN_CHARS, under the marker's own length
    conv = _conversation.Conversation()
    conv.notice("task")
    for tid in ("t1", "t2", "t3"):
        conv.assistant(
            [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "a.py"}}]
        )
        last = conv.turns[-1]
        assert isinstance(last, _conversation.AssistantTurn)
        conv.results(
            [
                _conversation.ToolResultItem(
                    tool_use_id=tid, content=payload, for_call=last.tool_uses[0]
                )
            ]
        )
    before = sum(
        len(item.content)
        for turn in conv.turns
        if isinstance(turn, _conversation.UserTurn)
        for item in turn.items
        if isinstance(item, _conversation.ToolResultItem)
    )

    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=1)

    after = sum(
        len(item.content)
        for turn in conv.turns
        if isinstance(turn, _conversation.UserTurn)
        for item in turn.items
        if isinstance(item, _conversation.ToolResultItem)
    )
    assert after <= before, f"compaction grew the conversation: {before} -> {after}"
    assert len(stats.deduped_calls) == 0


def test_tier1_dedup_alone_can_satisfy_the_budget() -> None:
    """When freeing duplicates gets the total under the threshold, nothing real is elided."""
    payload = "y" * 1_000
    conv = _conv_with_repeated_reads(payload)
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=2_500, keep_recent=1)
    assert len(stats.deduped_calls) >= 1
    assert len(stats.elided_calls) == 0


def test_tier1_dedup_skips_small_and_different_results() -> None:
    conv = _conversation.Conversation()
    conv.notice("task")
    for tid, body in (("t1", "tiny"), ("t2", "tiny"), ("t3", "z" * 900), ("t4", "w" * 900)):
        conv.assistant(
            [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "b.py"}}]
        )
        last = conv.turns[-1]
        conv.results(
            [
                _conversation.ToolResultItem(
                    tool_use_id=tid,
                    content=body,
                    for_call=last.tool_uses[0],  # type: ignore[union-attr]
                )
            ]
        )
    stats = _compaction.compact_old_tool_results(conv, max_total_bytes=100, keep_recent=1)
    # "tiny" is under the dedup floor; the 900-char bodies differ: no dedup.
    assert len(stats.deduped_calls) == 0


def test_strip_old_thinking_clears_all_but_the_newest_turns() -> None:
    """Old assistant turns lose their thinking blocks.

    The newest keep_thinking_turns keep theirs.
    """
    conv = _conversation.Conversation()
    conv.notice("task")
    for i in range(3):
        conv.assistant(
            [
                {"type": "thinking", "thinking": f"reasoning {i} " + "t" * 100},
                {"type": "text", "text": f"answer {i}"},
            ]
        )
    n_turns, n_chars = _compaction.strip_old_thinking(conv, keep_turns=1)
    assert n_turns == 2 and n_chars > 200
    assistants = [t for t in conv.turns if isinstance(t, _conversation.AssistantTurn)]
    kinds = [[b["type"] for b in t.raw_content] for t in assistants]
    assert kinds == [["text"], ["text"], ["thinking", "text"]]
    # Idempotent: nothing left to strip on the older turns.
    assert _compaction.strip_old_thinking(conv, keep_turns=1) == (0, 0)


def test_strip_thinking_preserves_tool_use_pairing() -> None:
    conv = _conversation.Conversation()
    conv.notice("task")
    conv.assistant(
        [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
        ]
    )
    last = conv.turns[-1]
    conv.results(
        [_conversation.ToolResultItem(tool_use_id="t1", content="body", for_call=last.tool_uses[0])]  # type: ignore[union-attr]
    )
    conv.assistant([{"type": "text", "text": "done"}])
    removed = conv.strip_thinking(1)
    assert removed > 0
    wire = conv.to_wire()
    # The tool_use block and its result still pair on the wire.
    assert any(
        b.get("type") == "tool_use" and b.get("id") == "t1"
        for m in wire
        if isinstance(m.get("content"), list)
        for b in m["content"]
    )


def test_restart_summary_parser_ignores_marker_text_inside_a_pin() -> None:
    """The restart summary parser ignores the restart label inside a verbatim operator pin."""
    from agent6.prompts import revision

    notice = revision.context_restart_notice("run", pins=("Preserve PROGRESS SUMMARY:\nverbatim",))
    summary = "actual prior progress"
    assert revision.progress_summary_from_notice(notice + summary) == summary


def test_restart_notice_re_shows_the_operator_rulings() -> None:
    """A compaction restart re-shows DECISIONS.md between the pins and the summary."""
    from agent6.prompts import revision

    notice = revision.context_restart_notice(
        "run", pins=("pin one",), decisions="- Q: modal?\n  A: no"
    )
    head, rulings, summary = notice.partition("OPERATOR RULINGS (recorded, still binding):")
    assert "pin one" in head and rulings and "- Q: modal?\n  A: no" in summary
    assert summary.index("A: no") < summary.index("PROGRESS SUMMARY")
    assert "OPERATOR RULINGS" not in revision.context_restart_notice("run")


def test_a_refused_checkoff_id_does_not_drop_the_rest(tmp_path: pathlib.Path) -> None:
    """A refused check-off id does not drop the ids and new tasks after it."""
    from unittest import mock

    from agent6.config import Config
    from agent6.graph import curator as graph_curator
    from agent6.graph import models
    from agent6.harness import loop
    from agent6.sessions import layout

    def draft(title: str) -> models.TaskNodeDraft:
        return models.TaskNodeDraft(title=title, depends_on=(), created_by="planner")

    curator = graph_curator.GraphCurator(
        layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    )
    root = curator.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=draft("run root")))
    container = curator.add_subtask(
        models.AddSubtaskIntent(parent_id=root.id, draft=draft("container"))
    )
    curator.add_subtask(models.AddSubtaskIntent(parent_id=container.id, draft=draft("open child")))
    done = curator.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=draft("finished")))
    wf = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=Config(),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        mode="run",
        curator=curator,
    )
    valid = {nid for nid, node in curator.nodes().items() if node.parent_id is not None}
    summary = (
        "progress so far\n\n```checkoff\n"
        f'{{"completed_ids": ["{container.id}", "{done.id}"],'
        ' "new_tasks": ["a newly discovered task"]}\n```\n'
    )

    wf.compactor.apply_checkoff(summary, valid_ids=valid, root_id=root.id)

    nodes = curator.nodes()
    assert nodes[container.id].status != "passed"  # the container refusal stands
    assert nodes[done.id].status == "passed"
    assert "a newly discovered task" in {node.title for node in nodes.values()}
