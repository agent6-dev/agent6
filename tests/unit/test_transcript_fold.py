# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""TranscriptFold: the event stream folds into the right conversation items."""

from __future__ import annotations

from agent6.ui.viewmodel import TranscriptItem, fold_transcript, salient_arg


def _read(path: str) -> list[dict[str, object]]:
    return [
        {"type": "role.call", "role": "worker"},
        {"type": "role.thinking_delta", "text": "let me look"},
        {"type": "role.result"},
        {"type": "tool.call", "name": "read_file", "args": {"path": path}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    ]


def test_tool_only_turn_has_no_empty_text_item() -> None:
    # A turn that reasons then calls a tool, emitting NO assistant text, must not
    # produce a blank `text` item -- the bug behind the empty response blocks.
    items = fold_transcript(_read("a.py"))
    kinds = [i.kind for i in items]
    assert kinds == ["thinking", "tool"]
    assert not any(i.kind == "text" and not i.body for i in items)
    tool = items[1]
    assert (tool.name, tool.arg, tool.ok, tool.detail) == ("read_file", "a.py", True, "12 bytes")


def test_verify_badge_folds_into_the_tool_item() -> None:
    events = [
        {"type": "tool.call", "name": "run_verify_command", "args": {}},
        {"type": "verify.start", "cmd": ["pytest"]},
        {"type": "verify.end", "exit_code": 0, "duration_s": 0.2},
        {"type": "tool.result", "name": "run_verify_command", "ok": True, "summary": "exit=0"},
    ]
    (item,) = fold_transcript(events)
    assert item.kind == "tool" and item.ok is True
    assert item.detail == "✓ pass · 0.2s"  # the verify badge, not the raw summary


def test_finish_tool_becomes_the_verdict_not_a_step() -> None:
    events = [
        {"type": "tool.call", "name": "finish_run", "args": {"summary": "all green"}},
        {"type": "tool.result", "name": "finish_run", "ok": True, "summary": "finish_run"},
        {"type": "run.end", "all_passed": True, "reason": "finish_run"},
    ]
    items = fold_transcript(events)
    assert [i.kind for i in items] == ["done"]
    done = items[0]
    assert done.ok is True and done.body == "all green"
    assert done.detail == "0 tools · 0 commits"


def test_failed_tool_keeps_a_tail() -> None:
    events = [
        {"type": "tool.call", "name": "run_command", "args": {"command": "ls /nope"}},
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": False,
            "summary": "exit=2",
            "stderr_tail": "ls: /nope: No such file",
        },
    ]
    (item,) = fold_transcript(events)
    assert item.ok is False and "No such file" in item.tail


def test_tool_output_ansi_is_stripped_from_the_fold() -> None:
    # The fold is plain data for non-terminal surfaces (web/saved transcripts):
    # colored tool output must not leak escape sequences as literal text.
    events = [
        {"type": "tool.call", "name": "run_command", "args": {"command": "pytest"}},
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "\x1b[32mok\x1b[0m",
            "stdout_tail": "\x1b[36m[Tach]\x1b[0m 10 tests \x1b[1mpass\x1b[0m",
        },
    ]
    (item,) = fold_transcript(events)
    assert "\x1b" not in item.tail and item.tail == "[Tach] 10 tests pass"
    assert "\x1b" not in item.detail and item.detail == "ok"


def test_salient_arg_prefers_a_primary_key() -> None:
    assert salient_arg({"recursive": True, "path": "src/x.py"}) == "src/x.py"
    assert salient_arg({}) == ""
    assert salient_arg({"n": 3}) == "n=3"
    assert isinstance(TranscriptItem("marker", body="reset"), TranscriptItem)


def test_salient_arg_renders_argv_as_a_shell_line() -> None:
    # Not a Python list repr: the operator reads it as a command; a token with a
    # space is quoted the way a shell needs.
    assert salient_arg({"argv": ["cargo", "build", "--release"]}) == "cargo build --release"
    assert salient_arg({"argv": ["echo", "a b"]}) == "echo 'a b'"


def test_salient_arg_renders_ask_user_questions_as_text() -> None:
    args = {"questions": [{"question": "Which theme?"}, {"question": "Apply to TUI?"}]}
    assert salient_arg(args) == "Which theme? (+1)"


def test_interleaved_tool_calls_pair_by_name() -> None:
    # A concurrent explore-tier review panel can interleave tool.call/tool.result
    # across tools; each result must pair with its own call by name, not with the
    # next pending call by position.
    events = [
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {"type": "tool.call", "name": "grep", "args": {"pattern": "def"}},
        {"type": "tool.result", "name": "grep", "ok": False, "summary": "no match"},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    ]
    tools = {i.name: i for i in fold_transcript(events) if i.kind == "tool"}
    assert len(tools) == 2  # both paired; none dropped or mislabelled
    assert tools["grep"].ok is False and tools["grep"].detail == "no match"
    assert tools["read_file"].ok is True and tools["read_file"].detail == "12 bytes"


def test_unmatched_tool_result_is_dropped() -> None:
    # A result with no matching pending call must not crash or emit a bogus item.
    assert fold_transcript([{"type": "tool.result", "name": "ghost", "ok": True}]) == []


def test_stopped_run_done_reads_as_stopped_not_failed() -> None:
    # A steer_abort run must render "stopped", not the raw "steer_abort" nor a
    # failure -- the CLI/TUI done line shows item.name for a not-ok run.
    (done,) = fold_transcript([{"type": "run.end", "reason": "steer_abort", "all_passed": False}])
    assert done.kind == "done" and done.ok is False and done.name == "stopped"


def test_operator_steer_text_becomes_an_operator_item() -> None:
    """The loop's steer injection (a typed steer, or the follow-up a resume was
    started with) shows in the conversation as an operator turn; old logs that
    carry only a char count yield nothing."""
    from agent6.ui.viewmodel.transcript import OPERATOR, TranscriptFold
    from agent6.ui.viewmodel.transcript_style import item_lines

    fold = TranscriptFold()
    items = fold.feed({"type": "loop.steer.injected", "chars": 9, "text": "try it\nagain"})
    assert [i.kind for i in items] == ["operator"]
    assert items[0].body == "try it\nagain"
    # Rendered at every detail level, glyph + the operator's own words.
    for level in ("hidden", "collapsed", "expanded"):
        lines = item_lines(items[0], detail=level)
        flat = "".join(chunk for line in lines for chunk, _ in line)
        assert f"{OPERATOR} try it" in flat and "again" in flat
    # An old log without the text field adds no item.
    assert fold.feed({"type": "loop.steer.injected", "chars": 9}) == []
