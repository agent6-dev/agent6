# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ConsoleView: the live CLI stream shows tools and never a blank response block."""

from __future__ import annotations

from io import StringIO

from agent6.cli._console_view import ConsoleView


def _render(events: list[dict[str, object]]) -> str:
    buf = StringIO()
    view = ConsoleView(buf, color=False)
    for event in events:
        view.feed(event)
    return buf.getvalue()


def test_reasoning_tool_call_and_result_all_render() -> None:
    out = _render(
        [
            {"type": "run.start", "user_task": "fix the failing test"},
            {"type": "role.call", "role": "worker"},
            {"type": "role.thinking_delta", "role": "worker", "text": "let me read the file"},
            {"type": "role.result", "role": "worker"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
            {"type": "run.end", "all_passed": True, "reason": "finish_run"},
        ]
    )
    assert "fix the failing test" in out
    assert "let me read the file" in out  # reasoning shown
    assert "→ read_file" in out and "a.py" in out  # the tool call, invisible before
    assert "⎿" in out and "12 bytes" in out  # its result
    assert "done" in out


def test_whitespace_only_text_prints_no_empty_block() -> None:
    # The turn streams only whitespace text then calls a tool. The old renderer
    # printed a "── worker: response ──" bar with nothing under it; this must not.
    out = _render(
        [
            {"type": "role.call", "role": "worker"},
            {"type": "role.text_delta", "role": "worker", "text": "  \n "},
            {"type": "role.result", "role": "worker"},
            {"type": "tool.call", "name": "apply_edit", "args": {}},
            {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "ok"},
        ]
    )
    assert "worker: response" not in out
    non_empty = [ln for ln in out.splitlines() if ln.strip()]
    assert non_empty and non_empty[0].strip().startswith("→ apply_edit")


def test_failed_tool_shows_its_output_tail() -> None:
    out = _render(
        [
            {"type": "tool.call", "name": "run_command", "args": {"command": "ls /nope"}},
            {
                "type": "tool.result",
                "name": "run_command",
                "ok": False,
                "summary": "exit=2",
                "stderr_tail": "ls: /nope: No such file or directory",
            },
        ]
    )
    assert "→ run_command" in out
    assert "No such file" in out


def test_steer_request_closes_open_dim_block() -> None:
    # A Ctrl-C pause message prints to the same terminal; the open dim thinking
    # block must be closed (reset) first so the message doesn't inherit the dim.
    buf = StringIO()
    view = ConsoleView(buf, color=True)
    view.feed({"type": "role.thinking_delta", "text": "pondering the fix"})
    assert not buf.getvalue().endswith("\033[0m\n")  # block still open
    view.feed({"type": "run.steer_requested", "source": "sigint"})
    assert buf.getvalue().endswith("\033[0m\n")  # closed + reset before the message prints
