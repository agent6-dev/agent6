# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ConsoleView: the live CLI stream shows tools and never a blank response block."""

from __future__ import annotations

from io import StringIO
from typing import Any

from agent6.ui.cli._console_view import ConsoleView


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


def _graph_event(nodes: dict[str, Any], cursor: str | None = None) -> dict[str, object]:
    return {"type": "graph.update", "nodes": nodes, "cursor": cursor}


def test_plan_block_prints_when_the_dag_is_seeded() -> None:
    nodes = {
        "01A": {
            "title": "root task",
            "status": "in_progress",
            "parent_id": None,
            "children": ["01B", "01C"],
        },
        "01B": {"title": "survey", "status": "passed", "parent_id": "01A", "children": []},
        "01C": {"title": "implement", "status": "pending", "parent_id": "01A", "children": []},
    }
    out = _render([_graph_event(nodes, cursor="01C")])
    assert "plan (3 tasks)" in out
    assert "root task" in out and "survey" in out and "implement" in out
    # Nesting: children are indented under the root.
    assert "  ✓ survey" in out


def test_plan_block_reprints_only_when_the_dag_grows() -> None:
    n1 = {
        "01A": {"title": "root", "status": "pending", "parent_id": None, "children": ["01B"]},
        "01B": {"title": "a", "status": "pending", "parent_id": "01A", "children": []},
    }
    n2 = dict(n1)  # same set of tasks -> no reprint
    n3 = {**n1, "01C": {"title": "b", "status": "pending", "parent_id": "01A", "children": []}}
    n1["01A"]["children"] = ["01B"]
    n3["01A"]["children"] = ["01B", "01C"]
    out = _render([_graph_event(n1), _graph_event(n2), _graph_event(n3)])
    assert out.count("plan (2 tasks)") == 1  # seeded once
    assert out.count("plan (3 tasks)") == 1  # reprinted when it grew, not on the no-op update


def test_single_root_task_is_not_a_plan_block() -> None:
    # A plain run seeds one root task; that is not a decomposition worth a block.
    out = _render(
        [
            _graph_event(
                {"01A": {"title": "t", "status": "in_progress", "parent_id": None, "children": []}}
            )
        ]
    )
    assert "plan (" not in out


class _FakeTTY:
    """A tty-like sink: isatty() True so ConsoleView starts its heartbeat thread."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, s: str) -> int:
        self.chunks.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def getvalue(self) -> str:
        return "".join(self.chunks)


# Comfortably past the module's stall threshold (1.5s) + a couple 0.5s ticks.
_STALL_WAIT_S = 3.0


def test_cli_heartbeat_shows_working_when_the_stream_stalls() -> None:
    """A turn that goes silent mid-flight (a stalled SSE stream) shows a ticking
    'working… Ns' line so the CLI never looks hung -- the user's exact symptom."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        view.feed({"type": "role.text_delta", "text": "Let me investigate"})
        time.sleep(_STALL_WAIT_S)  # let the stall register and the spinner tick
        assert "working…" in out.getvalue()
        # Output resuming clears the spinner (\r + erase) and shows the new text.
        view.feed({"type": "role.text_delta", "text": " the theme system"})
        assert "the theme system" in out.getvalue()
        assert "\x1b[2K" in out.getvalue()  # the spinner line was erased
    finally:
        view.close()


def test_cli_heartbeat_spins_during_a_long_tool_run() -> None:
    """A long verify / run_command executes between role.result and the next
    role.call; the heartbeat must still spin so a running test suite doesn't look
    frozen (gap: a role-only flag missed this)."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        view.feed({"type": "role.result", "role": "worker"})  # turn done...
        # ...now a tool.call starts a long jail command (no result yet).
        view.feed({"type": "tool.call", "name": "run_verify_command", "args": {}})
        time.sleep(_STALL_WAIT_S)
        assert "working…" in out.getvalue()  # spins through the command, not frozen
    finally:
        view.close()


def test_cli_heartbeat_silent_on_a_non_tty() -> None:
    """No spinner thread (and no spinner bytes) when the sink is not a terminal --
    a piped/redirected run or a test stays clean."""
    import time

    buf = StringIO()
    view = ConsoleView(buf, color=False)
    view.feed({"type": "role.call", "role": "worker", "model": "m"})
    time.sleep(_STALL_WAIT_S)
    assert "working…" not in buf.getvalue()
    view.close()


def test_notice_clears_the_spinner_before_printing() -> None:
    """A workflow notice (auto-commit, critic) routes through the ConsoleView so
    it clears the spinner line first and writes to the same stream -- no garble
    with the stderr heartbeat on a shared terminal."""
    import time

    out = _FakeTTY()
    view = ConsoleView(out, color=False)  # type: ignore[arg-type]
    try:
        view.feed({"type": "role.call", "role": "worker", "model": "m"})
        time.sleep(_STALL_WAIT_S)  # spinner up
        assert "working…" in out.getvalue()
        view.notice("[agent6]   auto-commit: abc123")
        v = out.getvalue()
        assert "auto-commit: abc123" in v
        assert "\x1b[2K" in v  # the spinner line was erased before the notice
    finally:
        view.close()
