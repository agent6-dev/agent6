# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The mid-run Ctrl-C menu maps operator input to a canonical steer action."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.ui.cli._steer_menu import normalize_steer_choice


def test_stop_keys_map_to_abort() -> None:
    for key in ("q", "Q", "quit", "stop", "abort", "  ABORT  "):
        assert normalize_steer_choice(key) == "abort"


def test_detach_keys_map_to_detach() -> None:
    for key in ("d", "D", "detach", " Detach "):
        assert normalize_steer_choice(key) == "detach"


def test_blank_continues() -> None:
    assert normalize_steer_choice("") == ""
    assert normalize_steer_choice("   ") == ""


def test_none_stays_none() -> None:
    assert normalize_steer_choice(None) is None


def test_instruction_passes_through() -> None:
    assert normalize_steer_choice("focus on the parser") == "focus on the parser"
    # a sentence that merely starts with a keyword is an instruction, not a command
    assert normalize_steer_choice("abort the current plan") == "abort the current plan"


def _feed(lines: list[str]) -> Callable[[str], str]:
    """An input_fn that replays *lines* then raises EOF (menu -> continue)."""
    it = iter(lines)

    def fn(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return fn


def test_pause_menu_slash_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Info commands print and re-prompt; action commands return the canonical
    steer values; free text passes through as the instruction."""
    import json

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "run.start", "user_task": "polish the TUI", "mode": "run"},
                {
                    "type": "graph.update",
                    "cursor": "t1",
                    "nodes": {
                        "t1": {
                            "title": "fix the bars",
                            "parent_id": None,
                            "status": "in_progress",
                            "children": [],
                        }
                    },
                },
                {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
                {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
            )
        ),
        encoding="utf-8",
    )
    # /help + /status + /tasks print, then the free text is the steer.
    out = pause_menu(tmp_path, input_fn=_feed(["/help", "/status", "/tasks", "focus on tests"]))
    assert out == "focus on tests"
    printed = capsys.readouterr().out
    assert "/detach" in printed  # help listed the commands
    assert "running" in printed and "1 tools" in printed  # status line
    assert "fix the bars" in printed  # the task graph

    assert pause_menu(tmp_path, input_fn=_feed(["/stop"])) == "abort"
    assert pause_menu(tmp_path, input_fn=_feed(["/detach"])) == "detach"
    assert pause_menu(tmp_path, input_fn=_feed(["/continue"])) == ""
    assert pause_menu(tmp_path, input_fn=_feed(["q"])) == "abort"  # bare words still work
    # Unknown slash command re-prompts (does not steer with a typo).
    out = pause_menu(tmp_path, input_fn=_feed(["/statsu", "real steer"]))
    assert out == "real steer"
    assert "unknown command" in capsys.readouterr().out
    # EOF (Ctrl-D) means continue.
    assert pause_menu(tmp_path, input_fn=_feed([])) is None


def test_pause_menu_completer_cycles_commands() -> None:
    from agent6.ui.cli._steer_menu import _complete  # pyright: ignore[reportPrivateUsage]

    matches = []
    state = 0
    while (m := _complete("/st", state)) is not None:
        matches.append(m)
        state += 1
    assert matches == ["/status", "/stop"]
    assert _complete("plain text", 0) is None  # only slash commands complete
