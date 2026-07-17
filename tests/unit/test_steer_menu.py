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
    # Bare keywords are gone: a plain word is a steering instruction now.
    assert pause_menu(tmp_path, input_fn=_feed(["q"])) == "q"
    # Unknown slash command re-prompts (does not steer with a typo).
    out = pause_menu(tmp_path, input_fn=_feed(["/statsu", "real steer"]))
    assert out == "real steer"
    assert "unknown command" in capsys.readouterr().out
    # EOF (Ctrl-D) means continue.
    assert pause_menu(tmp_path, input_fn=_feed([])) is None


def test_pause_menu_help_names_parallel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The mid-run steer help names `/parallel`, the directive the loop dispatches
    sibling lanes for (see agent6.directive.parse_directive)."""
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/help", "go"])) == "go"
    assert "/parallel" in capsys.readouterr().out


def test_pause_menu_parallel_directive_passes_through_verbatim(tmp_path: Path) -> None:
    """`/parallel <task>` has a space, so the pause menu sends it to the run
    verbatim (the loop's _maybe_handle_steer parses it); it is never swallowed as
    a menu command. This is why mid-run `/parallel` needs no composer change."""
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/parallel 2 add a greeting"])) == (
        "/parallel 2 add a greeting"
    )


def test_pause_menu_prefixes_and_word_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A unique prefix fires the command, an ambiguous one re-asks, and a line
    with spaces is always a steering instruction (no quoting needed)."""
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    # /sta is uniquely /status; /st matches /status and /stop -> re-ask.
    assert pause_menu(tmp_path, input_fn=_feed(["/sta", "/st", "/stop"])) == "abort"
    printed = capsys.readouterr().out
    assert "running" in printed  # /sta printed the status line
    assert "ambiguous" in printed and "/status" in printed and "/stop" in printed
    # A multi-word line starting with "/" is a steer, never a command.
    assert pause_menu(tmp_path, input_fn=_feed(["/stop hammering the API"])) == (
        "/stop hammering the API"
    )
    # /h is the /help alias.
    assert pause_menu(tmp_path, input_fn=_feed(["/h", "go"])) == "go"
    assert "/detach" in capsys.readouterr().out


def test_pause_menu_compact_requests_compaction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.bridge.approval import compact_request_pending
    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/compact"])) is None  # EOF -> continue
    assert compact_request_pending(tmp_path) is True
    assert "compaction requested" in capsys.readouterr().out


def test_pause_menu_status_shows_ctx_and_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """/status includes the context fill (tokens + % of the model window) and
    the sandbox profile the run started with."""
    import json

    from agent6.ui.cli._steer_menu import pause_menu

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "run.start", "user_task": "polish", "mode": "run"},
                {
                    "type": "role.call",
                    "role": "worker",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                },
                {"type": "role.result", "role": "worker", "tokens_in": 90_000, "tokens_out": 10},
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"profile": "paranoid"}), encoding="utf-8")
    assert pause_menu(tmp_path, input_fn=_feed(["/status"])) is None
    printed = capsys.readouterr().out
    assert "ctx 90,000 tok" in printed
    assert "(45%)" in printed  # 90k of the 200k sonnet window
    assert "profile paranoid" in printed


# --- skill slash commands ----------------------------------------------------


def _skill_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Install fake skills into an isolated data dir and chdir to tmp."""
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    for name in names:
        d = tmp_path / "data" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\nGRUNT {name}\n",
            encoding="utf-8",
        )


def test_skill_command_whole_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    out = pause_menu(tmp_path, input_fn=_feed(["/caveman"]))
    assert out is not None
    assert "GRUNT caveman" in out
    assert '<skill name="caveman">' in out


def test_skill_command_with_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    out = pause_menu(tmp_path, input_fn=_feed(["/caveman lite"]))
    assert out is not None
    assert "Skill arguments: lite" in out
    assert "GRUNT caveman" in out


def test_non_skill_line_with_spaces_stays_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    assert pause_menu(tmp_path, input_fn=_feed(["/focus on tests"])) == "/focus on tests"
    assert pause_menu(tmp_path, input_fn=_feed(["fix the parser"])) == "fix the parser"


def test_builtin_wins_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "status")
    # /status must still be the built-in info command (prints, re-prompts, EOF)
    assert pause_menu(tmp_path, input_fn=_feed(["/status"])) is None


def test_disabled_skill_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    _skill_env(tmp_path, monkeypatch, "caveman")
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "config.toml").write_text(
        '[skills.state]\ncaveman = "disabled"\n', encoding="utf-8"
    )
    out = pause_menu(tmp_path, input_fn=_feed(["/caveman", "steer text"]))
    # unknown command message printed, then the steer line is returned
    assert out == "steer text"


def test_skill_menu_table_lists_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.ui.cli._steer_menu import skill_menu_table

    _skill_env(tmp_path, monkeypatch, "caveman", "tidy")
    table = skill_menu_table()
    assert set(table) == {"/caveman", "/tidy"}
    assert table["/caveman"][0] == "Use when testing caveman."
