from __future__ import annotations

import json
import pathlib
import runpy

import pytest

from agent6.ui.tui import theme

_SCREENSHOTS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots"
_TAPES = sorted(_SCREENSHOTS.rglob("*.tape"))


@pytest.mark.parametrize("tape", _TAPES, ids=lambda tape: tape.name)
def test_tape_theme_matches_the_terminal_palette(tape: pathlib.Path) -> None:
    """Every VHS theme comes from the recording terminal palette."""
    rendered = json.loads(theme.render_vhs_theme())
    lines = [
        line
        for line in tape.read_text(encoding="utf-8").splitlines()
        if line.startswith("Set Theme ")
    ]
    assert lines, f"{tape.name} needs an explicit theme"
    for line in lines:
        assert json.loads(line.removeprefix("Set Theme ")) == rendered, (
            f"{tape.name} is stale; run uv run python docs/screenshots/sync_theme.py"
        )


def test_vhs_theme_is_independent_of_ansi_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the TUI's ANSI text colours leaves the recording terminal unchanged."""
    expected = {
        "name": "agent6-dark",
        "background": "#0f1414",
        "foreground": "#c0caf5",
        "cursor": "#06F5F3",
        "selection": "#1e2626",
        "black": "#161c1c",
        "red": "#f7768e",
        "green": "#9ece6a",
        "yellow": "#e0af68",
        "blue": "#7aa2f7",
        "magenta": "#bb9af7",
        "cyan": "#06f5f3",
        "white": "#c0caf5",
        "brightBlack": "#6e6e78",
        "brightRed": "#ff8fa3",
        "brightGreen": "#b4e07f",
        "brightYellow": "#ffa62b",
        "brightBlue": "#9cbcff",
        "brightMagenta": "#d3b4ff",
        "brightCyan": "#66fffd",
        "brightWhite": "#f2f4ff",
    }
    assert json.loads(theme.render_vhs_theme()) == expected
    monkeypatch.setattr(theme, "ANSI_THEME_DARK", theme.ANSI_THEME_LIGHT)
    assert json.loads(theme.render_vhs_theme()) == expected


def test_sync_rewrites_only_theme_lines_in_every_tape(tmp_path: pathlib.Path) -> None:
    """Sync finds nested tapes, preserves their commands and leaves a second run unchanged."""
    script = tmp_path / "sync_theme.py"
    script.write_bytes((_SCREENSHOTS / "sync_theme.py").read_bytes())
    nested = tmp_path / "nested"
    nested.mkdir()
    tapes = [tmp_path / "first.tape", nested / "second.tape"]
    for tape in tapes:
        tape.write_text('Set Width 1600\nSet Theme {"name": "stale"}\nSleep 1s\n', encoding="utf-8")
    unthemed = tmp_path / "unthemed.tape"
    unthemed.write_text("# Set Theme is a comment\nSleep 2s\n", encoding="utf-8")

    runpy.run_path(str(script), run_name="__main__")

    expected = f"Set Width 1600\nSet Theme {theme.render_vhs_theme()}\nSleep 1s\n"
    for tape in tapes:
        assert tape.read_text(encoding="utf-8") == expected
    assert unthemed.read_text(encoding="utf-8") == "# Set Theme is a comment\nSleep 2s\n"
    mtimes = [tape.stat().st_mtime_ns for tape in tapes]
    runpy.run_path(str(script), run_name="__main__")
    assert [tape.stat().st_mtime_ns for tape in tapes] == mtimes
