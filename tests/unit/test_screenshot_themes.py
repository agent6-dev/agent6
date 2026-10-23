from __future__ import annotations

import json
import pathlib
import runpy

import pytest

from agent6.ui.tui import theme

_SCREENSHOTS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots"
_TAPES = sorted(_SCREENSHOTS.rglob("*.tape"))


@pytest.mark.parametrize("tape", _TAPES, ids=lambda tape: tape.name)
def test_tape_theme_matches_the_tui_palette(tape: pathlib.Path) -> None:
    """Every VHS theme comes from the TUI palette, including readable bright black."""
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
