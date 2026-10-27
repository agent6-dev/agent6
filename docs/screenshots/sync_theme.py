"""Sync every VHS tape's theme with the recording terminal palette."""

from __future__ import annotations

import pathlib
import re

from agent6.ui.tui import theme


def main() -> None:
    """Rewrite Set Theme lines without changing the recording commands."""
    theme_line = f"Set Theme {theme.render_vhs_theme()}"
    for tape in sorted(pathlib.Path(__file__).resolve().parent.rglob("*.tape")):
        original = tape.read_text(encoding="utf-8")
        synced = re.sub(r"^Set Theme .*$", theme_line, original, flags=re.MULTILINE)
        if synced != original:
            tape.write_text(synced, encoding="utf-8")


if __name__ == "__main__":
    main()
