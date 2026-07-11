#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Snapshot an agent6 TUI screen to an image, for visual inspection without a
terminal (dev + agent use: "see" the dashboard while iterating on it).

Mounts the real ``Agent6TUI`` on a run's ``logs.jsonl`` via Textual's headless
Pilot, lets the reader thread fold the log, then exports the rendered screen.
Textual exports SVG; if the output path ends in ``.png`` and a converter is found
(a chromium binary, or ``rsvg-convert``), it is rasterised so tools that only read
PNG/JPG can view it.

Usage:
    uv run python docs/screenshots/tui_snapshot.py <run_dir> <out.(svg|png)> [screen]

    screen: transcript (default: the conversation the app opens on)
            | dashboard (Ctrl+D toggles it up) | log

    <run_dir> is any run directory holding a logs.jsonl, e.g.
    $XDG_STATE_HOME/agent6/<repo-id>/runs/<run-id>. Pair with llm_proxy.py's
    replay mode to snapshot a deterministic, key-free run.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from agent6.ui.tui.app import Agent6TUI

# Keypresses to reach each screen from the conversation the app opens on.
_KEYS: dict[str, list[str]] = {"transcript": [], "dashboard": ["ctrl+d"], "log": ["ctrl+d", "l"]}


def _find_chromium() -> str | None:
    for cand in (
        "chromium-browser",
        "chromium",
        "google-chrome",
        "google-chrome-stable",
    ):
        if found := shutil.which(cand):
            return found
    # Playwright's cached headless shell, if a browser was installed that way.
    cache = Path.home() / ".cache/ms-playwright"
    if cache.is_dir():
        for p in sorted(cache.rglob("chrome*")):
            if p.is_file() and p.name in ("chrome", "chrome-headless-shell"):
                return str(p)
    return None


def _svg_to_png(svg: Path, png: Path) -> bool:
    if rsvg := shutil.which("rsvg-convert"):
        subprocess.run([rsvg, "-o", str(png), str(svg)], check=False)
        return png.exists()
    if chrome := _find_chromium():
        subprocess.run(
            [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--force-device-scale-factor=1",
                "--window-size=1500,900",
                f"--screenshot={png}",
                f"file://{svg}",
            ],
            check=False,
            capture_output=True,
        )
        return png.exists()
    return False


async def _snapshot(run_dir: Path, out: Path, screen: str) -> None:
    app = Agent6TUI(run_dir)
    async with app.run_test(size=(150, 42)) as pilot:
        for _ in range(80):  # let the reader thread replay + fold the log
            await pilot.pause()
            if app.state.finished or app.state.tool_calls:
                break
        app._tick()  # force the coalesced repaint (same as the headless tests)
        await pilot.pause()
        for key in _KEYS.get(screen, []):
            await pilot.press(key)  # toggle to the dashboard / open the log
            await pilot.pause()
        app._tick()  # the dashboard repaints only while it is the top screen
        await pilot.pause()
        svg = out if out.suffix == ".svg" else out.with_suffix(".svg")
        app.save_screenshot(str(svg))
        if out.suffix == ".png":
            if _svg_to_png(svg, out):
                print(f"wrote {out}")
            else:
                print(f"wrote {svg} (no SVG->PNG converter found; open it in a browser)")
        else:
            print(f"wrote {svg}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    screen = sys.argv[3] if len(sys.argv) > 3 else "transcript"
    if not (run_dir / "logs.jsonl").is_file():
        print(f"ERROR: no logs.jsonl in {run_dir}", file=sys.stderr)
        return 2
    asyncio.run(_snapshot(run_dir, out, screen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
