# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Skip the `browser` tests where the Playwright browser they drive is not installed."""

from __future__ import annotations

import functools
import os
import pathlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import BrowserType


@functools.cache
def _missing() -> str:
    """Return why Chromium cannot run here, "" when it can."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "the playwright package is not installed"
    # The session's throwaway XDG_CACHE_HOME hides the operator's browsers; Playwright reads
    # this variable before it looks there.
    os.environ.setdefault(
        "PLAYWRIGHT_BROWSERS_PATH", str(pathlib.Path.home() / ".cache" / "ms-playwright")
    )
    with sync_playwright() as pw:
        exe = pathlib.Path(pw.chromium.executable_path)
    if exe.exists():
        return ""
    return f"Playwright's Chromium is not at {exe}: run `uv run playwright install chromium`"


def require_browser() -> None:
    """Skip the test when the browser is missing, naming what to run."""
    if reason := _missing():
        pytest.skip(reason)


def require_installed(engine: BrowserType) -> None:
    """Skip the test when a browser beside Chromium is missing, naming what to run."""
    if not pathlib.Path(engine.executable_path).exists():
        pytest.skip(
            f"Playwright's {engine.name.capitalize()} is not installed:"
            f" run `uv run playwright install {engine.name}`"
        )
