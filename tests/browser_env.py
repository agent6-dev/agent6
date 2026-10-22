# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Skip the `browser` tests where Playwright's Chromium cannot run."""

from __future__ import annotations

import functools
import os
import pathlib

import pytest


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
