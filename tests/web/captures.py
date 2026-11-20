from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page


def save(page: Page, name: str) -> None:
    """Save an optional capture of a browser regression's final state."""
    if directory := os.environ.get("WEB_CAPTURES"):
        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        width = page.evaluate("innerWidth")
        page.screenshot(path=str(target / f"{name}-{width}.png"), full_page=True)
