# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Measure several elements of a page in one step."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Locator


def boxes(*locators: Locator) -> list[dict[str, float]]:
    """Return each element's box, all read in one script run.

    Separate `bounding_box()` calls are separate round trips: content that arrives between
    two of them moves the later boxes, and a comparison of the two reads a gap that never was.
    """
    handles = [locator.element_handle() for locator in locators]
    return locators[0].page.evaluate(
        """elements => elements.map(element => {
          const box = element.getBoundingClientRect();
          return { x: box.x, y: box.y, width: box.width, height: box.height };
        })""",
        handles,
    )
