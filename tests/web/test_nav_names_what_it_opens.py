# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The nav rail names the thing it opens.

The tab lists every bucket -- runs, plans and asks -- under a card headed
"Sessions", while the rail beside it said "Runs". One of the two was wrong
about the same list.
"""

from __future__ import annotations

import re

from agent6.ui.web.page import PAGE_HTML


def _nav_labels() -> list[str]:
    """The rail's link text (both the wide rail and the mobile menu)."""
    return [
        re.sub(r"[^A-Za-z ]", "", re.sub(r"<[^>]+>", "", chunk)).strip()
        for chunk in re.findall(r'<a href="#/" data-tab="hub"[^>]*>(.*?)</a>', PAGE_HTML)
    ]


def test_the_hub_tab_is_not_called_runs() -> None:
    labels = _nav_labels()
    assert labels, "the hub nav links moved; this test is no longer reading them"
    assert all(label == "Sessions" for label in labels), labels


def test_nothing_on_the_page_still_calls_the_hub_runs() -> None:
    assert ">Runs<" not in PAGE_HTML
    assert 'title="Runs"' not in PAGE_HTML
