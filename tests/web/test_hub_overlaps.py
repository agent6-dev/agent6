# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web hub names the awaiting-merge runs that changed the same paths."""

from __future__ import annotations

from agent6.ui.web import page


def test_the_session_row_renders_the_overlaps_the_server_sends() -> None:
    js = page.CLIENT_JS
    assert "const overlaps = (q.overlaps || [])" in js
    assert "'same paths as ' + overlaps" in js
