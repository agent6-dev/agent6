# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web hub shows the server's merge note and keeps no copy of its wording."""

from __future__ import annotations

from agent6.ui.web import page


def test_the_session_row_renders_the_note_the_server_sends() -> None:
    js = page.CLIENT_JS
    assert "r.merge_note" in js
    assert "same paths as" not in js and "open findings" not in js
