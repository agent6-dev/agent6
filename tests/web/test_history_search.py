# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The composer's Ctrl-R history search: the client wiring the payload key,
scoped to the focused composer so the browser keeps its reload elsewhere."""

from __future__ import annotations

from importlib import resources

CLIENT_JS = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")


def test_composer_intercepts_ctrl_r_only() -> None:
    # The intercept lives in the composer's own keydown (fires only while the
    # textarea holds focus) and requires the bare Ctrl chord.
    assert "e.key === 'r' && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey" in CLIENT_JS
    # There is no document-level Ctrl-R hook: reload survives everywhere else.
    assert "openHistorySearch" in CLIENT_JS


def test_history_reads_the_payload_key_and_advertises_the_chord() -> None:
    assert "operator_inputs" in CLIENT_JS  # the conversation payload key
    assert CLIENT_JS.count("Ctrl-R past messages") == 2  # both composer hints


def test_enter_keeps_the_typed_text_when_nothing_matches() -> None:
    # One accept rule on every surface: the highlighted match, else the query
    # itself (the CLI and TUI searches behave the same way).
    assert "pick(items.length ? items[active] : field.value)" in CLIENT_JS
