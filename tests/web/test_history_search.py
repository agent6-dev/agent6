# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The composer's Ctrl-R history search wires the payload key, scoped to the focused composer.

The browser keeps its reload elsewhere.
"""

from __future__ import annotations

from agent6.ui.web import page  # the concatenated page-family files


def test_composer_intercepts_ctrl_r_only() -> None:
    # The intercept lives in the composer's own keydown and requires the bare Ctrl chord.
    assert "e.key === 'r' && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey" in page.CLIENT_JS
    # The only ctrlKey sites are that chord and the composer's Ctrl+Enter; reload works elsewhere.
    assert page.CLIENT_JS.count("ctrlKey") == 2
    assert "openHistorySearch" in page.CLIENT_JS


def test_history_reads_the_payload_key_and_advertises_the_chord() -> None:
    assert "operator_inputs" in page.CLIENT_JS  # the conversation payload key
    # every composer hint advertises it: steer, resume, and resume-needs-work
    assert page.CLIENT_JS.count("Ctrl-R past messages") == 3


def test_enter_keeps_the_typed_text_when_nothing_matches() -> None:
    # One accept rule on every surface: the highlighted match, else the query itself.
    assert "pick(items.length ? items[active] : field.value)" in page.CLIENT_JS
