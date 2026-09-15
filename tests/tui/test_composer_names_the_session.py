# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The conversation composer does not call every session a run.

One conversation view serves runs, plans and asks, so a fixed "continue the
run" is wrong two times in three -- the same wording the web page had, on the
other surface.
"""

from __future__ import annotations

from agent6.ui.tui.composer import composer_labels


def test_a_finished_session_offers_to_continue_the_session() -> None:
    title, keys = composer_labels("resume")
    assert "run" not in title, title
    assert "session" in title
    assert "resumes" in keys


def test_a_live_session_offers_to_steer_the_session() -> None:
    title, _keys = composer_labels("steer")
    assert "run" not in title, title
    assert "session" in title
    assert "/pin" in title and "/compact" in title


def test_a_finished_run_asks_for_new_work() -> None:
    """A run the agent finished green has nothing to continue; the web asks
    "what should it do next" and the TUI said "continue this session"."""
    assert composer_labels("resume", needs_new_work=True)[0] == "what should it do next"
    assert composer_labels("resume")[0] == "continue this session"
    assert composer_labels("resume", continue_as="f", needs_new_work=True)[0] == "continue as f"
