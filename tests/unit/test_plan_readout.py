# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One wording of a subscription account's reading: percent, reset, when it was read."""

from __future__ import annotations

import time

import pytest

from agent6 import budget
from agent6.viewmodel import format, state_types, state_view


@pytest.mark.parametrize(
    ("seconds", "words"),
    [
        (30, "under a minute"),
        (90, "2 min"),
        (3600 * 2 + 600, "2 h 10 min"),
        (86400 * 3 + 7200, "3 d 2 h"),
    ],
)
def test_a_duration_reads_in_its_largest_two_units(seconds: float, words: str) -> None:
    assert format.duration_words(seconds) == words


def test_the_readout_names_the_percent_the_reset_and_the_reading_time() -> None:
    now = 1_000_000.0
    assert format.plan_readout(73.0, 0.0, now=now) == "73%"
    assert format.plan_readout(73.0, now + 7800, now=now) == "73% · resets in 2 h 10 min"
    assert (
        format.plan_readout(73.0, now + 7800, now=now, observed_at=now - 240)
        == "73% · resets in 2 h 10 min · read 4 min ago"
    )
    assert format.plan_readout(73.0, now + 7800, now=now, observed_at=now - 5) == (
        "73% · resets in 2 h 10 min · read just now"
    )
    # A reading older than its window: the percent is as read, the reset has passed.
    assert format.plan_readout(52.0, now - 100, now=now, observed_at=now - 86400 * 5) == (
        "52% · window reset since · read 5 d 0 h ago"
    )


def test_the_snapshot_carries_the_server_rendered_plan_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web reads `budget.plan_text`; it keeps no copy of the wording."""
    monkeypatch.setattr(time, "time", lambda: 10_000.0)
    assert state_view.session_state_as_dict(state_types.SessionState())["budget"]["plan_text"] == ""
    account = state_types.AccountReading(
        provider="chatgpt",
        used_percent=52.0,
        resets_at=17_800.0,
        windows=(budget.PlanWindow("primary", 52.0, 10080, 17_800.0),),
        observed_at=9_760.0,
    )
    state = state_types.SessionState(budget=state_types.BudgetView(account=account))
    payload = state_view.session_state_as_dict(state)
    assert payload["budget"]["plan_text"] == "52% · resets in 2 h 10 min · read 4 min ago"
