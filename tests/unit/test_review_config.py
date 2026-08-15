# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validation + roster logic for the review panel's config surface.

Malformed `seats` are rejected, an unreachable quorum gate is caught at load
time, and a bare `trigger != off` config builds the simple-form roster (one
reviewer-model seat) rather than a dead gate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agent6.app.providers import build_review_seats
from agent6.config import Config, ReviewConfig


def test_review_seats_malformed_rejected() -> None:
    for bad in (("security@@x",), ("security@anthropic",), ("@/model",), ("  ",)):
        with pytest.raises(ValidationError):
            ReviewConfig(seats=bad)


def test_review_seats_valid_forms_accepted() -> None:
    rv = ReviewConfig(seats=("security", "@anthropic/claude-opus-4-8", "x@p/a/b"))
    assert len(rv.seats) == 3  # bare persona, @provider/model, model-with-slash


def test_quorum_gt1_needs_distinct_models() -> None:
    # Same-model panel can reach at most one block -> quorum=2 is unreachable.
    with pytest.raises(ValidationError, match="DISTINCT"):
        ReviewConfig(decision="quorum", quorum=2)
    # Two distinct models satisfy it.
    ok = ReviewConfig(
        decision="quorum",
        quorum=2,
        seats=("a@p1/m1", "b@p2/m2"),
    )
    assert ok.quorum == 2


def test_trigger_on_with_no_seats_builds_the_one_seat_roster() -> None:
    """A bare `trigger != off` config runs the panel in its simple form: the
    session wiring asks for n=1 and gets one reviewer-model seat with a
    built-in adversarial persona; no explicit `seats` are required."""
    cfg = Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {
                "worker": {"provider": "o", "model": "m"},
                "reviewer": {"provider": "o", "model": "rm"},
            },
            "review": {"trigger": "before_finish"},
        }
    )
    seats = build_review_seats(cfg, transcript_sink=MagicMock(), budget=MagicMock(), n=1)
    assert len(seats) == 1
    assert seats[0].model == "rm"
    assert seats[0].persona  # a built-in persona, not an empty stance
