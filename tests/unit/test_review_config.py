# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validation + opt-in logic for the review panel's config surface.

These lock in three pre-squash fixes: malformed `seats` are rejected, an
unreachable quorum gate is caught at load time, and a bare `trigger != off` config
keeps the legacy single critic instead of being silently downgraded to the
advisory panel.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent6.app.providers import review_panel_configured
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


def test_panel_configured_distinguishes_bare_critic_from_panel() -> None:
    # Bare critic, no panel keys -> NOT a panel (keeps the legacy gating critic).
    bare = Config.model_validate({"review": {"trigger": "before_finish"}})
    assert review_panel_configured(bare) is False
    # Any explicit panel opt-in -> panel.
    for rv in (
        {"panel_size": 3},
        {"decision": "veto"},
        {"personas": ["security"]},
        {"tier": "explore"},
        {"seats": ["security@p/m"]},
    ):
        cfg = Config.model_validate({"review": {"trigger": "before_finish", **rv}})
        assert review_panel_configured(cfg) is True
