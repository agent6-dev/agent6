# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validation + opt-in logic for the review panel's config surface.

These lock in three pre-squash fixes: malformed `review_seats` are rejected, an
unreachable quorum gate is caught at load time, and a bare `critic != off` config
keeps the legacy single critic instead of being silently downgraded to the
advisory panel.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent6.cli.providers import review_panel_configured
from agent6.config import Config, WorkflowConfig


def test_review_seats_malformed_rejected() -> None:
    for bad in (("security@@x",), ("security@anthropic",), ("@/model",), ("  ",)):
        with pytest.raises(ValidationError):
            WorkflowConfig(review_seats=bad)


def test_review_seats_valid_forms_accepted() -> None:
    wf = WorkflowConfig(review_seats=("security", "@anthropic/claude-opus-4-8", "x@p/a/b"))
    assert len(wf.review_seats) == 3  # bare persona, @provider/model, model-with-slash


def test_quorum_gt1_needs_distinct_models() -> None:
    # Same-model panel can reach at most one block -> quorum=2 is unreachable.
    with pytest.raises(ValidationError, match="DISTINCT"):
        WorkflowConfig(review_decision="quorum", review_quorum=2)
    # Two distinct models satisfy it.
    ok = WorkflowConfig(
        review_decision="quorum",
        review_quorum=2,
        review_seats=("a@p1/m1", "b@p2/m2"),
    )
    assert ok.review_quorum == 2


def test_panel_configured_distinguishes_bare_critic_from_panel() -> None:
    # Bare critic, no review_* keys -> NOT a panel (keeps the legacy gating critic).
    bare = Config.model_validate({"workflow": {"critic": "before_finish"}})
    assert review_panel_configured(bare) is False
    # Any explicit review_* opt-in -> panel.
    for wf in (
        {"review_panel_size": 3},
        {"review_decision": "veto"},
        {"review_personas": ["security"]},
        {"review_tier": "explore"},
        {"review_seats": ["security@p/m"]},
    ):
        cfg = Config.model_validate({"workflow": {"critic": "before_finish", **wf}})
        assert review_panel_configured(cfg) is True
