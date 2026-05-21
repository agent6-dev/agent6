# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for workflow profiles."""

from __future__ import annotations

import pytest

from agent6.workflows.profiles import DEFAULT_PROFILE, PROFILES, Profile, TaskClass


def test_profiles_table_covers_every_task_class() -> None:
    """PROFILES must have an entry for every value of TaskClass."""
    assert set(PROFILES.keys()) == set(TaskClass)


def test_trivial_profile_skips_critic_and_planner() -> None:
    p = PROFILES[TaskClass.TRIVIAL]
    assert p.skip_critic is True
    assert p.skip_planner is True
    assert p.max_step_retries >= 1


def test_single_step_profile_skips_critic_but_keeps_planner() -> None:
    p = PROFILES[TaskClass.SINGLE_STEP]
    assert p.skip_critic is True
    assert p.skip_planner is False


def test_multi_step_profile_keeps_full_pipeline() -> None:
    p = PROFILES[TaskClass.MULTI_STEP]
    assert p.skip_critic is False
    assert p.skip_planner is False
    assert p.enable_escalation is True


def test_exploration_profile_allows_more_retries() -> None:
    p = PROFILES[TaskClass.EXPLORATION]
    assert p.max_step_retries >= PROFILES[TaskClass.MULTI_STEP].max_step_retries


def test_default_profile_is_conservative() -> None:
    # The fallback must NOT skip critic/planner: a missing triage provider
    # must not silently downgrade the workflow.
    assert DEFAULT_PROFILE.skip_critic is False
    assert DEFAULT_PROFILE.skip_planner is False


def test_skip_planner_without_skip_critic_is_rejected() -> None:
    with pytest.raises(ValueError, match="skip_planner=True requires skip_critic=True"):
        Profile(
            skip_critic=False,
            skip_planner=True,
            enable_escalation=False,
            max_step_retries=1,
        )


def test_zero_retries_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_step_retries must be >= 1"):
        Profile(
            skip_critic=False,
            skip_planner=False,
            enable_escalation=False,
            max_step_retries=0,
        )


def test_zero_escalate_after_attempt_is_rejected() -> None:
    with pytest.raises(ValueError, match="escalate_after_attempt must be >= 1"):
        Profile(
            skip_critic=False,
            skip_planner=False,
            enable_escalation=True,
            max_step_retries=2,
            escalate_after_attempt=0,
        )


def test_multi_step_uses_tiered_escalation() -> None:
    """Multi-step must give the primary worker at least one retry before
    paying for opus escalation — see bench/results.md tier-4 rerun."""
    p = PROFILES[TaskClass.MULTI_STEP]
    assert p.enable_escalation is True
    assert p.escalate_after_attempt >= 2
    assert p.max_step_retries > p.escalate_after_attempt - 1


def test_trivial_does_not_escalate() -> None:
    """A trivial task is cheaper to abandon than to escalate."""
    p = PROFILES[TaskClass.TRIVIAL]
    assert p.enable_escalation is False
