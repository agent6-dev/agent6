# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Workflow profiles — task-class -> dispatch shape.

The triage sub-agent classifies an incoming user task into a ``TaskClass``;
the workflow then looks up the matching ``Profile`` to decide which
sub-agents to skip, how many retries to allow, and whether to use the
worker-escalation model on retry.

Rationale: on the bench, the critic + planner + reviewer pipeline costs
roughly the same on a one-line bug fix as on a four-step refactor, while
delivering most of its value only on the latter. Triaging up-front and
short-circuiting the cheap end of the distribution buys back ~50% of the
average run cost on tier 1-2 tasks without measurable quality regression
on tier 3-4. See ``bench/results.md`` for the empirical breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent6.agents.triage import TaskClass

__all__ = ["DEFAULT_PROFILE", "PROFILES", "Profile", "TaskClass"]


@dataclass(frozen=True, slots=True)
class Profile:
    """How the workflow should run for one ``TaskClass``.

    A profile is descriptive: every field maps to one branch in
    ``ImplementWorkflow.run`` / ``_run_step``. The profile is fixed at the
    start of the run; it does not adapt mid-flight.
    """

    skip_critic: bool
    """When True, the critic sub-agent is bypassed and the user's raw task
    is used as the refined task. Saves ~\\$0.005-0.05 per run depending on
    the critic model. Safe for TRIVIAL / SINGLE_STEP where the task text is
    already imperative and concrete."""

    skip_planner: bool
    """When True, the planner sub-agent is bypassed and the workflow
    synthesises a single-step Plan whose acceptance is the refined task
    itself. Implies the worker has to do the whole task in one shot. Only
    safe for TRIVIAL."""

    enable_escalation: bool
    """When True, the worker's retry attempt routes to the
    ``worker_escalation`` provider (typically opus-class). When False, the
    primary worker provider is reused on retry."""

    max_step_retries: int
    """Maximum number of worker attempts per step (including the first).
    Must be >= 1."""

    escalate_after_attempt: int = 1
    """Zero-based attempt index at which to start routing to
    ``worker_escalation``. ``1`` means attempt 0 uses the primary worker and
    attempts >= 1 use escalation (the historical behaviour). Bump to ``2``
    on harder tiers to give the primary worker a second cheap shot before
    paying for opus. Ignored when ``enable_escalation`` is False or no
    ``worker_escalation`` provider is configured. Must be >= 1."""

    def __post_init__(self) -> None:
        if self.max_step_retries < 1:
            raise ValueError(f"max_step_retries must be >= 1, got {self.max_step_retries}")
        if self.escalate_after_attempt < 1:
            raise ValueError(
                f"escalate_after_attempt must be >= 1, got {self.escalate_after_attempt}"
            )
        if self.skip_planner and not self.skip_critic:
            # Skipping the planner without skipping the critic is a strictly
            # worse shape (you pay for the critic but throw away its output's
            # structure). Reject the combination at construction time so the
            # PROFILES table can't drift into it silently.
            raise ValueError("skip_planner=True requires skip_critic=True")


PROFILES: dict[TaskClass, Profile] = {
    TaskClass.TRIVIAL: Profile(
        skip_critic=True,
        skip_planner=True,
        enable_escalation=False,
        # Two attempts: triage occasionally misjudges and a one-shot bugfix can
        # still need a retry when the worker emits a no-op edit on the first
        # try (e.g. when the synthesised single-step plan reaches the worker
        # before it has had a chance to internalise the file contents). Tier-1
        # bench at retries=1 regressed 2/8 trivial tasks (tasks 01 and 05);
        # bumping to 2 reclaims those without adding cost on the green path.
        max_step_retries=2,
    ),
    TaskClass.SINGLE_STEP: Profile(
        skip_critic=True,
        skip_planner=False,
        enable_escalation=True,
        max_step_retries=2,
        escalate_after_attempt=1,
    ),
    TaskClass.MULTI_STEP: Profile(
        skip_critic=False,
        skip_planner=False,
        enable_escalation=True,
        # Three attempts with tiered escalation: two cheap sonnet tries, then
        # opus on the third. Tier 4 rerun showed sonnet retries usually
        # succeed; jumping straight to opus on attempt 1 was a regression on
        # tasks where sonnet would have recovered on its own.
        max_step_retries=3,
        escalate_after_attempt=2,
    ),
    TaskClass.EXPLORATION: Profile(
        skip_critic=False,
        skip_planner=False,
        enable_escalation=True,
        max_step_retries=4,
        escalate_after_attempt=2,
    ),
}


# Conservative fallback used when no triage provider is configured or the
# triage call fails. Equivalent to the pre-profile behaviour (full pipeline,
# 2 attempts, escalation enabled if a worker_escalation provider is wired).
DEFAULT_PROFILE: Profile = PROFILES[TaskClass.MULTI_STEP]
