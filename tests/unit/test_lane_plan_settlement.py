# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import budget, paths
from agent6.app import lanes
from agent6.sessions import budget_events, event_log, journal, lane_dispatch, layout


def test_ancestor_folds_descendant_providers_without_recharging_inclusive_usage(
    tmp_path: pathlib.Path,
) -> None:
    """A worker's latest account reading cannot hide a panel provider's high-water mark."""
    state = paths.state_dir(tmp_path)
    worker = layout.bucket_dir(state, "runs") / "parent-p1-l1"
    leaf = layout.bucket_dir(state, "reviewers") / "parent-p1-l1-rpanel-l1"
    for directory, provider, parent in ((worker, "a", "parent"), (leaf, "b", worker.name)):
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "session_id": directory.name,
                    "mode": "run" if directory == worker else "reviewer",
                    "models": {"driver": {"provider": provider, "model": "base"}},
                    "parallel": {"coordinator": parent, "group": "g", "lane": 1},
                }
            )
        )
    parent = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=20)
    owner = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=20)
    panel = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=20)
    a0 = budget.PlanUsage.single(10, 10080, 2e9)
    a1 = budget.PlanUsage.single(11, 10080, 2e9)
    b0 = budget.PlanUsage.single(40, 10080, 2e9)
    b1 = budget.PlanUsage.single(44, 10080, 2e9)
    for tracker in (parent, owner):
        tracker.record_plan_preflight("a", a0)
        tracker.record_plan_preflight("b", b0)
    panel.record_plan_preflight("b", b1)
    leaf_sink = event_log.EventSink(leaf / "logs.jsonl")
    leaf_sink.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(panel)))
    leaf_usage = lanes.lane_usage(leaf)
    assert leaf_usage is not None
    owner.settle_lane(leaf.name, leaf_usage)
    owner.record_plan_preflight("a", a1)
    sink = event_log.EventSink(worker / "logs.jsonl")
    sink.emit(
        journal.BudgetUpdate(
            **budget_events.budget_update_fields(owner),
        )
    )
    lane_dispatch.write_record(
        worker,
        lane_dispatch.DispatchRecord(
            identity="panel",
            source="panel",
            join_policy="record_only",
            owner="worker",
            turn=1,
            base="",
            segments=(),
            children=(
                lane_dispatch.ChildRecord(
                    session_id=leaf.name,
                    lane=1,
                    task="review",
                    settled=True,
                    usage=lane_dispatch.UsageRecord(usd=0),
                ),
            ),
            state="complete",
        ),
        now=1,
    )
    usage = lanes.lane_usage(worker)
    assert usage is not None
    parent.settle_lane(worker.name, usage)
    parent.settle_lane(worker.name, usage)
    assert parent.snapshot().plan_observed_points == pytest.approx(4)
    assert parent.snapshot().plans["a"].usage == a1
    assert parent.snapshot().plans["b"].usage == b1
    assert parent.available().plan_points == pytest.approx(16)
    assert parent.estimate_usd()[0] == owner.estimate_usd()[0]
    assert parent.snapshot().unmetered_tokens == owner.snapshot().unmetered_tokens
