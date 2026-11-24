# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import budget, kinds, paths
from agent6.app import child_watch, children
from agent6.sessions import budget_events, child_dispatch, event_log, journal, layout


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
                    "parallel": {"parent": parent, "group": "g", "number": 1},
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
    leaf_usage = children.child_usage(leaf)
    assert leaf_usage is not None
    owner.settle_child(leaf.name, leaf_usage)
    owner.record_plan_preflight("a", a1)
    sink = event_log.EventSink(worker / "logs.jsonl")
    sink.emit(
        journal.BudgetUpdate(
            **budget_events.budget_update_fields(owner),
        )
    )
    child_dispatch.write_record(
        worker,
        child_dispatch.DispatchRecord(
            identity="panel",
            source="panel",
            merge_policy="record_only",
            owner="worker",
            turn=1,
            base="",
            children=(
                child_dispatch.ChildRecord(
                    session_id=leaf.name,
                    number=1,
                    task="review",
                    settled=True,
                    usage=child_dispatch.UsageRecord(usd=0),
                ),
            ),
            state="complete",
        ),
        now=1,
    )
    usage = children.child_usage(worker)
    assert usage is not None
    parent.settle_child(worker.name, usage)
    parent.settle_child(worker.name, usage)
    assert parent.snapshot().plan_observed_points == pytest.approx(4)
    assert parent.snapshot().plans["a"].usage == a1
    assert parent.snapshot().plans["b"].usage == b1
    assert parent.available().plan_points == pytest.approx(16)
    assert parent.estimate_usd()[0] == owner.estimate_usd()[0]
    assert parent.snapshot().unmetered_tokens == owner.snapshot().unmetered_tokens


def _balance(
    tracker: budget.BudgetTracker, provider: str, account: str, at: int, dollars: int
) -> None:
    tracker.record_plan_preflight(
        provider,
        budget.PlanUsage.single(
            20, 300, 0, account_id=account, observed_ns=at, credits_balance=f"${dollars}"
        ),
    )


def _call(tracker: budget.BudgetTracker, dollars: float) -> None:
    tracker.record(
        route=kinds.ModelRoute("api", "m"),
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=dollars,
    )


def test_running_and_settled_children_fold_credit_by_account_and_observation_time(
    tmp_path: pathlib.Path,
) -> None:
    """A late top-up separates two declines, even across provider aliases."""
    parent = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=-1)
    child = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=-1)
    child_dir = tmp_path / "child"
    sink = event_log.EventSink(child_dir / "logs.jsonl")
    _balance(parent, "p", "shared", 10, 10)
    _balance(parent, "p", "shared", 40, 8)
    _call(parent, 0.25)
    _balance(child, "alias", "shared", 20, 9)
    _balance(child, "alias", "shared", 30, 11)
    _balance(child, "other", "child-only", 10, 5)
    _balance(child, "other", "child-only", 20, 4)
    _call(child, 0.50)
    sink.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(child)))

    for _ in range(2):
        parent.note_running("child", child_watch.running_usage(child_dir))
        snap = parent.snapshot()
        assert snap.usd_observed_credits == 5  # 10→9, 11→8 and 5→4
        assert snap.usd_own_calls == 0.25 and snap.usd_running_children == 0.50
        assert parent.available().usd == 4.25
    usage = children.child_usage(child_dir)
    assert usage is not None and usage.usd == 1.50
    for _ in range(2):
        parent.settle_child("child", usage)
    snap = parent.snapshot()
    assert snap.usd_settled_children == 0.50 and snap.usd_running_children == 0
    assert snap.usd_total == 5.75 and parent.available().usd == 4.25


def test_a_continued_child_charges_only_new_calls_and_keeps_the_credit_series(
    tmp_path: pathlib.Path,
) -> None:
    """A continuation's first balance can complete a decline its first execution began."""
    parent = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=-1)
    child_dir = tmp_path / "child"
    sink = event_log.EventSink(child_dir / "logs.jsonl")
    for execution, at, balance, call_cost in (("first", 10, 8, 0.5), ("next", 30, 6, 0.25)):
        child = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=-1)
        if execution == "first":
            sink.emit(journal.SessionStart(user_task="t", mode="run", execution_id=execution))
        else:
            sink.emit(
                journal.SessionResume(
                    session_id=child_dir.name,
                    user_task="t",
                    steer="continue",
                    mode="run",
                    turn=2,
                    messages=2,
                    execution_id=execution,
                )
            )
        _balance(child, "p", "shared", at, balance)
        _balance(child, "p", "shared", at + 10, balance - 1)
        _call(child, call_cost)
        sink.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(child)))
        parent.note_running("child", child_watch.running_usage(child_dir))
        usage = children.child_usage(child_dir)
        assert usage is not None
        parent.settle_child("child", usage)
        if execution == "first":
            assert parent.available().usd == 8.5
            parent.child_execution("child", "child:next", prior=usage)
            parent.note_running("child", child_watch.running_usage(child_dir))
            assert parent.available().usd == 8.5
    assert parent.snapshot().usd_observed_credits == 3
    assert parent.snapshot().usd_settled_children == 0.75
    assert parent.available().usd == 6.25
