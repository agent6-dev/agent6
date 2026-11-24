# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import threading

import pytest

from agent6 import budget, kinds, paths
from agent6.app import child_watch, children
from agent6.sessions import budget_events, child_dispatch, event_log, journal, layout
from agent6.viewmodel.session import log as session_log
from agent6.viewmodel.session import view


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
                    "version": 0,
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
    budget_events.emit_budget_update(panel, leaf_sink.emit)
    leaf_usage = children.child_usage(leaf)
    assert leaf_usage is not None
    owner.settle_child(leaf.name, leaf_usage)
    owner.record_plan_preflight("a", a1)
    sink = event_log.EventSink(worker / "logs.jsonl")
    budget_events.emit_budget_update(owner, sink.emit)
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


def test_failed_budget_append_keeps_the_balance_for_child_settlement(
    tmp_path: pathlib.Path,
) -> None:
    """A missing baseline in the child journal would hide its decline from the parent."""
    child = budget.BudgetTracker(max_usd=2, max_tokens_fallback=-1, max_plan_points=-1)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    _balance(child, "p", "shared", 10, 10)
    sink.path.mkdir()
    with pytest.raises(event_log.EventWriteError, match="unwritable"):
        budget_events.emit_budget_update(child, sink.emit)
    sink.path.rmdir()
    _balance(child, "p", "shared", 20, 9)
    budget_events.emit_budget_update(child, sink.emit)
    usage = children.child_usage(tmp_path)
    assert usage is not None and usage.usd_observed_credits == 1
    parent = budget.BudgetTracker(max_usd=2, max_tokens_fallback=-1, max_plan_points=-1)
    parent.note_running("child", child_watch.running_usage(tmp_path))
    parent.settle_child("child", usage)
    assert parent.snapshot().usd_observed_credits == 1


def test_two_writers_cannot_append_an_older_budget_snapshot_last(tmp_path: pathlib.Path) -> None:
    """The meter a parent recovers must include a decline recorded during another write."""
    meter = budget.BudgetTracker(max_usd=2, max_tokens_fallback=-1, max_plan_points=-1)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    _balance(meter, "p", "shared", 10, 10)
    paused, competing, written = threading.Event(), threading.Event(), threading.Event()

    def first(row: journal.BudgetUpdate) -> None:
        paused.set()
        assert competing.wait(5)
        written.wait(0.25)  # Give the newer write a chance to overtake this one.
        sink.emit(row)

    def concurrent_write() -> None:
        assert paused.wait(5)
        _balance(meter, "p", "shared", 20, 8)
        competing.set()
        budget_events.emit_budget_update(meter, sink.emit)
        written.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as writers:
        older = writers.submit(budget_events.emit_budget_update, meter, first)
        newer = writers.submit(concurrent_write)
        older.result(timeout=10)
        newer.result(timeout=10)
    usage = children.child_usage(tmp_path)
    assert usage is not None and usage.usd == usage.usd_observed_credits == 2
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [row["usd_total"] for row in rows] == [0, 2]


def test_a_retried_append_confirms_only_its_own_readings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fsync can leave a readable row that the next append repeats."""
    child = budget.BudgetTracker(max_usd=3, max_tokens_fallback=-1, max_plan_points=-1)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    _balance(child, "p", "shared", 10, 10)

    def failed_sync(fd: int) -> None:
        raise OSError("sync failed")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", failed_sync)
        with pytest.raises(event_log.EventWriteError, match="sync failed"):
            budget_events.emit_budget_update(child, sink.emit)
    sink.subscribe(lambda _: _balance(child, "p", "shared", 30, 8))
    _balance(child, "p", "shared", 20, 9)
    for _ in range(2):
        budget_events.emit_budget_update(child, sink.emit)
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [[r["balance_usd"] for r in row["credit_readings"]] for row in rows] == [
        [10],
        [10, 9],
        [8],
    ]
    usage = children.child_usage(tmp_path)
    assert usage is not None and [r.balance_usd for r in usage.credit_readings] == [10, 9, 8]
    parent = budget.BudgetTracker(max_usd=3, max_tokens_fallback=-1, max_plan_points=-1)
    parent.settle_child("child", usage)
    assert parent.available().usd == 1


def test_an_unchanged_balance_stays_two_readings_through_the_journal_and_the_session_payload(
    tmp_path: pathlib.Path,
) -> None:
    """Each reply's balance stayed in the meter, the fold and every session payload."""
    child = budget.BudgetTracker(max_usd=10, max_tokens_fallback=-1, max_plan_points=-1)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    for at in range(1, 41):
        _balance(child, "p", "shared", at, 10)
        budget_events.emit_budget_update(child, sink.emit)
    _balance(child, "p", "shared", 41, 7)
    budget_events.emit_budget_update(child, sink.emit)
    assert child.snapshot().credit_reading_count == 3
    scan = session_log.scan_session_log(sink.path)
    assert [(r.observed_ns, r.balance_usd) for r in scan.credit_readings] == [
        (1, 10),
        (40, 10),
        (41, 7),
    ]
    assert "credit_readings" not in json.dumps(view.budget_dict(scan.budget))
    parent = budget.BudgetTracker(max_usd=10, max_tokens_fallback=-1, max_plan_points=-1)
    _balance(parent, "p", "shared", 20, 10)
    parent.note_running("child", child_watch.running_usage(tmp_path))
    parent.settle_child("child", children.child_usage(tmp_path))
    assert parent.snapshot().usd_observed_credits == 3 and parent.available().usd == 7


def test_credit_journal_distinguishes_accounts_without_publishing_their_header_values(
    tmp_path: pathlib.Path,
) -> None:
    tracker = budget.BudgetTracker(max_usd=2, max_tokens_fallback=-1, max_plan_points=-1)
    first, second = "billing-account-123", "billing-account-456"
    _balance(tracker, "p", first, 10, 10)
    _balance(tracker, "q", second, 20, 2)
    _balance(tracker, "alias", first, 30, 9)
    path = tmp_path / "logs.jsonl"
    budget_events.emit_budget_update(tracker, event_log.EventSink(path).emit)
    wire = path.read_text()
    assert first not in wire and second not in wire
    usage = children.child_usage(tmp_path)
    assert usage is not None and usage.usd_observed_credits == 1
    assert len({r.account for r in usage.credit_readings}) == 2


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
    budget_events.emit_budget_update(child, sink.emit)

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
        budget_events.emit_budget_update(child, sink.emit)
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
