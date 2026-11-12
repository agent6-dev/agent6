# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from agent6 import budget, directive, kinds, lane_records, paths
from agent6.app import (
    lane_launch,
    lanes,
    parallel_results,
    stop,
    worktrees,
)
from agent6.harness import loop_conversation, loop_state, parallel_dispatch, run_chain, subrun
from agent6.sessions import (
    budget_events,
    completion,
    event_log,
    ipc,
    journal,
    lane_dispatch,
    layout,
    lineage,
    process,
)
from agent6.sessions import manifest as session_manifest
from agent6.tools import dispatch, tool_errors
from agent6.ui.cli import parallel
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing, receipt
from tests import lane_env, subordinate_env


@pytest.mark.parametrize("panel_cost", [None, 2.0])
def test_worker_panel_uses_shared_executor_with_inherited_limits_and_one_charge(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, panel_cost: float | None
) -> None:
    """A private worker launch authorizes only panel leaves, whatever harness.parallel.tool is."""
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    subordinate_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    with lane_env.scripted_provider(("reviewed",), cost_usd=panel_cost) as (url, calls):
        cfg = subordinate_env.config(url)
        _path, launch = subordinate_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker_dir = subordinate_env.create_record(repo, launch.session_id, parent="parent")
        parent = budget.BudgetTracker(max_usd=8, max_tokens_fallback=1000, max_plan_points=20)
        share = parent.available()  # the worker starts under all its parent has left
        worker = budget.BudgetTracker(
            max_usd=share.usd, max_tokens_fallback=share.tokens, max_plan_points=share.plan_points
        )
        for cost in (1.0, None):
            worker.record(
                route=kinds.ModelRoute("p", "base"),
                input_tokens=1,
                output_tokens=4,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=cost,
            )
        event_log.EventSink(worker_dir / "logs.jsonl").emit(
            journal.BudgetUpdate(**budget_events.budget_update_fields(worker))
        )
        executor = subordinate_env.dispatcher(clone, worker_dir, cfg, worker)
        refused = executor.dispatch(
            loop_conversation.Conversation(),
            1,
            loop_state.LoopState(original_task="worker", tool_calls=0),
            [directive.Segment(task="not allowed", spec="1")],
        )
        assert not refused.dispatched and "subordinate" in refused.message
        contract = kinds.ConversationContract(
            system="Review only.",
            opening="Read this tree.",
            max_turns=2,
        )
        request = lane_records.LaneTask(
            "review",
            "reviewer",
            contract=contract,
            limits=lane_records.LaneLimits(deadline_at=time.time() + 1800),
        )
        result = executor.dispatch_panel(
            [request], panel_id="seat-test", at=subordinate_env.git(clone, "rev-parse", "HEAD")
        )
        assert result.dispatched and len(result.lanes) == 1, result.message
        assert result.lanes[0].status == "no_changes"
        record = lane_dispatch.read_records(worker_dir)[0]
        assert record.source == "panel" and record.join_policy == "record_only"
        assert record.version == 5
        child = record.children[0]
        assert child.limits is not None and launch.limits is not None
        assert child.limits.deadline_at == launch.limits.deadline_at
        assert child.settled
        panel_tokens = 2 if panel_cost is None else 0
        assert child.usage is not None and child.usage.tokens == panel_tokens
        assert worker.available().tokens == share.tokens - 5 - panel_tokens
        leaf = layout.bucket_dir(paths.state_dir(repo), "reviewers") / child.session_id
        assert completion.read_completion(leaf) is not None
        assert len(calls) == 1 and "parallel" not in json.dumps(calls[0].get("tools", []))
        emitted = [
            json.loads(line) for line in (worker_dir / "logs.jsonl").read_text().splitlines()
        ]
        charged = next(row for row in emitted if row.get("type") == "budget.turn")
        assert charged["cost_usd"] == 1.0 + (panel_cost or 0.0)
        # The worker can die here, before finalization publishes another budget update.
        parent.settle_lane(launch.session_id, lanes.lane_usage(worker_dir))
        assert parent.estimate_usd()[0] == 1.0 + (panel_cost or 0.0)
        assert parent.available().usd == 7.0 - (panel_cost or 0.0)
        assert parent.snapshot().unmetered_tokens == 5 + panel_tokens
        spec = lane_records.LaneSpec(1, worker_dir.name, clone, None)
        imported = parallel_results.import_lane(
            lane_records.LaneResult(
                spec, worker_dir, git_read.run_branch_for(worker_dir.name), True, ""
            ),
            repo,
            None,
        )
        worktrees.cleanup(
            [spec],
            origin=repo,
            record_state=paths.state_dir(repo),
            worktree_root=short_repo,
            base=short_repo,
            launch_dir=lane_launch.launch_dir(paths.state_dir(repo), "parent-p1"),
            cfg=cfg,
        )
        assert not clone.exists()
        retained = layout.bucket_dir(paths.state_dir(repo), "reviewers") / child.session_id
        assert completion.read_completion(retained) is not None
        assert receipt.read_receipt(retained, repo).session_id == child.session_id
        assert (
            receipt.read_receipt(imported.session_dir, repo).children[0].session_id
            == child.session_id
        )


class _Crash(BaseException):
    """Simulate a process exit outside the dispatcher's exception handler."""


@pytest.mark.parametrize(
    "boundary",
    [
        "prepared",
        "running",
        "imported",
        "accounting",
        "accounting_failed",
        "settling",
        "joining",
        "complete",
    ],
)
def test_panel_recovery_never_relaunches_merges_or_grants_twice(
    short_repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    clone = short_repo / "worker"
    with lane_env.scripted_provider() as (url, calls):
        cfg = subordinate_env.config(url)
        _path, launch = subordinate_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker_dir = subordinate_env.create_record(repo, launch.session_id, parent="parent")
        meter = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
        executor = subordinate_env.dispatcher(clone, worker_dir, cfg, meter)
        contract = kinds.ConversationContract(
            system="s",
            opening="o",
            max_turns=2,
        )
        original_emit = executor.emit

        def crash_after_accounting(event: journal.Event) -> None:
            if boundary == "accounting_failed" and isinstance(event, journal.BudgetUpdate):
                raise _Crash
            original_emit(event)
            if boundary == "accounting" and isinstance(event, journal.BudgetUpdate):
                raise _Crash

        executor.emit = crash_after_accounting
        original = parallel_dispatch.ParallelDispatcher.write
        spawner = executor.lane_spawner
        assert spawner is not None

        def crash_after_import(
            lanes: list[lane_records.LaneTask], group: str, *, at: str | None = None
        ) -> list[lane_records.LaneResult]:
            spawner(lanes, group, at=at)
            raise _Crash

        if boundary == "imported":
            executor.lane_spawner = crash_after_import

        def crash_after_write(
            self: parallel_dispatch.ParallelDispatcher, record: lane_dispatch.DispatchRecord
        ) -> None:
            original(self, record)
            if record.state == boundary:
                raise _Crash

        with monkeypatch.context() as patch:
            patch.setattr(parallel_dispatch.ParallelDispatcher, "write", crash_after_write)
            with pytest.raises(_Crash):
                executor.dispatch_panel(
                    [lane_records.LaneTask("review", "reviewer", contract=contract)],
                    panel_id="recovery",
                    at=subordinate_env.git(clone, "rev-parse", "HEAD"),
                )
        if boundary == "accounting_failed":
            # Finalization can still publish the charged meter after the dispatch emit failed.
            original_emit(journal.BudgetUpdate(**budget_events.budget_update_fields(meter)))
        before = len(calls)
        ancestor = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
        ancestor.settle_lane(worker_dir.name, lanes.lane_usage(worker_dir))
        if boundary != "complete":
            assert not ancestor.snapshot().settled_lanes, "an unfinished worker is not charged"
        event_log.EventSink(worker_dir / "logs.jsonl").emit(
            journal.SessionResume(
                session_id=worker_dir.name,
                user_task="worker",
                steer="",
                mode="run",
                turn=1,
                messages=0,
                execution_id="resumed-worker",
            )
        )
        recovered_budget = budget.BudgetTracker(
            max_usd=5, max_tokens_fallback=1000, max_plan_points=20
        )
        recovered = subordinate_env.dispatcher(clone, worker_dir, cfg, recovered_budget)
        recovered.lane_recoverer = lanes.build_lane_recoverer(
            cfg, clone, paths.state_dir(repo), mode="run"
        )
        state = loop_state.LoopState(original_task="worker", tool_calls=0)
        conversation = loop_conversation.Conversation()
        recovered.reconcile(state, conversation)
        recovered.reconcile(state, conversation)
        assert len(calls) == before
        assert recovered_budget.snapshot().usd_running_lanes == 0
        usage = lanes.lane_usage(worker_dir)
        assert usage is not None and usage.tokens == 2 * len(calls)
        ancestor.settle_lane(worker_dir.name, usage)
        ancestor.settle_lane(worker_dir.name, usage)
        assert ancestor.snapshot().unmetered_tokens == 2 * len(calls)
        assert ancestor.snapshot().settled_lanes == frozenset({worker_dir.name})
        record = lane_dispatch.read_records(worker_dir)[0]
        assert record.state == "complete" and record.join_policy == "record_only"
        assert record.delivered and all(child.settled for child in record.children)
        assert all(child.join is None or child.join.sha == "" for child in record.children)
        assert subordinate_env.git(clone, "rev-parse", "HEAD") == subordinate_env.git(
            repo, "rev-parse", "HEAD"
        )


def test_subordinate_bootstrap_refuses_model_and_steer_parallel_even_with_flag_on(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    clone = short_repo / "worker"
    steered = threading.Event()

    def queue_steer(_request: dict[str, Any]) -> None:
        if not steered.is_set():
            assert ipc.submit_steer(worker_dir, "/parallel 1 -- recurse via steer")
            steered.set()

    with lane_env.scripted_provider(
        [("parallel", {"task": "recurse", "spec": "1"}), lane_env.FINISH],
        before_reply=queue_steer,
    ) as (url, calls):
        cfg = subordinate_env.config(url, tool=True)
        _path, launch = subordinate_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker_dir = layout.bucket_dir(paths.state_dir(repo), "runs") / launch.session_id
        env = {**os.environ, "AGENT6_ALLOW_ROOT": "1", "AGENT6_STREAM_TO_LOG": "1"}
        proc = subprocess.run(
            lane_launch.bootstrap_argv(),
            cwd=clone,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    assert proc.returncode == 0, proc.stderr
    worker_dir = layout.bucket_dir(paths.state_dir(repo), "runs") / launch.session_id
    rows = [json.loads(line) for line in (worker_dir / "logs.jsonl").read_text().splitlines()]
    refusal = next(
        r for r in rows if r.get("type") == "tool.result" and r.get("name") == "parallel"
    )
    assert not refusal["ok"] and "inside a lane" in refusal["summary"]
    assert all("parallel" not in json.dumps(c.get("tools", [])) for c in calls)
    assert steered.is_set()
    assert any(
        "work dispatch is unavailable in a subordinate run; nothing dispatched."
        in json.dumps(call["messages"])
        for call in calls[1:]
    )
    assert lane_dispatch.read_records(worker_dir) == ()


def test_review_leaf_and_unproven_subrun_have_no_dispatch_authority(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    clone = short_repo / "worker"
    cfg = subordinate_env.config(tool=True)
    path, launch = subordinate_env.launch_worker(repo, clone, cfg, monkeypatch)
    contract = kinds.ConversationContract(system="s", opening="o", max_turns=1)
    lane_launch.write(path.parent, launch.model_copy(update={"contract": contract}))
    for mode in ("run", "reviewer"):
        assert (
            lanes.build_parent_spawner(
                cfg,
                clone,
                paths.state_dir(clone),
                explicit_leaves=frozenset(),
                mode=mode,
                session_id=launch.session_id,
                runtime=parallel.lane_runtime(),
            )
            is None
        )
    disp = dispatch.ToolDispatcher(root=clone, config=cfg, isolation="none", mode="reviewer")
    with pytest.raises(tool_errors.ToolError, match="not available"):
        disp.dispatch("parallel", {"task": "recurse"})
    monkeypatch.delenv(lane_launch.ENV)
    assert (
        lanes.build_parent_spawner(
            cfg,
            clone,
            paths.state_dir(clone),
            explicit_leaves=frozenset(),
            mode="run",
            session_id=launch.session_id,
            runtime=parallel.lane_runtime(),
        )
        is None
    )


def test_import_retains_descendant_records_receipts_and_target_objects(
    short_repo: pathlib.Path,
) -> None:
    """Removing a worker clone cannot remove its panel record or its immutable target."""
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    subordinate_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    subrun.clone_worktree(repo, clone)
    worker = subordinate_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = subordinate_env.create_record(
        repo, "parent-p1-l1-r1-l1", parent=worker.name, mode="reviewer"
    )
    evidence = completion.CompletionRecord(
        text="review evidence",
        stop_reason="end_turn",
        output_tokens=2,
        calls=1,
    )
    completion.write_completion(leaf, evidence)
    (leaf / "objects").mkdir()
    (leaf / "objects" / "evidence.json").write_text('{"evidence": "retained"}')
    chain = run_chain.RunChain(
        clone,
        ref=git_read.chain_ref_for(worker.name),
        fallback_parent=subordinate_env.git(clone, "rev-parse", "HEAD"),
    )
    (clone / "README.md").write_text("reviewed dirty tree\n")
    target = chain.review_target()
    assert target is not None
    target = chain.retain_review_target(target, session_id=worker.name)
    record = lane_dispatch.DispatchRecord(
        identity=worker.name + "-r1",
        source="panel",
        owner="e",
        turn=1,
        base=target.snapshot_commit,
        segments=(),
        children=(lane_dispatch.ChildRecord(session_id=leaf.name, lane=1, task="review"),),
    )
    lane_dispatch.write_record(worker, record, now=time.time())
    res = lane_records.LaneResult(
        lane_records.LaneSpec(1, worker.name, clone, None),
        worker,
        git_read.run_branch_for(worker.name),
        True,
        "",
    )
    imported = parallel_results.import_lane(res, repo, None)
    shutil.rmtree(clone)
    dest = layout.bucket_dir(paths.state_dir(repo), "reviewers") / leaf.name
    assert dest.is_dir() and not dest.is_symlink()
    assert completion.read_completion(dest) == evidence
    assert json.loads((dest / "objects" / "evidence.json").read_text()) == {"evidence": "retained"}
    saved = session_manifest.read_manifest(dest)
    assert saved.parallel is not None and saved.parallel.parent == worker.name
    assert lane_dispatch.read_records(imported.session_dir)[0].base == target.snapshot_commit
    assert subordinate_env.git(repo, "rev-parse", target.snapshot_ref) == target.snapshot_commit
    assert (
        subordinate_env.git(repo, "show", f"{target.snapshot_commit}:README.md")
        == "reviewed dirty tree"
    )
    # The seat is the worker's child by lineage and none of its lanes.
    assert [child.name for child in lineage.children_of(imported.session_dir)] == [leaf.name]
    assert listing.lanes_of(paths.state_dir(repo), worker.name) == []
    assert receipt.read_receipt(dest, repo).session_id == leaf.name


def test_cleanup_keeps_a_live_descendant_even_when_all_commits_are_reachable(
    short_repo: pathlib.Path,
) -> None:
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    clone = short_repo / "worker"
    subrun.clone_worktree(repo, clone)
    worker = subordinate_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = subordinate_env.create_record(
        repo, "parent-p1-l1-r1-l1", parent=worker.name, mode="reviewer"
    )
    process.write_worker_pid(leaf, os.getpid())
    kept: list[str] = []
    worktrees.cleanup(
        [lane_records.LaneSpec(1, worker.name, clone, None)],
        origin=repo,
        record_state=paths.state_dir(repo),
        worktree_root=short_repo,
        base=short_repo,
        launch_dir=lane_launch.launch_dir(paths.state_dir(repo), "parent-p1"),
        cfg=subordinate_env.config(),
        note_keep=lambda _p, why: kept.append(why),
    )
    assert clone.is_dir() and leaf.is_dir()
    assert any("live" in why for why in kept)


def test_an_unfinished_panel_keeps_the_ancestor_reservation(short_repo: pathlib.Path) -> None:
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    worker = subordinate_env.create_record(repo, "parent-p1-l1", parent="parent")
    record = lane_dispatch.DispatchRecord(
        identity=worker.name + "-r1",
        source="panel",
        owner="w",
        turn=1,
        base="",
        segments=(),
        children=(
            lane_dispatch.ChildRecord(session_id=worker.name + "-r1-l1", lane=1, task="review"),
        ),
        state="running",
    )
    lane_dispatch.write_record(worker, record, now=time.time())
    assert lanes.lane_usage(worker) is None


def test_a_settled_panel_keeps_the_ancestor_hold_until_the_journal_names_it(
    short_repo: pathlib.Path,
) -> None:
    """A worker that died before journaling a panel's charge must not settle without it."""
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    worker = subordinate_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = worker.name + "-r1-l1"
    record = lane_dispatch.DispatchRecord(
        identity=worker.name + "-r1",
        source="panel",
        join_policy="record_only",
        owner="w",
        turn=1,
        base="",
        segments=(),
        children=(
            lane_dispatch.ChildRecord(
                session_id=leaf,
                lane=1,
                task="review",
                settled=True,
                usage=lane_dispatch.UsageRecord(usd=2.0),
            ),
        ),
        state="complete",
    )
    lane_dispatch.write_record(worker, record, now=time.time())
    tracker = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=20)
    sink = event_log.EventSink(worker / "logs.jsonl")
    sink.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(tracker)))
    assert lanes.lane_usage(worker) is None
    tracker.settle_lane(leaf, budget.UsageSettlement(usd=2.0, usd_partial=False, tokens=0))
    sink.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(tracker)))
    usage = lanes.lane_usage(worker)
    assert usage is not None
    assert usage.usd == pytest.approx(2.0)


@pytest.mark.parametrize("worker_dead", [False, True])
def test_stop_follows_clone_state_and_readonly_bucket_and_leaves_the_worker_unsettled(
    short_repo: pathlib.Path,
    worker_dead: bool,
) -> None:
    """A live leaf is found through a worker symlink, even when the worker already died."""
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    parent = subordinate_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    subrun.clone_worktree(repo, clone)
    worker = subordinate_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = subordinate_env.create_record(
        repo, "parent-p1-l1-r1-l1", parent=worker.name, mode="reviewer"
    )
    ancestor = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
    lane_dispatch.write_record(
        worker,
        lane_dispatch.DispatchRecord(
            identity=worker.name + "-r1",
            source="panel",
            owner="w",
            turn=1,
            base="",
            segments=(),
            children=(lane_dispatch.ChildRecord(session_id=leaf.name, lane=1, task="review"),),
            state="running",
        ),
        now=time.time(),
    )
    children: list[subprocess.Popen[bytes]] = []
    try:
        for directory in [parent, leaf] if worker_dead else [parent, worker, leaf]:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            children.append(child)
            process.write_worker_pid(directory, child.pid)
        stopped = stop.stop_session(parent, wait_s=0, grace_s=0.05)
        assert stopped.ok
        assert all(not process.pid_alive(child.pid) for child in children)
        usage = lanes.lane_usage(worker)
        assert usage is None, "a worker with an unfinished descendant has no final usage"
        ancestor.settle_lane(worker.name, usage)
        assert not ancestor.snapshot().settled_lanes and ancestor.available().usd == 5
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)
