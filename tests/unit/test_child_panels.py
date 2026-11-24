# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import concurrent.futures
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

from agent6 import budget, child_records, directive, kinds, paths
from agent6.app import role_providers, stop, worktrees
from agent6.app.parallel import children
from agent6.app.parallel import launch as child_launch
from agent6.app.parallel import results as parallel_results
from agent6.harness import loop_conversation, loop_state, run_chain
from agent6.harness.parallel import dispatch as parallel_dispatch
from agent6.harness.parallel import panel as parallel_panel
from agent6.harness.parallel import recovery as parallel_recovery
from agent6.harness.parallel import repo as child_repo
from agent6.harness.review import gate as review_gate
from agent6.harness.review import panel as review_panel
from agent6.sessions import (
    budget_events,
    child_dispatch,
    completion,
    event_log,
    ipc,
    journal,
    layout,
    lineage,
    process,
)
from agent6.sessions import manifest as session_manifest
from agent6.tools import dispatch, tool_errors
from agent6.ui.cli import parallel
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing, receipt
from tests import child_env, launch_env, review_traces


@pytest.mark.parametrize("panel_cost", [None, 2.0])
def test_worker_panel_uses_shared_executor_with_inherited_limits_and_one_charge(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, panel_cost: float | None
) -> None:
    """A private worker launch authorizes only reviewers, whatever harness.parallel.tool is."""
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    child_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    with launch_env.scripted_provider(("reviewed",), cost_usd=panel_cost) as (url, calls):
        cfg = child_env.config(url)
        _path, launch = child_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker_dir = child_env.create_record(repo, launch.session_id, parent="parent")
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
        budget_events.emit_budget_update(
            worker, event_log.EventSink(worker_dir / "logs.jsonl").emit
        )
        executor = child_env.dispatcher(clone, worker_dir, cfg, worker)
        refused = executor.dispatch(
            loop_conversation.Conversation(),
            1,
            loop_state.LoopState(original_task="worker", tool_calls=0),
            [directive.Segment(task="not allowed", spec="1")],
        )
        assert not refused.children and "in a child" in refused.message
        contract = kinds.ConversationContract(
            system="Review only.",
            opening="Read this tree.",
            max_turns=2,
        )
        request = child_records.ChildTask(
            "review",
            "reviewer",
            contract=contract,
            limits=child_records.ChildLimits(deadline_at=time.time() + 1800),
        )
        result = parallel_panel.dispatch_panel(
            executor,
            [request],
            panel_id="reviewer-test",
            at=child_env.git(clone, "rev-parse", "HEAD"),
        )
        assert result.dispatched and len(result.children) == 1, result.message
        assert result.children[0] is not None and result.children[0].status == "no_changes"
        record = child_dispatch.read_records(worker_dir)[0]
        assert record.source == "panel" and record.merge_policy == "record_only"
        assert record.version == child_dispatch.RECORD_VERSION
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
        parent.settle_child(launch.session_id, children.child_usage(worker_dir))
        assert parent.estimate_usd()[0] == 1.0 + (panel_cost or 0.0)
        assert parent.available().usd == 7.0 - (panel_cost or 0.0)
        assert parent.snapshot().unmetered_tokens == 5 + panel_tokens
        # Two worker calls (1 in, 4 out each) plus the reviewer's one (1 in, 1 out).
        assert "Children: in=3 out=9" in parent.format_summary()
        spec = child_records.ChildSpec(1, worker_dir.name, clone, None)
        imported = parallel_results.import_child(
            child_records.ChildResult(
                spec, worker_dir, git_read.run_branch_for(worker_dir.name), True, ""
            ),
            repo,
            None,
            cfg,
        )
        worktrees.cleanup(
            [spec],
            origin=repo,
            record_state=paths.state_dir(repo),
            worktree_root=short_repo,
            base=short_repo,
            launch_dir=child_launch.launch_dir(paths.state_dir(repo), "parent-p1"),
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


@pytest.mark.parametrize("failure", ["launch", "watch"])
def test_a_partial_panel_keeps_the_reviewers_that_returned(
    short_repo: pathlib.Path, failure: str
) -> None:
    """A sibling's refusal or failed watch cannot erase a grounded block or its charge."""
    repo = short_repo / "repo"
    child_env.init_repo(repo)
    parent = child_env.create_record(repo, "parent")
    (repo / "README.md").write_text("base\nchanged\n")
    block = review_traces.BLOCK.replace('"correctness"', '"security"')
    with launch_env.scripted_provider((block,), cost_usd=0.1) as (url, _calls):
        cfg = child_env.config(url)
        meter = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
        executor = child_env.dispatcher(repo, parent, cfg, meter)
        spawn = executor.child_spawner
        assert spawn is not None

        def partial_batch(
            children: list[child_records.ChildTask], group: str, *, at: str | None = None
        ) -> list[child_records.StartedChild]:
            if group.endswith("-b2"):
                raise child_repo.ChildRepoError("second batch unavailable")
            started = spawn(children, group, at=at)
            if failure == "watch":
                child_env.results_of(started)
                lost: concurrent.futures.Future[child_records.ChildResult] = (
                    concurrent.futures.Future()
                )
                lost.set_exception(OSError("first watch failed"))
                started[0] = child_records.StartedChild(started[0].session_id, lost)
            return started

        executor.child_spawner = partial_batch
        target = executor.chain.review_target()
        assert target is not None
        (reviewer,) = role_providers.resolve_reviewers(cfg)
        result = review_gate.run_panel(
            [reviewer, reviewer],
            review_panel.ReviewContext(task="review", diff=child_env.git(repo, "diff")),
            executor=executor,
            target=executor.chain.retain_review_target(target, "parent"),
            block_on="any",
            min_blocking_models=2,
            panel_id="example",
            max_turns=1,
            tools=("read_file",),
            concurrency=1 if failure == "launch" else 2,
        )
    rows = [json.loads(line) for line in (parent / "logs.jsonl").read_text().splitlines()]
    (returned,) = [row for row in rows if row["type"] == "parallel.returned"]
    sid = "parent-rexample-b1-l1" if failure == "launch" else "parent-rexample-l2"
    assert [(c["session_id"], c["status"]) for c in returned["children"]] == [(sid, "no_changes")]
    (record,) = child_dispatch.read_records(parent)
    done, missing = record.children if failure == "launch" else record.children[::-1]
    assert record.state == "complete" and done.settled and not missing.settled
    assert done.merge is not None and done.merge.status == "no_changes" and missing.merge is None
    assert done.usage is not None and done.usage.usd == 0.1
    assert meter.snapshot().settled_children == {sid} and meter.estimate_usd()[0] == 0.1
    assert result.blocked and result.n_abstain == 1
    block_vote, abstention = (
        result.per_reviewer if failure == "launch" else result.per_reviewer[::-1]
    )
    assert (block_vote.session_id, block_vote.verdict) == (sid, "block")
    assert str(block_vote.findings[0].location) == "README.md:2"
    diagnostic = "second batch unavailable" if failure == "launch" else "first watch failed"
    assert abstention.verdict == "abstain" and diagnostic in (abstention.error or "")


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
        "merging",
        "complete",
    ],
)
def test_panel_recovery_never_relaunches_merges_or_grants_twice(
    short_repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    clone = short_repo / "worker"
    with launch_env.scripted_provider() as (url, calls):
        cfg = child_env.config(url)
        _path, launch = child_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker_dir = child_env.create_record(repo, launch.session_id, parent="parent")
        meter = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
        executor = child_env.dispatcher(clone, worker_dir, cfg, meter)
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
        spawner = executor.child_spawner
        assert spawner is not None

        def crash_after_import(
            children: list[child_records.ChildTask], group: str, *, at: str | None = None
        ) -> list[child_records.StartedChild]:
            child_env.results_of(spawner(children, group, at=at))
            raise _Crash

        if boundary == "imported":
            executor.child_spawner = crash_after_import

        def crash_after_write(
            self: parallel_dispatch.ParallelDispatcher, record: child_dispatch.DispatchRecord
        ) -> None:
            original(self, record)
            if record.state == boundary:
                raise _Crash

        with monkeypatch.context() as patch:
            patch.setattr(parallel_dispatch.ParallelDispatcher, "write", crash_after_write)
            with pytest.raises(_Crash):
                parallel_panel.dispatch_panel(
                    executor,
                    [child_records.ChildTask("review", "reviewer", contract=contract)],
                    panel_id="recovery",
                    at=child_env.git(clone, "rev-parse", "HEAD"),
                )
        if boundary == "accounting_failed":
            # Finalization can still publish the charged meter after the dispatch emit failed.
            budget_events.emit_budget_update(meter, original_emit)
        before = len(calls)
        ancestor = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
        ancestor.settle_child(worker_dir.name, children.child_usage(worker_dir))
        if boundary != "complete":
            assert not ancestor.snapshot().settled_children, "an unfinished worker is not charged"
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
        recovered = child_env.dispatcher(clone, worker_dir, cfg, recovered_budget)
        recovered.child_recoverer = children.build_child_recoverer(
            cfg, clone, paths.state_dir(repo), mode="run"
        )
        state = loop_state.LoopState(original_task="worker", tool_calls=0)
        conversation = loop_conversation.Conversation()
        parallel_recovery.reconcile(recovered, state, conversation, turn=1)
        parallel_recovery.reconcile(recovered, state, conversation, turn=1)
        assert len(calls) == before
        assert recovered_budget.snapshot().usd_running_children == 0
        usage = children.child_usage(worker_dir)
        assert usage is not None and usage.tokens == 2 * len(calls)
        ancestor.settle_child(worker_dir.name, usage)
        ancestor.settle_child(worker_dir.name, usage)
        assert ancestor.snapshot().unmetered_tokens == 2 * len(calls)
        assert ancestor.snapshot().settled_children == frozenset({worker_dir.name})
        record = child_dispatch.read_records(worker_dir)[0]
        assert record.state == "complete" and record.merge_policy == "record_only"
        assert record.delivered and all(child.settled for child in record.children)
        assert all(child.merge is None or child.merge.sha == "" for child in record.children)
        assert child_env.git(clone, "rev-parse", "HEAD") == child_env.git(repo, "rev-parse", "HEAD")


def test_a_child_bootstrap_refuses_model_and_steer_parallel_even_with_flag_on(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    clone = short_repo / "worker"
    steered = threading.Event()

    def queue_steer(_request: dict[str, Any]) -> None:
        if not steered.is_set():
            assert ipc.submit_steer(worker_dir, "/parallel 1 -- recurse via steer")
            steered.set()

    with launch_env.scripted_provider(
        [("parallel", {"task": "recurse", "spec": "1"}), launch_env.FINISH],
        before_reply=queue_steer,
    ) as (url, calls):
        cfg = child_env.config(url, tool=True)
        _path, launch = child_env.launch_worker(repo, clone, cfg, monkeypatch)
        worker_dir = layout.bucket_dir(paths.state_dir(repo), "runs") / launch.session_id
        env = {**os.environ, "AGENT6_ALLOW_ROOT": "1", "AGENT6_STREAM_TO_LOG": "1"}
        proc = subprocess.run(
            child_launch.bootstrap_argv(),
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
    assert not refusal["ok"] and "inside a child" in refusal["summary"]
    assert all("parallel" not in json.dumps(c.get("tools", [])) for c in calls)
    assert steered.is_set()
    assert any(
        "work dispatch is unavailable in a child; nothing dispatched."
        in json.dumps(call["messages"])
        for call in calls[1:]
    )
    assert child_dispatch.read_records(worker_dir) == ()


def test_review_leaf_and_unproven_child_have_no_dispatch_authority(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    clone = short_repo / "worker"
    cfg = child_env.config(tool=True)
    path, launch = child_env.launch_worker(repo, clone, cfg, monkeypatch)
    contract = kinds.ConversationContract(system="s", opening="o", max_turns=1)
    child_launch.write(path.parent, launch.model_copy(update={"contract": contract}))
    for mode in ("run", "reviewer"):
        assert (
            children.build_parent_spawner(
                cfg,
                clone,
                paths.state_dir(clone),
                explicit_leaves=frozenset(),
                mode=mode,
                session_id=launch.session_id,
                runtime=parallel.child_runtime(),
            )
            is None
        )
    disp = dispatch.ToolDispatcher(root=clone, config=cfg, isolation="none", mode="reviewer")
    with pytest.raises(tool_errors.ToolError, match="not available"):
        disp.dispatch("parallel", {"task": "recurse"})
    monkeypatch.delenv(child_launch.ENV)
    assert (
        children.build_parent_spawner(
            cfg,
            clone,
            paths.state_dir(clone),
            explicit_leaves=frozenset(),
            mode="run",
            session_id=launch.session_id,
            runtime=parallel.child_runtime(),
        )
        is None
    )


def test_import_retains_descendant_records_receipts_and_target_objects(
    short_repo: pathlib.Path,
) -> None:
    """Removing a worker clone cannot remove its panel record or its immutable target."""
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    child_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    child_repo.clone_worktree(repo, clone)
    worker = child_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = child_env.create_record(repo, "parent-p1-l1-r1-l1", parent=worker.name, mode="reviewer")
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
        fallback_parent=child_env.git(clone, "rev-parse", "HEAD"),
    )
    (clone / "README.md").write_text("reviewed dirty tree\n")
    target = chain.review_target()
    assert target is not None
    target = chain.retain_review_target(target, session_id=worker.name)
    record = child_dispatch.DispatchRecord(
        identity=worker.name + "-r1",
        source="panel",
        owner="e",
        turn=1,
        base=target.snapshot_commit,
        children=(child_dispatch.ChildRecord(session_id=leaf.name, number=1, task="review"),),
    )
    child_dispatch.write_record(worker, record, now=time.time())
    res = child_records.ChildResult(
        child_records.ChildSpec(1, worker.name, clone, None),
        worker,
        git_read.run_branch_for(worker.name),
        True,
        "",
    )
    imported = parallel_results.import_child(res, repo, None, child_env.config())
    shutil.rmtree(clone)
    dest = layout.bucket_dir(paths.state_dir(repo), "reviewers") / leaf.name
    assert dest.is_dir() and not dest.is_symlink()
    assert completion.read_completion(dest) == evidence
    assert json.loads((dest / "objects" / "evidence.json").read_text()) == {"evidence": "retained"}
    saved = session_manifest.read_manifest(dest)
    assert saved.parallel is not None and saved.parallel.parent == worker.name
    assert child_dispatch.read_records(imported.session_dir)[0].base == target.snapshot_commit
    assert child_env.git(repo, "rev-parse", target.snapshot_ref) == target.snapshot_commit
    assert (
        child_env.git(repo, "show", f"{target.snapshot_commit}:README.md") == "reviewed dirty tree"
    )
    # The reviewer is the worker's child by lineage and none of its children.
    assert [child.name for child in lineage.children_of(imported.session_dir)] == [leaf.name]
    assert listing.children_of(paths.state_dir(repo), worker.name) == []
    assert receipt.read_receipt(dest, repo).session_id == leaf.name


def test_cleanup_keeps_a_live_descendant_even_when_all_commits_are_reachable(
    short_repo: pathlib.Path,
) -> None:
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    clone = short_repo / "worker"
    child_repo.clone_worktree(repo, clone)
    worker = child_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = child_env.create_record(repo, "parent-p1-l1-r1-l1", parent=worker.name, mode="reviewer")
    process.write_worker_pid(leaf, os.getpid())
    kept: list[str] = []
    worktrees.cleanup(
        [child_records.ChildSpec(1, worker.name, clone, None)],
        origin=repo,
        record_state=paths.state_dir(repo),
        worktree_root=short_repo,
        base=short_repo,
        launch_dir=child_launch.launch_dir(paths.state_dir(repo), "parent-p1"),
        cfg=child_env.config(),
        note_keep=lambda _p, why: kept.append(why),
    )
    assert clone.is_dir() and leaf.is_dir()
    assert any("live" in why for why in kept)


def test_an_unfinished_panel_keeps_the_ancestor_reservation(short_repo: pathlib.Path) -> None:
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    worker = child_env.create_record(repo, "parent-p1-l1", parent="parent")
    record = child_dispatch.DispatchRecord(
        identity=worker.name + "-r1",
        source="panel",
        owner="w",
        turn=1,
        base="",
        children=(
            child_dispatch.ChildRecord(session_id=worker.name + "-r1-l1", number=1, task="review"),
        ),
        state="running",
    )
    child_dispatch.write_record(worker, record, now=time.time())
    assert children.child_usage(worker) is None


def test_a_settled_panel_keeps_the_ancestor_hold_until_the_journal_names_it(
    short_repo: pathlib.Path,
) -> None:
    """A worker that died before journaling a panel's charge must not settle without it."""
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    worker = child_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = worker.name + "-r1-l1"
    record = child_dispatch.DispatchRecord(
        identity=worker.name + "-r1",
        source="panel",
        merge_policy="record_only",
        owner="w",
        turn=1,
        base="",
        children=(
            child_dispatch.ChildRecord(
                session_id=leaf,
                number=1,
                task="review",
                settled=True,
                usage=child_dispatch.UsageRecord(usd=2.0),
            ),
        ),
        state="complete",
    )
    child_dispatch.write_record(worker, record, now=time.time())
    tracker = budget.BudgetTracker(max_usd=10, max_tokens_fallback=100, max_plan_points=20)
    sink = event_log.EventSink(worker / "logs.jsonl")
    budget_events.emit_budget_update(tracker, sink.emit)
    assert children.child_usage(worker) is None
    tracker.settle_child(leaf, budget.UsageSettlement(usd=2.0, usd_partial=False, tokens=0))
    budget_events.emit_budget_update(tracker, sink.emit)
    usage = children.child_usage(worker)
    assert usage is not None
    assert usage.usd == pytest.approx(2.0)


@pytest.mark.parametrize("worker_dead", [False, True])
def test_stop_follows_clone_state_and_readonly_bucket_and_leaves_the_worker_unsettled(
    short_repo: pathlib.Path,
    worker_dead: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live leaf is found through a worker symlink, even when the worker already died."""
    # These stand-ins never import their children's work or answer a stop request.
    monkeypatch.setattr(stop, "CHILDREN_WAIT_S", 0.0)
    repo = short_repo / "origin"
    child_env.init_repo(repo)
    parent = child_env.create_record(repo, "parent")
    clone = short_repo / "worker"
    child_repo.clone_worktree(repo, clone)
    worker = child_env.create_record(repo, "parent-p1-l1", parent="parent")
    leaf = child_env.create_record(repo, "parent-p1-l1-r1-l1", parent=worker.name, mode="reviewer")
    ancestor = budget.BudgetTracker(max_usd=5, max_tokens_fallback=1000, max_plan_points=20)
    child_dispatch.write_record(
        worker,
        child_dispatch.DispatchRecord(
            identity=worker.name + "-r1",
            source="panel",
            owner="w",
            turn=1,
            base="",
            children=(child_dispatch.ChildRecord(session_id=leaf.name, number=1, task="review"),),
            state="running",
        ),
        now=time.time(),
    )
    procs: list[subprocess.Popen[bytes]] = []
    try:
        for directory in [parent, leaf] if worker_dead else [parent, worker, leaf]:
            proc = subprocess.Popen(
                [sys.executable, "-c", "import signal; signal.pause()"], start_new_session=True
            )
            procs.append(proc)
            process.write_worker_pid(directory, proc.pid)
        stopped = stop.stop_session(parent, wait_s=0, grace_s=0.05)
        assert stopped.ok
        assert all(not process.pid_alive(proc.pid) for proc in procs)
        usage = children.child_usage(worker)
        assert usage is None, "a worker with an unfinished descendant has no final usage"
        ancestor.settle_child(worker.name, usage)
        assert not ancestor.snapshot().settled_children and ancestor.available().usd == 5
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=5)
