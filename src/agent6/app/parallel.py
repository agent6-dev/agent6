# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The fan-out orchestrator for `agent6 run --parallel` and the `/parallel` dispatch.

Each lane is a disposable clone of the repo running its own detached `agent6 run`.
The live lanes are symlinked into the origin's sessions, awaited, and each finished
lane's branch and session dir imported back; the fan-out then ranks them and
prints a report. Nothing is merged: the operator picks a winner. The origin is
untouched until an import lands a branch, and clones and lane state are torn down
after import. The git plumbing lives in `harness.subrun`, the ranking in
`app.compare`; the front-end injects a `LaneRuntime`, so this module never
imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import functools
import pathlib
import uuid
from collections.abc import Sequence

from agent6 import errors, kinds, lane_records
from agent6.app import compare as app_compare
from agent6.app import (
    lane_launch,
    lane_watch,
    parallel_results,
    preflight,
    stamps,
    workdirs,
    workspace_setup,
)
from agent6.app import lanes as app_lanes
from agent6.app import reporter as app_reporter
from agent6.app.session import runtime as session_runtime
from agent6.app.session import setup as session_setup
from agent6.config import Config
from agent6.context import memory
from agent6.providers import TranscriptSink
from agent6.sessions import event_log, ipc, journal, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read
from agent6.viewmodel import wire


def run_parallel(
    task: str,
    lanes: list[lane_records.LaneSpec],
    *,
    cfg: Config,
    explicit_leaves: frozenset[str],
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    runtime: app_lanes.LaneRuntime,
    spawner: lane_records.LaneSpawner | None = None,
    max_usd: float | None = None,
    fanout_id: str | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    pins: Sequence[str] = (),
    spec: str = "",
    operator_task: str | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> int:
    """Run the lanes to completion, import them, and print a ranked comparison.

    The fan-out is a session of its own under the origin's runs: a manifest with
    the fan-out stamp and no run branch, a journal, and a worker pid while the
    lanes run and the judge ranks them. `stop <fanout_id>` ends it with its lanes.

    Args:
        task: The task every lane runs.
        lanes: The planned lanes.
        cfg: The origin's effective config.
        explicit_leaves: The dotted leaves the origin's config layers set.
        origin: The repository.
        origin_state: The origin's state dir.
        runtime: The front-end's primitives.
        spawner: Clones and spawns a lane; the bridge spawner by default.
        max_usd: Each lane's spend cap.
        fanout_id: The fan-out's session id; derived from the first lane's by default.
        auto_approve: Forwarded to every lane's argv.
        lane_away: Every lane's away-mode.
        pins: The pins every lane starts with.
        spec: The `--parallel` argument as typed, stamped on the manifest.
        operator_task: The words before composition; None means `task` is uncomposed.
        reporter: Receives the notes and the report.

    Returns:
        The lanes' exit code, 2 with no lanes or an unreadable origin, 130 on an
        interrupt or a stop request.

    Raises:
        KeyboardInterrupt: The operator interrupted past the await; the end is journaled
            first.
    """
    operator_task = task if operator_task is None else operator_task
    if not lanes:
        reporter.error("no lanes to run")
        return 2
    if fanout_id is None:
        fanout_id = lanes[0].session_id.rsplit("-l", 1)[0]
    # The lanes are seeded from the origin's rulings file, which a run start would move first.
    if note := memory.migrate_legacy_decisions(origin_state):
        reporter.note(note)
    try:
        origin_status = git_read.status(origin)
    except errors.GitError as exc:
        reporter.error(str(exc))
        return 2
    base_sha = origin_status.head_sha
    if spawner is None:
        cfg = workspace_setup.infer_if_unset(cfg, origin, reporter=reporter, at=base_sha)
    # Before any clone: a lane's clone holds tracked files only.
    refusal = workspace_setup.gate_refusal(
        cfg,
        origin,
        workspace_setup.origin_gate(cfg, origin),
        workspace="a lane's clone",
        at=base_sha,
    )
    if refusal is not None:
        reporter.refuse(refusal)
        return 2

    layout = sessions_layout.SessionLayout(
        state_dir=origin_state, session_id=fanout_id, subdir=kinds.session_bucket("run")
    )
    layout.ensure()
    session_runtime.note_state_disk_space(origin_state, reporter=reporter)
    stamps.write_session_manifest(
        layout,
        session_id=fanout_id,
        user_task=task,
        operator_task=operator_task,
        base_sha=base_sha,
        base_branch=origin_status.branch,
        run_branch=None,
        cfg=cfg,
        fanout=sessions_manifest.FanoutStamp(lanes=len(lanes), spec=spec),
    )
    execution_id = uuid.uuid4().hex
    events = event_log.EventSink(layout.logs_path)
    process.emit_session_start(
        events,
        layout.session_dir,
        journal.SessionStart(
            execution_id=execution_id,
            session_id=fanout_id,
            user_task=operator_task,
            mode="run",
        ),
    )
    events.emit(
        journal.ParallelDispatched(group=fanout_id, lanes=len(lanes), tasks=[operator_task])
    )
    configured_gate = bool(cfg.harness.verify.command)
    if spawner is None:
        # A clone lacks the untracked signals verify inference reads in the origin.
        cfg = preflight.infer_verify_if_unset(
            cfg,
            origin,
            mode="run",
            events=events,
            transcript_sink=TranscriptSink(layout.transcripts_dir, objects_dir=layout.objects_dir),
            budget=session_setup.budget_tracker(cfg),
            reporter=reporter,
        )
    gate_source = ""
    if cfg.harness.verify.command:
        gate_source = "configured" if configured_gate else "inferred"
    # The gate every lane runs, pinned on the coordinator as a run pins its own.
    stamps.stamp_verify_gate(
        layout.session_dir,
        sessions_manifest.GateStamp(command=cfg.harness.verify.command, source=gate_source),
    )
    if spawner is None:
        spawner = functools.partial(
            app_lanes.bridge_spawner,
            pins=pins,
            cfg=cfg,
            origin=origin,
            max_usd=max_usd,
            auto_approve=auto_approve,
            lane_away=lane_away,
            fanout_id=fanout_id,
            coordinator=fanout_id,
            gate_source=gate_source,
            explicit_leaves=explicit_leaves,
            runtime=runtime,
            operator_task=operator_task,
        )
    try:
        return _drive_fanout(
            task,
            lanes,
            spawner=spawner,
            cfg=cfg,
            origin=origin,
            origin_state=origin_state,
            runtime=runtime,
            max_usd=max_usd,
            fanout_id=fanout_id,
            base_sha=base_sha,
            events=events,
            execution_id=execution_id,
            coordinator_dir=layout.session_dir,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        # An interrupt past the await is journaled, so the record reads stopped, never stale.
        with contextlib.suppress(event_log.EventWriteError):
            events.emit(
                journal.SessionStop(
                    execution_id=execution_id,
                    reason="interrupted",
                    turns=0,
                    verification=None,
                )
            )
        raise
    except Exception:
        with contextlib.suppress(event_log.EventWriteError):
            events.emit(
                journal.SessionStop(
                    execution_id=execution_id,
                    reason="crashed",
                    turns=0,
                    verification=None,
                )
            )
        raise
    finally:
        process.clear_worker_pid(layout.session_dir)


def _drive_fanout(
    task: str,
    lanes: list[lane_records.LaneSpec],
    *,
    spawner: lane_records.LaneSpawner,
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    runtime: app_lanes.LaneRuntime,
    max_usd: float | None,
    fanout_id: str,
    base_sha: str,
    events: event_log.EventSink,
    execution_id: str,
    coordinator_dir: pathlib.Path,
    reporter: app_reporter.Reporter,
) -> int:
    """Spawn, await, import, rank and report the lanes, journaling into the fan-out's session.

    Args:
        task: The task every lane runs.
        lanes: The planned lanes.
        spawner: Clones and spawns a lane.
        cfg: The origin's effective config.
        origin: The repository.
        origin_state: The origin's state dir.
        runtime: The front-end's primitives.
        max_usd: Each lane's spend cap.
        fanout_id: The fan-out's session id.
        base_sha: The origin's HEAD at the start.
        events: The fan-out's journal.
        execution_id: The coordinator's execution identifier.
        coordinator_dir: The fan-out's session dir, where a stop request lands.
        reporter: Receives the notes and the report.

    Returns:
        The fan-out's exit code.
    """
    reporter.note(f"parallel fan-out {fanout_id}: {len(lanes)} lanes")
    if max_usd is not None:
        # The judge is one more capped call series, so the total includes it.
        reporter.note(
            f"budget: ${max_usd:g}/lane x {len(lanes)} + judge"
            f" = ${max_usd * (len(lanes) + 1):g} total"
        )

    results: list[lane_records.LaneResult] = []
    try:
        for spec in lanes:
            if ipc.stop_request_pending(coordinator_dir):
                break
            res = spawner(spec, task)
            results.append(res)
            if res.ok:
                lane_watch.symlink_lane(origin_state, res)
                lane_watch.print_lane_status(spec, "started", "", reporter=reporter)
            else:
                reporter.note(f"lane {spec.lane} [{spec.session_id}]: FAILED to start: {res.error}")
        interrupted = lane_watch.await_lanes(
            [r for r in results if r.ok],
            should_stop=lambda: ipc.stop_request_pending(coordinator_dir),
            stop_is_now=lambda: ipc.stop_request_is_now(coordinator_dir),
            reporter=reporter,
        )
    except KeyboardInterrupt:
        # An interrupt mid-spawn routes the started lanes through the same stop grace.
        interrupted = lane_watch.await_lanes(
            [r for r in results if r.ok],
            already_interrupted=True,
            reporter=reporter,
        )
    if len(results) < len(lanes):
        first = lanes[len(results)].lane
        span = f"lane {first}" if first == len(lanes) else f"lanes {first}-{len(lanes)}"
        reporter.note(f"stopped before {span} started")

    # The stop request is consumed with the session.
    ipc.clear_stop_request(coordinator_dir)
    batch = parallel_results.import_lanes(
        results,
        origin=origin,
        origin_state=origin_state,
        base_sha=base_sha,
        task=task,
        reporter=reporter,
    )
    if batch.failed:
        events.emit(
            journal.ParallelFailed(
                group=fanout_id,
                fanout=True,
                lanes=[
                    {"session_id": res.spec.session_id, "detail": detail}
                    for res, detail in batch.failed
                ],
            )
        )
    workdirs.cleanup(
        batch.imported,
        origin=origin,
        note_keep=lambda path, why: reporter.note(f"kept {path}: {why}"),
        workdir_root=lanes[0].workdir.parent,
        base=workdirs.workdir_base(cfg, origin),
        launch_dir=lane_launch.launch_dir(origin_state, fanout_id),
        cfg=cfg,
    )

    outcome = app_compare.rank(
        cfg,
        batch.candidates,
        transcript_dir=origin_state / "parallel" / fanout_id,
        build_provider=runtime.build_provider,
        judging_status=runtime.judging_status,
        max_usd=max_usd,
        events=events,
        reporter=reporter,
    )
    parallel_results.stamp_compare_outcomes(
        batch.candidates,
        outcome,
        origin_state=origin_state,
        coordinator_dir=coordinator_dir,
        group=fanout_id,
        reporter=reporter,
    )
    model_refs = {}
    for spec in batch.imported:
        manifest = sessions_manifest.read_manifest(
            lane_watch.lane_link(origin_state, spec.session_id, mode=spec.mode)
        )
        if manifest.git_control == "model":
            model_refs[spec.session_id] = wire.commits_ref(manifest, origin)
    parallel_results.print_report(
        batch.candidates,
        outcome,
        batch.failed,
        rows=batch.rows,
        model_refs=model_refs,
        # A lane that made no commit imports no branch.
        committed={
            spec.session_id: git_read.branch_exists(
                origin, git_read.run_branch_for(spec.session_id)
            )
            for spec in batch.imported
        },
        fanout_id=fanout_id,
        reporter=reporter,
    )
    for line in batch.plan_lines:
        reporter.out(line)
    by_id = {c.session_id: c for c in batch.candidates}
    events.emit(
        journal.ParallelCompared(
            group=fanout_id,
            ranked_by=outcome.ranked_by,
            rationale=outcome.rationale,
            ranking=[
                {
                    "session_id": rid,
                    "verify": app_compare.verify_word(by_id[rid].verification),
                    "cost_usd": by_id[rid].cost_usd,
                    # The listings' cell: plan points for a lane on a subscription plan.
                    "cost": batch.rows[rid].usage_cell if rid in batch.rows else "",
                }
                for rid in outcome.ranking
            ],
            judge_cost_usd=outcome.judge_cost_usd,
            judge_cost_partial=outcome.judge_cost_partial,
        )
    )
    if outcome.judge_cost_usd > 0 or outcome.judge_cost_partial:
        # The judge is the fan-out's own spend; each lane's rides its own row.
        events.emit(
            journal.BudgetUpdate(
                usd_total=outcome.judge_cost_usd,
                usd_partial=outcome.judge_cost_partial,
            )
        )
    rc = (
        130
        if interrupted
        else parallel_results.fanout_exit_code(
            batch.candidates, all_budget_exhausted=batch.all_budget_exhausted
        )
    )
    reason, verification = parallel_results.fanout_end(rc, batch.candidates)
    events.emit(
        journal.SessionStop(
            execution_id=execution_id, reason=reason, turns=0, verification=verification
        )
    )
    return rc
