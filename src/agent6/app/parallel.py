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
from collections.abc import Sequence

from agent6 import errors, event_log, kinds, memory, task_text
from agent6.app import (
    _lane_watch,
    _parallel_results,
    _setup,
    _workdirs,
    preflight,
    stamps,
    workspace_setup,
)
from agent6.app import compare as app_compare
from agent6.app import lanes as app_lanes
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import subrun
from agent6.providers import TranscriptSink
from agent6.sessions import ipc, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_read


def run_parallel(
    task: str,
    lanes: list[subrun.LaneSpec],
    *,
    cfg: Config,
    explicit_leaves: frozenset[str],
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    runtime: app_lanes.LaneRuntime,
    spawner: subrun.LaneSpawner | None = None,
    max_usd: float | None = None,
    fanout_id: str | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    pins: Sequence[str] = (),
    spec: str = "",
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
        reporter: Receives the notes and the report.

    Returns:
        The lanes' exit code, 2 with no lanes or an unreadable origin, 130 on an
        interrupt or a stop request.

    Raises:
        KeyboardInterrupt: The operator interrupted past the await; the end is journaled
            first.
    """
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
    stamps.write_session_manifest(
        layout,
        session_id=fanout_id,
        user_task=task,
        base_sha=base_sha,
        base_branch=origin_status.branch,
        run_branch=None,
        cfg=cfg,
        fanout=sessions_manifest.FanoutStamp(lanes=len(lanes), spec=spec),
    )
    events = event_log.EventSink(layout.logs_path)
    display_task = task_text.task_display_text(task)
    process.emit_session_start(
        events,
        layout.session_dir,
        "session.start",
        session_id=fanout_id,
        user_task=display_task,
        mode="run",
    )
    events.emit("loop.parallel.dispatched", group=fanout_id, lanes=len(lanes), tasks=[display_task])
    configured_gate = bool(cfg.harness.verify_command)
    if spawner is None:
        # A clone lacks the untracked signals verify inference reads in the origin.
        cfg = preflight.infer_verify_if_unset(
            cfg,
            origin,
            mode="run",
            events=events,
            transcript_sink=TranscriptSink(layout.transcripts_dir),
            budget=_setup.budget_tracker(cfg),
            reporter=reporter,
        )
    verify_origin = ""
    if cfg.harness.verify_command:
        verify_origin = "configured" if configured_gate else "inferred"
    # The gate every lane runs, pinned on the coordinator as a run pins its own.
    stamps.stamp_verify_gate(layout.session_dir, cfg.harness.verify_command, verify_origin)
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
            verify_origin=verify_origin,
            explicit_leaves=explicit_leaves,
            runtime=runtime,
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
            coordinator_dir=layout.session_dir,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        # An interrupt past the await is journaled, so the record reads stopped, never stale.
        with contextlib.suppress(event_log.EventWriteError):
            events.emit("session.end", reason="interrupted", turns=0, all_passed=False)
        raise
    except Exception:
        with contextlib.suppress(event_log.EventWriteError):
            events.emit("session.end", reason="crashed", turns=0, all_passed=False)
        raise
    finally:
        process.clear_worker_pid(layout.session_dir)


def _drive_fanout(
    task: str,
    lanes: list[subrun.LaneSpec],
    *,
    spawner: subrun.LaneSpawner,
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    runtime: app_lanes.LaneRuntime,
    max_usd: float | None,
    fanout_id: str,
    base_sha: str,
    events: event_log.EventSink,
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

    results: list[subrun.LaneResult] = []
    try:
        for spec in lanes:
            if ipc.stop_request_pending(coordinator_dir):
                break
            res = spawner(spec, task)
            results.append(res)
            if res.ok:
                _lane_watch.symlink_lane(origin_state, res)
                _lane_watch.print_lane_status(spec, "started", "", reporter=reporter)
            else:
                reporter.note(f"lane {spec.lane} [{spec.session_id}]: FAILED to start: {res.error}")
        interrupted = _lane_watch.await_lanes(
            [r for r in results if r.ok],
            already_interrupted=ipc.stop_request_pending(coordinator_dir),
            should_stop=lambda: ipc.stop_request_pending(coordinator_dir),
            reporter=reporter,
        )
    except KeyboardInterrupt:
        # An interrupt mid-spawn routes the started lanes through the same stop grace.
        interrupted = _lane_watch.await_lanes(
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
    batch = _parallel_results.import_lanes(
        results,
        origin=origin,
        origin_state=origin_state,
        base_sha=base_sha,
        task=task,
        reporter=reporter,
    )
    if batch.failed:
        events.emit(
            "loop.parallel.failed",
            group=fanout_id,
            fanout=True,
            lanes=[
                {"session_id": res.spec.session_id, "detail": detail}
                for res, detail in batch.failed
            ],
        )
    _workdirs.cleanup(
        batch.imported,
        workdir_root=lanes[0].workdir.parent,
        base=_workdirs.workdir_base(cfg, origin),
        cfg=cfg,
    )

    outcome = app_compare.rank(
        cfg,
        batch.candidates,
        transcript_dir=origin_state / "parallel" / fanout_id,
        build_provider=runtime.build_provider,
        judging_status=runtime.judging_status,
        max_usd=max_usd,
        reporter=reporter,
    )
    _parallel_results.stamp_compare_outcomes(
        batch.candidates,
        outcome,
        origin_state=origin_state,
        reporter=reporter,
    )
    _parallel_results.print_report(
        batch.candidates,
        outcome,
        batch.failed,
        rows=batch.rows,
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
        "loop.parallel.compared",
        group=fanout_id,
        ranked_by=outcome.ranked_by,
        ranking=[
            {
                "session_id": rid,
                "verify": app_compare.verify_word(by_id[rid].verify_ok),
                "cost_usd": by_id[rid].cost_usd,
                # The listings' cell: plan points for a lane on a subscription plan.
                "cost": batch.rows[rid].cost_cell if rid in batch.rows else "",
            }
            for rid in outcome.ranking
        ],
        judge_cost_usd=outcome.judge_cost_usd,
        judge_cost_partial=outcome.judge_cost_partial,
    )
    if outcome.judge_cost_usd > 0 or outcome.judge_cost_partial:
        # The judge is the fan-out's own spend; each lane's rides its own row.
        events.emit(
            "budget.update",
            usd_total=outcome.judge_cost_usd,
            usd_partial=outcome.judge_cost_partial,
        )
    rc = (
        130
        if interrupted
        else _parallel_results.fanout_exit_code(
            batch.candidates, all_budget_exhausted=batch.all_budget_exhausted
        )
    )
    reason, all_passed = _parallel_results.fanout_end(rc, batch.candidates)
    events.emit("session.end", reason=reason, turns=0, all_passed=all_passed)
    return rc
