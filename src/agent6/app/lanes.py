# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The lanes of a fan-out: the front-end runtime, planning, spawning, running and dispatch."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import os
import pathlib
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from agent6 import budget as agent6_budget
from agent6 import directive, errors, kinds, lane_records, paths
from agent6.app import compare as app_compare
from agent6.app import lane_launch, lane_watch, parallel_results, stop, workdirs
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError
from agent6.context import memory
from agent6.harness import subrun
from agent6.models import validate
from agent6.sessions import ipc, journal, lane_dispatch, lineage
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_ops, git_read
from agent6.viewmodel import (
    events,
    fold_session,
    format,
    produced_result,
    session_is_live,
    session_log,
    summarize_session_dir,
    tail_events,
)


class ParallelError(Exception):
    """The fan-out could not be set up: over the lane cap, or a route that names nothing."""


class SpawnRun(Protocol):
    """Spawn a detached agent6 command and locate its session dir: the front-end's primitive."""

    def __call__(
        self,
        argv: list[str],
        cwd: pathlib.Path,
        *,
        before: set[pathlib.Path],
        list_dirs: Callable[[], list[pathlib.Path]],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        """Spawn the command and wait for its session dir.

        Args:
            argv: The full command: the interpreter, its flags and the bootstrap module.
            cwd: Where to spawn.
            before: The session dirs that existed before the spawn.
            list_dirs: Lists the candidate session dirs.
            env: The child's environment.

        Returns:
            `(session_dir, pid, "")` once a new dir with a log appears, else
            `(None, pid, error)`, the pid 0 when nothing started.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class LaneRuntime:
    """The front-end primitives the parallel pipeline drives.

    Attributes:
        spawn: Launches a detached agent6 run and locates its session dir.
        build_provider: Builds the reviewer provider the auto-compare's judge uses.
        judging_status: Reports the judge's progress.
    """

    spawn: SpawnRun
    build_provider: app_compare.BuildProvider
    judging_status: app_compare.JudgingStatus


def build_lane_specs(
    spec: str,
    *,
    cfg: Config,
    origin: pathlib.Path,
    fanout_id: str,
    workdir_root: pathlib.Path | None = None,
) -> list[lane_records.LaneSpec]:
    """Plan the lanes for a `--parallel` fan-out, refusing over-cap up front.

    Args:
        spec: The `--parallel` argument as typed.
        cfg: The run's config.
        origin: The repository.
        fanout_id: The fan-out's session id.
        workdir_root: Where the lane clones go; the fan-out's subordinate dir by default.

    Returns:
        One spec per lane, numbered from 1.

    Raises:
        ConfigError: A lane's model names nothing.
    """
    if workdir_root is None:
        workdir_root = workdirs.subordinate_workdir_root(cfg, origin, fanout_id)
    settings = lane_settings(cfg, directive.parse_spec(spec, limit=cfg.parallel.max_lanes))
    return [
        lane_records.LaneSpec(
            lane=i,
            session_id=f"{fanout_id}-l{i}",
            workdir=workdir_root / f"lane-{i}",
            settings=chosen,
        )
        for i, chosen in enumerate(settings, start=1)
    ]


def lane_settings(cfg: Config, selectors: Sequence[str | None]) -> list[kinds.ModelSettings | None]:
    """Resolve each lane's selector to its complete settings, in lane order.

    Args:
        cfg: The run's config.
        selectors: One entry per lane; None is the worker's own settings.

    Returns:
        The settings in lane order, None where the lane runs the worker's own.

    Raises:
        ConfigError: A selector names nothing.
    """
    return [cfg.lane_settings(selector) for selector in selectors]


def bridge_spawner(
    spec: lane_records.LaneSpec,
    task: str,
    *,
    operator_task: str | None = None,
    pins: Sequence[str] = (),
    cfg: Config,
    explicit_leaves: frozenset[str],
    origin: pathlib.Path,
    max_usd: float | None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    at: str | None = None,
    fanout_id: str,
    coordinator: str,
    gate_source: sessions_manifest.GateSource = "",
    limits: lane_records.LaneLimits | None = None,
    allowance: agent6_budget.BudgetAllowance | None = None,
    contract: kinds.ConversationContract | None = None,
    runtime: LaneRuntime,
) -> lane_records.LaneResult:
    """Clone the origin and start the lane bootstrap detached in the clone.

    The task, pins, caps and lineage travel in the lane's launch file under the origin's
    state dir; the host argv is the fixed bootstrap.

    Args:
        spec: The lane.
        task: The lane's task.
        pins: The coordinator's pins, passed out of band of the task.
        operator_task: The words before composition; None means `task` is uncomposed.
        cfg: The origin's effective config.
        explicit_leaves: The dotted leaves the origin's config layers set.
        origin: The repository.
        max_usd: The lane's spend cap.
        auto_approve: The fan-out's own `--auto-approve`, forwarded.
        lane_away: The lane's away-mode: "wait" parks a question for `agent6 attach`,
            "deny" answers it empty.
        at: A commit to detach the clone at; the coordinator's chain tip.
        fanout_id: The group the lane belongs to.
        coordinator: The session every listing nests the lane under.
        gate_source: Where the gate in the lane's config came from, "" when it has none.
        limits: The turn cap and deadline the lane runs under; None for the config's own.
        allowance: What the coordinator had left when the lane started, which sets every
            cap; None for the config's.
        contract: The harness-owned read-only conversation; None for work.
        runtime: The front-end's spawn primitive.

    Returns:
        The lane's result once its session dir is located, the run still going;
        `ok=False` when the clone or spawn failed, with the child's pid when one started.
    """
    branch = git_read.run_branch_for(spec.session_id)
    try:
        subrun.clone_workspace(origin, spec.workdir)
        if at is not None:
            # A local clone hardlinks the whole odb, so the chain's commits are present.
            git_ops.checkout_detached(spec.workdir, at)
    except (subrun.SubrunError, errors.GitError) as exc:
        return lane_records.LaneResult(
            spec=spec, session_dir=spec.workdir, branch=branch, ok=False, error=str(exc)
        )
    # The lane's own manifest records its lineage, so a coordinator death orphans nothing; the
    # gate's origin was resolved in the origin, where the repo signals are.
    private = lane_launch.launch_dir(paths.state_dir(origin), fanout_id)
    try:
        config = workdirs.write_lane_config(
            cfg, spec, explicit_leaves=explicit_leaves, into=private
        )
        launch_path = lane_launch.write(
            private,
            lane_launch.LaneLaunch(
                workdir=spec.workdir,
                session_id=spec.session_id,
                task=task,
                pins=tuple(pins),
                max_usd=max_usd,
                auto_approve=auto_approve,
                away=lane_away,
                config=config.path,
                notes=config.notes,
                coordinator=coordinator,
                group=fanout_id,
                lane=spec.lane,
                gate_source=gate_source,
                operator_task=operator_task,
                limits=limits,
                allowance=allowance,
                contract=contract,
            ),
        )
    except (lane_launch.LaunchError, OSError) as exc:
        return lane_records.LaneResult(
            spec=spec, session_dir=spec.workdir, branch=branch, ok=False, error=str(exc)
        )
    lane_state = paths.state_dir(spec.workdir)
    # The clone's state dir is empty; the origin's memory and rulings are seeded into it.
    memory.seed_store(paths.state_dir(origin), lane_state)
    lane_bucket = sessions_layout.bucket_dir(lane_state, kinds.session_bucket(spec.mode))

    def list_dirs() -> list[pathlib.Path]:
        if not lane_bucket.is_dir():
            return []
        return [p for p in lane_bucket.iterdir() if p.is_dir()]

    # Subordinate models cannot dispatch work; only the harness may schedule panel leaves.
    markers = {
        "AGENT6_STREAM_TO_LOG": "1",
        "AGENT6_SUBRUN": "1",
        lane_launch.ENV: str(launch_path),
    }
    session_dir, pid, err = runtime.spawn(
        lane_launch.bootstrap_argv(),
        spec.workdir,
        before=set(),
        list_dirs=list_dirs,
        env={**os.environ, **markers},
    )
    if session_dir is None:
        return lane_records.LaneResult(
            spec=spec,
            session_dir=lane_bucket / spec.session_id,
            branch=branch,
            ok=False,
            error=err,
            pid=pid,
        )
    return lane_records.LaneResult(
        spec=spec, session_dir=session_dir, branch=branch, ok=True, error="", pid=pid
    )


def run_lane_to_completion(
    spec: lane_records.LaneSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    explicit_leaves: frozenset[str],
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    group: str,
    coordinator: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    gate_source: sessions_manifest.GateSource = "",
    spawner: lane_records.LaneSpawner | None = None,
    at: str | None = None,
    import_lock: threading.Lock | None = None,
    poll_interval_s: float = lane_watch.POLL_INTERVAL_S,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    should_stop: Callable[[], bool] | None = None,
    stop_is_now: Callable[[], bool] | None = None,
    hard_stop: threading.Event | None = None,
    limits: lane_records.LaneLimits | None = None,
    allowance: agent6_budget.BudgetAllowance | None = None,
    contract: kinds.ConversationContract | None = None,
    clock: Callable[[], float] = time.time,
    watch: Callable[[pathlib.Path, str, str], bool] | None = None,
) -> lane_records.LaneResult:
    """Run one subordinate lane to its end and import it into the origin.

    The coordinator runs a group of these on a thread pool, so each is
    self-contained. A lane that failed to start, was still running at teardown or its
    deadline, refused its import, or ended without a result comes back `ok=False`; an
    imported branch is safe in the origin either way, and a clone that was not imported
    keeps its work. The lane's process is owned to its end: one that does not stop when
    asked is killed and reaped, never left running.

    Args:
        spec: The lane.
        task: The lane's task.
        pins: The coordinator's pins.
        cfg: The origin's effective config.
        explicit_leaves: The dotted leaves the origin's config layers set.
        origin: The repository.
        origin_state: The origin's state dir.
        group: The group the lane belongs to.
        coordinator: The session every listing nests the lane under.
        runtime: The front-end's primitives.
        max_usd: The lane's spend cap.
        auto_approve: The coordinator's own `--auto-approve`, forwarded.
        lane_away: The lane's away-mode.
        gate_source: Where the gate in the config came from, "" when it has none.
        spawner: Clones and spawns the lane; the bridge spawner by default.
        at: A commit to detach the clone at.
        import_lock: Serializes the import across the group, since concurrent fetches
            into one repo race.
        poll_interval_s: How often the await polls.
        reporter: Receives the notes.
        should_stop: Polled to interrupt the await; the lane then gets a stop request
            with the coordinator's urgency.
        stop_is_now: Whether that request carries the `now` urgency; None reads as `now`. A
            plain request lets the lane finish its turn and waits for it, escalating to
            `now` when one arrives; `now` hands its command back and waits a bounded grace.
        hard_stop: Set to skip the grace.
        limits: The turn cap and deadline the lane runs under; None for the config's own.
        allowance: What the coordinator had left when the lane started, as the lane's caps;
            None for the config's caps.
        contract: The harness-owned read-only conversation; None for work.
        clock: The wall clock the deadline is read against.
        watch: The coordinator's side of the wait, called on each poll with the lane's dir,
            its label and its status: it counts the lane's spend and answers its prompt,
            and returns whether the coordinator's budget is spent. None leaves the lane's
            prompt to whoever attaches to it.

    Returns:
        The lane's result, its `session_dir` the imported dir on success.
    """
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner,
            pins=pins,
            cfg=cfg,
            origin=origin,
            max_usd=max_usd,
            auto_approve=auto_approve,
            lane_away=lane_away,
            at=at,
            explicit_leaves=explicit_leaves,
            fanout_id=group,
            coordinator=coordinator,
            gate_source=gate_source,
            limits=limits,
            allowance=allowance,
            contract=contract,
            runtime=runtime,
        )
    res = spawner(spec, task)
    if not res.ok:
        # A child that never reported its session dir is still a process of ours.
        how = lane_watch.stop_child(res.pid)
        if how != "gone":
            res = dataclasses.replace(res, error=f"{res.error}; its process {how}")
        return res
    # The live symlink lets a hub answer the lane's prompts while it runs.
    lane_watch.symlink_lane(origin_state, res)
    deadline_at = limits.deadline_at if limits is not None else None
    overdue = over_budget = False
    label = f"{'seat' if contract is not None else 'lane'} {spec.lane} {spec.session_id}"

    def on_poll(status: str) -> bool:
        nonlocal over_budget
        over_budget = watch is not None and watch(res.session_dir, label, status)
        return over_budget

    if not lane_watch.await_lane(
        res,
        poll_interval_s=poll_interval_s,
        should_stop=should_stop,
        deadline_at=deadline_at,
        clock=clock,
        on_poll=on_poll,
    ):
        overdue = deadline_at is not None and clock() >= deadline_at
        immediate = overdue or over_budget or stop_is_now is None or stop_is_now()
        asked = ipc.request_stop(res.session_dir, now=immediate)
        if not immediate:

            def escalate() -> bool:
                # The lane finishes its turn; a later `now` or a teardown escalates.
                torn_down = hard_stop is not None and hard_stop.is_set()
                return torn_down or (stop_is_now is not None and stop_is_now())

            if lane_watch.await_lane(res, poll_interval_s=poll_interval_s, should_stop=escalate):
                asked = True
            else:
                asked = ipc.request_stop(res.session_dir, now=True)
                immediate = True
        if immediate and not lane_watch.drain_lane(
            res, poll_interval_s=poll_interval_s, hard_stop=hard_stop
        ):
            # Not reading its requests: ended by signal. The clone keeps its branch.
            how = lane_watch.stop_child(res.pid)
            return lane_records.LaneResult(
                spec=spec,
                session_dir=res.session_dir,
                branch=res.branch,
                ok=False,
                error=(_stopped_by(overdue=overdue, over_budget=over_budget) or "interrupted")
                + "; lane "
                + ("was asked to stop and" if asked else "could not be asked to stop and")
                + f" did not end within the grace; its process {how}; not imported"
                " (its clone keeps its work)",
                pid=res.pid,
            )
    try:
        imported = parallel_results.import_lane(res, origin, origin_state, import_lock)
    except subrun.SubrunError as exc:
        return lane_records.LaneResult(
            spec=res.spec, session_dir=res.session_dir, branch=res.branch, ok=False, error=str(exc)
        )
    return land_lane(
        res,
        imported,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group=group,
        reporter=reporter,
        stopped_by=_stopped_by(overdue=overdue, over_budget=over_budget),
    )


def _stopped_by(*, overdue: bool, over_budget: bool) -> str:
    """Return what ended a lane the coordinator stopped, "" for a stop the operator asked."""
    if overdue:
        return "ended at its deadline"
    return "stopped at the run's budget" if over_budget else ""


def land_lane(
    res: lane_records.LaneResult,
    imported: lane_records.Imported,
    *,
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    group: str,
    reporter: app_reporter.Reporter,
    stopped_by: str = "",
) -> lane_records.LaneResult:
    """Finish an imported lane: carry its memory back, clean its clone, read its outcome.

    Args:
        res: The lane as it ran.
        imported: What the import landed.
        cfg: The origin's effective config.
        origin: The repository.
        origin_state: The origin's state dir.
        group: The group the lane belongs to.
        reporter: Receives the notes.
        stopped_by: What ended the lane when the coordinator stopped it, "" otherwise.

    Returns:
        The lane's result over the imported dir, its usage read from its record.
    """
    dest = imported.session_dir
    parallel_results.carry_back(
        paths.state_dir(res.spec.workdir),
        origin_state,
        dest,
        lane=res.spec.lane,
        reporter=reporter,
    )
    # Each lane removes its own dirs only; the group dir goes with whichever empties it last.
    workdirs.cleanup(
        [res.spec],
        origin=origin,
        note_keep=lambda path, why: reporter.note(f"kept {path}: {why}"),
        workdir_root=res.spec.workdir.parent,
        base=workdirs.workdir_base(cfg, origin),
        launch_dir=lane_launch.launch_dir(origin_state, group),
        cfg=cfg,
    )
    return recorded_result(
        res.spec, res.branch, dest, imported, origin=origin, stopped_by=stopped_by
    )


def recorded_result(
    spec: lane_records.LaneSpec,
    branch: str,
    dest: pathlib.Path,
    imported: lane_records.Imported,
    *,
    origin: pathlib.Path,
    stopped_by: str = "",
) -> lane_records.LaneResult:
    """Return a lane's result from its imported record: its outcome, its work, its usage.

    Args:
        spec: The lane.
        branch: Its branch.
        dest: Its imported run dir.
        imported: What the import landed.
        origin: The repository.
        stopped_by: What ended the lane when the coordinator stopped it, "" otherwise.

    Returns:
        `ok` only for a deliberate end; the usage is read either way.
    """
    summary = summarize_session_dir(dest)
    usage = lane_usage(dest)
    if not produced_result(summary.status):
        # Only a deliberate end joins the coordinator as a success.
        return lane_records.LaneResult(
            spec=spec,
            session_dir=dest,
            branch=branch,
            ok=False,
            error=(f"{stopped_by}: " if stopped_by else "")
            + f"no result ({format.status_label(summary.status, summary.detail)});"
            f" {parallel_results.imported_branch_note(origin, branch)}",
            imported_ref=imported.ref,
            imported_tip=imported.tip,
            usage=usage,
        )
    return lane_records.LaneResult(
        spec=spec,
        session_dir=dest,
        branch=branch,
        ok=True,
        error="",
        imported_ref=imported.ref,
        imported_tip=imported.tip,
        has_commits=lane_has_commits(dest, imported),
        usage=usage,
        report=fold_session(
            tail_events(dest / sessions_layout.LOGS_NAME, follow=False)
        ).finish_summary,
    )


def lane_usage(session_dir: pathlib.Path) -> agent6_budget.UsageSettlement | None:
    """Read what a lane used from its journal, as its parent settles it.

    Args:
        session_dir: The lane's imported run dir.

    Returns:
        The inclusive spend and per-provider plan observations; None when descendant
        records are incomplete or unreadable, which leaves the lane unsettled.
    """
    try:
        records = lane_dispatch.read_records(session_dir)
    except lane_dispatch.DispatchRecordError:
        return None
    if any(
        record.state != "complete" or any(not child.settled for child in record.children)
        for record in records
    ):
        return None
    # The journal includes settled descendants already. Never sum their journals again.
    scan = session_log.scan_session_log(session_dir / sessions_layout.LOGS_NAME)
    if any(
        child.session_id not in scan.budget.session.settled_lanes
        for record in records
        if record.source == "panel"
        for child in record.children
    ):
        return None
    readings: list[tuple[float, str, agent6_budget.PlanUsage]] = []
    for directory in (session_dir, *lineage.descendants_of(session_dir)):
        for raw in tail_events(directory / sessions_layout.LOGS_NAME, follow=False):
            if raw.get("type") != "budget.update":
                continue
            event = events.parse_event(raw)
            if not isinstance(event, journal.BudgetUpdate) or event.account is None:
                continue
            account = event.account
            provider = account.provider or parallel_results.lane_provider(directory)
            if not provider or not account.windows:
                continue
            # Credits are already in the inclusive USD total; only replay plan windows.
            plan = agent6_budget.PlanUsage(
                windows=tuple(
                    agent6_budget.PlanWindow(w.name, w.used_percent, w.window_minutes, w.resets_at)
                    for w in account.windows
                )
            )
            readings.append((events.event_epoch(raw.get("ts")) or 0.0, provider, plan))
    return agent6_budget.UsageSettlement(
        usd=scan.budget.session.usd,
        usd_partial=scan.budget.session.usd_partial,
        tokens=scan.budget.session.unmetered_tokens,
        plan_readings=tuple((p, r) for _, p, r in sorted(readings, key=lambda item: item[0])),
    )


def recover_lanes(
    group: str,
    children: Sequence[str],
    *,
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> list[lane_records.LaneResult]:
    """Return an interrupted group's lanes: the live stopped, the finished imported.

    A lane with no session dir under the origin never started. A live one is stopped as
    `agent6 stop` stops a run and waited for. One whose record is still only in its clone
    (the live symlink) is imported; one already imported is read as it stands.

    Args:
        group: The group id.
        children: The lanes' session ids, in lane order.
        cfg: The origin's effective config.
        origin: The repository.
        origin_state: The origin's state dir.
        reporter: Receives the notes.

    Returns:
        One result per lane that has a session dir, in lane order.
    """
    workdir_root = workdirs.subordinate_workdir_root(cfg, origin, group)
    results: list[lane_records.LaneResult] = []
    for lane, session_id in enumerate(children, start=1):
        matches = sessions_layout.session_matches(origin_state, session_id)
        if not matches:
            continue
        lane_dir = matches[0].session_dir
        mode = "read_only" if matches[0].subdir == "read_only" else "run"
        spec = lane_records.LaneSpec(
            lane, session_id, workdir_root / f"lane-{lane}", None, mode=mode
        )
        branch = git_read.run_branch_for(session_id)
        if session_is_live(lane_dir):
            outcome = stop.stop_session(lane_dir, wait_s=lane_watch.STOP_GRACE_S)
            reporter.note(f"lane {lane} [{session_id}]: {outcome.message}")
        if lane_dir.is_symlink():
            real = lane_dir.resolve()
            res = lane_records.LaneResult(spec, real, branch, True, "")
            try:
                imported = parallel_results.import_lane(res, origin, origin_state, None)
            except subrun.SubrunError as exc:
                results.append(dataclasses.replace(res, ok=False, error=str(exc)))
                continue
            results.append(
                land_lane(
                    res,
                    imported,
                    cfg=cfg,
                    origin=origin,
                    origin_state=origin_state,
                    group=group,
                    reporter=reporter,
                )
            )
            continue
        ref = branch if git_read.branch_exists(origin, branch) else ""
        tip = (git_read.branch_tip_sha(origin, branch) or "") if ref else ""
        imported = lane_records.Imported(lane_dir, ref, tip)
        results.append(recorded_result(spec, branch, lane_dir, imported, origin=origin))
    return results


def build_lane_recoverer(
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    *,
    mode: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> lane_records.LaneRecoverer | None:
    """Build the recoverer a resumed run's loop reconciles interrupted dispatches with.

    Args:
        cfg: The run's config.
        origin: The repository.
        origin_state: The origin's state dir.
        mode: The session mode; only a run dispatches.
        reporter: Receives the notes.

    Returns:
        The recoverer, or None where no dispatch could have happened.
    """
    if mode != "run":
        return None

    def recover(group: str, children: Sequence[str]) -> list[lane_records.LaneResult]:
        return recover_lanes(
            group, children, cfg=cfg, origin=origin, origin_state=origin_state, reporter=reporter
        )

    return recover


def lane_has_commits(session_dir: pathlib.Path, imported: lane_records.Imported) -> bool:
    """Return whether the imported tip holds a commit beyond the lane's recorded base.

    Args:
        session_dir: The imported lane's run dir, whose manifest records its base.
        imported: What the import landed.

    Returns:
        False for a record-only import or a tip at the base; True otherwise, and when the
        base is unrecorded, so an unknown lane is merged rather than dropped.
    """
    if not imported.tip:
        return False
    try:
        base = sessions_manifest.read_manifest(session_dir).base_sha
    except sessions_manifest.ManifestError:
        return True
    return imported.tip != base


def build_lane_spawner(
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    *,
    explicit_leaves: frozenset[str],
    coordinator_session_id: str,
    coordinator_mode: str = "run",
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    watch: Callable[[pathlib.Path, str, str], bool] | None = None,
) -> lane_records.GroupLaneSpawner:
    """Build the group dispatcher the loop calls at a `/parallel` boundary.

    One call runs every lane on its own thread, imports each under a shared lock
    and returns the results in dispatch order. Lane ids are
    `<coordinator>-<group>-l<i>`.

    Args:
        cfg: The coordinator's config.
        origin: The repository.
        origin_state: The origin's state dir.
        explicit_leaves: The dotted leaves the coordinator's config layers set.
        coordinator_session_id: The coordinator's session id.
        coordinator_mode: The coordinator's session bucket owner.
        runtime: The front-end's primitives.
        max_usd: Each lane's spend cap.
        auto_approve: The coordinator's own `--auto-approve`, forwarded to every lane.
        lane_away: Every lane's away-mode, without a watch.
        reporter: Receives the notes.
        watch: The coordinator's side of each lane's wait (`run_lane_to_completion`); with it
            every lane waits for the coordinator's operator to answer its prompts.

    Returns:
        The dispatcher.
    """
    if watch is not None:
        lane_away = "wait"

    def dispatch(
        lanes: list[lane_records.LaneTask], group: str, *, at: str | None = None
    ) -> list[lane_records.LaneResult]:
        # The lane cap binds before any clone; the refusal reaches the coordinator as feedback.
        if len(lanes) > cfg.parallel.max_lanes:
            raise ParallelError(
                f"/parallel requests {len(lanes)} lanes but [parallel].max_lanes ="
                f" {cfg.parallel.max_lanes}. Request fewer, or raise [parallel].max_lanes."
            )
        # The routes are validated before any clone; no model cache warns and proceeds.
        try:
            settings = [lane.settings or cfg.lane_settings(lane.model) for lane in lanes]
        except ConfigError as exc:
            raise ParallelError(str(exc)) from exc
        verdict = validate.validate_spec_models(
            [s.route if s is not None else None for s in settings], cfg
        )
        if verdict.refused:
            raise ParallelError(validate.refusal_message(verdict, directive=True))
        if verdict.warned:
            reporter.warn(validate.warning_message(verdict))
        # One name for the workdir, the lane ids and the stamped lineage: the sweeps derive
        # the clone path from it.
        group_id = f"{coordinator_session_id}-{group}"
        workdir_root = workdirs.subordinate_workdir_root(cfg, origin, group_id)
        specs = [
            lane_records.LaneSpec(
                lane=i,
                session_id=f"{group_id}-l{i}",
                workdir=workdir_root / f"lane-{i}",
                settings=chosen,
                mode="read_only" if lane.contract is not None else "run",
            )
            for i, (chosen, lane) in enumerate(zip(settings, lanes, strict=True), start=1)
        ]
        paths.mkdir_for_real_user(sessions_layout.bucket_dir(origin_state, "runs"))
        import_lock = threading.Lock()
        coord_dir = (
            sessions_layout.bucket_dir(origin_state, kinds.session_bucket(coordinator_mode))
            / coordinator_session_id
        )
        hard_stop = threading.Event()

        def should_stop() -> bool:
            # Both stop channels interrupt the wait; the loop consumes their markers after.
            return (
                hard_stop.is_set()
                or ipc.steer_answer_is_abort(coord_dir)
                or ipc.stop_request_pending(coord_dir)
            )

        def stop_is_now() -> bool:
            return (
                hard_stop.is_set()
                or ipc.steer_answer_is_abort(coord_dir)
                or ipc.stop_request_is_now(coord_dir)
            )

        # The coordinator's manifest pins the gate the lanes' config carries.
        gate_source = ""
        if cfg.harness.verify_command:
            try:
                gate_source = sessions_manifest.read_manifest(coord_dir).harness.gate.source
            except sessions_manifest.ManifestError:
                gate_source = ""

        def one(
            pair: tuple[lane_records.LaneSpec, lane_records.LaneTask],
        ) -> lane_records.LaneResult:
            spec, lane = pair
            return run_lane_to_completion(
                spec,
                lane.task,
                pins=lane.pins,
                at=at,
                cfg=cfg,
                gate_source=gate_source,
                explicit_leaves=explicit_leaves,
                origin=origin,
                origin_state=origin_state,
                group=group_id,
                coordinator=coordinator_session_id,
                runtime=runtime,
                max_usd=max_usd,
                auto_approve=auto_approve,
                lane_away=lane_away,
                import_lock=import_lock,
                reporter=reporter,
                should_stop=should_stop,
                stop_is_now=stop_is_now,
                hard_stop=hard_stop,
                limits=lane.limits,
                allowance=lane.allowance,
                contract=lane.contract,
                watch=watch,
            )

        pairs = list(zip(specs, lanes, strict=True))
        if len(pairs) > 1:
            # Not a with-block: its exit would join every lane await on an interrupt.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs))
            try:
                futures = [pool.submit(one, p) for p in pairs]
                # A lane thread that raises (a bug, not a lane failure) aborts the group now.
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_EXCEPTION
                )
                for f in done:
                    exc = f.exception()
                    if exc is not None:
                        raise exc
                results = [f.result() for f in futures]  # submit order = lane order
            except BaseException:
                # The lane threads request a stop and exit; the lanes keep running detached.
                hard_stop.set()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
            return results
        return [one(p) for p in pairs]

    return dispatch


def build_coordinator_spawner(
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    *,
    explicit_leaves: frozenset[str],
    mode: str,
    session_id: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    watch: Callable[[pathlib.Path, str, str], bool] | None = None,
) -> lane_records.GroupLaneSpawner | None:
    """Build the `/parallel` dispatcher for a run's loop, or None when dispatch is unavailable.

    A subordinate worker retains the executor only for its configured harness panels.

    Args:
        cfg: The run's config.
        origin: The repository.
        origin_state: The origin's state dir.
        explicit_leaves: The dotted leaves the run's config layers set.
        mode: The session mode.
        session_id: The run's session id.
        runtime: The front-end's primitives.
        max_usd: Each lane's spend cap.
        auto_approve: The run's own `--auto-approve`, forwarded to every lane.
        lane_away: Every lane's away-mode, without a watch.
        reporter: Receives the notes.
        watch: The run's side of each lane's wait: its prompts and its spend.

    Returns:
        The dispatcher, or None.
    """
    authority = lane_launch.dispatch_authority(cfg, mode=mode, session_id=session_id, root=origin)
    if not (authority.work or authority.panels):
        return None
    return build_lane_spawner(
        cfg,
        origin,
        origin_state,
        explicit_leaves=explicit_leaves,
        coordinator_session_id=session_id,
        coordinator_mode=mode,
        runtime=runtime,
        max_usd=max_usd,
        auto_approve=auto_approve,
        lane_away=lane_away,
        reporter=reporter,
        watch=watch,
    )
