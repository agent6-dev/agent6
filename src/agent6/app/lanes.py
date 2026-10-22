# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The lanes of a fan-out: the front-end runtime, planning, spawning, running and dispatch."""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
import threading
from collections.abc import Callable, Sequence
from concurrent import futures as concurrent_futures
from typing import Protocol

from agent6 import directive, errors, kinds, memory, paths
from agent6.app import _lane_watch, _parallel_results, _workdirs
from agent6.app import compare as app_compare
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError
from agent6.harness import subrun
from agent6.models import validate
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_ops
from agent6.viewmodel import format, produced_result, summarize_session_dir


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
    ) -> tuple[pathlib.Path | None, str]:
        """Spawn the command and wait for its session dir.

        Args:
            argv: The agent6 subcommand and flags, without the executable.
            cwd: Where to spawn.
            before: The session dirs that existed before the spawn.
            list_dirs: Lists the candidate session dirs.
            env: The child's environment.

        Returns:
            `(session_dir, "")` once a new dir with a log appears, else `(None, error)`.
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
) -> list[subrun.LaneSpec]:
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
        workdir_root = _workdirs.subordinate_workdir_root(cfg, origin, fanout_id)
    routes = lane_routes(cfg, directive.parse_spec(spec, limit=cfg.parallel.max_lanes))
    return [
        subrun.LaneSpec(
            lane=i,
            session_id=f"{fanout_id}-l{i}",
            workdir=workdir_root / f"lane-{i}",
            route=route,
        )
        for i, route in enumerate(routes, start=1)
    ]


def lane_routes(cfg: Config, models: Sequence[str | None]) -> list[kinds.ModelRoute | None]:
    """Resolve each lane's `[provider/]model` text against the worker route.

    Args:
        cfg: The run's config.
        models: One entry per lane; None keeps the worker's own route.

    Returns:
        The routes in lane order.

    Raises:
        ConfigError: A value names nothing.
    """
    return [cfg.model_route("worker", m) if m else None for m in models]


def bridge_spawner(
    spec: subrun.LaneSpec,
    task: str,
    *,
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
    verify_origin: str = "",
    runtime: LaneRuntime,
) -> subrun.LaneResult:
    """Clone the origin and spawn a detached `agent6 run` in the clone.

    Args:
        spec: The lane.
        task: The lane's task.
        pins: The coordinator's pins, passed out of band of the task.
        cfg: The origin's effective config.
        explicit_leaves: The dotted leaves the origin's config layers set.
        origin: The repository.
        max_usd: The lane's spend cap, forwarded to its argv.
        auto_approve: The fan-out's own `--auto-approve`, forwarded to its argv.
        lane_away: The lane's away-mode: "wait" parks a question for `agent6 attach`,
            "deny" answers it empty.
        at: A commit to detach the clone at; the coordinator's chain tip.
        fanout_id: The group the lane belongs to.
        coordinator: The session every listing nests the lane under.
        verify_origin: Where the gate in the lane's config came from, "" when it has none.
        runtime: The front-end's spawn primitive.

    Returns:
        The lane's result once its session dir is located, the run still going;
        `ok=False` when the clone or spawn failed.
    """
    branch = git_ops.run_branch_for(spec.session_id)
    try:
        subrun.clone_workspace(origin, spec.workdir)
        if at is not None:
            # A local clone hardlinks the whole odb, so the chain's commits are present.
            git_ops.checkout_detached(spec.workdir, at)
    except (subrun.SubrunError, errors.GitError) as exc:
        return subrun.LaneResult(
            spec=spec, session_dir=spec.workdir, branch=branch, ok=False, error=str(exc)
        )
    config_path = _workdirs.write_lane_config(cfg, spec, explicit_leaves=explicit_leaves)
    lane_state = paths.state_dir(spec.workdir)
    # The clone's state dir is empty; the origin's memory and rulings are seeded into it.
    memory.seed_store(paths.state_dir(origin), lane_state)
    lane_runs = sessions_layout.bucket_dir(lane_state, "runs")

    def list_dirs() -> list[pathlib.Path]:
        if not lane_runs.is_dir():
            return []
        return [p for p in lane_runs.iterdir() if p.is_dir()]

    argv = [
        "run",
        "--session-id",
        spec.session_id,
        "--config",
        str(config_path),
    ]
    if max_usd is not None:
        argv += ["--max-usd", f"{max_usd:g}"]
    if auto_approve:
        argv += ["--auto-approve"]
    for pin in pins:
        # Pins ride out of band, so the lane's manifest task stays the task.
        argv += ["--pin", pin]
    # `--` so a task that looks like a flag is never parsed as one.
    argv += ["--", task]
    # AGENT6_SUBRUN keeps a lane from fanning out itself: depth 1 by construction.
    markers = {
        "AGENT6_STREAM_TO_LOG": "1",
        "AGENT6_DETACHED_AWAY": lane_away,
        "AGENT6_SUBRUN": "1",
        # The lane's own manifest records its lineage, so a coordinator death orphans nothing.
        "AGENT6_PARALLEL_LINEAGE": f"{coordinator}:{fanout_id}:{spec.lane}",
        # Resolved in the origin, where the repo signals are; the lane stamps the same origin.
        "AGENT6_LANE_VERIFY_ORIGIN": verify_origin,
    }
    session_dir, err = runtime.spawn(
        argv, spec.workdir, before=set(), list_dirs=list_dirs, env={**os.environ, **markers}
    )
    if session_dir is None:
        return subrun.LaneResult(
            spec=spec, session_dir=lane_runs / spec.session_id, branch=branch, ok=False, error=err
        )
    return subrun.LaneResult(spec=spec, session_dir=session_dir, branch=branch, ok=True, error="")


def run_lane_to_completion(
    spec: subrun.LaneSpec,
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
    verify_origin: str = "",
    spawner: subrun.LaneSpawner | None = None,
    at: str | None = None,
    import_lock: threading.Lock | None = None,
    poll_interval_s: float = _lane_watch.POLL_INTERVAL_S,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    should_stop: Callable[[], bool] | None = None,
    hard_stop: threading.Event | None = None,
) -> subrun.LaneResult:
    """Run one subordinate lane to its end and import it into the origin.

    The coordinator runs a group of these on a thread pool, so each is
    self-contained. A lane that failed to start, was still running at teardown,
    refused its import, or ended without a result comes back `ok=False`; an
    imported branch is safe in the origin either way.

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
        verify_origin: Where the gate in the config came from, "" when it has none.
        spawner: Clones and spawns the lane; the bridge spawner by default.
        at: A commit to detach the clone at.
        import_lock: Serializes the import across the group, since concurrent fetches
            into one repo race.
        poll_interval_s: How often the await polls.
        reporter: Receives the notes.
        should_stop: Polled to interrupt the await; the lane then gets a stop request
            and a bounded grace.
        hard_stop: Set to skip the grace.

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
            verify_origin=verify_origin,
            runtime=runtime,
        )
    res = spawner(spec, task)
    if not res.ok:
        return res
    # The live symlink lets a hub answer the lane's prompts while it runs.
    _lane_watch.symlink_lane(origin_state, res)
    if not _lane_watch.await_lane(res, poll_interval_s=poll_interval_s, should_stop=should_stop):
        asked = ipc.request_stop(res.session_dir, now=True)
        if not _lane_watch.drain_lane(res, poll_interval_s=poll_interval_s, hard_stop=hard_stop):
            # Still running: the clone and symlink hold the only copy of its branch.
            return subrun.LaneResult(
                spec=spec,
                session_dir=res.session_dir,
                branch=res.branch,
                ok=False,
                error="interrupted; lane "
                + ("was asked to stop and" if asked else "could not be asked to stop and")
                + " keeps running detached; not imported",
            )
    try:
        dest = _parallel_results.import_lane(res, origin, origin_state, import_lock)
    except subrun.SubrunError as exc:
        return subrun.LaneResult(
            spec=res.spec, session_dir=res.session_dir, branch=res.branch, ok=False, error=str(exc)
        )
    _parallel_results.carry_back(
        paths.state_dir(res.spec.workdir),
        origin_state,
        dest,
        lane=res.spec.lane,
        reporter=reporter,
    )
    # Each lane removes its own dirs only; the group dir goes with whichever empties it last.
    _workdirs.cleanup(
        [res.spec],
        workdir_root=res.spec.workdir.parent,
        base=_workdirs.workdir_base(cfg, origin),
        cfg=cfg,
    )
    summary = summarize_session_dir(dest)
    if not produced_result(summary.status):
        # Only a deliberate end joins the coordinator as a success.
        return subrun.LaneResult(
            spec=spec,
            session_dir=dest,
            branch=res.branch,
            ok=False,
            error=f"no result ({format.status_label(summary.status, summary.reason)});"
            f" {_parallel_results.imported_branch_note(origin, res.branch)}",
        )
    return subrun.LaneResult(spec=spec, session_dir=dest, branch=res.branch, ok=True, error="")


def build_lane_spawner(
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    *,
    explicit_leaves: frozenset[str],
    coordinator_session_id: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> subrun.GroupLaneSpawner:
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
        runtime: The front-end's primitives.
        max_usd: Each lane's spend cap.
        auto_approve: The coordinator's own `--auto-approve`, forwarded to every lane.
        lane_away: Every lane's away-mode.
        reporter: Receives the notes.

    Returns:
        The dispatcher.
    """

    def dispatch(
        lanes: list[subrun.LaneTask], group: str, *, at: str | None = None
    ) -> list[subrun.LaneResult]:
        # The lane cap binds before any clone; the refusal reaches the coordinator as feedback.
        if len(lanes) > cfg.parallel.max_lanes:
            raise ParallelError(
                f"/parallel requests {len(lanes)} lanes but [parallel].max_lanes ="
                f" {cfg.parallel.max_lanes}. Request fewer, or raise [parallel].max_lanes."
            )
        # The routes are validated before any clone; no model cache warns and proceeds.
        try:
            routes = lane_routes(cfg, [lane.model for lane in lanes])
        except ConfigError as exc:
            raise ParallelError(str(exc)) from exc
        verdict = validate.validate_spec_models(routes, cfg)
        if verdict.refused:
            raise ParallelError(validate.refusal_message(verdict, directive=True))
        if verdict.warned:
            reporter.warn(validate.warning_message(verdict))
        # One name for the workdir, the lane ids and the stamped lineage: the sweeps derive
        # the clone path from it.
        group_id = f"{coordinator_session_id}-{group}"
        workdir_root = _workdirs.subordinate_workdir_root(cfg, origin, group_id)
        specs = [
            subrun.LaneSpec(
                lane=i,
                session_id=f"{group_id}-l{i}",
                workdir=workdir_root / f"lane-{i}",
                route=route,
            )
            for i, route in enumerate(routes, start=1)
        ]
        paths.mkdir_for_real_user(sessions_layout.bucket_dir(origin_state, "runs"))
        import_lock = threading.Lock()
        coord_dir = sessions_layout.bucket_dir(origin_state, "runs") / coordinator_session_id
        hard_stop = threading.Event()

        def should_stop() -> bool:
            # Both stop channels interrupt the wait; the loop consumes their markers after.
            return (
                hard_stop.is_set()
                or ipc.steer_answer_is_abort(coord_dir)
                or ipc.stop_request_pending(coord_dir)
            )

        # The coordinator's manifest pins the gate the lanes' config carries.
        verify_origin = ""
        if cfg.harness.verify_command:
            try:
                verify_origin = sessions_manifest.read_manifest(coord_dir).harness.verify_origin
            except sessions_manifest.ManifestError:
                verify_origin = ""

        def one(pair: tuple[subrun.LaneSpec, subrun.LaneTask]) -> subrun.LaneResult:
            spec, lane = pair
            return run_lane_to_completion(
                spec,
                lane.task,
                pins=lane.pins,
                at=at,
                cfg=cfg,
                verify_origin=verify_origin,
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
                hard_stop=hard_stop,
            )

        pairs = list(zip(specs, lanes, strict=True))
        if len(pairs) > 1:
            # Not a with-block: its exit would join every lane await on an interrupt.
            pool = concurrent_futures.ThreadPoolExecutor(max_workers=len(pairs))
            try:
                futures = [pool.submit(one, p) for p in pairs]
                # A lane thread that raises (a bug, not a lane failure) aborts the group now.
                done, _ = concurrent_futures.wait(
                    futures, return_when=concurrent_futures.FIRST_EXCEPTION
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
) -> subrun.GroupLaneSpawner | None:
    """Build the `/parallel` dispatcher for a run's loop, or None when dispatch is unavailable.

    A non-write mode and a run inside a subordinate lane get none.

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
        lane_away: Every lane's away-mode.
        reporter: Receives the notes.

    Returns:
        The dispatcher, or None.
    """
    if mode != "run" or os.environ.get("AGENT6_SUBRUN"):
        return None
    return build_lane_spawner(
        cfg,
        origin,
        origin_state,
        explicit_leaves=explicit_leaves,
        coordinator_session_id=session_id,
        runtime=runtime,
        max_usd=max_usd,
        auto_approve=auto_approve,
        lane_away=lane_away,
        reporter=reporter,
    )
