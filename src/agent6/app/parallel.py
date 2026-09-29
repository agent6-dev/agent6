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
import dataclasses
import functools
import os
import pathlib
import shutil
import threading
from collections.abc import Callable, Sequence
from concurrent import futures as concurrent_futures
from typing import Protocol

from agent6 import directive, event_log, git_ops, kinds, memory, paths
from agent6.app import _lane_watch, finalize
from agent6.app import compare as app_compare
from agent6.app import manifest as app_manifest
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError, layer
from agent6.harness import judge, subrun
from agent6.models import validate
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
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


# ---------------------------------------------------------------------------
# Lane planning
# ---------------------------------------------------------------------------


def subordinate_workdir_root(cfg: Config, origin: pathlib.Path, group: str) -> pathlib.Path:
    """Return the base dir for a group of subordinate working trees.

    Fan-out ids, `machine-<id>` and fork ids are all groups, so one prune sweep
    covers them; the dir is scoped by repo id, so another repo's clones never enter
    a sweep.

    Args:
        cfg: The run's config.
        origin: The repository.
        group: The group's name.

    Returns:
        `<workdir base>/<repo id>/<group>`.
    """
    return workdir_base(cfg, origin) / group


def workdir_base(cfg: Config, origin: pathlib.Path) -> pathlib.Path:
    """Return the per-repo dir every subordinate working tree of the origin sits in.

    Args:
        cfg: The run's config.
        origin: The repository.

    Returns:
        `[parallel].workdir`, or the cache's `parallel` dir, plus the repo id.
    """
    base = (
        pathlib.Path(cfg.parallel.workdir)
        if cfg.parallel.workdir
        else paths.cache_dir() / "parallel"
    )
    return base / paths.repo_id(origin)


def adopt_orphan_lane(
    origin: pathlib.Path,
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
) -> str | None:
    """Import an orphaned fan-out lane so an ordinary merge can land it.

    A coordinator death leaves a finished lane's branch only in its clone, with
    the origin still holding the live-view symlink.

    Args:
        origin: The repository.
        cfg: The run's config.
        layout: The lane's layout in the origin.
        manifest: The lane's manifest.

    Returns:
        The note to print, or None when the session is not an orphaned lane.

    Raises:
        SubrunError: The import failed; the symlink is restored.
    """
    if (
        not manifest.run_branch
        or manifest.parallel is None
        or git_ops.branch_exists(origin, manifest.run_branch)
        or not layout.session_dir.is_symlink()
    ):
        return None
    lineage = manifest.parallel
    clone = subordinate_workdir_root(cfg, origin, lineage.group) / f"lane-{lineage.lane}"
    if not (clone / ".git").exists() or not git_ops.branch_exists(clone, manifest.run_branch):
        return None
    real = layout.session_dir.resolve()
    layout.session_dir.unlink()
    try:
        subrun.import_run(origin, clone, manifest.run_branch, real, layout.state_dir)
    except subrun.SubrunError:
        layout.session_dir.symlink_to(real)
        raise
    return f"imported orphaned lane branch {manifest.run_branch} from {clone}"


def _recorded_merged(origin: pathlib.Path, clone: pathlib.Path, tip: str) -> bool:
    """Tell whether a session's manifest records the tip as merged into a live branch.

    A squash-merged lane's commits are unreachable by design, so this is the second
    proof that its clone may go.

    Args:
        origin: The repository.
        clone: The lane's clone.
        tip: The commit to prove.

    Returns:
        Whether a merge stamp covers it.
    """
    state = paths.state_dir(origin)
    for branch in git_ops.list_run_branches(clone):
        layout = sessions_layout.session_layout(state, branch.removeprefix("agent6/"))
        if layout is None:
            continue
        try:
            merged = sessions_manifest.read_manifest(layout.session_dir).merged
        except sessions_manifest.ManifestError:
            continue
        if merged is not None and merged.tip == tip and git_ops.branch_exists(origin, merged.into):
            return True
    return False


def _lane_may_run(clone: pathlib.Path) -> bool:
    """Tell whether a clone still belongs to a lane: a live worker, or no session dir yet.

    Args:
        clone: The lane's clone.

    Returns:
        Whether the lane may still be running.
    """
    state = paths.state_dir(clone)
    dirs = [
        d
        for bucket in sessions_layout.HUB_BUCKETS
        for d in sessions_layout.bucket_dir(state, bucket).glob("*")
        if d.is_dir()
    ]
    return not dirs or any(ipc.worker_is_alive(d) for d in dirs)


def sweep_fanout_clones(origin: pathlib.Path, cfg: Config) -> tuple[int, int]:
    """Delete fan-out clone dirs whose every tip the origin already reaches.

    A clone holding a commit the origin lacks, or a live lane, keeps its whole
    group dir: it may be the only copy. Only a dir holding `lane-*` clones is a
    group; a fork's worktree or an operator's dir is left alone.

    Args:
        origin: The repository.
        cfg: The run's config.

    Returns:
        `(swept, kept)` group counts.
    """
    base = workdir_base(cfg, origin)
    if not base.is_dir():
        return 0, 0
    swept = kept = 0
    for fanout in sorted(p for p in base.iterdir() if p.is_dir()):
        clones = [c for c in sorted(fanout.glob("lane-*")) if (c / ".git").is_dir()]
        if not clones:
            continue
        safe = True
        for clone in clones:
            try:
                tips = [git_ops.chain_tip(clone, br) for br in git_ops.list_run_branches(clone)]
                # A machine-state clone's work rides its chain ref, not a branch.
                tips += [sha for _ref, sha in git_ops.list_chain_refs(clone)]
            except git_ops.GitError:
                safe = False
                break
            if not tips and _lane_may_run(clone):
                # A starting lane's work is its working tree alone.
                safe = False
                break
            # Reachability, not existence: a loose object is one `git gc` from gone.
            if any(
                tip is not None
                and not git_ops.commit_is_reachable(origin, tip)
                and not _recorded_merged(origin, clone, tip)
                for tip in tips
            ):
                safe = False
                break
        if safe:
            shutil.rmtree(fanout, ignore_errors=True)
            swept += 1
        else:
            kept += 1
    return swept, kept


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
        workdir_root = subordinate_workdir_root(cfg, origin, fanout_id)
    routes = _lane_routes(cfg, directive.parse_spec(spec, limit=cfg.parallel.max_lanes))
    return [
        subrun.LaneSpec(
            lane=i,
            session_id=f"{fanout_id}-l{i}",
            workdir=workdir_root / f"lane-{i}",
            route=route,
        )
        for i, route in enumerate(routes, start=1)
    ]


def _lane_routes(cfg: Config, models: Sequence[str | None]) -> list[kinds.ModelRoute | None]:
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


# ---------------------------------------------------------------------------
# The real (bridge) spawner: clone, write a lane config, spawn detached, locate
# ---------------------------------------------------------------------------


def _write_lane_config(cfg: Config, spec: subrun.LaneSpec) -> pathlib.Path:
    """Write the lane's config file: the origin's effective config, the worker re-routed.

    The clone's own per-repo config is empty, so the origin's settings ride in the
    file the lane loads with `--config`.

    Args:
        cfg: The origin's effective config.
        spec: The lane.

    Returns:
        The config file's path.
    """
    lane_cfg = cfg.with_model_route("worker", spec.route) if spec.route else cfg
    # The import fetches the lane's branch, so branch_per_run stays on whatever the origin says.
    lane_cfg = lane_cfg.model_copy(
        update={"git": lane_cfg.git.model_copy(update={"branch_per_run": True})}
    )
    config_path = spec.workdir.parent / f"lane-{spec.lane}-config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(layer.materialize(lane_cfg), encoding="utf-8")
    return config_path


def bridge_spawner(
    spec: subrun.LaneSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    origin: pathlib.Path,
    max_usd: float | None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
    at: str | None = None,
    fanout_id: str,
    coordinator: str,
    runtime: LaneRuntime,
) -> subrun.LaneResult:
    """Clone the origin and spawn a detached `agent6 run` in the clone.

    Args:
        spec: The lane.
        task: The lane's task.
        pins: The coordinator's pins, passed out of band of the task.
        cfg: The origin's effective config.
        origin: The repository.
        max_usd: The lane's spend cap, forwarded to its argv.
        auto_approve: The fan-out's own `--auto-approve`, forwarded to its argv.
        lane_away: The lane's away-mode: "wait" parks a question for `agent6 attach`,
            "deny" answers it empty.
        at: A commit to detach the clone at; the coordinator's chain tip.
        fanout_id: The group the lane belongs to.
        coordinator: The session every listing nests the lane under.
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
    except (subrun.SubrunError, git_ops.GitError) as exc:
        return subrun.LaneResult(
            spec=spec, session_dir=spec.workdir, branch=branch, ok=False, error=str(exc)
        )
    config_path = _write_lane_config(cfg, spec)
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
    }
    session_dir, err = runtime.spawn(
        argv, spec.workdir, before=set(), list_dirs=list_dirs, env={**os.environ, **markers}
    )
    if session_dir is None:
        return subrun.LaneResult(
            spec=spec, session_dir=lane_runs / spec.session_id, branch=branch, ok=False, error=err
        )
    return subrun.LaneResult(spec=spec, session_dir=session_dir, branch=branch, ok=True, error="")


# ---------------------------------------------------------------------------
# Coordinator dispatch: one lane to completion + a group spawner for the loop
# ---------------------------------------------------------------------------


def run_lane_to_completion(
    spec: subrun.LaneSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    group: str,
    coordinator: str,
    runtime: LaneRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    lane_away: ipc.AwayMode = "wait",
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
        origin: The repository.
        origin_state: The origin's state dir.
        group: The group the lane belongs to.
        coordinator: The session every listing nests the lane under.
        runtime: The front-end's primitives.
        max_usd: The lane's spend cap.
        auto_approve: The coordinator's own `--auto-approve`, forwarded.
        lane_away: The lane's away-mode.
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
            fanout_id=group,
            coordinator=coordinator,
            runtime=runtime,
        )
    res = spawner(spec, task)
    if not res.ok:
        return res
    # The live symlink lets a hub answer the lane's prompts while it runs.
    _lane_watch.symlink_lane(origin_state, res)
    if not _lane_watch.await_lane(res, poll_interval_s=poll_interval_s, should_stop=should_stop):
        asked = ipc.request_stop(res.session_dir)
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
        dest = _import_lane(res, origin, origin_state, import_lock)
    except subrun.SubrunError as exc:
        return subrun.LaneResult(
            spec=res.spec, session_dir=res.session_dir, branch=res.branch, ok=False, error=str(exc)
        )
    carry_back(
        paths.state_dir(res.spec.workdir),
        origin_state,
        dest,
        lane=res.spec.lane,
        reporter=reporter,
    )
    # Each lane removes its own dirs only; the group dir goes with whichever empties it last.
    _cleanup(
        [res.spec], workdir_root=res.spec.workdir.parent, base=workdir_base(cfg, origin), cfg=cfg
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
            f" branch imported as {res.branch}",
        )
    return subrun.LaneResult(spec=spec, session_dir=dest, branch=res.branch, ok=True, error="")


def build_lane_spawner(
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    *,
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
            routes = _lane_routes(cfg, [lane.model for lane in lanes])
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
        workdir_root = subordinate_workdir_root(cfg, origin, group_id)
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

        def one(pair: tuple[subrun.LaneSpec, subrun.LaneTask]) -> subrun.LaneResult:
            spec, lane = pair
            return run_lane_to_completion(
                spec,
                lane.task,
                pins=lane.pins,
                at=at,
                cfg=cfg,
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
        coordinator_session_id=session_id,
        runtime=runtime,
        max_usd=max_usd,
        auto_approve=auto_approve,
        lane_away=lane_away,
        reporter=reporter,
    )


# ---------------------------------------------------------------------------
# Import + auto-compare
# ---------------------------------------------------------------------------


def _stamp(session_dir: pathlib.Path, **updates: object) -> str | None:
    """Apply field updates to an imported lane's manifest.

    Args:
        session_dir: The lane's imported dir.
        **updates: The manifest fields to set.

    Returns:
        The error when the manifest cannot be read or written, else None; the import
        stands either way.
    """
    mpath = session_dir / sessions_manifest.MANIFEST_NAME
    try:
        m = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError as exc:
        return f"could not read {mpath}: {exc}"
    try:
        app_manifest.write_manifest(mpath, m.model_copy(update=updates))
    except (OSError, sessions_manifest.ManifestError) as exc:
        # The import stands; the remaining lanes keep importing.
        return f"could not write {mpath}: {exc}"
    return None


def _stamp_compare_outcomes(
    candidates: list[judge.CandidateBrief],
    outcome: app_compare.RankOutcome,
    *,
    origin_state: pathlib.Path,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Stamp the auto-compare outcome into each ranked lane's manifest.

    The rationale and judge cost describe the whole group's ranking and land on
    every lane. Rank 1 is the winner unless every gate ran red, and, when any lane
    verified green, only if it is one: the judge's ranking is untrusted.

    Args:
        candidates: The ranked lanes.
        outcome: The ranking.
        origin_state: The origin's state dir.
        reporter: Receives a note per stamp failure; the others still land.
    """
    of = len(candidates)
    text = outcome.rationale[:2000] if outcome.ranked_by == "judge" else ""
    crown = fanout_exit_code(candidates) != finalize.EXIT_VERIFY_FAILED
    any_verified = any(c.verify_ok is True for c in candidates)
    by_id = {c.session_id: c for c in candidates}
    for rank_pos, session_id in enumerate(outcome.ranking, start=1):
        winner = (
            rank_pos == 1 and crown and (not any_verified or by_id[session_id].verify_ok is True)
        )
        compare = sessions_manifest.CompareStamp(
            rank=rank_pos,
            of=of,
            winner=winner,
            ranked_by=outcome.ranked_by,
            rationale=text,
            judge_cost_usd=outcome.judge_cost_usd,
            judge_cost_partial=outcome.judge_cost_partial,
        )
        err = _stamp(_lane_watch.lane_link(origin_state, session_id), compare=compare)
        if err is not None:
            reporter.note(f"lane [{session_id}]: imported, but the compare stamp failed: {err}")


def carry_back(
    lane_state: pathlib.Path,
    origin_state: pathlib.Path,
    dest: pathlib.Path,
    *,
    lane: int,
    reporter: app_reporter.Reporter,
) -> None:
    """Carry a lane's recorded decisions and memory into the origin's store.

    The lane's state dir is torn down after import, so this runs first; files that
    changed in both stores are held back under the imported dir.

    Args:
        lane_state: The lane's state dir.
        origin_state: The origin's state dir.
        dest: The lane's imported session dir.
        lane: The lane number the note names.
        reporter: Receives the note.
    """
    carried, known = memory.merge_decisions(lane_state, origin_state)
    if carried or known:
        already = f", {known} already recorded" if known else ""
        reporter.note(f"lane {lane}: {carried} recorded decision(s) carried over{already}")
    held_dir = dest / "memory-held"
    merge = memory.merge_memory(lane_state, origin_state, held_dir=held_dir)
    used, read = memory.merge_use(
        lane_state, origin_state, written=(*merge.carried, *merge.updated)
    )
    parts = [
        f"{len(names)} {word}"
        for word, names in (
            ("carried", merge.carried),
            ("updated", merge.updated),
            ("deleted", merge.deleted),
        )
        if names
    ]
    if merge.held:
        kept = f", kept at {held_dir}" if held_dir.is_dir() else ""
        parts.append(
            f"held back (changed here too, or the name is taken): {', '.join(merge.held)}{kept}"
        )
    if used or read:
        parts.append(f"use record carried for {max(used, read)}")
    if parts:
        reporter.note(f"lane {lane}: memory {', '.join(parts)}")


def _import_lane(
    res: subrun.LaneResult,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    lock: threading.Lock | None,
) -> pathlib.Path:
    """Land a finished lane's session dir in the origin.

    The live symlink drops so the real dir can take its place, and comes back
    when the import refuses.

    Args:
        res: The lane's result.
        origin: The repository.
        origin_state: The origin's state dir.
        lock: Serializes the import across a group, when one is shared.

    Returns:
        The imported dir.

    Raises:
        SubrunError: The import failed; nothing moved.
    """
    link = _lane_watch.lane_link(origin_state, res.spec.session_id)
    had_link = link.is_symlink()
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    try:
        with lock if lock is not None else contextlib.nullcontext():
            return subrun.import_run(
                origin, res.spec.workdir, res.branch, res.session_dir, origin_state
            )
    except subrun.SubrunError:
        if had_link:
            _lane_watch.symlink_lane(origin_state, res)
        raise


def _import_lanes(
    results: list[subrun.LaneResult],
    *,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    base_sha: str,
    task: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> tuple[list[judge.CandidateBrief], list[tuple[subrun.LaneResult, str]], list[subrun.LaneSpec]]:
    """Import each finished lane into the origin and build the candidate briefs.

    A lane that failed to start, is still running or refused its import is recorded
    as failed and keeps its clone, state and symlink; an imported lane without a
    result is recorded as failed too, its work safe in the origin.

    Args:
        results: The spawned lanes.
        origin: The repository.
        origin_state: The origin's state dir.
        base_sha: The commit the candidate diffs are taken from.
        task: The fan-out's task, for a candidate whose manifest names none.
        reporter: Receives the notes.

    Returns:
        The candidates, the failed lanes with their reasons, and the imported specs
        (the only ones safe to clean up).
    """
    candidates: list[judge.CandidateBrief] = []
    failed: list[tuple[subrun.LaneResult, str]] = []
    imported: list[subrun.LaneSpec] = []
    for res in results:
        if not res.ok:
            failed.append((res, f"failed to start: {res.error}"))
            continue
        if ipc.worker_is_alive(res.session_dir):
            failed.append(
                (
                    res,
                    "still running; left in place"
                    f" (watch: agent6 attach {res.spec.session_id};"
                    f" stop: agent6 stop {res.spec.session_id})",
                )
            )
            continue
        try:
            dest = _import_lane(res, origin, origin_state, None)
        except subrun.SubrunError as exc:
            failed.append((res, str(exc)))
            continue
        imported.append(res.spec)
        carry_back(
            paths.state_dir(res.spec.workdir),
            origin_state,
            dest,
            lane=res.spec.lane,
            reporter=reporter,
        )
        summary = summarize_session_dir(dest)
        if not produced_result(summary.status):
            # Only a deliberate end is rankable work.
            failed.append(
                (
                    res,
                    f"no result ({format.status_label(summary.status, summary.reason)});"
                    f" branch imported as {res.branch}, not ranked",
                )
            )
            continue
        candidates.append(
            judge.CandidateBrief(
                session_id=res.spec.session_id,
                task=app_compare.manifest_task(dest, task),
                diff=git_ops.diff_since(res.spec.workdir, base_sha),
                verify_ok=summary.verify_ok,
                cost_usd=summary.cost_usd,
            )
        )
    return candidates, failed, imported


def _cleanup(
    imported: list[subrun.LaneSpec], *, workdir_root: pathlib.Path, base: pathlib.Path, cfg: Config
) -> None:
    """Tear down the clone, state dir and config of each imported lane, best-effort.

    Every level from the group dir up to the per-repo base is removed once empty,
    never the dir above it.

    Args:
        imported: The lanes whose import landed.
        workdir_root: The group dir.
        base: The per-repo base dir.
        cfg: The run's config.
    """
    for spec in imported:
        shutil.rmtree(paths.state_dir(spec.workdir), ignore_errors=True)
        shutil.rmtree(spec.workdir, ignore_errors=True)
        (spec.workdir.parent / f"lane-{spec.lane}-config.toml").unlink(missing_ok=True)
    level = workdir_root
    while level.is_relative_to(base):
        with contextlib.suppress(OSError):
            level.rmdir()  # succeeds only when nothing was kept
        if level == base:
            break
        level = level.parent


def fanout_exit_code(candidates: list[judge.CandidateBrief]) -> int:
    """Map the lanes' gate verdicts to the fan-out's exit code.

    Args:
        candidates: The rankable lanes.

    Returns:
        1 nothing rankable, 0 a lane verified green or none had a gate, 4 gates ran
        and none passed.
    """
    if not candidates:
        return 1
    if any(c.verify_ok is True for c in candidates):
        return 0
    return finalize.EXIT_VERIFY_FAILED if any(c.verify_ok is False for c in candidates) else 0


def _print_report(
    candidates: list[judge.CandidateBrief],
    outcome: app_compare.RankOutcome,
    failed: list[tuple[subrun.LaneResult, str]],
    *,
    fanout_id: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Print the ranked candidate table, a merge line per candidate and the failed lanes.

    Args:
        candidates: The ranked lanes.
        outcome: The ranking.
        failed: The failed lanes with their reasons.
        fanout_id: The fan-out's session id.
        reporter: Receives the lines.
    """
    reporter.out(
        f"\n[agent6] parallel fan-out {fanout_id} complete: {len(candidates)} candidate(s)"
    )
    app_compare.print_ranked_candidates(candidates, outcome, reporter=reporter)
    if failed:
        reporter.out("\nfailed lanes:")
        for res, err in failed:
            reporter.out(f"  - lane {res.spec.lane} [{res.spec.session_id}]: {err}")
            kept = [p for p in (res.spec.workdir, res.session_dir) if p.exists()]
            if kept:
                reporter.out(f"    kept: {', '.join(str(p) for p in kept)}")


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------


def run_parallel(
    task: str,
    lanes: list[subrun.LaneSpec],
    *,
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    runtime: LaneRuntime,
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
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner,
            pins=pins,
            cfg=cfg,
            origin=origin,
            max_usd=max_usd,
            auto_approve=auto_approve,
            lane_away=lane_away,
            fanout_id=fanout_id,
            coordinator=fanout_id,
            runtime=runtime,
        )
    try:
        origin_status = git_ops.status(origin)
    except git_ops.GitError as exc:
        reporter.error(str(exc))
        return 2
    base_sha = origin_status.head_sha

    layout = sessions_layout.SessionLayout(
        state_dir=origin_state, session_id=fanout_id, subdir=kinds.session_bucket("run")
    )
    layout.ensure()
    app_manifest.write_session_manifest(
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
    ipc.emit_session_start(
        events,
        layout.session_dir,
        "session.start",
        session_id=fanout_id,
        user_task=task[:200],
        mode="run",
    )
    events.emit("loop.parallel.dispatched", group=fanout_id, lanes=len(lanes), tasks=[task[:200]])
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
            events.emit("session.end", reason="interrupted", iterations=0, all_passed=False)
        raise
    except Exception:
        with contextlib.suppress(event_log.EventWriteError):
            events.emit("session.end", reason="crashed", iterations=0, all_passed=False)
        raise
    finally:
        ipc.clear_worker_pid(layout.session_dir)


def _drive_fanout(
    task: str,
    lanes: list[subrun.LaneSpec],
    *,
    spawner: subrun.LaneSpawner,
    cfg: Config,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    runtime: LaneRuntime,
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
                _lane_watch.print_lane_status(spec, "started", 0.0, reporter=reporter)
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
    candidates, failed, imported = _import_lanes(
        results,
        origin=origin,
        origin_state=origin_state,
        base_sha=base_sha,
        task=task,
        reporter=reporter,
    )
    if failed:
        events.emit(
            "loop.parallel.failed",
            group=fanout_id,
            fanout=True,
            lanes=[{"session_id": res.spec.session_id, "detail": detail} for res, detail in failed],
        )
    _cleanup(
        imported, workdir_root=lanes[0].workdir.parent, base=workdir_base(cfg, origin), cfg=cfg
    )

    outcome = app_compare.rank(
        cfg,
        candidates,
        transcript_dir=origin_state / "parallel" / fanout_id,
        build_provider=runtime.build_provider,
        judging_status=runtime.judging_status,
        max_usd=max_usd,
        reporter=reporter,
    )
    _stamp_compare_outcomes(
        candidates,
        outcome,
        origin_state=origin_state,
        reporter=reporter,
    )
    _print_report(
        candidates,
        outcome,
        failed,
        fanout_id=fanout_id,
        reporter=reporter,
    )
    by_id = {c.session_id: c for c in candidates}
    events.emit(
        "loop.parallel.compared",
        group=fanout_id,
        ranked_by=outcome.ranked_by,
        ranking=[
            {
                "session_id": rid,
                "verify": app_compare.verify_word(by_id[rid].verify_ok),
                "cost_usd": by_id[rid].cost_usd,
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
    rc = 130 if interrupted else fanout_exit_code(candidates)
    reason, all_passed = _fanout_end(rc, candidates)
    events.emit("session.end", reason=reason, iterations=0, all_passed=all_passed)
    return rc


def _fanout_end(rc: int, candidates: list[judge.CandidateBrief]) -> tuple[str, bool | None]:
    """Map the fan-out's exit code to its `session.end` reason and gate verdict.

    Args:
        rc: The exit code.
        candidates: The rankable lanes.

    Returns:
        The reason and `all_passed`: True when a lane's gate went green, None when
        no lane had a gate.
    """
    if rc == 130:
        return "interrupted", False
    if rc == 0:
        return "finish_session", True if any(c.verify_ok for c in candidates) else None
    return ("no_lane_passed" if rc == finalize.EXIT_VERIFY_FAILED else "no_lane_result"), False
