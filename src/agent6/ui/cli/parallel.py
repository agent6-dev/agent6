# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fan-out orchestrator for `agent6 run --parallel`.

Spawn N isolated lanes -- each a disposable clone of the repo running its own
detached `agent6 run` -- symlink the live lanes into `agent6 runs` for
visibility, await them, import each finished lane's branch + run dir back into
the origin, then auto-compare and print a ranked report. Nothing is merged: the
operator picks a winner and runs `agent6 runs merge <id>`.

The origin repo is never mutated (no branch cut, no run dir, no commits) until
`import_run` lands a lane's branch. Clones + lane state are torn down after
import. The heavy git plumbing lives in `workflows.subrun`; the ranking in
`workflows.judge`; this module orchestrates them over `ui.bridge.spawn`.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent6.config import Config
from agent6.config.layer import materialize, resolved_state_dir
from agent6.directive import DirectiveError, parse_spec
from agent6.git_ops import GitError, diff_since, dirty_paths
from agent6.git_ops import status as git_status
from agent6.models.validate import refusal_message, validate_spec_models, warning_message
from agent6.paths import cache_dir, state_dir
from agent6.portable import atomic_write
from agent6.runs.id import new_friendly_id
from agent6.ui.bridge.approval import request_stop, worker_is_alive
from agent6.ui.bridge.spawn import agent6_exe, spawn_and_locate
from agent6.ui.cli._common import _explicit_usd_flag_error
from agent6.ui.cli._compare import (
    manifest_task as _manifest_task,
)
from agent6.ui.cli._compare import (
    print_ranked_candidates as _print_ranked_candidates,
)
from agent6.ui.cli._compare import (
    rank as _rank,
)
from agent6.ui.cli._compare import (
    verify_ok as _verify_ok,
)
from agent6.viewmodel import summarize_run_dir
from agent6.workflows.judge import CandidateBrief
from agent6.workflows.subrun import (
    GroupLaneSpawner,
    LaneResult,
    LaneSpawner,
    LaneSpec,
    LaneTask,
    SubrunError,
    clone_workspace,
    import_run,
)

# How often the await loop polls lane liveness, and how long Ctrl+C waits for a
# stop-requested lane to finish its in-flight step before giving up on it.
_POLL_INTERVAL_S = 2.0
_STOP_GRACE_S = 30.0


class ParallelError(Exception):
    """The fan-out could not be set up (over the [parallel].max_lanes cap)."""


# ---------------------------------------------------------------------------
# Lane planning
# ---------------------------------------------------------------------------


def _workdir_root(cfg: Config, fanout_id: str) -> Path:
    """Base dir for this fan-out's lane clones: `[parallel].workdir` (or
    `<cache_dir>/parallel`) / `<fanout-id>`."""
    base = Path(cfg.parallel.workdir) if cfg.parallel.workdir else cache_dir() / "parallel"
    return base / fanout_id


def build_lane_specs(
    spec: str, *, cfg: Config, fanout_id: str, workdir_root: Path
) -> list[LaneSpec]:
    """Plan the lanes for a `--parallel` fan-out, refusing over-cap up front."""
    models = parse_spec(spec)
    if len(models) > cfg.parallel.max_lanes:
        raise ParallelError(
            f"--parallel requests {len(models)} lanes but [parallel].max_lanes ="
            f" {cfg.parallel.max_lanes}. Request fewer, or raise [parallel].max_lanes."
        )
    return [
        LaneSpec(
            lane=i,
            run_id=f"{fanout_id}-l{i}",
            workdir=workdir_root / f"lane-{i}",
            model=model,
        )
        for i, model in enumerate(models, start=1)
    ]


# ---------------------------------------------------------------------------
# The real (bridge) spawner: clone, write a lane config, spawn detached, locate
# ---------------------------------------------------------------------------


def _write_lane_config(cfg: Config, spec: LaneSpec) -> Path:
    """Materialize the origin's effective config (worker model overridden for a
    per-lane model) to a file the lane loads with `--config`.

    The clone's path-keyed repo id yields an EMPTY per-repo config, so the lane
    would otherwise lose every origin repo setting; a full materialized config
    layered over the (shared) global config restores them. `for_repo=True` drops
    the global-only `[agent6].state_dir`, which `--config` forbids. Global config
    + secrets apply automatically."""
    lane_cfg = cfg.with_machine_agent_overrides(model=spec.model) if spec.model else cfg
    config_path = spec.workdir.parent / f"lane-{spec.lane}-config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(materialize(lane_cfg, for_repo=True), encoding="utf-8")
    return config_path


def bridge_spawner(
    spec: LaneSpec,
    task: str,
    *,
    cfg: Config,
    origin: Path,
    max_usd: float | None,
    auto_approve: bool = False,
) -> LaneResult:
    """Clone the origin, spawn a detached `agent6 run` in the clone, and return a
    LaneResult once its run dir is located (the run keeps going in the
    background). `ok=False` when the clone or spawn fails; the orchestrator
    records it and moves on. `auto_approve` forwards the coordinator/fan-out's
    own `--auto-approve` to the lane's argv, same as `max_usd`."""
    branch = f"agent6/{spec.run_id}"
    try:
        clone_workspace(origin, spec.workdir)
    except SubrunError as exc:
        return LaneResult(spec=spec, run_dir=spec.workdir, branch=branch, ok=False, error=str(exc))
    config_path = _write_lane_config(cfg, spec)
    lane_runs = state_dir(spec.workdir, cfg.agent6.state_dir) / "runs"

    def list_dirs() -> list[Path]:
        if not lane_runs.is_dir():
            return []
        return [p for p in lane_runs.iterdir() if p.is_dir()]

    argv = [
        agent6_exe(),
        "run",
        "--run-id",
        spec.run_id,
        "--config",
        str(config_path),
    ]
    if max_usd is not None:
        argv += ["--max-usd", f"{max_usd:g}"]
    if auto_approve:
        argv += ["--auto-approve"]
    # `--` before the task so a task that looks like a flag (`--allow-root ...`)
    # is never parsed as one. Flags all precede it.
    argv += ["--", task]
    # AGENT6_SUBRUN marks the lane as a subordinate run: run.py leaves its
    # coordinator `lane_spawner` unwired and the `--parallel` flag refuses under
    # it, so a lane can never itself fan out or dispatch (depth 1 by construction,
    # for both the CLI fan-out and the coordinator's `/parallel` groups).
    env = {
        **os.environ,
        "AGENT6_STREAM_TO_LOG": "1",
        "AGENT6_DETACHED_AWAY": "wait",
        "AGENT6_SUBRUN": "1",
    }
    run_dir, err = spawn_and_locate(argv, spec.workdir, before=set(), list_dirs=list_dirs, env=env)
    if run_dir is None:
        return LaneResult(
            spec=spec, run_dir=lane_runs / spec.run_id, branch=branch, ok=False, error=err
        )
    return LaneResult(spec=spec, run_dir=run_dir, branch=branch, ok=True, error="")


# ---------------------------------------------------------------------------
# Coordinator dispatch: one lane to completion + a group spawner for the loop
# ---------------------------------------------------------------------------


def _lane_terminal(run_dir: Path, status: str) -> bool:
    """Terminal gate for an awaited lane: the fold left "running" AND the worker
    pid is cleared/dead. run.end lands in logs.jsonl before the lane's teardown
    clears worker.pid, so status alone races the teardown, and importing inside
    that window would misread a finished lane as still running. A lane that dies
    WITHOUT a run.end cannot hang this gate: the fold flips a dead recorded pid
    to "stale" at once, a pid-less silent lane to "stale" after its bounded
    silence window, and a lane that never wrote logs reads "?" (see
    `summarize_run_dir`)."""
    return status != "running" and not worker_is_alive(run_dir)


def _await_lane(res: LaneResult, *, poll_interval_s: float = _POLL_INTERVAL_S) -> None:
    """Block until *res*'s lane is terminal (`_lane_terminal`), awaited on its
    REAL run dir. Same gate as the fan-out's `_await_lanes`, for a single lane."""
    while True:
        summary = summarize_run_dir(res.run_dir)
        if _lane_terminal(res.run_dir, summary.status):
            return
        time.sleep(poll_interval_s)


def run_lane_to_completion(
    spec: LaneSpec,
    task: str,
    *,
    cfg: Config,
    origin: Path,
    origin_state: Path,
    group: str,
    max_usd: float | None = None,
    auto_approve: bool = False,
    spawner: LaneSpawner | None = None,
    import_lock: threading.Lock | None = None,
    poll_interval_s: float = _POLL_INTERVAL_S,
) -> LaneResult:
    """Run ONE subordinate lane fully and import it into *origin*.

    Clone + spawn (via *spawner*, default the bridge spawner), await the lane to
    terminal, then import its branch + run dir into the coordinator's repo and
    stamp `<group>` lineage. Returns a LaneResult whose `run_dir` is the imported
    dir on success; `ok=False` (nothing imported, *origin* untouched for this
    lane) when the lane failed to start, was still running at teardown, or its
    import was refused. The coordinator runs a group of these on a thread pool, so
    each is self-contained per lane; *import_lock*, when given, serializes the
    git-mutating import step across that group (concurrent fetches into one repo
    race on refs/objects). Tests inject a fake *spawner*."""
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner, cfg=cfg, origin=origin, max_usd=max_usd, auto_approve=auto_approve
        )
    res = spawner(spec, task)
    if not res.ok:
        return res
    # Symlink the live lane into the origin's runs/ (same as the fan-out path) so
    # a hub can see it and answer its approvals/asks while it runs, not just at
    # import. Dropped just before import so import_run can place the real dir.
    _symlink_lane(origin_state, res)
    _await_lane(res, poll_interval_s=poll_interval_s)
    lock = import_lock if import_lock is not None else contextlib.nullcontext()
    link = _lane_link(origin_state, res.spec.run_id)
    had_link = link.is_symlink()
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    try:
        with lock:
            dest = import_run(origin, spec.workdir, res.branch, res.run_dir, origin_state)
    except SubrunError as exc:
        if had_link:
            _symlink_lane(origin_state, res)  # restore the live view; nothing moved
        return LaneResult(
            spec=spec, run_dir=res.run_dir, branch=res.branch, ok=False, error=str(exc)
        )
    stamp_err = _stamp_lineage(dest, group, spec.lane)
    if stamp_err is not None:
        print(
            f"[agent6] lane {spec.lane} [{spec.run_id}]: imported, but the lineage"
            f" stamp failed: {stamp_err}",
            file=sys.stderr,
        )
    return LaneResult(spec=spec, run_dir=dest, branch=res.branch, ok=True, error="")


def build_lane_spawner(
    cfg: Config,
    origin: Path,
    origin_state: Path,
    *,
    coordinator_run_id: str,
    max_usd: float | None = None,
    auto_approve: bool = False,
) -> GroupLaneSpawner:
    """Build the coordinator's group dispatcher: the `GroupLaneSpawner` the loop
    calls at a `/parallel` steer boundary.

    One call clones + spawns each lane on its own model, awaits them all to
    terminal on a thread pool (one thread per lane, like the review panel's
    seats), imports each into *origin* (serialized by a shared lock), and returns
    per-lane LaneResults in dispatch order. Lane run ids are
    `<coordinator_run_id>-<group>-l<i>`; lane workspaces live under the same
    `[parallel].workdir` cache the fan-out uses, in a `<group>` subdir. The bridge
    spawner tags each lane `AGENT6_SUBRUN=1`, so a lane can never itself dispatch
    (depth 1 by construction). `auto_approve` forwards the coordinator's own
    `--auto-approve` to every lane, same as `max_usd`."""

    def dispatch(lanes: list[LaneTask], group: str) -> list[LaneResult]:
        # Validate the per-lane models before any clone: a refusal raises, and the
        # loop's group-failure feedback delivers the message to the coordinator
        # (keeping workflows free of a models dependency); no cache = warn + proceed.
        verdict = validate_spec_models([lane.model for lane in lanes], cfg)
        if verdict.refused:
            raise ParallelError(refusal_message(verdict, directive=True))
        if verdict.warned:
            print(f"[agent6] WARNING: {warning_message(verdict)}", file=sys.stderr)
        workdir_root = _workdir_root(cfg, coordinator_run_id) / group
        specs = [
            LaneSpec(
                lane=i,
                run_id=f"{coordinator_run_id}-{group}-l{i}",
                workdir=workdir_root / f"lane-{i}",
                model=lane.model,
            )
            for i, lane in enumerate(lanes, start=1)
        ]
        (origin_state / "runs").mkdir(parents=True, exist_ok=True)
        import_lock = threading.Lock()

        def one(pair: tuple[LaneSpec, LaneTask]) -> LaneResult:
            spec, lane = pair
            return run_lane_to_completion(
                spec,
                lane.task,
                cfg=cfg,
                origin=origin,
                origin_state=origin_state,
                group=group,
                max_usd=max_usd,
                auto_approve=auto_approve,
                import_lock=import_lock,
            )

        pairs = list(zip(specs, lanes, strict=True))
        if len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
                return list(pool.map(one, pairs))  # map preserves input order
        return [one(p) for p in pairs]

    return dispatch


def build_coordinator_spawner(
    cfg: Config,
    origin: Path,
    origin_state: Path,
    *,
    mode: str,
    run_id: str,
    max_usd: float | None = None,
    auto_approve: bool = False,
) -> GroupLaneSpawner | None:
    """The `/parallel` group dispatcher to wire into a run's loop, or None when
    dispatch is unavailable: a non-write mode (plan/ask make no commits to clone),
    or a run already inside a subordinate lane (`AGENT6_SUBRUN` set), which keeps
    dispatch depth 1 by construction. run.py / resume.py call this to build the
    loop's `lane_spawner`, passing the coordinator run's own effective
    `--auto-approve` (same as `max_usd`) so a lane never sits on an approval
    nothing detached can answer."""
    if mode != "run" or os.environ.get("AGENT6_SUBRUN"):
        return None
    return build_lane_spawner(
        cfg,
        origin,
        origin_state,
        coordinator_run_id=run_id,
        max_usd=max_usd,
        auto_approve=auto_approve,
    )


# ---------------------------------------------------------------------------
# Live view + await
# ---------------------------------------------------------------------------


def _lane_link(origin_state: Path, run_id: str) -> Path:
    return origin_state / "runs" / run_id


def _symlink_lane(origin_state: Path, res: LaneResult) -> None:
    """Symlink a located lane's (clone-side) run dir into the origin's `runs/` so
    `agent6 runs`/hub shows it live. Replaced by the real imported dir at import."""
    link = _lane_link(origin_state, res.spec.run_id)
    link.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    with contextlib.suppress(OSError):
        link.symlink_to(res.run_dir)


def _await_lanes(started: list[LaneResult], *, already_interrupted: bool = False) -> bool:
    """Poll every started lane's REAL run dir (in the clone's state; the origin
    symlink is a view for the hub, never the source of truth) until it is
    terminal (`_lane_terminal`), printing one line per lane on a status/cost
    change. Returns True if interrupted (Ctrl+C): request a clean stop on each
    still-running lane, wait a bounded grace for them to finish their in-flight
    step, then return so the caller imports what landed.

    `already_interrupted=True` (a Ctrl+C the spawn loop caught before the await
    even began) skips the normal poll and goes straight into that same stop-grace
    path, so a mid-spawn interrupt stops the already-started lanes identically."""
    pending = {r.spec.run_id: r for r in started}
    seen: dict[str, tuple[str, str, float]] = {}

    def poll_once() -> None:
        for rid, res in list(pending.items()):
            summary = summarize_run_dir(res.run_dir)
            # A "running" lane may actually be blocked on an approval/question no
            # detached lane can answer; surface it so the operator opens the hub.
            waiting = _pending_prompt(res.run_dir) if summary.status == "running" else ""
            key = (summary.status, waiting, round(summary.cost_usd, 4))
            if seen.get(rid) != key:
                seen[rid] = key
                _print_lane_status(res.spec, summary.status, summary.cost_usd, waiting=waiting)
            if _lane_terminal(res.run_dir, summary.status):
                pending.pop(rid)

    def stop_and_drain() -> None:
        print("\n[agent6] interrupted; stopping lanes...", file=sys.stderr)
        for res in pending.values():
            request_stop(res.run_dir)
        deadline = time.monotonic() + _STOP_GRACE_S
        with contextlib.suppress(KeyboardInterrupt):
            while pending and time.monotonic() < deadline:
                poll_once()
                if pending:
                    time.sleep(_POLL_INTERVAL_S)

    if already_interrupted:
        stop_and_drain()
        return True
    try:
        while pending:
            poll_once()
            if pending:
                time.sleep(_POLL_INTERVAL_S)
        return False
    except KeyboardInterrupt:
        stop_and_drain()
        return True


# The two prompt/answer event pairs a lane can block on, for `_pending_prompt`.
_PROMPT_KIND = {"approval.prompt": "approval", "question.prompt": "a question"}
_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


def _pending_prompt(run_dir: Path) -> str:
    """ "approval" / "a question" if the lane is blocked on an unanswered prompt,
    else "". The worker emits `approval.prompt`/`question.prompt` then BLOCKS on
    its `*.answer` (lanes run with AGENT6_DETACHED_AWAY=wait, so a prompt with no
    hub attached waits rather than denies), so the LAST prompt/answer event in
    logs.jsonl decides it -- a cheap trailing scan, no `*.request` marker exists
    for approvals/questions. Deliberately not the heavyweight RunState fold; the
    fan-out status line needs only this one bit."""
    try:
        lines = (run_dir / "logs.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        if "approval." not in raw and "question." not in raw:
            continue  # fast reject before json.loads
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        etype = ev.get("type") if isinstance(ev, dict) else None
        if etype in _ANSWER_EVENTS:
            return ""
        if etype in _PROMPT_KIND:
            return _PROMPT_KIND[etype]
    return ""


def _print_lane_status(spec: LaneSpec, status: str, cost: float, *, waiting: str = "") -> None:
    model = f" ({spec.model})" if spec.model else ""
    cost_s = f"  ${cost:.4f}" if cost > 0 else ""
    wait_s = f" · waiting on {waiting} (answer via the web or TUI hub)" if waiting else ""
    print(
        f"[agent6] lane {spec.lane} [{spec.run_id}]{model}: {status}{wait_s}{cost_s}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Import + auto-compare
# ---------------------------------------------------------------------------


def _rewrite_manifest(run_dir: Path, updates: dict[str, object]) -> str | None:
    """Merge *updates* into an imported lane's manifest.json (read, add, atomic
    rewrite). Returns an error string when the manifest cannot be read/parsed or
    written (the import itself stands; the caller reports the degradation). The
    one stamping helper: `_stamp_lineage` (post-import) and `_stamp_compare`
    (post-ranking) both go through it, so the atomic rewrite + loud-degrade
    contract lives in one place."""
    mpath = run_dir / "manifest.json"
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"could not read {mpath}: {exc}"
    if not isinstance(manifest, dict):
        return f"manifest is not a JSON object: {mpath}"
    manifest.update(updates)
    try:
        atomic_write(mpath, json.dumps(manifest, indent=2) + "\n")
    except OSError as exc:
        # Disk full / read-only mount: the import already stands, so report the
        # degradation and let the loop keep importing/stamping the remaining lanes.
        return f"could not write {mpath}: {exc}"
    return None


def _stamp_lineage(run_dir: Path, fanout_id: str, lane: int) -> str | None:
    """Record fan-out lineage on an imported lane's manifest. The lane process
    wrote the manifest not knowing it was a lane, so the orchestrator adds
    `parallel_id`/`lane` post-import."""
    return _rewrite_manifest(run_dir, {"parallel_id": fanout_id, "lane": lane})


def _stamp_compare_outcomes(
    candidates: list[CandidateBrief],
    ranking: tuple[str, ...],
    *,
    origin_state: Path,
    fanout_id: str,
    ranked_by: str,
    rationale: str,
) -> None:
    """Stamp the auto-compare outcome into EACH ranked lane's manifest, so every
    run view can show where a lane placed and why. ONE writer: only the fan-out's
    auto-compare stamps this (`runs compare` stays stateless; the coordinator
    never compares its lanes). The imported lanes sit at `<origin_state>/runs/<id>`
    (import_run's contract); the same rationale is recorded on every lane (it is
    the judge's ranking of the whole group), truncated to bound the manifest, and
    empty for a mechanical ranking. A per-lane stamp failure degrades loudly and
    never blocks the others."""
    of = len(candidates)
    text = rationale[:2000] if ranked_by == "judge" else ""
    for rank_pos, run_id in enumerate(ranking, start=1):
        compare: dict[str, object] = {
            "group": fanout_id,
            "rank": rank_pos,
            "of": of,
            "winner": rank_pos == 1,
            "ranked_by": ranked_by,
            "rationale": text,
        }
        err = _rewrite_manifest(_lane_link(origin_state, run_id), {"compare": compare})
        if err is not None:
            print(
                f"[agent6] lane [{run_id}]: imported, but the compare stamp failed: {err}",
                file=sys.stderr,
            )


def _import_lanes(
    results: list[LaneResult],
    *,
    origin: Path,
    origin_state: Path,
    base_sha: str,
    fanout_id: str,
    task: str,
) -> tuple[list[CandidateBrief], list[tuple[LaneResult, str]], list[LaneSpec]]:
    """Import each finished lane's branch + run dir into the origin, stamp its
    lineage, and build a candidate brief from it. Returns (candidates, failed,
    imported specs); only imported lanes are safe to clean up. A failed-to-start,
    still-running, or import-refused lane is recorded and never blocks the
    others; its clone, state, and live symlink stay in place (they may hold the
    only copy of its work). Candidate diffs come from the clone (still on the
    run branch) before cleanup.
    """
    candidates: list[CandidateBrief] = []
    failed: list[tuple[LaneResult, str]] = []
    imported: list[LaneSpec] = []
    for res in results:
        if not res.ok:
            failed.append((res, res.error))
            continue
        link = _lane_link(origin_state, res.spec.run_id)
        if worker_is_alive(res.run_dir):
            failed.append(
                (
                    res,
                    "still running; left in place"
                    f" (watch: agent6 attach {res.spec.run_id};"
                    f" stop: agent6 runs stop {res.spec.run_id})",
                )
            )
            continue
        had_link = link.is_symlink()
        with contextlib.suppress(FileNotFoundError):
            link.unlink()  # drop the live symlink so import can place the real dir
        try:
            dest = import_run(origin, res.spec.workdir, res.branch, res.run_dir, origin_state)
        except SubrunError as exc:
            if had_link:
                _symlink_lane(origin_state, res)  # restore the live view; nothing moved
            failed.append((res, str(exc)))
            continue
        imported.append(res.spec)
        stamp_err = _stamp_lineage(dest, fanout_id, res.spec.lane)
        if stamp_err is not None:
            print(
                f"[agent6] lane {res.spec.lane} [{res.spec.run_id}]: imported, but the"
                f" lineage stamp failed: {stamp_err}",
                file=sys.stderr,
            )
        summary = summarize_run_dir(dest)
        candidates.append(
            CandidateBrief(
                run_id=res.spec.run_id,
                task=_manifest_task(dest, task),
                diff=diff_since(res.spec.workdir, base_sha),
                verify_ok=_verify_ok(summary.status),
                cost_usd=summary.cost_usd,
            )
        )
    return candidates, failed, imported


def _cleanup(imported: list[LaneSpec], *, workdir_root: Path, cfg: Config) -> None:
    """Tear down clone + state dir + lane config for IMPORTED lanes only; a lane
    that did not import keeps everything it has (its clone may hold the only
    copy of its branch, and a live lane must never lose its workspace). The
    fan-out workdir root is removed only once it is empty. Best-effort: a
    leftover clone is disk waste, never corruption."""
    for spec in imported:
        shutil.rmtree(state_dir(spec.workdir, cfg.agent6.state_dir), ignore_errors=True)
        shutil.rmtree(spec.workdir, ignore_errors=True)
        (spec.workdir.parent / f"lane-{spec.lane}-config.toml").unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        workdir_root.rmdir()  # only succeeds when nothing was kept


def _print_report(
    candidates: list[CandidateBrief],
    ranking: tuple[str, ...],
    failed: list[tuple[LaneResult, str]],
    *,
    fanout_id: str,
    rationale: str,
) -> None:
    """Print the ranked candidate table + a `runs merge` line per candidate, and
    list any failed lanes. Nothing is merged automatically."""
    print(f"\n[agent6] parallel fan-out {fanout_id} complete: {len(candidates)} candidate(s)")
    _print_ranked_candidates(candidates, ranking, rationale)
    if failed:
        print("\nfailed lanes (nothing of theirs was deleted):")
        for res, err in failed:
            print(f"  - lane {res.spec.lane} [{res.spec.run_id}]: {err}")
            kept = [p for p in (res.spec.workdir, res.run_dir) if p.exists()]
            if kept:
                print(f"    kept: {', '.join(str(p) for p in kept)}")


# ---------------------------------------------------------------------------
# Orchestrator entry points
# ---------------------------------------------------------------------------


def run_parallel(
    task: str,
    lanes: list[LaneSpec],
    *,
    cfg: Config,
    origin: Path,
    origin_state: Path,
    spawner: LaneSpawner | None = None,
    max_usd: float | None = None,
    fanout_id: str | None = None,
    auto_approve: bool = False,
) -> int:
    """Run *lanes* to completion, import them, and print a ranked comparison.

    Returns 0 when at least one lane imported, 1 when none did, 130 on Ctrl+C.
    *spawner* defaults to the real bridge spawner; tests inject a fake.
    `auto_approve` forwards to every lane's argv, same as `max_usd`.
    """
    if not lanes:
        print("ERROR: no lanes to run", file=sys.stderr)
        return 2
    if fanout_id is None:
        fanout_id = lanes[0].run_id.rsplit("-l", 1)[0]
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner, cfg=cfg, origin=origin, max_usd=max_usd, auto_approve=auto_approve
        )
    try:
        base_sha = git_status(origin).head_sha
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    (origin_state / "runs").mkdir(parents=True, exist_ok=True)
    print(f"[agent6] parallel fan-out {fanout_id}: {len(lanes)} lanes", file=sys.stderr)
    if max_usd is not None:
        print(
            f"[agent6] budget: ${max_usd:g}/lane x {len(lanes)} = ${max_usd * len(lanes):g} total",
            file=sys.stderr,
        )

    results: list[LaneResult] = []
    try:
        for spec in lanes:
            res = spawner(spec, task)
            results.append(res)
            if res.ok:
                _symlink_lane(origin_state, res)
                _print_lane_status(spec, "started", 0.0)
            else:
                print(
                    f"[agent6] lane {spec.lane} [{spec.run_id}]: FAILED to start: {res.error}",
                    file=sys.stderr,
                )
        interrupted = _await_lanes([r for r in results if r.ok])
    except KeyboardInterrupt:
        # Ctrl+C mid-spawn (before the await): route the already-started lanes
        # into the same stop-grace path, then import-what-exists + report below.
        interrupted = _await_lanes([r for r in results if r.ok], already_interrupted=True)

    candidates, failed, imported = _import_lanes(
        results,
        origin=origin,
        origin_state=origin_state,
        base_sha=base_sha,
        fanout_id=fanout_id,
        task=task,
    )
    _cleanup(imported, workdir_root=lanes[0].workdir.parent, cfg=cfg)

    ranking, rationale, ranked_by = _rank(
        cfg, candidates, transcript_dir=origin_state / "parallel" / fanout_id
    )
    _stamp_compare_outcomes(
        candidates,
        ranking,
        origin_state=origin_state,
        fanout_id=fanout_id,
        ranked_by=ranked_by,
        rationale=rationale,
    )
    _print_report(candidates, ranking, failed, fanout_id=fanout_id, rationale=rationale)

    if interrupted:
        return 130
    return 0 if candidates else 1


def dispatch_parallel(
    cfg: Config,
    task: str,
    spec: str,
    *,
    cwd: Path,
    max_usd: float | None = None,
    auto_approve: bool = False,
) -> int:
    """Preflight and route `agent6 run --parallel`: refuse an unenforceable
    --max-usd or a dirty origin (lanes clone committed HEAD only), plan the
    lanes, then hand off to `run_parallel`. Called from `run.py`. `auto_approve`
    forwards `--auto-approve` to every lane, same as `max_usd`."""
    origin = cwd
    origin_state = resolved_state_dir(origin)
    usd_err = _explicit_usd_flag_error(max_usd, cfg)
    if usd_err is not None:
        print(f"REFUSING: {usd_err}", file=sys.stderr)
        return 2
    try:
        st = git_status(origin)
    except GitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not st.is_clean and cfg.git.require_clean_worktree:
        dirty = dirty_paths(origin)
        listed = "\n".join(f"    {p}" for p in dirty)
        more = "\n    ..." if len(dirty) >= 10 else ""
        print(
            "REFUSING: working tree is not clean:\n"
            f"{listed}{more}\n"
            "Commit, stash, or discard your changes, or set"
            " [git].require_clean_worktree=false to override. Lanes clone from"
            " committed HEAD, so uncommitted work is not carried into them.",
            file=sys.stderr,
        )
        return 2

    fanout_id = new_friendly_id()
    try:
        lanes = build_lane_specs(
            spec, cfg=cfg, fanout_id=fanout_id, workdir_root=_workdir_root(cfg, fanout_id)
        )
    except (DirectiveError, ParallelError) as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    # Validate the named models before any clone/spawn (lanes are plain specs so
    # far, no workdir touched): refuse a typo when a cache exists to check
    # against, else warn and proceed (a fresh/offline machine is never blocked).
    verdict = validate_spec_models([ln.model for ln in lanes], cfg)
    if verdict.refused:
        print(f"REFUSING: {refusal_message(verdict, directive=False)}", file=sys.stderr)
        return 2
    if verdict.warned:
        print(f"[agent6] WARNING: {warning_message(verdict)}", file=sys.stderr)
    return run_parallel(
        task,
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        max_usd=max_usd,
        fanout_id=fanout_id,
        auto_approve=auto_approve,
    )
