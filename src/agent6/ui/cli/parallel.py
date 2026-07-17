# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI adapter for `agent6 run --parallel` and the coordinator `/parallel`
dispatch.

The fan-out / coordinator pipeline is headless in `agent6.app.parallel`; this
module is the front-end seam. It supplies the `LaneRuntime` the pipeline drives
(the detached process spawn from `ui.spawn`, and the reviewer provider +
judging spinner from `_compare`), and holds the CLI-side preflight +
refusal messages for `run --parallel`. run.py routes here; run.py / resume.py
wire the coordinator spawner from here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from agent6.app._setup import explicit_usd_flag_error as _explicit_usd_flag_error
from agent6.app.parallel import (
    LaneRuntime,
    ParallelError,
    build_lane_specs,
    run_parallel,
)
from agent6.app.parallel import (
    build_coordinator_spawner as _app_build_coordinator_spawner,
)
from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.directive import DirectiveError
from agent6.git_ops import GitError, dirty_paths
from agent6.git_ops import status as git_status
from agent6.models.validate import refusal_message, validate_spec_models, warning_message
from agent6.runs.id import new_friendly_id
from agent6.ui.cli._compare import _judging_status, _reviewer_provider
from agent6.ui.spawn import agent6_exe, spawn_and_locate
from agent6.workflows.subrun import GroupLaneSpawner


def lane_runtime() -> LaneRuntime:
    """The front-end primitives the parallel pipeline drives: the detached
    process spawn (`ui.spawn`) and the reviewer provider + judging spinner
    (`_compare`). Injected so `agent6.app` never imports `agent6.ui`. Lane
    liveness/stop is the run-dir bridge, imported directly by `agent6.app.parallel`
    (no longer routed through this seam)."""

    def spawn(
        argv: list[str],
        cwd: Path,
        *,
        before: set[Path],
        list_dirs: Callable[[], list[Path]],
        env: dict[str, str],
    ) -> tuple[Path | None, str]:
        return spawn_and_locate(
            [agent6_exe(), *argv], cwd, before=before, list_dirs=list_dirs, env=env
        )

    return LaneRuntime(
        spawn=spawn,
        build_provider=_reviewer_provider,
        judging_status=_judging_status,
    )


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
    dispatch is unavailable (non-write mode, or a run already inside a lane).
    Injects the CLI's `LaneRuntime` into the headless pipeline. run.py / resume.py
    call this to build the loop's `lane_spawner`, passing the coordinator run's
    own effective `--auto-approve` (same as `max_usd`)."""
    return _app_build_coordinator_spawner(
        cfg,
        origin,
        origin_state,
        mode=mode,
        run_id=run_id,
        runtime=lane_runtime(),
        max_usd=max_usd,
        auto_approve=auto_approve,
    )


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
    lanes, then hand off to the headless `run_parallel`. Called from `run.py`.
    `auto_approve` forwards `--auto-approve` to every lane, same as `max_usd`."""
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
        lanes = build_lane_specs(spec, cfg=cfg, fanout_id=fanout_id)
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
        runtime=lane_runtime(),
        max_usd=max_usd,
        fanout_id=fanout_id,
        auto_approve=auto_approve,
    )
