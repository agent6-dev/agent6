# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Start one fan-out lane from its launch file.

Invoked as `python -P -E -m agent6.ui.cli.lane_agent` with `AGENT6_LANE` naming the file:
validates it and the config snapshot beside it, loads the lane's config and drives
`agent6.app.run.run_task` through the CLI's session seam. The task reaches the run as the
file holds it, never parsed as options or directives.
"""

from __future__ import annotations

import os
import pathlib
import signal
import sys
import threading
import time
from collections.abc import Callable

from agent6 import kinds
from agent6.app import lane_launch, preflight, stop
from agent6.app import run as app_run
from agent6.app.session import setup as session_setup
from agent6.config import ConfigError
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import common, entry, run


def arm_deadline(
    session_dir: pathlib.Path,
    deadline_at: float,
    *,
    grace_s: float = stop.KILL_GRACE_S,
    clock: Callable[[], float] = time.time,
) -> tuple[threading.Timer, threading.Timer]:
    """Arm the lane's own backstop: the stop bridges at the deadline, SIGTERM a grace later.

    The run reads the abort steer and the `now` stop marker as `agent6 stop` writes them;
    a run that is not reading them takes the SIGTERM as a Ctrl-C stop and journals its end.

    Args:
        session_dir: The lane's run dir, which may not exist yet.
        deadline_at: The wall-clock second the lane must have ended by.
        grace_s: From the bridges to the signal.
        clock: The wall clock.

    Returns:
        The two daemon timers, for a caller that cancels them.
    """

    def ask() -> None:
        ipc.submit_steer(session_dir, "abort", now=True)
        ipc.request_stop(session_dir, now=True)

    def term() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    delay = max(0.0, deadline_at - clock())
    timers = (threading.Timer(delay, ask), threading.Timer(delay + grace_s, term))
    for timer in timers:
        timer.daemon = True
        timer.start()
    return timers


def trusted_launch() -> lane_launch.LaneLaunch:
    """Return the launch `AGENT6_LANE` names, its config snapshot checked beside it.

    Returns:
        The lane's facts.

    Raises:
        LaunchError: The variable is unset, or either file fails the private-file checks.
    """
    launch = lane_launch.read()
    if launch is None:
        raise lane_launch.LaunchError(
            f"{lane_launch.ENV} is unset: the lane agent starts from a launch file"
        )
    # The snapshot is trusted only beside the launch file, under the same private checks.
    snapshot = pathlib.Path(os.environ[lane_launch.ENV]).parent / launch.config.name
    if launch.config != snapshot:
        raise lane_launch.LaunchError(
            f"invalid lane launch: the config snapshot {launch.config} is not beside it"
        )
    lane_launch.read_private(launch.config)
    return launch


def main() -> int:
    """Start the lane `AGENT6_LANE` describes.

    Returns:
        The run's exit code; 2 when the launch file, the snapshot or the config refuses.
    """
    try:
        launch = trusted_launch()
    except ConfigError as exc:
        common.error(f"{exc}")
        return 2
    # The away mode is the operator's resolved policy; the run's prompts read the marker.
    os.environ["AGENT6_CHILD"] = "1"
    os.environ["AGENT6_DETACHED_AWAY"] = launch.away
    os.chdir(launch.worktree)
    if not preflight.require_git_repo(launch.worktree):
        return 2
    budget = (
        session_setup.BudgetOverrides(
            max_usd=launch.allowance.usd,
            max_tokens_fallback=launch.allowance.tokens,
            max_plan_points=launch.allowance.plan_points,
        )
        if launch.allowance is not None
        else session_setup.BudgetOverrides(max_usd=launch.max_usd)
    )
    sandbox = session_setup.SandboxOverrides(auto_approve=launch.auto_approve)
    # A contract makes the lane a bounded read-only conversation; without one it is a run.
    mode: kinds.LoopMode = "reviewer" if launch.contract is not None else "run"
    try:
        effective = session_setup.load_session_config(
            launch.worktree,
            (launch.config,),
            mode=mode,
            budget_overrides=budget,
            sandbox_overrides=sandbox,
        )
    except ConfigError as exc:
        common.error(f"{exc}")
        return 2
    cfg = lane_launch.apply_limits(effective.config, launch.limits)
    if launch.limits is not None and launch.limits.deadline_at is not None:
        bucket = sessions_layout.bucket_dir(launch.record_state, kinds.session_bucket(mode))
        arm_deadline(bucket / launch.session_id, launch.limits.deadline_at)
    return app_run.run_task(
        cfg,
        launch.task,
        frontend=run.session_frontend((launch.config,)),
        cwd=launch.worktree,
        started_at=time.time(),
        operator_task=launch.operator_task if launch.operator_task is not None else launch.task,
        session_id=launch.session_id,
        mode=mode,
        contract=launch.contract,
        budget_overrides=budget,
        sandbox_overrides=sandbox,
        pins=launch.pins,
        explicit_leaves=effective.explicit_leaves,
        config_paths=(launch.config,),
        config_sources=effective.sources,
    )


if __name__ == "__main__":
    sys.exit(entry.run_guarded(main))
