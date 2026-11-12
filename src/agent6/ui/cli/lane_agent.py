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
import sys
import time

from agent6.app import lane_launch, preflight, session_setup
from agent6.app import run as app_run
from agent6.config import ConfigError
from agent6.ui.cli import common, entry, run


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
    os.environ["AGENT6_DETACHED_AWAY"] = launch.away
    os.chdir(launch.workdir)
    if not preflight.require_git_repo(launch.workdir):
        return 2
    budget = session_setup.BudgetOverrides(max_usd=launch.max_usd)
    sandbox = session_setup.SandboxOverrides(auto_approve=launch.auto_approve)
    try:
        effective = session_setup.load_session_config(
            launch.workdir,
            (launch.config,),
            mode="run",
            budget_overrides=budget,
            sandbox_overrides=sandbox,
        )
    except ConfigError as exc:
        common.error(f"{exc}")
        return 2
    return app_run.run_task(
        effective.config,
        launch.task,
        frontend=run.session_frontend((launch.config,)),
        cwd=launch.workdir,
        started_at=time.time(),
        operator_task=launch.operator_task if launch.operator_task is not None else launch.task,
        session_id=launch.session_id,
        mode="run",
        budget_overrides=budget,
        sandbox_overrides=sandbox,
        pins=launch.pins,
        explicit_leaves=effective.explicit_leaves,
        config_paths=(launch.config,),
        config_sources=effective.sources,
    )


if __name__ == "__main__":
    sys.exit(entry.run_guarded(main))
