# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The CLI adapter for `agent6 run --parallel` and the parent `/parallel` dispatch.

The lanes are headless in `agent6.app.lanes`; this module supplies the `LaneRuntime` they
drive (the detached spawn from `ui.spawn`, the reviewer provider and judging spinner from
`compare_ranking`) and the check of a `--parallel` spec. `run.py` routes here.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from agent6 import directive
from agent6.app import lanes as app_lanes
from agent6.config import Config, ConfigError
from agent6.models import validate
from agent6.ui import spawn as ui_spawn
from agent6.ui.cli import common, compare_ranking


def lane_runtime() -> app_lanes.LaneRuntime:
    """Return the front-end primitives the parallel pipeline drives.

    Injected so `agent6.app` never imports `agent6.ui`. Lane liveness and stop are the
    session-dir bridge, which `agent6.app.lanes` imports directly.

    Returns:
        The primitives.
    """

    def spawn(
        argv: list[str],
        cwd: pathlib.Path,
        *,
        before: set[pathlib.Path],
        list_dirs: Callable[[], list[pathlib.Path]],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        """Spawn a detached lane and locate its session dir.

        Args:
            argv: The full command, the lane bootstrap.
            cwd: Where to spawn.
            before: The session dirs that existed before the spawn.
            list_dirs: Lists the candidate session dirs.
            env: The child's environment.

        Returns:
            The session dir, the pid and an error text, from `spawn_lane`.
        """
        return ui_spawn.spawn_lane(argv, cwd, before=before, list_dirs=list_dirs, env=env)

    return app_lanes.LaneRuntime(
        spawn=spawn,
        build_provider=compare_ranking._reviewer_provider,
        judging_status=compare_ranking._judging_status,
    )


def spec_refusal(cfg: Config, spec: str) -> str | None:
    """Return why a `--parallel` spec cannot run, before any session exists.

    A model the cache cannot check warns and proceeds.

    Args:
        cfg: The effective config.
        spec: The `--parallel` lane spec.

    Returns:
        The refusal: a bad count, a lane over `[harness.parallel].max_lanes`, or a selector
        that names no model; None when the spec can run.
    """
    try:
        settings = app_lanes.lane_settings(
            cfg, directive.parse_spec(spec, limit=cfg.harness.parallel.max_lanes)
        )
    except (ConfigError, directive.DirectiveError) as exc:
        return str(exc)
    verdict = validate.validate_spec_models(
        [chosen.route if chosen is not None else None for chosen in settings], cfg
    )
    if verdict.refused:
        return validate.refusal_message(verdict, directive=False)
    if verdict.warned:
        common.warn(f"{validate.warning_message(verdict)}")
    return None
