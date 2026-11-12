# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions merge`: land a run's chain on its base."""

from __future__ import annotations

import pathlib

from agent6.app import merge_plan
from agent6.app import reporter as app_reporter
from agent6.ui.cli import sessions_cmds


def _cmd_merge(
    *,
    session_id: str,
    method: str | None,
    into: str | None,
    message: str | None,
    config_paths: tuple[pathlib.Path, ...],
) -> int:
    """Land a run's work on a target branch with the chosen method.

    Args:
        session_id: The run, or "" for the newest.
        method: The merge method; None takes `git.merge_method`.
        into: The target branch; None takes the branch the run was cut from.
        message: The merge commit message; None takes the default.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The exit code: 0 landed or nothing to land, 1 a conflict, 2 a refusal.
    """
    cwd = pathlib.Path.cwd()
    res = sessions_cmds._resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    layout, manifest = res
    return merge_plan.run_merge(
        cwd,
        layout,
        manifest,
        method=method,
        into=into,
        message=message,
        config_paths=config_paths,
        reporter=app_reporter.STDIO_REPORTER,
    )
