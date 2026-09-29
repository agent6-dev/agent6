# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 resume`: adapt argv and hand the lifecycle to `agent6.app.resume.resume_task`.

The presentation seam is the one `agent6 run` injects (`ui.cli.run.session_frontend`).
"""

from __future__ import annotations

import time
from pathlib import Path

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
)
from agent6.app.resume import resume_task
from agent6.ui.cli.run import session_frontend


def _cmd_resume(
    config_path: Path | None,
    session_id: str,
    *,
    force: bool,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    steer: str = "",
    interactive: bool = False,
    model: str = "",
) -> int:
    """Resume a paused or crashed run from its snapshot.

    Args:
        config_path: The `--config` file, if any.
        session_id: The session, or "" for the newest resumable one.
        force: Resume past a head mismatch.
        tui: Open the dashboard.
        budget_overrides: The budget flags.
        sandbox_overrides: The sandbox flags.
        preset: The `--preset` name.
        steer: A steering instruction for the first safe boundary.
        interactive: Stay attached for a conversation.
        model: The `--model` override.

    Returns:
        The exit code from `app.resume.resume_task`.
    """
    return resume_task(
        config_path,
        session_id,
        frontend=session_frontend(config_path),
        force=force,
        started_at=time.time(),
        tui=tui,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        steer=steer,
        interactive=interactive,
        model=model,
    )
