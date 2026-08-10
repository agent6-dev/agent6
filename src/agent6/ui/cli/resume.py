# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 resume`: adapt argv and hand the lifecycle to
`agent6.app.resume.resume_task` with the same injected presentation seam
`agent6 run` uses (`ui.cli.run.session_frontend`)."""

from __future__ import annotations

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
) -> int:
    """Resume a paused/crashed run from its snapshot (see `app.resume`)."""
    return resume_task(
        config_path,
        session_id,
        frontend=session_frontend(),
        force=force,
        tui=tui,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        steer=steer,
        interactive=interactive,
    )
