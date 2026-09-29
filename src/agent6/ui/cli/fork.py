# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 fork`: adapt argv, materialize the fork, then continue it over the resume path.

`--no-run` stops after `agent6.app.fork` has created the fork.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from agent6.app._setup import BudgetOverrides, SandboxOverrides
from agent6.app.fork import create_fork
from agent6.app.preflight import headless_approval_refusal
from agent6.app.resume import resume_task
from agent6.config import Config
from agent6.kinds import session_kind
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import error, resolve_or_newest_layout
from agent6.ui.cli.run import session_frontend
from agent6.viewmodel.listing import finished_needs_new_work


def _cmd_fork(
    config_path: Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    no_run: bool = False,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    steer: str = "",
) -> int:
    """Create a session cloned from a source at a checkpoint, then continue it.

    The default forks the latest checkpoint and continues from that turn; `--steer` seeds
    the new direction at the first safe boundary; `--no-run` only creates the fork dir.

    Args:
        config_path: The `--config` file, if any.
        source_session_id: The session to fork, or "" for the newest.
        at_turn: The checkpoint turn; None takes the latest.
        new_session_id: The fork's id; "" allocates one.
        no_run: Create the fork without continuing it.
        tui: Open the dashboard for the continuation.
        budget_overrides: The budget flags.
        sandbox_overrides: The sandbox flags.
        steer: A steering instruction for the continuation's first safe boundary.

    Returns:
        The exit code; 2 when the flags conflict or the source already finished.
    """
    if no_run and steer.strip():
        error(
            "--steer seeds the immediate continuation, which --no-run skips."
            " Drop --no-run, or start the fork later with `agent6 resume <id> --steer ...`."
        )
        return 2
    if not no_run and not steer.strip() and at_turn is None:
        # `resume` cannot see a finished parent from the child's empty log, so refuse here.
        try:
            source = resolve_or_newest_layout(Path.cwd(), source_session_id)
        except SessionIdError:
            source = None
        if source is not None and finished_needs_new_work(source.session_dir):
            error(
                f"run {source.session_id!r} already finished (the agent called"
                " finish_session), so a fork of its last turn has nothing to do."
                ' Give the fork new work with --steer "<what to do next>",'
                " or fork an earlier turn with --at-turn N."
            )
            return 2
    frontend = session_frontend(config_path)

    def refuse_continuation(cfg: Config, mode: str) -> str | None:
        # The resume below would refuse the same way, after the fork existed.
        """Return why the continuation would refuse, before the fork exists."""
        return headless_approval_refusal(
            cfg,
            tui_enabled=frontend.should_spawn_tui(tui, False, mode),
            away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
            can_ask=frontend.capabilities.can_ask,
            clamped=session_kind(mode).clamps_commands,
        )

    child_id, rc = create_fork(
        config_path,
        source_session_id,
        at_turn=at_turn,
        new_session_id=new_session_id,
        cwd=Path.cwd(),
        sandbox_overrides=None if no_run else sandbox_overrides,
        refuse_continuation=None if no_run else refuse_continuation,
    )
    if rc != 0:
        return rc

    if no_run:
        print(f"[agent6] fork created (not started): {child_id}", file=sys.stderr)
        print(f"  resume it with: agent6 resume {child_id}", file=sys.stderr)
        return 0

    # The fork's branch sits at the checkpoint's sha, so the head guard passes; force stays off.
    return resume_task(
        config_path,
        child_id,
        frontend=frontend,
        force=False,
        started_at=time.time(),
        tui=tui,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        steer=steer,
    )
