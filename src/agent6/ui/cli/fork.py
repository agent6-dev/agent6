# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 fork`: adapt argv, materialize the fork, then continue it over the resume path.

`--no-run` stops after `agent6.app.fork` has created the fork.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

from agent6 import kinds
from agent6.app import _setup, fork, preflight, resume
from agent6.config import Config
from agent6.sessions import id
from agent6.ui.cli import _common, run
from agent6.viewmodel import listing


def _cmd_fork(
    config_path: pathlib.Path | None,
    source_session_id: str,
    *,
    at_turn: int | None = None,
    new_session_id: str = "",
    no_run: bool = False,
    tui: bool = False,
    budget_overrides: _setup.BudgetOverrides | None = None,
    sandbox_overrides: _setup.SandboxOverrides | None = None,
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
        _common.error(
            "--steer seeds the immediate continuation, which --no-run skips."
            " Drop --no-run, or start the fork later with `agent6 resume <id> --steer ...`."
        )
        return 2
    if not no_run and not steer.strip() and at_turn is None:
        # `resume` cannot see a finished parent from the child's empty log, so refuse here.
        try:
            source = _common.resolve_or_newest_layout(pathlib.Path.cwd(), source_session_id)
        except id.SessionIdError:
            source = None
        if source is not None and listing.finished_needs_new_work(source.session_dir):
            _common.error(
                f"run {source.session_id!r} already finished (the agent called"
                " finish_session), so a fork of its last turn has nothing to do."
                ' Give the fork new work with --steer "<what to do next>",'
                " or fork an earlier turn with --at-turn N."
            )
            return 2
    frontend = run.session_frontend(config_path)

    def refuse_continuation(cfg: Config, mode: str) -> str | None:
        # The resume below would refuse the same way, after the fork existed.
        """Return why the continuation would refuse, before the fork exists."""
        return preflight.headless_approval_refusal(
            cfg,
            tui_enabled=frontend.should_spawn_tui(tui, False, mode),
            away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
            can_ask=frontend.capabilities.can_ask,
            clamped=kinds.session_kind(mode).clamps_commands,
        )

    child_id, rc = fork.create_fork(
        config_path,
        source_session_id,
        at_turn=at_turn,
        new_session_id=new_session_id,
        cwd=pathlib.Path.cwd(),
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
    return resume.resume_task(
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
