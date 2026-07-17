# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 fork`: adapt argv, materialize the fork (`agent6.app.fork`), then
(unless `--no-run`) continue the new run from its forked turn over the resume
path."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.app._setup import BudgetOverrides
from agent6.app.fork import create_fork
from agent6.ui.cli.resume import _cmd_resume


def _cmd_fork(
    config_path: Path | None,
    source_run_id: str,
    *,
    at_turn: int | None = None,
    new_run_id: str = "",
    no_run: bool = False,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
) -> int:
    """Create a new run cloned from *source_run_id* at checkpoint *at_turn*.

    Default: fork from the latest checkpoint and immediately continue the new run
    from that turn (resume-like). ``--no-run`` just creates the fork dir.
    """
    child_id, rc = create_fork(
        config_path, source_run_id, at_turn=at_turn, new_run_id=new_run_id, cwd=Path.cwd()
    )
    if rc != 0:
        return rc

    if no_run:
        print(f"[agent6] fork created (not started): {child_id}", file=sys.stderr)
        print(f"  resume it with: agent6 resume {child_id}", file=sys.stderr)
        return 0

    # Continue the new run from turn N by reusing the resume path. The fork just
    # cloned the checkpoint (its head_sha) and cut agent6/<child> at that same
    # sha, so the resume head guard passes by construction; force stays off so a
    # real mismatch (a broken fork) still refuses.
    return _cmd_resume(
        config_path,
        child_id,
        force=False,
        tui=tui,
        budget_overrides=budget_overrides,
    )
