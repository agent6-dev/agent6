# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Find a completed child's actual gate pass for the parent's final content."""

from __future__ import annotations

import pathlib

from agent6.sessions import child_dispatch, journal, layout
from agent6.viewmodel.session import log


def covering_pass(
    session_dir: pathlib.Path, tree: str, command: tuple[str, ...]
) -> journal.ChildVerification | None:
    """Find a full pass over this tree by this command, never a child's summary or branch tip.

    Args:
        session_dir: The parent's session directory.
        tree: Its final worktree content id.
        command: Its gate in force.

    Returns:
        The child and attempt to cite, or None without matching evidence.
    """
    if not tree or not command:
        return None
    try:
        records = child_dispatch.read_records(session_dir)
    except child_dispatch.DispatchRecordError:
        return None
    candidates = [
        child
        for record in records
        if record.state == "complete" and record.source != "panel"
        for child in record.children
        if child.join is not None
        and child.join.status not in ("failed", "conflict", "stopped")
        and layout.is_safe_session_id(child.session_id)
    ]
    # If several children tested identical content, cite the one whose work joined first.
    for child in sorted(candidates, key=lambda c: not (c.join and c.join.status == "joined")):
        scan = log.scan_session_log(session_dir.parent / child.session_id / layout.LOGS_NAME)
        if (
            not scan.finished
            or scan.verification != "passed"
            or scan.verify_scoped
            or scan.end_tree != tree
            or not scan.final_verify_id
            or scan.evidence_notes
        ):
            continue
        attempts = [v for v in scan.verifications if v.verify_id == scan.final_verify_id]
        if len(attempts) != 1:
            continue
        (attempt,) = attempts
        if (
            attempt.certifies(tree)
            and attempt.exit_code == 0
            and not (attempt.baseline or attempt.scoped or attempt.selected_paths)
            and attempt.command == command
        ):
            return journal.ChildVerification(
                session_id=child.session_id,
                verify_id=scan.final_verify_id,
                tree=tree,
                command=command,
            )
    return None
