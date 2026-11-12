# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Import a completed child, carry its memory back, and stamp a ranking's outcome."""

from __future__ import annotations

import contextlib
import pathlib
import threading
from collections.abc import Sequence

from agent6 import child_records
from agent6.app import child_wait, finalize, stamps
from agent6.app import compare as app_compare
from agent6.app import reporter as app_reporter
from agent6.context import memory
from agent6.harness import child_repo, judge
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read


def imported_branch_note(origin: pathlib.Path, branch: str) -> str:
    """Return the branch a successful import landed, or say the child made no commit.

    Args:
        origin: The repository the child was imported into.
        branch: The child's branch.

    Returns:
        "branch imported as <branch>", or "no branch (the run made no commit)" when the
        origin has no such branch.
    """
    if git_read.branch_exists(origin, branch):
        return f"branch imported as {branch}"
    return "no branch (the run made no commit)"


def stamp(session_dir: pathlib.Path, **updates: object) -> str | None:
    """Apply field updates to a session's manifest.

    Args:
        session_dir: The session's directory.
        **updates: The manifest fields to set.

    Returns:
        The error when the manifest cannot be read or written, else None; the import
        stands either way.
    """
    mpath = session_dir / sessions_manifest.MANIFEST_NAME
    try:
        m = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError as exc:
        return f"could not read {mpath}: {exc}"
    try:
        stamps.write_manifest(mpath, m.model_copy(update=updates))
    except (OSError, sessions_manifest.ManifestError) as exc:
        # The import stands; the remaining children keep importing.
        return f"could not write {mpath}: {exc}"
    return None


def stamp_compare_outcomes(
    candidates: list[judge.CandidateBrief],
    outcome: app_compare.RankOutcome,
    *,
    winner: str,
    record_state: pathlib.Path,
    parent_dir: pathlib.Path,
    group: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Stamp the group on its parent and each child's placement on the child.

    Args:
        candidates: The ranked children.
        outcome: The ranking.
        winner: The child the ranking crowns (`ranked_winner`), or "".
        record_state: The state dir the run's and its children's records live under.
        parent_dir: The manifest owner for the group's ranking and judge spend.
        group: The comparison group id.
        reporter: Receives a note per stamp failure; the others still land.
    """
    comparison = sessions_manifest.ComparisonStamp(
        group=group,
        ranking=outcome.ranking,
        ranked_by=outcome.ranked_by,
        rationale=outcome.rationale[:2000] if outcome.ranked_by == "judge" else "",
        judge_cost_usd=outcome.judge_cost_usd,
        judge_cost_partial=outcome.judge_cost_partial,
    )
    err = stamp(parent_dir, comparison=comparison)
    if err is not None:
        reporter.note(f"group [{group}]: the comparison stamp failed: {err}")
    of = len(candidates)
    for rank_pos, session_id in enumerate(outcome.ranking, start=1):
        placement = sessions_manifest.PlacementStamp(
            rank=rank_pos, of=of, winner=session_id == winner, group=group
        )
        # A judged group ranks run children.
        err = stamp(child_wait.child_dir(record_state, session_id, mode="run"), placement=placement)
        if err is not None:
            reporter.note(f"child [{session_id}]: imported, but the placement stamp failed: {err}")


def ranked_winner(
    candidates: list[judge.CandidateBrief], ranking: Sequence[str], *, harness_gate: bool
) -> str:
    """Return the child a ranking crowns, or "" when it crowns none.

    The judge's order is model output: rank 1 wins unless a gate ran red and none passed, and
    only a verified child wins when any verified. An unverified finish counts as red only under
    a harness gate, the rule one run's exit code follows.

    Args:
        candidates: The ranked children.
        ranking: Their ids, best first.
        harness_gate: Whether the harness runs the gate (`harness.verify.when` is not `never`).

    Returns:
        The winner's session id, or "".
    """
    if not ranking:
        return ""
    first = next(c for c in candidates if c.session_id == ranking[0])
    verdicts = {c.verification for c in candidates}
    if "passed" in verdicts:
        return first.session_id if first.verification == "passed" else ""
    red = {"failed", "unverified"} if harness_gate else {"failed"}
    return "" if verdicts & red else first.session_id


def carry_back(
    child_state: pathlib.Path,
    origin_state: pathlib.Path,
    dest: pathlib.Path,
    *,
    number: int,
    reporter: app_reporter.Reporter,
) -> None:
    """Carry a child's recorded decisions and memory into the origin's store.

    The child's state dir is torn down after import, so this runs first; files that
    changed in both stores are held back under the imported dir.

    Args:
        child_state: The child's state dir.
        origin_state: The origin's state dir.
        dest: The child's imported session dir.
        number: The child number the note names.
        reporter: Receives the note.
    """
    finalize.print_child_config_notes(dest, reporter=reporter, prefix=f"child {number}: ")
    carried, known = memory.merge_decisions(child_state, origin_state)
    if carried or known:
        already = f", {known} already recorded" if known else ""
        reporter.note(f"child {number}: {carried} recorded decision(s) carried over{already}")
    held_dir = dest / "memory-held"
    merge = memory.merge_memory(child_state, origin_state, held_dir=held_dir)
    used, read = memory.merge_use(
        child_state, origin_state, written=(*merge.carried, *merge.updated)
    )
    parts = [
        f"{len(names)} {word}"
        for word, names in (
            ("carried", merge.carried),
            ("updated", merge.updated),
            ("deleted", merge.deleted),
        )
        if names
    ]
    if merge.held:
        kept = f", kept at {held_dir}" if held_dir.is_dir() else ""
        parts.append(
            f"held back (changed here too, or the name is taken): {', '.join(merge.held)}{kept}"
        )
    if used or read:
        parts.append(f"use record carried for {max(used, read)}")
    if parts:
        reporter.note(f"child {number}: memory {', '.join(parts)}")


def import_child(
    res: child_records.ChildResult, origin: pathlib.Path, lock: threading.Lock | None
) -> child_records.Imported:
    """Land a finished child's work in the origin; its record is there from its first event.

    Args:
        res: The child's result.
        origin: The repository.
        lock: Serializes the import across a group, when one is shared.

    Returns:
        The child's dir and the ref the work landed in.

    Raises:
        ChildRepoError: The import failed; nothing moved.
    """
    with lock if lock is not None else contextlib.nullcontext():
        manifest = sessions_manifest.read_manifest(res.session_dir)
        model_control = manifest.git_control == "model"
        branch = git_read.status(res.spec.worktree).branch if model_control else ""
        free_branch = branch and branch != "HEAD" and not git_read.branch_exists(origin, branch)
        imported = child_repo.import_run(
            origin, res.spec.worktree, res.branch, res.session_dir, model_control=model_control
        )
        if free_branch:
            stamps.write_manifest(
                imported.session_dir / sessions_manifest.MANIFEST_NAME,
                manifest.model_copy(update={"run_branch": branch}),
            )
        return imported
