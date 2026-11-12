# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Import a completed lane, carry its memory back, and stamp a ranking's outcome."""

from __future__ import annotations

import contextlib
import pathlib
import threading
from collections.abc import Sequence

from agent6 import lane_records
from agent6.app import compare as app_compare
from agent6.app import finalize, lane_watch, stamps
from agent6.app import reporter as app_reporter
from agent6.context import memory
from agent6.harness import judge, subrun
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import read as git_read


def imported_branch_note(origin: pathlib.Path, branch: str) -> str:
    """Return the branch a successful import landed, or say the lane made no commit.

    Args:
        origin: The repository the lane was imported into.
        branch: The lane's branch.

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
        # The import stands; the remaining lanes keep importing.
        return f"could not write {mpath}: {exc}"
    return None


def stamp_compare_outcomes(
    candidates: list[judge.CandidateBrief],
    outcome: app_compare.RankOutcome,
    *,
    winner: str,
    record_state: pathlib.Path,
    coordinator_dir: pathlib.Path,
    group: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Stamp the group on its coordinator and each lane's placement on the lane.

    Args:
        candidates: The ranked lanes.
        outcome: The ranking.
        winner: The lane the ranking crowns (`ranked_winner`), or "".
        record_state: The state dir the run's and its lanes' records live under.
        coordinator_dir: The manifest owner for the group's ranking and judge spend.
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
    err = stamp(coordinator_dir, comparison=comparison)
    if err is not None:
        reporter.note(f"group [{group}]: the comparison stamp failed: {err}")
    of = len(candidates)
    for rank_pos, session_id in enumerate(outcome.ranking, start=1):
        placement = sessions_manifest.PlacementStamp(
            rank=rank_pos, of=of, winner=session_id == winner, group=group
        )
        # A judged fan-out ranks run lanes.
        err = stamp(lane_watch.lane_dir(record_state, session_id, mode="run"), placement=placement)
        if err is not None:
            reporter.note(f"lane [{session_id}]: imported, but the placement stamp failed: {err}")


def ranked_winner(
    candidates: list[judge.CandidateBrief], ranking: Sequence[str], *, harness_gate: bool
) -> str:
    """Return the lane a ranking crowns, or "" when it crowns none.

    The judge's order is model output: rank 1 wins unless a gate ran red and none passed, and
    only a verified lane wins when any verified. An unverified finish counts as red only under
    a harness gate, the rule one run's exit code follows.

    Args:
        candidates: The ranked lanes.
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
    lane_state: pathlib.Path,
    origin_state: pathlib.Path,
    dest: pathlib.Path,
    *,
    lane: int,
    reporter: app_reporter.Reporter,
) -> None:
    """Carry a lane's recorded decisions and memory into the origin's store.

    The lane's state dir is torn down after import, so this runs first; files that
    changed in both stores are held back under the imported dir.

    Args:
        lane_state: The lane's state dir.
        origin_state: The origin's state dir.
        dest: The lane's imported session dir.
        lane: The lane number the note names.
        reporter: Receives the note.
    """
    finalize.print_lane_config_notes(dest, reporter=reporter, prefix=f"lane {lane}: ")
    carried, known = memory.merge_decisions(lane_state, origin_state)
    if carried or known:
        already = f", {known} already recorded" if known else ""
        reporter.note(f"lane {lane}: {carried} recorded decision(s) carried over{already}")
    held_dir = dest / "memory-held"
    merge = memory.merge_memory(lane_state, origin_state, held_dir=held_dir)
    used, read = memory.merge_use(
        lane_state, origin_state, written=(*merge.carried, *merge.updated)
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
        reporter.note(f"lane {lane}: memory {', '.join(parts)}")


def import_lane(
    res: lane_records.LaneResult, origin: pathlib.Path, lock: threading.Lock | None
) -> lane_records.Imported:
    """Land a finished lane's work in the origin; its record is there from its first event.

    Args:
        res: The lane's result.
        origin: The repository.
        lock: Serializes the import across a group, when one is shared.

    Returns:
        The lane's dir and the ref the work landed in.

    Raises:
        SubrunError: The import failed; nothing moved.
    """
    with lock if lock is not None else contextlib.nullcontext():
        manifest = sessions_manifest.read_manifest(res.session_dir)
        model_control = manifest.git_control == "model"
        branch = git_read.status(res.spec.worktree).branch if model_control else ""
        free_branch = branch and branch != "HEAD" and not git_read.branch_exists(origin, branch)
        imported = subrun.import_run(
            origin, res.spec.worktree, res.branch, res.session_dir, model_control=model_control
        )
        if free_branch:
            stamps.write_manifest(
                imported.session_dir / sessions_manifest.MANIFEST_NAME,
                manifest.model_copy(update={"run_branch": branch}),
            )
        return imported
