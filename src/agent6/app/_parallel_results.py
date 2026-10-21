# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Import completed lanes and report their comparison outcomes."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import threading
from collections.abc import Mapping

from agent6 import budget as agent6_budget
from agent6 import git_ops, memory, paths
from agent6.app import (
    _lane_watch,
    finalize,
    stamps,
)
from agent6.app import compare as app_compare
from agent6.app import reporter as app_reporter
from agent6.harness import judge, subrun
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.viewmodel import _session_log, format, listing, produced_result, summarize_session_dir


def imported_branch_note(origin: pathlib.Path, branch: str) -> str:
    """Return the branch a successful import landed, or say the lane made no commit.

    Args:
        origin: The repository the lane was imported into.
        branch: The lane's branch.

    Returns:
        "branch imported as <branch>", or "no branch (the run made no commit)" when the
        origin has no such branch.
    """
    if git_ops.branch_exists(origin, branch):
        return f"branch imported as {branch}"
    return "no branch (the run made no commit)"


def stamp(session_dir: pathlib.Path, **updates: object) -> str | None:
    """Apply field updates to an imported lane's manifest.

    Args:
        session_dir: The lane's imported dir.
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
    origin_state: pathlib.Path,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Stamp the auto-compare outcome into each ranked lane's manifest.

    The rationale and judge cost describe the whole group's ranking and land on
    every lane. Rank 1 is the winner unless every gate ran red, and, when any lane
    verified green, only if it is one: the judge's ranking is untrusted.

    Args:
        candidates: The ranked lanes.
        outcome: The ranking.
        origin_state: The origin's state dir.
        reporter: Receives a note per stamp failure; the others still land.
    """
    of = len(candidates)
    text = outcome.rationale[:2000] if outcome.ranked_by == "judge" else ""
    crown = fanout_exit_code(candidates) != finalize.EXIT_VERIFY_FAILED
    any_verified = any(c.verify_ok is True for c in candidates)
    by_id = {c.session_id: c for c in candidates}
    for rank_pos, session_id in enumerate(outcome.ranking, start=1):
        winner = (
            rank_pos == 1 and crown and (not any_verified or by_id[session_id].verify_ok is True)
        )
        compare = sessions_manifest.CompareStamp(
            rank=rank_pos,
            of=of,
            winner=winner,
            ranked_by=outcome.ranked_by,
            rationale=text,
            judge_cost_usd=outcome.judge_cost_usd,
            judge_cost_partial=outcome.judge_cost_partial,
        )
        err = stamp(_lane_watch.lane_link(origin_state, session_id), compare=compare)
        if err is not None:
            reporter.note(f"lane [{session_id}]: imported, but the compare stamp failed: {err}")


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
    res: subrun.LaneResult,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    lock: threading.Lock | None,
) -> pathlib.Path:
    """Land a finished lane's session dir in the origin.

    The live symlink drops so the real dir can take its place, and comes back
    when the import refuses.

    Args:
        res: The lane's result.
        origin: The repository.
        origin_state: The origin's state dir.
        lock: Serializes the import across a group, when one is shared.

    Returns:
        The imported dir.

    Raises:
        SubrunError: The import failed; nothing moved.
    """
    link = _lane_watch.lane_link(origin_state, res.spec.session_id)
    had_link = link.is_symlink()
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    try:
        with lock if lock is not None else contextlib.nullcontext():
            return subrun.import_run(
                origin, res.spec.workdir, res.branch, res.session_dir, origin_state
            )
    except subrun.SubrunError:
        if had_link:
            _lane_watch.symlink_lane(origin_state, res)
        raise


@dataclasses.dataclass(frozen=True, slots=True)
class ImportedLanes:
    """The imported candidates, failures, cleanup specs and all-lane budget outcome."""

    candidates: list[judge.CandidateBrief]
    rows: dict[str, listing.SessionSummary]
    failed: list[tuple[subrun.LaneResult, str]]
    imported: list[subrun.LaneSpec]
    all_budget_exhausted: bool
    plan_lines: list[str]


def import_lanes(
    results: list[subrun.LaneResult],
    *,
    origin: pathlib.Path,
    origin_state: pathlib.Path,
    base_sha: str,
    task: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> ImportedLanes:
    """Import each finished lane into the origin and build the candidate briefs.

    A lane that failed to start, is still running or refused its import is recorded
    as failed and keeps its clone, state and symlink; an imported lane without a
    result is recorded as failed too, its work safe in the origin.

    Args:
        results: The spawned lanes.
        origin: The repository.
        origin_state: The origin's state dir.
        base_sha: The commit the candidate diffs are taken from.
        task: The fan-out's task, for a candidate whose manifest names none.
        reporter: Receives the notes.

    Returns:
        The candidates, listing rows, failures, cleanup specs and whether every lane
        ended at its budget and imported successfully.
    """
    candidates: list[judge.CandidateBrief] = []
    rows: dict[str, listing.SessionSummary] = {}
    budget_stops = 0
    failed: list[tuple[subrun.LaneResult, str]] = []
    imported: list[subrun.LaneSpec] = []
    # A reading is account-wide, so each provider's newest one among the lanes is the fan-out's.
    readings: dict[str, tuple[float, agent6_budget.PlanUsage]] = {}
    for res in results:
        if not res.ok:
            failed.append((res, f"failed to start: {res.error}"))
            continue
        if process.worker_is_alive(res.session_dir):
            failed.append(
                (
                    res,
                    "still running; left in place"
                    f" (watch: agent6 attach {res.spec.session_id};"
                    f" stop: agent6 stop {res.spec.session_id})",
                )
            )
            continue
        try:
            dest = import_lane(res, origin, origin_state, None)
        except subrun.SubrunError as exc:
            failed.append((res, str(exc)))
            continue
        landed = dataclasses.replace(res, session_dir=dest)
        imported.append(res.spec)
        carry_back(
            paths.state_dir(res.spec.workdir),
            origin_state,
            dest,
            lane=res.spec.lane,
            reporter=reporter,
        )
        summary = summarize_session_dir(dest)
        scan = _session_log.scan_session_log(dest / sessions_layout.LOGS_NAME)
        budget_stops += scan.end_reason == "budget_exhausted"
        if scan.plan_usage is not None and (provider := lane_provider(dest)):
            seen = (scan.plan_usage_at, scan.plan_usage)
            readings[provider] = max(readings.get(provider, seen), seen, key=lambda r: r[0])
        if not produced_result(summary.status):
            # Only a deliberate end is rankable work.
            failed.append(
                (
                    landed,
                    f"no result ({format.status_label(summary.status, summary.reason)});"
                    f" {imported_branch_note(origin, res.branch)}, not ranked",
                )
            )
            continue
        if scan.setup_output:
            reporter.note(f"lane {res.spec.lane} [{res.spec.session_id}]: {scan.setup_output}")
        rows[res.spec.session_id] = summary
        candidates.append(
            judge.CandidateBrief(
                session_id=res.spec.session_id,
                task=app_compare.manifest_task(dest, task),
                diff=git_ops.diff_since(res.spec.workdir, base_sha),
                verify_ok=summary.verify_ok,
                cost_usd=summary.cost_usd,
            )
        )
    return ImportedLanes(
        candidates,
        rows,
        failed,
        imported,
        all_budget_exhausted=bool(results) and budget_stops == len(results),
        plan_lines=[
            agent6_budget.format_plan_reading(provider, usage)
            for provider, (_at, usage) in sorted(readings.items())
        ],
    )


def lane_provider(session_dir: pathlib.Path) -> str:
    """Return the provider entry that drove a lane, "" when its manifest names none."""
    try:
        driver = sessions_manifest.read_manifest(session_dir).models.driver
    except sessions_manifest.ManifestError:
        return ""
    return driver.provider if driver is not None else ""


def fanout_exit_code(
    candidates: list[judge.CandidateBrief], *, all_budget_exhausted: bool = False
) -> int:
    """Map the lanes' gate verdicts and budget stops to the fan-out's exit code.

    Args:
        candidates: The rankable lanes.
        all_budget_exhausted: Every lane ended at its budget and imported successfully.

    Returns:
        3 every lane ended at its budget, 1 nothing rankable otherwise, 0 a lane
        verified green or none had a gate, 4 gates ran and none passed.
    """
    if not candidates:
        return 3 if all_budget_exhausted else 1
    if any(c.verify_ok is True for c in candidates):
        return 0
    return finalize.EXIT_VERIFY_FAILED if any(c.verify_ok is False for c in candidates) else 0


def print_report(
    candidates: list[judge.CandidateBrief],
    outcome: app_compare.RankOutcome,
    failed: list[tuple[subrun.LaneResult, str]],
    *,
    rows: Mapping[str, listing.SessionSummary],
    committed: Mapping[str, bool],
    fanout_id: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Print the ranked candidate table, a merge line per candidate and the failed lanes.

    Args:
        candidates: The ranked lanes.
        outcome: The ranking.
        failed: The failed lanes with their reasons.
        rows: Each candidate's listing row, by session id.
        committed: Whether each lane's import landed a branch, by session id.
        fanout_id: The fan-out's session id.
        reporter: Receives the lines.
    """
    reporter.out(
        f"\n[agent6] parallel fan-out {fanout_id} complete: {len(candidates)} candidate(s)"
    )
    app_compare.print_ranked_candidates(
        candidates, outcome, rows=rows, committed=committed, reporter=reporter
    )
    if failed:
        reporter.out("\nfailed lanes:")
        for res, err in failed:
            reporter.out(f"  - lane {res.spec.lane} [{res.spec.session_id}]: {err}")
            scan = _session_log.scan_session_log(res.session_dir / sessions_layout.LOGS_NAME)
            for line in scan.setup_output.splitlines():
                reporter.out(f"    {line}")
            if scan.end_reason == "budget_exhausted":
                reporter.out(
                    "    raise the cap on the fork (--max-usd, --max-percent or"
                    " --max-tokens-fallback), then continue with:"
                    f" agent6 fork {res.spec.session_id}"
                )
            kept = [p for p in (res.spec.workdir, res.session_dir) if p.exists()]
            if kept:
                reporter.out(f"    kept: {', '.join(str(p) for p in kept)}")


def fanout_end(rc: int, candidates: list[judge.CandidateBrief]) -> tuple[str, bool | None]:
    """Map the fan-out's exit code to its `session.end` reason and gate verdict.

    Args:
        rc: The exit code.
        candidates: The rankable lanes.

    Returns:
        The reason and `all_passed`: True when a lane's gate went green, None when
        no lane had a gate.
    """
    if rc == 130:
        return "interrupted", False
    if rc == 3:
        return "budget_exhausted", False
    if rc == 0:
        return "lanes_ranked", True if any(c.verify_ok for c in candidates) else None
    return ("no_lane_passed" if rc == finalize.EXIT_VERIFY_FAILED else "no_lane_result"), False
