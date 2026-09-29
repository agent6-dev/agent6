# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions compare`: the advisory ranked comparison across already-run candidates.

Its own module so `sessions list` does not load the judge.
"""

from __future__ import annotations

import contextlib
import pathlib

from agent6 import git_ops, paths
from agent6.app import compare
from agent6.config import layer
from agent6.harness import judge
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import _common, _compare, sessions_cmds
from agent6.viewmodel import (
    LIVE_STATUS_WORDS,
    died_without_end,
    format,
    summarize_session_dir,
)


def _candidate_diff(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> tuple[str, bool]:
    """Return the diff a run introduced, read-only, without checking out its branch.

    A pruned branch reads from the recorded merge: its merged tip while the objects exist,
    else the commit it landed as.

    Args:
        cwd: The repo.
        manifest: The run's manifest.

    Returns:
        `(diff, from_merge)`; the diff is "" when nothing records the change, which never
        blocks the comparison.
    """
    base_sha, run_branch = manifest.base_sha, manifest.run_branch or ""
    if not base_sha:
        return "", False
    if run_branch and git_ops.branch_exists(cwd, run_branch):
        return git_ops.diff_range(cwd, base_sha, run_branch), False
    merged = manifest.merged
    if merged is None:
        return "", False
    for ref in (merged.tip, merged.sha):
        if ref and ref != sessions_manifest.NO_MERGE_COMMIT:
            try:
                return git_ops.diff_range(cwd, base_sha, ref), True
            except git_ops.GitError:
                continue
    return "", False


def _screen_candidates(
    cwd: pathlib.Path,
    resolved: list[tuple[sessions_layout.SessionLayout, sessions_manifest.SessionManifest]],
) -> tuple[list[judge.CandidateBrief], list[str]]:
    """Return briefs for the comparable runs and a note for each excluded one.

    A run without a session end has no verdict and a truncated spend, so ranking it would
    float it to first place and offer a merge of a branch still moving; it is excluded and
    named rather than silently dropped.

    Args:
        cwd: The repo.
        resolved: The runs with their manifests.

    Returns:
        The candidates and the notes to print.
    """
    candidates: list[judge.CandidateBrief] = []
    notes: list[str] = []
    for layout, manifest in resolved:
        summary = summarize_session_dir(layout.session_dir)
        if summary.status in LIVE_STATUS_WORDS:
            notes.append(
                f"note: {layout.session_id} is still {summary.status};"
                " excluded (stop it or let it finish first)"
            )
            continue
        if died_without_end(summary.status):
            notes.append(
                f"note: {layout.session_id} never finished ({summary.status});"
                " excluded from the ranking"
            )
            continue
        diff, from_merge = _candidate_diff(cwd, manifest)
        if from_merge:
            notes.append(
                f"note: {layout.session_id}'s branch is pruned; its change is read from"
                " the recorded merge"
            )
        candidates.append(
            judge.CandidateBrief(
                session_id=layout.session_id,
                task=compare.manifest_task(layout.session_dir, fallback=layout.session_id),
                diff=diff,
                verify_ok=summary.verify_ok,
                cost_usd=summary.cost_usd,
            )
        )
    return candidates, notes


def _fanout_lanes(cwd: pathlib.Path, parallel_id: str) -> tuple[str, ...]:
    """Return the lane ids of a fan-out in lane order; empty when no run names it.

    Args:
        cwd: The repo.
        parallel_id: The fan-out's id, as each lane's manifest records it.
    """
    lanes: list[tuple[int, str]] = []
    runs = _common._runs_dir(cwd)
    if runs.is_dir():
        for d in runs.iterdir():
            with contextlib.suppress(sessions_manifest.ManifestError):
                m = sessions_manifest.read_manifest(d)
                if m.parallel is not None and m.parallel.group == parallel_id:
                    lanes.append((m.parallel.lane, d.name))
    return tuple(name for _, name in sorted(lanes))


def _recorded_outcome(
    resolved: list[tuple[sessions_layout.SessionLayout, sessions_manifest.SessionManifest]],
    candidates: list[judge.CandidateBrief],
) -> compare.RankOutcome | None:
    """Return a fan-out's stamped verdict as a `RankOutcome`, for the ranking table.

    Args:
        resolved: The runs with their manifests.
        candidates: The comparable runs.

    Returns:
        The outcome, or None when any candidate lacks an auto-compare stamp (then it is judged).
    """
    ids = {c.session_id for c in candidates}
    stamps = {
        layout.session_id: manifest.compare
        for layout, manifest in resolved
        if layout.session_id in ids and manifest.compare is not None and manifest.compare.rank
    }
    if len(stamps) != len(ids):
        return None
    first = stamps[min(stamps, key=lambda sid: stamps[sid].rank)]
    return compare.RankOutcome(
        ranking=tuple(sorted(stamps, key=lambda sid: stamps[sid].rank)),
        rationale=first.rationale,
        ranked_by="judge" if first.ranked_by == "judge" else "mechanical",
        judge_cost_usd=first.judge_cost_usd,
        judge_cost_partial=first.judge_cost_partial,
    )


def _cmd_compare(
    *, session_ids: tuple[str, ...], config_path: pathlib.Path | None, rejudge: bool = False
) -> int:
    """Print an advisory ranking of two or more already-run candidates.

    The same ranked report `--parallel`'s auto-compare prints (the reviewer model as judge
    when configured, else the mechanical verify-then-cost ranking), for runs picked by hand,
    not necessarily from one fan-out or one task. Read-only: no merges, no stamps. A
    fan-out id prints the verdict it recorded; ids named one by one are judged; `rejudge`
    judges either way, and can rank differently from the stamp the listings read.

    Args:
        session_ids: Two or more run ids, or one fan-out id.
        config_path: The `--config` file, if any.
        rejudge: Judge afresh even for a fan-out with a recorded verdict.

    Returns:
        The exit code; 2 when the ids do not name two comparable runs.
    """
    cwd = pathlib.Path.cwd()
    by_fanout = False
    if len(session_ids) == 1:
        # One id is a fan-out's, comparing its lanes; anything else is one run, too few.
        lanes = _fanout_lanes(cwd, session_ids[0])
        by_fanout = bool(lanes)
        session_ids = lanes or session_ids
    if len(session_ids) < 2:
        _common.error(
            "sessions compare needs 2 or more run ids, or one --parallel fan-out id"
            f" (its lanes); got {len(session_ids)}."
        )
        return 2
    resolved: list[tuple[sessions_layout.SessionLayout, sessions_manifest.SessionManifest]] = []
    seen: set[str] = set()
    for query in session_ids:
        res = sessions_cmds._resolve_session_manifest(cwd, query)
        if isinstance(res, int):
            return res
        layout, manifest = res
        if layout.session_id in seen:
            _common.error(f"run {layout.session_id!r} was given more than once.")
            return 2
        seen.add(layout.session_id)
        resolved.append((layout, manifest))
    cfg = layer.load_effective(cwd, config_path).config

    candidates, notes = _screen_candidates(cwd, resolved)
    for note in notes:
        print(note)
    if not candidates:
        _common.error("no comparable runs; every run given is still live or never finished.")
        return 2

    merged = {
        layout.session_id: manifest.merged.into
        for layout, manifest in resolved
        if manifest.merged is not None and manifest.merged.into
    }
    recorded = _recorded_outcome(resolved, candidates) if by_fanout and not rejudge else None
    if recorded is not None:
        print(f"[agent6] the recorded verdict for {_common.plural(len(candidates), 'lane')}:")
        compare.print_ranked_candidates(candidates, recorded, merged_into=merged)
        print("\n(recorded when the fan-out ran; `--rejudge` spends a fresh judge call)")
        return 0

    reviewer = cfg.models.resolve("reviewer")
    # Advisory and stateless: only the fan-out's auto-compare stamps a manifest.
    outcome = _compare.rank(cfg, candidates, transcript_dir=paths.state_dir(cwd) / "compare")
    print(f"[agent6] comparing {len(candidates)} runs:")
    compare.print_ranked_candidates(candidates, outcome, merged_into=merged)
    # Re-judging one fan-out's lanes can contradict its stamp, which the listings read: say so.
    groups = {manifest.parallel.group if manifest.parallel else None for _, manifest in resolved}
    if outcome.ranking and len(groups) == 1 and None not in groups:
        stamped = next(
            (
                layout.session_id
                for layout, manifest in resolved
                if manifest.compare is not None and manifest.compare.winner
            ),
            None,
        )
        if stamped is not None and stamped != outcome.ranking[0]:
            print(
                f"\nnote: the recorded fan-out verdict picked {stamped}"
                f" (the {format.WINNER_GLYPH} in listings); this fresh ranking is advisory"
                " and nothing was re-stamped."
            )
    if reviewer is None:
        print(
            "\n(no reviewer model configured; ranked mechanically: verify-pass first, then"
            " lower cost)"
        )
    return 0
