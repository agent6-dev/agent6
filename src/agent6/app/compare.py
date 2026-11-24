# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rank candidates and print the ranked table.

Shared by a run's rank merge and `sessions compare`. Each of `[harness.parallel].judges` is a
reviewer child that ranks once, and the lowest sum of ranks wins; with no order from a judge
the order is the mechanical one. A run starts its judges through its dispatcher
(`harness.parallel_rank`); `rank` here starts them for a command that has no run, and waits.
The caller injects the child spawner and the in-flight status (`contextlib.nullcontext`
shows nothing), so `app` never imports `ui`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import time
from collections.abc import Callable, Mapping
from typing import Literal

from agent6 import budget as agent6_budget
from agent6 import child_records, kinds
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import parallel_panel, parallel_rank
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import format, listing
from agent6.viewmodel.session import status as session_status

# The CLI's progress indicator while the judges rank the candidates.
JudgingStatus = Callable[[], contextlib.AbstractContextManager[None]]


@dataclasses.dataclass(frozen=True, slots=True)
class RankOutcome:
    """Hold a ranking and which path produced it.

    Attributes:
        ranking: The candidate ids, best first.
        rationale: The judge's reasoning, or each judge's under its selector when several
            ranked; empty on the mechanical path.
        ranked_by: "judge" when at least one judge's ranking ordered them, else "mechanical".
        judge_cost_usd: The judges' spend, billed even for a failed call; 0.0 without a call.
        judge_cost_partial: The spend is a lower bound (a judge's model is unpriced).
    """

    ranking: tuple[str, ...]
    rationale: str
    ranked_by: Literal["judge", "mechanical"]
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


def manifest_task(session_dir: pathlib.Path, fallback: str) -> str:
    """Return the run's recorded `user_task`, else the fallback."""
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return fallback
    return manifest.user_task or fallback


def outcome_of(ranking: child_records.Ranking) -> RankOutcome:
    """Return a ranking as the table and the stamps read it."""
    return RankOutcome(
        ranking.order,
        ranking.rationale,
        ranking.ranked_by,
        ranking.judge_cost_usd,
        ranking.judge_cost_partial,
    )


def rank(
    cfg: Config,
    candidates: list[child_records.CandidateBrief],
    *,
    spawner: child_records.GroupChildSpawner | None,
    judging_status: JudgingStatus,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> RankOutcome:
    """Rank candidates best first, for a command that hosts the ranking itself.

    Each of `[harness.parallel].judges` runs as a reviewer child of the spawner's host and
    this waits for them. The lowest sum of ranks wins and a tie goes to the mechanical
    order. The mechanical order alone applies with one candidate, with no judge, with no
    spawner, or when no judge returns a usable order.

    Args:
        cfg: The resolved config.
        candidates: The candidates to rank.
        spawner: Starts the judges, each a reviewer child filed under the spawner's host;
            None ranks mechanically.
        judging_status: Shown while the judges run.
        reporter: Where a failed judge is reported.

    Returns:
        The ranking and which path produced it.
    """
    mechanical = parallel_rank.mechanical_ranking(candidates)
    judges = parallel_rank.configured_judges(cfg, log=reporter.err)
    if len(candidates) < 2 or not judges or spawner is None:
        return RankOutcome(mechanical, "", "mechanical")
    tasks = [
        dataclasses.replace(
            task,
            limits=child_records.ChildLimits(
                max_turns=cfg.review.max_turns,
                deadline_at=time.time() + parallel_panel.PANEL_BATCH_DEADLINE_S,
            ),
        )
        for task in parallel_rank.judge_tasks(
            judges, candidates, base="", max_turns=cfg.review.max_turns, tools=None
        )
    ]
    ids = [c.session_id for c in candidates]
    replies: list[parallel_rank.JudgeReply] = []
    spent, unknown = 0.0, False
    with judging_status():
        try:
            results: list[child_records.ChildResult] = [
                child.result.result() for child in spawner(tasks, "r1", at=None)
            ]
        except Exception as exc:
            # A judge that cannot start is reported: the table must not read as ranked by it.
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            reporter.err(f"the judges could not start ({detail})")
            results = []
    for judge, res in zip(judges, results, strict=False):
        reply = parallel_rank.read_reply(
            judge.selector,
            res.session_dir,
            ids,
            max_turns=cfg.review.max_turns,
            failed="" if res.ok else res.error or "its run failed",
        )
        if reply.order is None:
            reporter.err(f"judge {judge.selector} failed ({reply.said})")
        replies.append(reply)
        spent += res.usage.usd if res.usage is not None else 0.0
        unknown = unknown or res.usage is None or res.usage.usd_partial
    ranking = parallel_rank.combined(candidates, replies)
    if ranking.ranked_by == "mechanical":
        spent_s = (
            f"; judge spend {agent6_budget.format_usd(spent, partial=unknown)}"
            if spent > 0 or unknown
            else ""
        )
        reporter.err(f"no judge ranked the candidates; ranked mechanically{spent_s}")
    return dataclasses.replace(
        outcome_of(ranking), judge_cost_usd=spent, judge_cost_partial=unknown
    )


def verify_word(verification: kinds.Verification | None) -> str:
    """Return a candidate's gate verdict as the report and the journal word it."""
    return verification or "unknown"


def print_ranked_candidates(
    candidates: list[child_records.CandidateBrief],
    outcome: RankOutcome,
    *,
    rows: Mapping[str, listing.SessionSummary],
    merged_into: Mapping[str, str] | None = None,
    committed: Mapping[str, bool] | None = None,
    model_refs: Mapping[str, str] | None = None,
    merge_parents: Mapping[str, str] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Print the ranked table, the total spend and the judge's rationale.

    Each line carries the candidate's status label and cost cell as the listings spell
    them. Prints nothing when the ranking is empty.

    Args:
        candidates: The ranked candidates.
        outcome: The ranking.
        rows: Each candidate's listing row, by session id.
        merged_into: Where a candidate is already merged, by session id.
        committed: Whether a candidate's branch holds a commit, by session id; one named
            False has nothing to merge.
        model_refs: Imported work locations for model-controlled children.
        merge_parents: The target parent for a pending merge, keyed by the winner's id.
        reporter: The output channels.
    """
    if not outcome.ranking:
        return
    labels = {
        rid: format.status_label(rows[rid].status, rows[rid].detail) for rid in outcome.ranking
    }
    width = max(len(label) for label in labels.values())
    reporter.out("ranked candidates (best first):")
    for rnk, rid in enumerate(outcome.ranking, start=1):
        cost = rows[rid].usage_cell
        into = (merged_into or {}).get(rid, "")
        if into:
            merge_note = f"merged into {into}"
        elif model_refs is not None and rid in model_refs:
            merge_note = (
                f"work is on {model_refs[rid]}"
                if model_refs[rid]
                else "no imported work ref (git was model-controlled)"
            )
        elif not (committed or {}).get(rid, True):
            merge_note = "nothing to merge (no commit)"
        elif parent := (merge_parents or {}).get(rid):
            merge_note = f"selected for {parent}'s chain (merge pending)"
        else:
            merge_note = f"merge with: agent6 sessions merge {rid}"
        reporter.out(f"  {rnk}. {rid}  {labels[rid]:<{width}}  {cost}   {merge_note}")
    if len(candidates) > 1:
        cand_total = sum(c.cost_usd for c in candidates)
        cand_rows = [rows[c.session_id] for c in candidates if c.session_id in rows]
        cand_partial = any(row.usd_partial for row in cand_rows)
        plan_total = sum(row.plan_observed_points for row in cand_rows)
        plan_read = max((row.plan_used_percent for row in cand_rows), default=0.0)
        candidates_cell = session_status.usage_cell(
            cand_total,
            partial=cand_partial,
            plan_observed_points=plan_total,
            plan_used_percent=plan_read,
        )
        judge = outcome.judge_cost_usd
        partial = outcome.judge_cost_partial
        line = f"total: candidates {candidates_cell}"
        if judge > 0 or partial:
            line += f" + judge {agent6_budget.format_usd(judge, partial=partial)}"
            if plan_total <= 0 and plan_read <= 0:
                # Points and dollars do not add; the sum is for an all-dollar line.
                lower = partial or cand_partial
                line += f" = {agent6_budget.format_usd(cand_total + judge, partial=lower)}"
        reporter.out(line)
    if outcome.rationale:
        reporter.out(f"\njudge: {outcome.rationale}")
