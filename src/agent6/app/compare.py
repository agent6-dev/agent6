# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rank candidates and print the ranked table.

Shared by the `--parallel` auto-compare and `sessions compare`. The judge is the reviewer model
when one resolves, else the mechanical ranking. The caller injects the provider builder and
the in-flight status (`contextlib.nullcontext` shows nothing), so `app` never imports `ui`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import budget_tracker
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.harness.judge import CandidateBrief, JudgeError, compare, mechanical_ranking
from agent6.providers import Provider, ProviderError, TranscriptSink
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.viewmodel.format import format_usd

# Builds the reviewer provider the judge call uses.
BuildProvider = Callable[[Config, TranscriptSink, BudgetTracker], Provider]
# Shown around the judge call, which is silent for 50 to 60 seconds.
JudgingStatus = Callable[[], AbstractContextManager[None]]


@dataclass(frozen=True, slots=True)
class RankOutcome:
    """Hold a ranking and which path produced it.

    Attributes:
        ranking: The candidate ids, best first.
        rationale: The judge's reasoning; empty on the mechanical path.
        ranked_by: "judge" when the reviewer model ordered them, else "mechanical".
        judge_cost_usd: The judge call's spend, billed even when it failed; 0.0 without a call.
        judge_cost_partial: The spend is a lower bound (the reviewer model is unpriced).
    """

    ranking: tuple[str, ...]
    rationale: str
    ranked_by: Literal["judge", "mechanical"]
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


def manifest_task(session_dir: Path, fallback: str) -> str:
    """Return the run's recorded `user_task`, else the fallback."""
    try:
        manifest = read_manifest(session_dir)
    except ManifestError:
        return fallback
    return manifest.user_task or fallback


def rank(
    cfg: Config,
    candidates: list[CandidateBrief],
    *,
    transcript_dir: Path,
    build_provider: BuildProvider,
    judging_status: JudgingStatus,
    max_usd: float | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> RankOutcome:
    """Rank candidates best first.

    The reviewer model judges when it resolves; the mechanical ranking applies when it is
    unset, there is one candidate, or the judge call fails.

    Args:
        cfg: The resolved config.
        candidates: The candidates to rank.
        transcript_dir: Where the judge call's transcript is written.
        build_provider: Builds the reviewer provider.
        judging_status: Shown around the judge call.
        max_usd: Caps the judge like one more lane; None uses the config budget.
        reporter: Where a failed judge is reported.

    Returns:
        The ranking and which path produced it.
    """
    reviewer = cfg.models.resolve("reviewer")
    if len(candidates) > 1 and reviewer is not None:
        sink = TranscriptSink(transcript_dir)
        budget = budget_tracker(cfg, max_usd=max_usd)
        try:
            provider: Provider = build_provider(cfg, sink, budget)
            with judging_status():
                verdict = compare(provider, reviewer.model, candidates)
            spent, unknown = budget.estimate_usd()
            return RankOutcome(verdict.ranking, verdict.rationale, "judge", spent, unknown)
        except (ProviderError, JudgeError) as exc:
            # A failed judge is reported, with its spend: the table must not read as judged.
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            spent, unknown = budget.estimate_usd()
            spent_s = (
                f"; judge spend {format_usd(spent, partial=unknown)}"
                if spent > 0 or unknown
                else ""
            )
            reporter.err(f"judge failed ({detail}); ranked mechanically{spent_s}")
            return RankOutcome(mechanical_ranking(candidates), "", "mechanical", spent, unknown)
    return RankOutcome(mechanical_ranking(candidates), "", "mechanical")


def verify_word(verify_ok: bool | None) -> str:
    """Return a candidate's gate verdict as the report and the journal word it."""
    return "passed" if verify_ok else "failed" if verify_ok is False else "no-verify"


def print_ranked_candidates(
    candidates: list[CandidateBrief],
    outcome: RankOutcome,
    *,
    merged_into: Mapping[str, str] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print the ranked table, the total spend and the judge's rationale.

    Prints nothing when the ranking is empty.

    Args:
        candidates: The ranked candidates.
        outcome: The ranking.
        merged_into: Where a candidate is already merged, by session id.
        reporter: The output channels.
    """
    if not outcome.ranking:
        return
    by_id = {c.session_id: c for c in candidates}
    reporter.out("ranked candidates (best first):")
    for rnk, rid in enumerate(outcome.ranking, start=1):
        c = by_id[rid]
        verify = verify_word(c.verify_ok)
        into = (merged_into or {}).get(rid, "")
        landing = f"merged into {into}" if into else f"merge with: agent6 sessions merge {rid}"
        reporter.out(f"  {rnk}. {rid}  {verify:<9} {format_usd(c.cost_usd)}   {landing}")
    if len(candidates) > 1:
        cand_total = sum(c.cost_usd for c in candidates)
        judge = outcome.judge_cost_usd
        partial = outcome.judge_cost_partial
        if judge > 0 or partial:
            reporter.out(
                f"total: candidates {format_usd(cand_total)}"
                f" + judge {format_usd(judge, partial=partial)}"
                f" = {format_usd(cand_total + judge, partial=partial)}"
            )
        else:
            reporter.out(f"total: candidates {format_usd(cand_total)}")
    if outcome.rationale:
        reporter.out(f"\njudge: {outcome.rationale}")
