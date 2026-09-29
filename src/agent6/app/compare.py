# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rank candidates and print the ranked table.

Shared by the `--parallel` auto-compare and `sessions compare`. The judge is the reviewer model
when one resolves, else the mechanical ranking. The caller injects the provider builder and
the in-flight status (`contextlib.nullcontext` shows nothing), so `app` never imports `ui`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Callable, Mapping
from typing import Literal

from agent6 import budget as agent6_budget
from agent6.app import _setup
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import judge as harness_judge
from agent6.providers import Provider, ProviderError, TranscriptSink
from agent6.sessions import manifest as sessions_manifest

# Builds the reviewer provider the judge call uses.
BuildProvider = Callable[[Config, TranscriptSink, agent6_budget.BudgetTracker], Provider]
# Shown around the judge call, which is silent for 50 to 60 seconds.
JudgingStatus = Callable[[], contextlib.AbstractContextManager[None]]


@dataclasses.dataclass(frozen=True, slots=True)
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


def manifest_task(session_dir: pathlib.Path, fallback: str) -> str:
    """Return the run's recorded `user_task`, else the fallback."""
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return fallback
    return manifest.user_task or fallback


def rank(
    cfg: Config,
    candidates: list[harness_judge.CandidateBrief],
    *,
    transcript_dir: pathlib.Path,
    build_provider: BuildProvider,
    judging_status: JudgingStatus,
    max_usd: float | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
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
        budget = _setup.budget_tracker(cfg, max_usd=max_usd)
        try:
            provider: Provider = build_provider(cfg, sink, budget)
            with judging_status():
                verdict = harness_judge.compare(provider, reviewer.model, candidates)
            spent, unknown = budget.estimate_usd()
            return RankOutcome(verdict.ranking, verdict.rationale, "judge", spent, unknown)
        except (ProviderError, harness_judge.JudgeError) as exc:
            # A failed judge is reported, with its spend: the table must not read as judged.
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            spent, unknown = budget.estimate_usd()
            spent_s = (
                f"; judge spend {agent6_budget.format_usd(spent, partial=unknown)}"
                if spent > 0 or unknown
                else ""
            )
            reporter.err(f"judge failed ({detail}); ranked mechanically{spent_s}")
            return RankOutcome(
                harness_judge.mechanical_ranking(candidates), "", "mechanical", spent, unknown
            )
    return RankOutcome(harness_judge.mechanical_ranking(candidates), "", "mechanical")


def verify_word(verify_ok: bool | None) -> str:
    """Return a candidate's gate verdict as the report and the journal word it."""
    return "passed" if verify_ok else "failed" if verify_ok is False else "no-verify"


def print_ranked_candidates(
    candidates: list[harness_judge.CandidateBrief],
    outcome: RankOutcome,
    *,
    merged_into: Mapping[str, str] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
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
        reporter.out(
            f"  {rnk}. {rid}  {verify:<9} {agent6_budget.format_usd(c.cost_usd)}   {landing}"
        )
    if len(candidates) > 1:
        cand_total = sum(c.cost_usd for c in candidates)
        judge = outcome.judge_cost_usd
        partial = outcome.judge_cost_partial
        if judge > 0 or partial:
            reporter.out(
                f"total: candidates {agent6_budget.format_usd(cand_total)}"
                f" + judge {agent6_budget.format_usd(judge, partial=partial)}"
                f" = {agent6_budget.format_usd(cand_total + judge, partial=partial)}"
            )
        else:
            reporter.out(f"total: candidates {agent6_budget.format_usd(cand_total)}")
    if outcome.rationale:
        reporter.out(f"\njudge: {outcome.rationale}")
