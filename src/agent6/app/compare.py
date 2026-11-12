# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rank candidates and print the ranked table.

Shared by the `--parallel` auto-compare and `sessions compare`. Each of
`[harness.parallel].judges` ranks once and the lowest sum of ranks wins; with no ranking from
a judge the order is the mechanical one. The caller injects the provider builder and the
in-flight status (`contextlib.nullcontext` shows nothing), so `app` never imports `ui`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Callable, Mapping
from typing import Literal

from agent6 import budget as agent6_budget
from agent6 import kinds
from agent6.app import reporter as app_reporter
from agent6.app import role_providers
from agent6.app.session import setup as session_setup
from agent6.config import Config, ConfigError
from agent6.harness import judge as harness_judge
from agent6.providers import Provider, ProviderError, TranscriptSink
from agent6.sessions import event_log
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import format, listing
from agent6.viewmodel.session import status as session_status

# Builds the reviewer provider a judge call uses; `rank` sets that role to each judge in turn.
BuildProvider = Callable[
    [Config, TranscriptSink, agent6_budget.BudgetTracker, event_log.EventSink | None], Provider
]
# The CLI's progress indicator while the judge ranks the candidates.
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


def rank(
    cfg: Config,
    candidates: list[harness_judge.CandidateBrief],
    *,
    transcript_dir: pathlib.Path,
    build_provider: BuildProvider,
    judging_status: JudgingStatus,
    max_usd: float | None = None,
    events: event_log.EventSink | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> RankOutcome:
    """Rank candidates best first.

    Each of `[harness.parallel].judges` ranks once, one after another under one budget.
    The lowest sum of ranks wins and a tie goes to the mechanical order. The mechanical
    order alone applies with one candidate, with no judge, or when no judge returns a ranking.

    Args:
        cfg: The resolved config.
        candidates: The candidates to rank.
        transcript_dir: Where the judges' transcripts are written.
        build_provider: Builds the reviewer provider.
        judging_status: Shown around the judge calls.
        max_usd: Caps the judging like one more lane; None uses the config budget.
        events: The coordinator's journal; when present, a judge streams role events into it.
        reporter: Where a failed judge is reported.

    Returns:
        The ranking and which path produced it.
    """
    mechanical = harness_judge.mechanical_ranking(candidates)
    # A role with no model route judges nothing: the way a config has no judge by default.
    unrouted = {role for role in kinds.MODEL_ROLES if cfg.models.resolve(role) is None}
    selectors = [sel for sel in cfg.harness.parallel.judges if sel not in unrouted]
    if len(candidates) < 2 or not selectors:
        return RankOutcome(mechanical, "", "mechanical")
    sink = TranscriptSink(transcript_dir, objects_dir=transcript_dir / "objects")
    budget = session_setup.budget_tracker(cfg, max_usd=max_usd)
    verdicts: list[tuple[str, harness_judge.CompareVerdict]] = []
    with judging_status():
        for selector in selectors:
            try:
                # The judge sits in the reviewer role, the one `build_provider` builds.
                seated = cfg.with_model_settings("reviewer", cfg.lane_settings(selector))
                verdicts.append(
                    (
                        selector,
                        _judge_once(seated, candidates, sink, budget, events, build_provider),
                    )
                )
            except (ConfigError, ProviderError, harness_judge.JudgeError) as exc:
                # A failed judge is reported: the table must not read as ranked by it.
                detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
                reporter.err(f"judge {selector} failed ({detail})")
    spent, unknown = budget.estimate_usd()
    if not verdicts:
        spent_s = (
            f"; judge spend {agent6_budget.format_usd(spent, partial=unknown)}"
            if spent > 0 or unknown
            else ""
        )
        reporter.err(f"no judge ranked the candidates; ranked mechanically{spent_s}")
        return RankOutcome(mechanical, "", "mechanical", spent, unknown)
    if len(verdicts) == 1:
        return RankOutcome(
            verdicts[0][1].ranking, verdicts[0][1].rationale, "judge", spent, unknown
        )
    # A candidate's place in each ranking, summed; the mechanical order settles a tie.
    total = {
        rid: sum(verdict.ranking.index(rid) for _sel, verdict in verdicts) for rid in mechanical
    }
    ranking = tuple(sorted(mechanical, key=lambda rid: (total[rid], mechanical.index(rid))))
    rationale = "\n".join(
        [f"lowest sum of ranks over {len(verdicts)} judges"]
        + [f"  {selector}: {verdict.rationale}" for selector, verdict in verdicts]
    )
    return RankOutcome(ranking, rationale, "judge", spent, unknown)


def _judge_once(
    cfg: Config,
    candidates: list[harness_judge.CandidateBrief],
    sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink | None,
    build_provider: BuildProvider,
) -> harness_judge.CompareVerdict:
    """Ask one judge for its ranking.

    Args:
        cfg: The config whose reviewer role is this judge.
        candidates: The candidates to rank.
        sink: The judges' transcript sink.
        budget: The judging's tracker.
        events: The coordinator's journal, or None.
        build_provider: Builds the reviewer provider.

    Returns:
        The judge's verdict.

    Raises:
        ProviderError: The reviewer role has no model.
    """
    reviewer = cfg.models.resolve("reviewer")
    if reviewer is None:
        raise ProviderError("no model configured for the judge")
    provider: Provider = build_provider(cfg, sink, budget, events)
    if events is not None:
        provider = role_providers.InstrumentedProvider(
            inner=provider,
            role="judge",
            model=reviewer.model,
            provider_name=reviewer.provider,
            events=events,
            budget=budget,
            stream_text=True,
        )
    return harness_judge.compare(provider, reviewer.model, candidates)


def verify_word(verification: kinds.Verification | None) -> str:
    """Return a candidate's gate verdict as the report and the journal word it."""
    return verification or "unknown"


def print_ranked_candidates(
    candidates: list[harness_judge.CandidateBrief],
    outcome: RankOutcome,
    *,
    rows: Mapping[str, listing.SessionSummary],
    merged_into: Mapping[str, str] | None = None,
    committed: Mapping[str, bool] | None = None,
    model_refs: Mapping[str, str] | None = None,
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
        model_refs: Imported work locations for model-controlled lanes.
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
        # A clean $0 is a blank cell in a listing; a report line spells it out.
        cost = rows[rid].usage_cell or agent6_budget.format_usd(0.0)
        into = (merged_into or {}).get(rid, "")
        if into:
            landing = f"merged into {into}"
        elif model_refs is not None and rid in model_refs:
            landing = (
                f"work is on {model_refs[rid]}"
                if model_refs[rid]
                else "no imported work ref (git was model-controlled)"
            )
        elif (committed or {}).get(rid, True):
            landing = f"merge with: agent6 sessions merge {rid}"
        else:
            landing = "nothing to merge (no commit)"
        reporter.out(f"  {rnk}. {rid}  {labels[rid]:<{width}}  {cost}   {landing}")
    if len(candidates) > 1:
        cand_total = sum(c.cost_usd for c in candidates)
        plan_total = sum(
            rows[c.session_id].plan_observed_points for c in candidates if c.session_id in rows
        )
        candidates_cell = session_status.usage_cell(
            cand_total, partial=False, plan_observed_points=plan_total, plan_used_percent=0.0
        ) or agent6_budget.format_usd(0.0)
        judge = outcome.judge_cost_usd
        partial = outcome.judge_cost_partial
        line = f"total: candidates {candidates_cell}"
        if judge > 0 or partial:
            line += f" + judge {agent6_budget.format_usd(judge, partial=partial)}"
            if plan_total <= 0:
                # Points and dollars do not add; the sum is for an all-dollar line.
                line += f" = {agent6_budget.format_usd(cand_total + judge, partial=partial)}"
        reporter.out(line)
    if outcome.rationale:
        reporter.out(f"\njudge: {outcome.rationale}")
