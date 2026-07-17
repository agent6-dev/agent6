# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Candidate-ranking core shared by `--parallel`'s auto-compare (`app.parallel`)
and the standalone `runs compare` (`ui/cli/runs_cmds.py`): rank candidates (judge
via the reviewer model when one is built, else the deterministic mechanical
fallback) and print the ranked table. One implementation so the two callers can
never drift.

The reviewer-provider wiring and the `judging...` in-flight status are injected
by the caller (ui/cli supplies the console spinner + the role-provider builder),
so `app` never imports `ui`. A caller that shows nothing passes
`contextlib.nullcontext` as *judging_status*.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.providers import Provider, ProviderError, TranscriptSink
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.workflows.judge import CandidateBrief, JudgeError, compare, mechanical_ranking

# The reviewer provider the judge call uses, built by the caller from the
# configured `reviewer` role (ui/cli wires it via `build_role_provider`).
BuildProvider = Callable[[Config, TranscriptSink, BudgetTracker], Provider]
# A no-arg context manager shown around the (~50-60s, otherwise silent) judge
# call. ui/cli supplies the console spinner; `nullcontext` shows nothing.
JudgingStatus = Callable[[], AbstractContextManager[None]]


def verify_ok(status: str) -> bool | None:
    """Map a run's folded status to the judge's verify tri-state."""
    if status == "passed":
        return True
    if status == "failed":
        return False
    return None


def manifest_task(run_dir: Path, fallback: str) -> str:
    """The run's own recorded `user_task`, else *fallback*."""
    try:
        manifest = read_manifest(run_dir)
    except ManifestError:
        return fallback
    task = manifest.get("user_task")
    return task if isinstance(task, str) and task else fallback


def rank(
    cfg: Config,
    candidates: list[CandidateBrief],
    *,
    transcript_dir: Path,
    build_provider: BuildProvider,
    judging_status: JudgingStatus,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[tuple[str, ...], str, str]:
    """Rank candidates best-first. Use the configured reviewer model as the
    compare judge when one resolves; fall back to the deterministic mechanical
    ranking when it is unset or the judge call fails.

    Returns ``(ranking, rationale, ranked_by)`` where ``ranked_by`` is ``"judge"``
    only when the reviewer model actually produced the order, else ``"mechanical"``
    (reviewer unset, one candidate, or the judge call failed) -- the honest signal
    the compare stamp records, not a guess from whether the rationale is empty."""
    reviewer = cfg.models.resolve("reviewer")
    if len(candidates) > 1 and reviewer is not None:
        try:
            sink = TranscriptSink(transcript_dir)
            budget = BudgetTracker(
                max_input_tokens=cfg.budget.max_input_tokens,
                max_output_tokens=cfg.budget.max_output_tokens,
                max_usd=cfg.budget.best_effort_usd_limit,
            )
            provider: Provider = build_provider(cfg, sink, budget)
            with judging_status():
                verdict = compare(provider, reviewer.model, candidates)
            return verdict.ranking, verdict.rationale, "judge"
        except (ProviderError, JudgeError) as exc:
            # A configured reviewer that fails must not degrade to the mechanical
            # table silently: say so, so the report isn't mistaken for a judged one.
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            reporter.err(f"judge failed ({detail}); ranked mechanically")
    return mechanical_ranking(candidates), "", "mechanical"


def print_ranked_candidates(
    candidates: list[CandidateBrief],
    ranking: tuple[str, ...],
    rationale: str,
    *,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print the ranked table (best first) + a `runs merge` line per
    candidate, then the judge's rationale if there is one. Prints nothing when
    *ranking* is empty."""
    if not ranking:
        return
    by_id = {c.run_id: c for c in candidates}
    reporter.out("ranked candidates (best first):")
    for rnk, rid in enumerate(ranking, start=1):
        c = by_id[rid]
        verify = "passed" if c.verify_ok else "failed" if c.verify_ok is False else "no-verify"
        reporter.out(
            f"  {rnk}. {rid}  {verify:<9} ${c.cost_usd:.4f}   merge with: agent6 runs merge {rid}"
        )
    if rationale:
        reporter.out(f"\njudge: {rationale}")
