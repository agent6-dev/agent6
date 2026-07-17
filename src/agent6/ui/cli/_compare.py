# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared candidate-ranking + report pieces for `--parallel`'s auto-compare
(`parallel.py`) and the standalone `runs compare` (`runs_cmds.py`): rank
candidates (judge via the reviewer model when configured, else the
deterministic mechanical fallback) and print the ranked table. One
implementation so the two callers can never drift, including the `judging...`
status shown while the judge call is in flight (`_judging_status`).
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
from collections.abc import Generator
from pathlib import Path

from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.providers import Provider, ProviderError, TranscriptSink
from agent6.ui.cli._console_view import _HEARTBEAT_TICK_S, _SPINNER
from agent6.ui.cli.providers import _build_role_provider
from agent6.workflows.judge import CandidateBrief, JudgeError, compare, mechanical_ranking


@contextlib.contextmanager
def _judging_status() -> Generator[None]:
    """Show progress around the (~50-60s, otherwise silent) judge call: a real
    terminal gets the SAME spinner glyphs/cadence as the run stream's
    provider-call heartbeat (`_console_view`'s `_SPINNER`/`_HEARTBEAT_TICK_S`);
    a non-tty (piped, detached orchestrator) gets one plain line so logs stay
    truthful -- no animation frames written to a file."""
    if not sys.stdout.isatty():
        print("judging...")
        yield
        return
    stop = threading.Event()

    def spin() -> None:
        i = 0
        while True:
            sys.stdout.write(f"\r\x1b[2K{_SPINNER[i % len(_SPINNER)]} judging...")
            sys.stdout.flush()
            i += 1
            if stop.wait(_HEARTBEAT_TICK_S):
                return

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()


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
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    task = manifest.get("user_task") if isinstance(manifest, dict) else None
    return task if isinstance(task, str) and task else fallback


def rank(
    cfg: Config, candidates: list[CandidateBrief], *, transcript_dir: Path
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
            provider: Provider = _build_role_provider(
                cfg, "reviewer", transcript_sink=sink, budget=budget
            )
            with _judging_status():
                verdict = compare(provider, reviewer.model, candidates)
            return verdict.ranking, verdict.rationale, "judge"
        except (ProviderError, JudgeError) as exc:
            # A configured reviewer that fails must not degrade to the mechanical
            # table silently: say so, so the report isn't mistaken for a judged one.
            detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            print(f"judge failed ({detail}); ranked mechanically", file=sys.stderr)
    return mechanical_ranking(candidates), "", "mechanical"


def print_ranked_candidates(
    candidates: list[CandidateBrief], ranking: tuple[str, ...], rationale: str
) -> None:
    """Print the ranked table (best first) + a `runs merge` line per
    candidate, then the judge's rationale if there is one. Prints nothing when
    *ranking* is empty."""
    if not ranking:
        return
    by_id = {c.run_id: c for c in candidates}
    print("ranked candidates (best first):")
    for rnk, rid in enumerate(ranking, start=1):
        c = by_id[rid]
        verify = "passed" if c.verify_ok else "failed" if c.verify_ok is False else "no-verify"
        print(
            f"  {rnk}. {rid}  {verify:<9} ${c.cost_usd:.4f}   merge with: agent6 runs merge {rid}"
        )
    if rationale:
        print(f"\njudge: {rationale}")
