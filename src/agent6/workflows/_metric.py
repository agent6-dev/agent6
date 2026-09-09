# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Metric-driven optimisation helpers for the agent loop.

For runs with a configured [workflow.metric], the loop measures a continuous
score after each verified step and feeds the trajectory back to the worker.
This module owns the pure pieces of that: the `MetricSample` record, parsing a
score and the unmet thresholds out of metric output, deciding whether a sample
is a new best / at a provable ceiling, formatting the feedback block, and the
plateau detection + budget-scaled nudge selection. The loop owns the policy of
when to measure and when to stop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class MetricSample:
    label: str
    score: float | None
    returncode: int | None
    sha: str = ""
    error: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    # Comparison thresholds parsed from the metric command output (e.g.
    # `assert cycles() < 1487` lines); they point the worker at the next
    # unmet target rather than a vague "go faster". See
    # `extract_metric_targets`.
    targets: tuple[float, ...] = ()
    # True when the grader reported the score as a maxed-out fraction
    # (`SCORE: 27/27`): the metric is at its provable ceiling and cannot
    # be improved. See `metric_at_fraction_ceiling`.
    at_ceiling: bool = False


def coerce_metric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


# A comparison operator followed by a numeric literal, e.g. the
# `< 1487` in `assert cycles() < 1487`. Underscores in the literal
# (Python int separators) are tolerated and stripped.
# (?<![-=<>!]) rejects the '>' inside '->' / '=>' arrows (and the tail of
# '>>' / '!>'): a grader progress log like 'epoch 2 -> 27.0' would otherwise
# be captured as a threshold, fabricating an unmeetable 'drive the metric
# above <current>' directive from the grader's own echo of the score.
METRIC_TARGET_RE = re.compile(r"(?<![-=<>!])(<=|>=|<|>)\s*([0-9][0-9_]*(?:\.[0-9]+)?)")


def extract_metric_targets(
    text: str,
    *,
    goal: Literal["minimize", "maximize"],
) -> tuple[float, ...]:
    """Pull threshold numbers out of metric-command output.

    For `goal="minimize"` we want upper bounds the score must get
    *under* (`<` / `<=` thresholds); for `"maximize"` we want lower
    bounds it must get *over* (`>` / `>=`). Benchmarks commonly print
    these as `assert <expr> < N` lines (one per unmet speed tier), so
    extracting them turns "go faster" into a concrete next target.
    Order-preserving and de-duplicated.
    """
    wanted = {"<", "<="} if goal == "minimize" else {">", ">="}
    seen: set[float] = set()
    out: list[float] = []
    for op, num in METRIC_TARGET_RE.findall(text):
        if op not in wanted:
            continue
        try:
            value = float(num.replace("_", ""))
        except ValueError:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def next_metric_target(
    targets: tuple[float, ...],
    current: float | None,
    goal: Literal["minimize", "maximize"],
) -> float | None:
    """The nearest threshold the current score has not yet met. A target is
    met only when the score is STRICTLY beyond it in the improving direction:
    equality stays unmet, matching a strict `assert x < N` (and holding the
    conservative reading for a `<=` bound). Returns the largest
    not-yet-undercut `<` bound (minimize) or the smallest not-yet-exceeded
    `>` bound (maximize); None when all are met or there is nothing to aim
    at."""
    if not targets or current is None:
        return None
    if goal == "minimize":
        unmet = [t for t in targets if t <= current]
        return max(unmet) if unmet else None
    unmet = [t for t in targets if t >= current]
    return min(unmet) if unmet else None


# A fraction in metric output, e.g. the `27/27` in `SCORE: 27/27`. A
# maxed-out fraction means the metric is at its provable ceiling. See
# `metric_at_fraction_ceiling`.
METRIC_FRACTION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def metric_at_fraction_ceiling(text: str, score: float, *, pattern: str) -> bool:
    """True if `text` reports `score` as a maxed-out `X/Y` fraction.

    Many graders print a bounded score as `X/Y` (`SCORE: 27/27`,
    `passed 27/27`). When the numerator equals both the parsed score and
    the denominator, the metric is provably at its ceiling: no further edit
    can push it higher. Detecting this lets a `maximize` run stop cleanly
    instead of treating the unbeatable plateau as a local optimum worth
    spending the rest of the budget pivoting away from. Conservative: only
    fires on an exact `score/score` match, so partial scores (`26/27`)
    and unbounded metrics (raw cycle counts, which never print a
    denominator) are unaffected.

    `pattern` is the metric score regex (`[workflow.metric].pattern`, the
    one the score was parsed with): only fractions on the line of the score
    match count, so an incidental fraction elsewhere in the output (a tqdm
    `100/100` in stderr) cannot latch the ceiling for the run.
    """
    try:
        m = re.search(pattern, text)
    except re.error:  # mirror parse_metric_score: a bad pattern means no score line
        return False
    if m is None:
        return False
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.end())
    scan = text[start:] if end == -1 else text[start:end]
    for num_s, den_s in METRIC_FRACTION_RE.findall(scan):
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:  # pragma: no cover - regex already constrains digits
            continue
        if num == score and num == den:
            return True
    return False


def metric_is_better(
    candidate: float,
    incumbent: float,
    goal: Literal["minimize", "maximize"],
) -> bool:
    if goal == "minimize":
        return candidate < incumbent
    return candidate > incumbent


def best_metric_sample(
    samples: list[MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
) -> MetricSample | None:
    parsed = [sample for sample in samples if sample.score is not None]
    if not parsed:
        return None
    best = parsed[0]
    for sample in parsed[1:]:
        assert sample.score is not None
        assert best.score is not None
        if metric_is_better(sample.score, best.score, goal):
            best = sample
    return best


def format_metric_sample(sample: MetricSample) -> str:
    score = "unparsed" if sample.score is None else f"{sample.score:g}"
    parts = [f"{sample.label}: score={score}"]
    if sample.returncode is not None:
        parts.append(f"exit={sample.returncode}")
    if sample.sha:
        parts.append(f"sha={sample.sha[:12]}")
    if sample.error:
        parts.append(f"error={sample.error[:200]}")
    return ", ".join(parts)


def metric_goal(metric_cfg: Any) -> Literal["minimize", "maximize"] | None:
    goal = getattr(metric_cfg, "goal", None)
    if goal in ("minimize", "maximize"):
        return goal
    return None


def format_metric_feedback(
    history: list[MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
) -> str:
    latest = history[-1]
    best = best_metric_sample(history, goal=goal)
    previous_best = best_metric_sample(history[:-1], goal=goal)
    best_line = format_metric_sample(best) if best is not None else "none parsed yet"

    if latest.score is None:
        verdict = "latest metric score was not parsed; inspect output before trusting this edit"
    elif previous_best is None:
        verdict = "first parsed metric sample"
    else:
        assert previous_best.score is not None
        verdict = (
            "new best"
            if metric_is_better(latest.score, previous_best.score, goal)
            else "not a new best"
        )

    lines = [
        "[harness metric]",
        f"goal: {goal} ({'lower' if goal == 'minimize' else 'higher'} is better)",
        f"latest: {format_metric_sample(latest)}",
        f"best: {best_line}",
        f"verdict: {verdict}",
        "trajectory (last 5):",
    ]
    lines.extend(f"- {format_metric_sample(sample)}" for sample in history[-5:])
    next_target = next_metric_target(latest.targets, latest.score, goal)
    if next_target is not None and latest.score is not None:
        direction = "below" if goal == "minimize" else "above"
        lines.append(
            f"next target: {direction} {next_target:g} (current {latest.score:g}), the"
            " nearest threshold not yet cleared"
        )
    if latest.score is None:
        if latest.stdout_tail:
            lines.append(f"stdout tail: {latest.stdout_tail[-500:]}")
        if latest.stderr_tail:
            lines.append(f"stderr tail: {latest.stderr_tail[-500:]}")
    return "\n".join(lines)


# How many plateau notices a run in its final budget slice gets before the
# loop ends it. The detector fires the first time a verified metric ties the
# prior best, and a tie with budget to spare is not an end.
METRIC_PLATEAU_PATIENCE = 3

# A metric plateau only becomes a terminal condition once the run has
# entered its final budget slice. While more than this fraction of the
# token budget remains, a plateau is treated as a local optimum worth
# pivoting away from rather than a reason to quit: stopping with most of
# the budget unspent leaves measurable gains (and money) on the table.
# Only consulted when a real BudgetTracker is wired in; with no budget
# signal the loop falls back to the fixed `METRIC_PLATEAU_PATIENCE`.
METRIC_PLATEAU_STOP_BELOW_BUDGET = 0.25

# The plateau notice states the fact and the budget; the loop ends a run on
# a plateau only below METRIC_PLATEAU_STOP_BELOW_BUDGET.
METRIC_PLATEAU_NUDGE = (
    "[harness plateau] The recent verified edits did not improve the metric;"
    " {budget}. The best commit stands."
)


# An early finish_session on an optimisation run is deferred, a bounded
# number of times, so the notice states the deferral and its bound.
METRIC_EARLY_FINISH_PATIENCE = 3
METRIC_FINISH_NUDGE = (
    "[harness budget] finish_session deferred: this is an optimisation run with"
    " most of its budget unspent, and the metric's best commit stands. The call"
    f" is honoured after {METRIC_EARLY_FINISH_PATIENCE} deferrals, or once the"
    " budget is nearly spent."
)


def metric_plateau_nudge(budget_remaining: float | None) -> str:
    """The plateau notice with the run's remaining budget (unknown with no
    tracker wired in)."""
    budget = (
        f"{budget_remaining:.0%} of the budget remains"
        if budget_remaining is not None
        else "the remaining budget is unknown"
    )
    return METRIC_PLATEAU_NUDGE.format(budget=budget)


def metric_plateau_summary(
    history: list[MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
    min_parsed_samples: int = 5,
) -> str | None:
    parsed = [sample for sample in history if sample.score is not None]
    if len(parsed) < min_parsed_samples:
        return None
    latest = parsed[-1]
    previous_best = best_metric_sample(parsed[:-1], goal=goal)
    if previous_best is None or latest.score is None or previous_best.score is None:
        return None
    if latest.score != previous_best.score:
        return None
    best = format_metric_sample(previous_best)
    latest_text = format_metric_sample(latest)
    return (
        "metric plateau: latest verified metric tied the prior best after "
        f"{len(parsed)} parsed samples; stopping to preserve performance per dollar. "
        f"latest={latest_text}; best={best}"
    )
