# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Score a metric run and decide when it plateaus or may finish.

For a run with a `[harness.metric]`, the loop measures a score after each verified step and
feeds the trajectory back to the worker. This module holds the pure pieces: the sample record,
the score and threshold parsing, the feedback block, the plateau stop and the early-finish gate.
The loop decides when to measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent6.harness._advice import Nudge, Refusal, Stop, TurnContext, with_open_tasks
from agent6.harness._snapshot import End

if TYPE_CHECKING:
    from agent6.harness._loop_state import LoopState, TurnState


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One metric reading.

    Attributes:
        label: The sample's name in the feedback block.
        score: The parsed score, or None when the output held none.
        returncode: The metric command's exit code, or None when it did not run.
        sha: The commit the reading covers, or "".
        error: Why the reading failed, or "".
        stdout_tail: The command's last stdout bytes.
        stderr_tail: The command's last stderr bytes.
        targets: The unmet thresholds parsed from the output (`extract_metric_targets`).
        at_ceiling: Whether the output reported the score as a maxed-out fraction
            (`metric_at_fraction_ceiling`).
    """

    label: str
    score: float | None
    returncode: int | None
    sha: str = ""
    error: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    targets: tuple[float, ...] = ()
    at_ceiling: bool = False


def coerce_metric_score(value: Any) -> float | None:
    """Return a number as a float, or None for a bool or a non-number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


# An operator and a numeric literal, as in `assert cycles() < 1487`; underscores are tolerated.
# The lookbehind rejects the `>` of an arrow (`epoch 2 -> 27.0`): a score echo, not a target.
METRIC_TARGET_RE = re.compile(r"(?<![-=<>!])(<=|>=|<|>)\s*([0-9][0-9_]*(?:\.[0-9]+)?)")


def extract_metric_targets(
    text: str,
    *,
    goal: Literal["minimize", "maximize"],
) -> tuple[float, ...]:
    """Pull the threshold numbers out of metric-command output.

    A `minimize` goal takes the `<` and `<=` bounds, a `maximize` goal the `>` and `>=` bounds.
    Benchmarks print these as `assert <expr> < N` lines, one per unmet tier.

    Args:
        text: The metric command's output.
        goal: The metric's direction.

    Returns:
        The thresholds in order of appearance, deduplicated.
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
    """Return the nearest threshold the current score has not met.

    A target is met only when the score is strictly beyond it in the improving direction,
    matching a strict `assert x < N`.

    Args:
        targets: The parsed thresholds.
        current: The latest score, or None.
        goal: The metric's direction.

    Returns:
        The largest `<` bound not yet undercut (minimize) or the smallest `>` bound not yet
        exceeded (maximize); None when all are met or there is nothing to aim at.
    """
    if not targets or current is None:
        return None
    if goal == "minimize":
        unmet = [t for t in targets if t <= current]
        return max(unmet) if unmet else None
    unmet = [t for t in targets if t >= current]
    return min(unmet) if unmet else None


# A fraction in metric output, as the `27/27` in `SCORE: 27/27`.
METRIC_FRACTION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def metric_at_fraction_ceiling(text: str, score: float, *, pattern: str) -> bool:
    """Return whether the text reports the score as a maxed-out `X/Y` fraction.

    Graders print a bounded score as `SCORE: 27/27`: a numerator equal to both the parsed score
    and the denominator puts the metric at its provable ceiling. Only fractions on the line the
    score pattern matched count, so a progress bar's `100/100` elsewhere cannot latch it.

    Args:
        text: The metric command's output.
        score: The parsed score.
        pattern: The `[harness.metric].pattern` regex the score was parsed with.

    Returns:
        Whether the score is at its ceiling; a bad pattern reads as no score line.
    """
    try:
        m = re.search(pattern, text)
    except re.error:  # as parse_metric_score: a bad pattern means no score line
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
    """Return whether the candidate beats the incumbent in the goal's direction."""
    if goal == "minimize":
        return candidate < incumbent
    return candidate > incumbent


def best_metric_sample(
    samples: list[MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
) -> MetricSample | None:
    """Return the best parsed sample, or None when none parsed."""
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
    """Return one sample rendered as a feedback line."""
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
    """Return the metric config's goal, or None without a metric."""
    goal = getattr(metric_cfg, "goal", None)
    if goal in ("minimize", "maximize"):
        return goal
    return None


def format_metric_feedback(
    history: list[MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
) -> str:
    """Return the feedback block the model reads after a reading."""
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


# Plateau notices a run in its final budget slice gets before the loop ends it.
METRIC_PLATEAU_PATIENCE = 3

# A plateau ends the run only once at most this fraction of the token budget remains.
# With no budget signal the loop falls back to `METRIC_PLATEAU_PATIENCE` alone.
METRIC_PLATEAU_STOP_BELOW_BUDGET = 0.25

# The notice states the fact and the budget; only the final slice ends a run on a plateau.
METRIC_PLATEAU_NUDGE = (
    "[harness plateau] The recent verified edits did not improve the metric;"
    " {budget}. The best commit stands."
)


# An early finish_session on an optimisation run is deferred a bounded number of times.
METRIC_EARLY_FINISH_PATIENCE = 3
METRIC_FINISH_NUDGE = (
    "[harness budget] finish_session deferred: this is an optimisation run with"
    " most of its budget unspent, and the metric's best commit stands. The call"
    f" is honoured after {METRIC_EARLY_FINISH_PATIENCE} deferrals, or once the"
    " budget is nearly spent."
)


def metric_plateau_nudge(budget_remaining: float | None) -> str:
    """Return the plateau notice with the run's remaining budget.

    Args:
        budget_remaining: The fraction of the budget left, or None with no tracker wired in.

    Returns:
        The notice text.
    """
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
    """Return the plateau summary when the latest parsed reading only ties the prior best.

    Args:
        history: The run's samples.
        goal: The metric's direction.
        min_parsed_samples: The fewest parsed samples before a tie counts.

    Returns:
        The summary, or None before the threshold or when the latest reading differs.
    """
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


@dataclass(slots=True)
class MetricGuard:
    """A metric run's readings and its patience counters.

    Attributes:
        history: The run's samples.
        tree: The worktree state the metric was last sampled on, one reading per state.
        denied: Whether the operator's no withholds the automatic metric for the rest of the run.
        plateau_nudges_used: Final-slice plateau notices delivered.
        finish_nudges_used: Early finishes rejected while runway remained.
    """

    history: list[MetricSample] = field(default_factory=list)
    tree: str = ""
    denied: bool = False
    plateau_nudges_used: int = 0
    finish_nudges_used: int = 0

    def rearm(self) -> None:
        """Restart the plateau patience once a standing goal absorbed the stop."""
        self.plateau_nudges_used = 0

    def at_ceiling(self) -> bool:
        """Return whether any verified sample reached the metric's provable ceiling."""
        return any(sample.at_ceiling for sample in self.history)


def metric_plateau(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | Stop | None:
    """Decide the metric run's end after a reading that only ties the best.

    While the run has runway the tie draws the plateau notice; in the final budget slice, or
    with no budget signal, the notice counts against `METRIC_PLATEAU_PATIENCE`, and past it the
    run stops. A metric at its ceiling stops at once. The stop is an ending the end gates judge
    and a standing task may absorb.

    Args:
        turn: The turn's state; `metric_plateau_finish` carries the tie.
        state: The loop state holding the metric guard.
        ctx: The turn's context: budget, open tasks.

    Returns:
        The nudge, the stop, or None when this turn's reading is not a tie.
    """
    finish = turn.metric_plateau_finish
    if finish is None:
        return None
    remaining = ctx.budget_remaining()
    in_final_slice = remaining is None or remaining <= METRIC_PLATEAU_STOP_BELOW_BUDGET
    guard = state.metric

    def end() -> End:
        return End(
            "metric_plateau",
            with_open_tasks(finish, ctx.open_subtasks()),
            completed=True,
            verdict="grounded",
        )

    log = f"LOOP: metric_plateau at iter {turn.iteration}"
    if guard.at_ceiling():
        return Stop(
            end,
            soft="metric_plateau",
            declared="metric_plateau",
            event="loop.metric_ceiling.stop",
            fields={"iteration": turn.iteration},
            log=log,
        )
    if in_final_slice and guard.plateau_nudges_used >= METRIC_PLATEAU_PATIENCE:
        return Stop(end, soft="metric_plateau", declared="metric_plateau", log=log)
    # Patience counts final-slice notices only; a tie with runway left is a local optimum to leave.
    if in_final_slice:
        guard.plateau_nudges_used += 1
    budget_note = "n/a" if remaining is None else f"{remaining:.0%} left"
    return Nudge(
        metric_plateau_nudge(remaining),
        event="loop.metric_plateau.nudge",
        fields={
            "iteration": turn.iteration,
            "nudges_used": guard.plateau_nudges_used,
            "budget_remaining": remaining,
        },
        log=(
            f"  metric_plateau notice at iter {turn.iteration} (budget {budget_note};"
            f" final-slice patience {guard.plateau_nudges_used}/{METRIC_PLATEAU_PATIENCE})"
        ),
    )


def metric_early_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """Refuse an early finish on an optimisation run while runway remains.

    Above the final budget slice an early finish is rejected `METRIC_EARLY_FINISH_PATIENCE`
    times. A metric at its ceiling, a run in the final slice, or no budget signal lets it
    through, so a finish can never deadlock.

    Args:
        turn: The turn's state; `ending` names the finish call or the silent finish.
        state: The loop state holding the metric guard.
        ctx: The turn's context: mode, metric, budget.

    Returns:
        The refusal, or None when the finish goes through.
    """
    if ctx.mode != "run" or not ctx.metric or state.metric.at_ceiling():
        return None
    remaining = ctx.budget_remaining()
    if remaining is None or remaining <= METRIC_PLATEAU_STOP_BELOW_BUDGET:
        return None
    if state.metric.finish_nudges_used >= METRIC_EARLY_FINISH_PATIENCE:
        return None
    state.metric.finish_nudges_used += 1
    used = state.metric.finish_nudges_used
    trigger = turn.ending if turn.ending not in (None, "finish_session") else ""
    return Refusal(
        METRIC_FINISH_NUDGE,
        event="loop.metric_early_finish.rejected",
        fields={
            "iteration": turn.iteration,
            "nudges_used": used,
            "budget_remaining": remaining,
            **({"trigger": trigger} if trigger else {}),
        },
        log=(
            f"  metric early-finish{' (silent)' if trigger else ''} rejected #{used}"
            f" at iter {turn.iteration} (budget {remaining:.0%} left)"
        ),
    )
