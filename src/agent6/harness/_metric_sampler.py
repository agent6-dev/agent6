# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The metric readings of a run with a `[harness.metric]`: the sample a
`run_metric_command` result becomes (the model's own call, or the harness's
after a green verify), the feedback block the model reads, and the plateau
summary. The pure rules live in `_metric`; this object does the call and
the events, and the loop decides when to measure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent6.config import MetricConfig
from agent6.harness._metric import (
    MetricSample,
    coerce_metric_score,
    extract_metric_targets,
    format_metric_feedback,
    metric_at_fraction_ceiling,
    metric_goal,
    metric_plateau_summary,
)
from agent6.tools.dispatch import ToolDeniedError, ToolDispatcher, ToolError
from agent6.tools.results import MetricResult

if TYPE_CHECKING:
    from agent6.harness._loop_state import LoopState


@dataclass(frozen=True, slots=True)
class MetricSampler:
    """`settings` is `[harness.metric]`; `enabled` is run mode (plan and ask
    never sample); `dispatcher` runs the harness's own `run_metric_command`."""

    settings: MetricConfig | None
    enabled: bool
    dispatcher: ToolDispatcher
    log: Callable[[str], None]
    emit: Callable[..., None]

    @property
    def goal(self) -> Literal["minimize", "maximize"] | None:
        """The configured direction, or None without a metric."""
        return metric_goal(self.settings)

    @property
    def active(self) -> bool:
        """Whether this run measures: run mode with a configured goal."""
        return self.enabled and self.goal is not None

    def record(
        self,
        history: list[MetricSample],
        result: MetricResult,
        *,
        iteration: int,
        label: str,
        sha: str,
    ) -> str | None:
        """A `run_metric_command` result as the run's next sample, with its
        event; the feedback block for the model, or None without a metric."""
        goal = self.goal
        if goal is None or self.settings is None:
            return None
        score = coerce_metric_score(result.score)
        combined = f"{result.stdout}\n{result.stderr}"
        sample = MetricSample(
            label=label,
            score=score,
            returncode=result.returncode,
            sha=sha,
            stdout_tail=result.stdout[-500:],
            stderr_tail=result.stderr[-500:],
            targets=extract_metric_targets(combined, goal=goal),
            # Only an X/Y ceiling reported on the score-match line counts, so
            # an incidental "100/100" progress bar elsewhere cannot latch it.
            at_ceiling=goal == "maximize"
            and score is not None
            and metric_at_fraction_ceiling(combined, score, pattern=self.settings.pattern),
        )
        history.append(sample)
        self.emit(
            "loop.metric.sample",
            iteration=iteration,
            label=label,
            score=score,
            returncode=result.returncode,
            sha=sha[:12],
        )
        return format_metric_feedback(history, goal=goal)

    def auto_feedback(self, state: LoopState, *, iteration: int, sha: str) -> str | None:
        """The harness's own reading after a green verify, as feedback text;
        a failed reading is a sample with its error, and a denied one also
        withholds the automatic metric for the rest of the run. An
        unexecutable operator command raises through, as the model's own call
        would."""
        goal = self.goal
        if not self.active or goal is None:
            return None
        history = state.metric.history
        self.log(f"LOOP: auto metric after verify-pass at iter {iteration}")
        self.emit("loop.metric.auto_call", iteration=iteration, sha=sha[:12])
        try:
            result = self.dispatcher.dispatch("run_metric_command", {})
        except ToolError as exc:
            error = str(exc)
            if isinstance(exc, ToolDeniedError):
                state.metric.denied = True
                error += "; the automatic metric is withheld for the rest of the run"
            history.append(
                MetricSample(
                    label=f"auto iter {iteration}",
                    score=None,
                    returncode=None,
                    sha=sha,
                    error=error,
                )
            )
            self.emit("loop.metric.auto_failed", iteration=iteration, error=error[:200])
            return format_metric_feedback(history, goal=goal)
        assert isinstance(result, MetricResult)  # run_metric_command's result type
        return self.record(
            history, result, iteration=iteration, label=f"auto iter {iteration}", sha=sha
        )

    def plateau_finish(self, history: list[MetricSample]) -> str | None:
        """The plateau summary for this run, when it measures."""
        goal = self.goal
        if not self.active or goal is None:
            return None
        return metric_plateau_summary(history, goal=goal)
