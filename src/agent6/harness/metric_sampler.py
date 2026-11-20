# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Take the metric readings of a run with an operator-set metric.

A `run_metric_command` result, the model's own call or the harness's after each committed edit,
becomes a sample with its event and the feedback block the model reads. The pure rules live
in `metric_rules`; the loop decides when to measure.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from agent6 import directive
from agent6.harness import metric_rules
from agent6.sessions import journal
from agent6.tools import dispatch, results, tool_errors
from agent6.viewmodel.session import log as session_log

if TYPE_CHECKING:
    from agent6.harness import loop_state


@dataclasses.dataclass(slots=True)
class MetricSampler:
    """The run's metric sampler.

    Attributes:
        enabled: Run mode; plan and ask never sample.
        dispatcher: Runs the harness's own `run_metric_command`.
        log: The run's text logger.
        emit: The run's event sink.
    """

    enabled: bool
    dispatcher: dispatch.ToolDispatcher
    log: Callable[[str], None]
    emit: Callable[..., None]

    revision: int = 0

    def change(self, spec: directive.MetricSpec | None) -> None:
        """Journal one operator change before using the new command."""
        revision = self.revision + 1
        self.emit(journal.MetricSet(revision=revision, spec=spec.text() if spec else None))
        self.dispatcher.metric = spec
        self.revision = revision

    def restore(self, logs: pathlib.Path) -> None:
        """Bind this execution to the last operator setting in its own journal."""
        setting = session_log.read_metric(logs)
        self.dispatcher.metric = (
            directive.parse_metric_spec(setting.spec) if setting and setting.spec else None
        )
        self.revision = setting.revision if setting else 0

    def notice(self) -> str:
        """Return the metric instruction that supersedes the frozen system prompt."""
        if not self.active or self.dispatcher.metric is None:
            return "[harness metric] No active metric. Disregard any earlier metric instructions."
        return f"[harness metric] The operator's current metric: {self.dispatcher.metric.text()}"

    @property
    def goal(self) -> Literal["minimize", "maximize"] | None:
        """The configured direction, or None without a metric."""
        return self.dispatcher.metric.goal if self.dispatcher.metric else None

    @property
    def active(self) -> bool:
        """Whether this run measures: run mode with a configured goal."""
        return self.enabled and self.goal is not None and self.dispatcher.command_policy() != "deny"

    def record(
        self,
        history: list[metric_rules.MetricSample],
        result: results.MetricResult,
        *,
        turn: int,
        label: str,
        sha: str,
    ) -> str | None:
        """Record a `run_metric_command` result as the run's next sample and emit its event.

        Args:
            history: The run's samples so far; the new one is appended.
            result: The command's result.
            turn: The turn the reading belongs to.
            label: The sample's name in the feedback block.
            sha: The commit the reading covers, or "".

        Returns:
            The feedback block for the model, or None without a metric.
        """
        goal = self.goal
        if goal is None or self.dispatcher.metric is None:
            return None
        score = metric_rules.coerce_metric_score(result.score)
        combined = f"{result.stdout}\n{result.stderr}"
        sample = metric_rules.MetricSample(
            label=label,
            score=score,
            returncode=result.returncode,
            sha=sha,
            stdout_tail=result.stdout[-500:],
            stderr_tail=result.stderr[-500:],
            targets=metric_rules.extract_metric_targets(combined, goal=goal),
            # Only an X/Y ceiling on the score line counts, never a progress bar elsewhere.
            at_ceiling=goal == "maximize"
            and score is not None
            and metric_rules.metric_at_fraction_ceiling(
                combined, score, pattern=self.dispatcher.metric.pattern
            ),
        )
        history.append(sample)
        self.emit(
            journal.MetricSample(
                turn=turn,
                label=label,
                score=score,
                exit_code=result.returncode,
                sha=sha[:12],
            )
        )
        return metric_rules.format_metric_feedback(history, goal=goal)

    def auto_feedback(self, state: loop_state.LoopState, *, turn: int, sha: str) -> str | None:
        """Take the harness's own reading after a committed edit, verified or a checkpoint.

        A failed reading is a sample with its error; a denied one also withholds the automatic
        metric for the rest of the run. An unexecutable operator command raises through, as the
        model's own call would.

        Args:
            state: The loop state holding the metric history and the denial flag.
            turn: The turn the reading belongs to.
            sha: The commit the reading covers, or "".

        Returns:
            The feedback block for the model, or None when this run does not measure.
        """
        goal = self.goal
        if not self.active or goal is None:
            return None
        history = state.metric.history
        self.log(f"LOOP: auto metric after the committed edit at turn {turn}")
        self.emit(journal.MetricAutoCall(turn=turn, sha=sha[:12]))
        try:
            result = self.dispatcher.dispatch("run_metric_command", {})
        except tool_errors.ToolError as exc:
            error = str(exc)
            if isinstance(exc, tool_errors.ToolDeniedError):
                state.metric.denied = True
                error += "; the automatic metric is withheld for the rest of the run"
            history.append(
                metric_rules.MetricSample(
                    label=f"auto turn {turn}",
                    score=None,
                    returncode=None,
                    sha=sha,
                    error=error,
                )
            )
            self.emit(journal.MetricAutoFailed(turn=turn, error=error[:200]))
            return metric_rules.format_metric_feedback(history, goal=goal)
        assert isinstance(result, results.MetricResult)  # run_metric_command's result type
        return self.record(history, result, turn=turn, label=f"auto turn {turn}", sha=sha)

    def plateau_finish(self, history: list[metric_rules.MetricSample]) -> str | None:
        """Return the plateau summary for this run, or None when it does not measure."""
        goal = self.goal
        if not self.active or goal is None:
            return None
        return metric_rules.metric_plateau_summary(history, goal=goal)
