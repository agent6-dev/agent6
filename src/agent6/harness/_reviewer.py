# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Schedule the review panel and apply its verdict to the run."""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import pydantic

from agent6 import budget, errors, review_records
from agent6.config import ReviewScope
from agent6.harness import _chain, _context, _panel, _review_history, _review_seat
from agent6.providers import Provider, ToolDefinition
from agent6.vcs import git_read

if TYPE_CHECKING:
    from agent6.harness import _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class CritiqueResult:
    """Hold the panel's verdict as the triggers consume it.

    Attributes:
        text: The findings text injected for the worker.
        satisfied: False only when a blocking block_on mode rejects.
        capped: The panel blocked but the run's rejected finishes reached the cap, so the end
            stands with the findings.
    """

    text: str
    satisfied: bool
    capped: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewSeat:
    """Bind one focus to a provider and model.

    Attributes:
        focus: The reviewer's stance.
        model: The model name, for events.
        provider: The provider called.
        scope: `diff` for one call over the diff, `repo` for a read-only tool loop first.
        temperature: The sampling temperature sent on every call.
    """

    focus: str
    model: str
    provider: Provider
    scope: ReviewScope = "diff"
    temperature: float | None = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewSettings:
    """Hold the in-loop review panel's settings, from `[review]`.

    The findings return to the model on the next user message whatever the blocking rule.

    Attributes:
        trigger: When the panel sits: after a verify failure, before a finish, or periodically.
        interval_turns: The turns between periodic panels.
        reviewers: The resolved seats.
        block_on: `none` advises; `any`, `all` and `model_quorum` can reject an end.
        min_blocking_models: The distinct blocking routes a `model_quorum` rejection needs.
        max_review_rejections: The ends the gate may reject in the run; from then on the panel
            advises and every end stands.
        min_remaining_budget_fraction: Skip panels below this remaining budget fraction.
        concurrency: The seat calls run at once.
        repo_max_turns: The provider calls each repo-scope seat may make, tools included.
    """

    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    interval_turns: int = 10
    reviewers: Sequence[ReviewSeat] = ()
    block_on: _panel.ReviewBlockOn = "none"
    min_blocking_models: int = 2
    max_review_rejections: int = 4
    min_remaining_budget_fraction: float = 0.25
    concurrency: int = 1
    repo_max_turns: int = 20


def run_panel(
    seats: Sequence[ReviewSeat],
    ctx: _panel.ReviewContext,
    *,
    block_on: _panel.ReviewBlockOn,
    min_blocking_models: int,
    panel_id: str,
    repo_max_turns: int,
    concurrency: int = 1,
    tools: list[ToolDefinition] | None = None,
    dispatch: _review_seat.ReviewDispatch | None = None,
) -> _panel.PanelResult:
    """Run every seat over the same context and aggregate the verdicts in seat order.

    Args:
        seats: The seats.
        ctx: The review context; each seat gets its own focus substituted.
        block_on: The blocking rule.
        min_blocking_models: The distinct blocking routes a `model_quorum` rejection needs.
        panel_id: The panel's id, for the result.
        repo_max_turns: The provider calls each repo-scope seat may make, tools included.
        concurrency: The seat calls run at once.
        tools: The read-only tools for repo seats.
        dispatch: The read-only dispatch for repo seats.

    Returns:
        The aggregated panel result.
    """

    def _run(i: int, s: ReviewSeat) -> review_records.ReviewVerdict:
        seat_ctx = dataclasses.replace(ctx, focus=s.focus)
        if s.scope == "repo" and tools is not None and dispatch is not None:
            return _review_seat.repo_review(
                s.provider,
                seat_ctx,
                seat_index=i,
                focus=s.focus,
                model_route=s.model,
                tools=tools,
                dispatch=dispatch,
                max_turns=repo_max_turns,
                temperature=s.temperature,
            )
        return _review_seat.structured_review(
            s.provider,
            seat_ctx,
            seat_index=i,
            focus=s.focus,
            model_route=s.model,
            temperature=s.temperature,
        )

    if concurrency > 1 and len(seats) > 1:
        verdicts = _run_seats_concurrently(seats, _run, concurrency)
    else:
        verdicts = [_run(i, s) for i, s in enumerate(seats)]
    return _panel.aggregate_verdicts(
        verdicts, ctx, block_on=block_on, min_blocking_models=min_blocking_models, panel_id=panel_id
    )


def _run_seats_concurrently(
    seats: Sequence[ReviewSeat],
    run_seat: Callable[[int, ReviewSeat], review_records.ReviewVerdict],
    concurrency: int,
) -> list[review_records.ReviewVerdict]:
    """Run the seat calls on daemon threads, results in seat order.

    A thread pool's workers are joined at exit and a seat call has no abort hook, so Ctrl-C
    would wait for every seat; daemon threads die with the process and the polling wait lets
    KeyboardInterrupt land.

    Args:
        seats: The seats.
        run_seat: The call that runs one seat.
        concurrency: The seat calls run at once.

    Returns:
        One verdict per seat.

    Raises:
        RuntimeError: When a thread ended with neither a verdict nor an error.
    """
    slots: list[review_records.ReviewVerdict | None] = [None] * len(seats)
    errors: list[BaseException] = []
    gate = threading.Semaphore(min(concurrency, len(seats)))
    done = threading.Semaphore(0)

    def work(i: int, seat: ReviewSeat) -> None:
        with gate:
            try:
                slots[i] = run_seat(i, seat)
            except BaseException as exc:  # surfaced below; a pool would do the same
                errors.append(exc)
            finally:
                done.release()

    threads = [
        threading.Thread(target=work, args=(i, s), name=f"review-seat-{i}", daemon=True)
        for i, s in enumerate(seats)
    ]
    for t in threads:
        t.start()
    for _ in seats:
        while not done.acquire(timeout=0.2):
            pass
    if errors:
        raise errors[0]
    verdicts = [v for v in slots if v is not None]
    if len(verdicts) != len(seats):  # a worker ended with neither verdict nor error
        raise RuntimeError("review seat thread ended without a verdict")
    return verdicts


# The before-finish panel's rejection by the ending it rejected; the findings follow.
# Room in the review notice for a rejection prefix and the "more findings" line.
_NOTICE_ROOM = 256

REVIEW_REJECTED = {
    "finish_session": (
        "The review panel rejected your finish_session call. Address the"
        " issues below before calling finish_session again.\n\n"
    ),
    "silent_finish": (
        "The review panel rejected your silent finish (no tool_use, just"
        " text). Address the issues below and continue the task.\n\n"
    ),
    "settled": (
        "The review panel rejected the settled end. Address the issues"
        " below; the run ends when it settles again or finish_session"
        " passes.\n\n"
    ),
    "metric_plateau": (
        "The review panel rejected the end at the metric plateau. Address the"
        " issues below; the run ends when the plateau holds again or"
        " finish_session passes.\n\n"
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class Reviewer:
    """Sit the in-loop review panel for one run.

    Attributes:
        settings: The panel's settings.
        chain: The run's chain: the diff the panel grounds on and the AGENTS.md it reads.
        hide: `[sandbox].hide_paths`, which no AGENTS.md is read from.
        review_tools: Builds the read-only tools and dispatch a repo seat gets.
        budget_remaining: The fraction of the budget left, or None without a tracker.
        log: The run's text logger.
        emit: The run's event emitter.
    """

    settings: ReviewSettings
    chain: _chain.RunChain
    hide: tuple[str, ...]
    review_tools: Callable[[], tuple[list[ToolDefinition], _review_seat.ReviewDispatch]]
    budget_remaining: Callable[[], float | None]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def triggers(self, state: _loop_state.LoopState, turn: _loop_state.TurnState) -> None:
        """Sit the observe-only panels after a verify failure or at the configured interval.

        The before-finish panel, which can revoke an end, is `end_rejected`.

        Args:
            state: The execution's state.
            turn: The turn, which receives the findings text.
        """
        if (
            self.settings.trigger == "on_verify_fail"
            and turn.verify_just_failed
            and self.available()
        ):
            critique = self.critique(state, trigger="verify_failed", turn=turn.number)
            if critique is not None:
                turn.review_text = critique.text
        elif (
            self.settings.trigger == "periodic"
            and self.available()
            and turn.number % max(1, self.settings.interval_turns) == 0
        ):
            critique = self.critique(state, trigger="periodic", turn=turn.number)
            if critique is not None:
                turn.review_text = critique.text

    def end_rejected(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, *, ending: str
    ) -> bool:
        """Sit the before-finish panel over an end, once per turn.

        After `max_review_rejections` rejected ends the panel advises and the end goes through
        with the findings injected. A turn that declares two ends gets one verdict for both.

        Args:
            state: The execution's state.
            turn: The turn, which receives the findings text.
            ending: The end declared, a key of `REVIEW_REJECTED`.

        Returns:
            True when the panel rejected the end and the run carries on.
        """
        if turn.end_rejected is None:
            turn.end_rejected = self._judge_end(state, turn, ending=ending)
        return turn.end_rejected

    def _judge_end(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, *, ending: str
    ) -> bool:
        if not (self.settings.trigger == "before_finish" and self.available()):
            return False
        critique = self.critique(state, trigger="before_finish", turn=turn.number)
        if critique is None:
            return False
        if not critique.satisfied:
            state.gates.review_rejections += 1
            self.log(f"  review rejected {ending} at turn {turn.number}")
            self.emit("loop.review.rejected_finish", turn=turn.number, ending=ending)
            turn.review_text = REVIEW_REJECTED[ending] + critique.text
            return True
        if critique.capped:
            self.log(
                f"  review rejected {ending} at turn {turn.number} but the rejection cap"
                f" ({self.settings.max_review_rejections}) is reached: the end stands"
            )
            self.emit(
                "loop.review.rejection_cap_reached",
                turn=turn.number,
                rejections=state.gates.review_rejections,
            )
            turn.review_text = (
                "The review panel flagged issues but the rejection cap was"
                " reached; the end stands. Findings:\n\n" + critique.text
            )
        else:
            self.log(f"  review approved {ending}")
        return False

    def available(self) -> bool:
        """Return whether the panel has seats; every in-loop trigger gates on it."""
        return bool(self.settings.reviewers)

    def critique(
        self, state: _loop_state.LoopState, *, trigger: str, turn: int
    ) -> CritiqueResult | None:
        """Run the panel over the run diff.

        Past `max_review_rejections` rejected ends the panel still sits and its findings still
        reach the model, but nothing blocks: a gating panel cannot stall the run.

        Args:
            state: The execution's state.
            trigger: What called the panel, for the events.
            turn: The current turn.

        Returns:
            The critique, `satisfied` False only when the panel blocks and the gate is armed,
            `capped` when it blocks past the cap; None when the panel was skipped (no diff, a
            scarce budget, a spent budget).
        """
        panel_id = uuid.uuid4().hex
        identity = dict(
            panel_id=panel_id, execution_id=state.execution_id, turn=turn, trigger=trigger
        )
        try:
            target = self.chain.review_target()
            diff = (
                git_read.diff_range(self.chain.root, target.base_tree, target.head_tree)
                if target
                else ""
            )
        except (errors.GitError, OSError) as exc:
            self.emit("loop.review.skipped", **identity, reason="target_unreadable", error=str(exc))
            return None
        if target is None or not diff.strip():
            self.emit(
                "loop.review.skipped",
                **identity,
                reason="no_target" if target is None else "no_diff",
            )
            return None
        remaining = self.budget_remaining()
        if remaining is not None and remaining < self.settings.min_remaining_budget_fraction:
            self.emit(
                "loop.review.skipped",
                **identity,
                reason="min_remaining_budget_fraction",
                remaining=round(remaining, 3),
            )
            return None
        # Only the before-finish panel gates.
        block_on: _panel.ReviewBlockOn = (
            self.settings.block_on if trigger == "before_finish" else "none"
        )
        known, prior = _review_history.matching_findings(
            self.chain.root, target, state.review_panels
        )
        ctx = _panel.ReviewContext(
            task=state.original_task,
            agents_md=_context.agents_md_text(self.chain.root, self.hide),
            diff=diff,
            verify_ok=state.verify.last_ok,
            verify_output=state.verify.last_tail,
            prior_findings=prior,
        )
        self.emit(
            "loop.review.start",
            **identity,
            target=pydantic.TypeAdapter(review_records.ReviewTarget).dump_python(
                target, mode="json"
            ),
            seats=[
                dict(
                    seat_index=i,
                    focus=s.focus,
                    model_route=s.model,
                    scope=s.scope,
                    temperature=s.temperature,
                )
                for i, s in enumerate(self.settings.reviewers)
            ],
        )
        tools: list[ToolDefinition] | None = None
        dispatch: _review_seat.ReviewDispatch | None = None
        if any(s.scope == "repo" for s in self.settings.reviewers):
            tools, dispatch = self.review_tools()
        try:
            result = run_panel(
                self.settings.reviewers,
                ctx,
                block_on=block_on,
                min_blocking_models=self.settings.min_blocking_models,
                panel_id=panel_id,
                repo_max_turns=self.settings.repo_max_turns,
                concurrency=self.settings.concurrency,
                tools=tools,
                dispatch=dispatch,
            )
        except budget.BudgetExceededError:
            self.emit("loop.review.skipped", **identity, reason="budget")
            return None
        for verdict in result.per_seat:
            self.emit(
                "loop.review.seat",
                **identity,
                **pydantic.TypeAdapter(review_records.ReviewVerdict).dump_python(
                    verdict, mode="json"
                ),
            )
        disarmed = state.gates.review_rejections >= self.settings.max_review_rejections
        effective_blocked = result.blocked and not disarmed
        inconclusive = _panel.panel_is_inconclusive(result)
        shown = (
            ()
            if inconclusive
            else _panel.findings_within(
                result.merged_findings, _panel.REVIEW_NOTICE_BYTES - _NOTICE_ROOM
            )
        )
        observations = _review_history.observations(panel_id, result.per_seat, known)
        shown_keys = {review_records.finding_key(f) for f in shown}
        record = review_records.ReviewPanel(
            panel_id=panel_id,
            execution_id=state.execution_id,
            turn=turn,
            trigger=trigger,
            target=target,
            findings=observations,
            injected_finding_ids=tuple(
                o.finding_id
                for o in observations
                if review_records.finding_key(o.finding) in shown_keys
            ),
            block_on=block_on,
            blocked=effective_blocked,
            raw_blocked=result.blocked,
            disarmed=disarmed,
            n_block=result.n_block,
            n_abstain=result.n_abstain,
            completion="inconclusive" if inconclusive else "complete",
        )
        self.emit("loop.review.panel", **record.model_dump(mode="json"))
        state.review_panels += (record,)
        if inconclusive:
            text = _panel.inconclusive_note(result)
        else:
            text = _panel.render_findings(shown) or "No new findings."
            if cut := len(result.merged_findings) - len(shown):
                text += f"\n[{cut} more finding{'s' if cut > 1 else ''} not shown]"
        return CritiqueResult(
            text=text,
            satisfied=not effective_blocked,
            capped=result.blocked and disarmed and trigger == "before_finish",
        )
