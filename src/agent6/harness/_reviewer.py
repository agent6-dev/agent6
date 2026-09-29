# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the review panel's seats.

A seat is one reviewer: a grounded call over the diff and verify result that returns a
`ReviewVerdict`. The network calls live here; the grounding and aggregation stay in `_panel`,
testable without a provider.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from agent6 import budget
from agent6.config import ReviewTier
from agent6.harness import _chain, _context, _llm_json, _panel
from agent6.prompts import review
from agent6.providers import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    output_cap_truncated,
)
from agent6.tools import results

if TYPE_CHECKING:
    from agent6.harness import _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class CritiqueResult:
    """Hold the panel's verdict as the triggers consume it.

    Attributes:
        text: The findings text injected for the worker.
        satisfied: False only when a blocking decision mode rejects.
    """

    text: str
    satisfied: bool


# A read-only dispatch callable for explore-tier seats: (tool_name, input) -> result.
ReviewDispatch = Callable[[str, dict[str, Any]], results.ToolResult]


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewSeat:
    """Bind one persona to a provider and model.

    Attributes:
        persona: The reviewer's stance.
        model: The model name, for events.
        provider: The provider called.
        tier: `diff` for one call over the diff, `explore` for a read-only tool loop first.
    """

    persona: str
    model: str
    provider: Provider
    tier: ReviewTier = "diff"


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewSettings:
    """Hold the in-loop review panel's settings, from `[review]`.

    The findings return to the model on the next user turn whatever the decision mode.

    Attributes:
        trigger: When the panel sits: on a verify failure, before a finish, or every `period`
            iterations; `off` never.
        period: The iterations between periodic panels.
        seats: The seats.
        decision: `advisory` only injects the findings; `veto` and `quorum` can reject an end.
        quorum: The blocks a `quorum` rejection needs.
        max_total_rejections: The blocks after which the gate disarms to advisory for the run.
        budget_fraction: The remaining budget fraction below which the panel is skipped.
        concurrency: The seat calls run at once.
        max_consecutive_rejections: The back-to-back rejections after which the next end is
            accepted; 0 disables the cap.
    """

    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    period: int = 10
    seats: Sequence[ReviewSeat] = ()
    decision: _panel.ReviewDecision = "advisory"
    quorum: int = 2
    max_total_rejections: int = 4
    budget_fraction: float = 0.25
    concurrency: int = 1
    max_consecutive_rejections: int = 2


def _build_user_message(ctx: _panel.ReviewContext) -> str:
    parts: list[str] = [f"TASK:\n{ctx.task.strip()[:4000]}"]
    if ctx.agents_md.strip():
        parts.append(f"AGENTS.md:\n{ctx.agents_md.strip()[:8000]}")
    if ctx.verify_ok is None:
        parts.append("VERIFY: not run.")
    else:
        status = "PASSED" if ctx.verify_ok else "FAILED"
        out = ctx.verify_output.strip()[-2000:]
        parts.append(f"VERIFY: {status}\n{out}" if out else f"VERIFY: {status}")
    if ctx.prior_findings:
        already = "; ".join(f"{f.file_line} {f.category}" for f in ctx.prior_findings[:20])
        parts.append(f"ALREADY RAISED (do not repeat): {already}")
    parts.append(f"DIFF:\n{ctx.diff}")
    return "\n\n".join(parts)


def _coerce_findings(raw: object) -> tuple[_panel.Finding, ...]:
    out: list[_panel.Finding] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "other"))
        if category not in _panel.ALL_CATEGORIES:
            category = "other"
        severity = str(item.get("severity", "warn"))
        if severity not in ("block", "warn", "nit"):
            severity = "warn"
        out.append(
            _panel.Finding(
                category=category,
                severity=severity,  # type: ignore[arg-type]
                file_line=" ".join(str(item.get("file_line", "")).split()),
                title=" ".join(str(item.get("title", "")).split())[:200],
                detail=" ".join(str(item.get("detail", "")).split())[:1000],
            )
        )
    return tuple(out)


def _no_verdict_error(resp: ProviderResponse) -> str:
    """Return why a seat produced no verdict JSON: the output cap, no content, or junk.

    Args:
        resp: The seat's response.

    Returns:
        The error text for the abstaining verdict.
    """
    if output_cap_truncated(resp):
        detail = (
            "before emitting any content (likely all reasoning)"
            if not resp.text.strip()
            else "mid-answer, before the verdict JSON completed"
        )
        return (
            f"output hit the cap {detail}"
            f" (stop_reason={resp.stop_reason}, {resp.output_tokens} output tokens);"
            " raise max_tokens or use a model with more output headroom"
        )
    if not resp.text.strip():
        # The reasoning produced tells a starved reasoner from an upstream error.
        thought = sum(
            len(str(block.get("thinking") or ""))
            for block in (resp.raw.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "thinking"
        )
        channel = f", {thought:,} chars of it in the reasoning channel" if thought else ""
        return (
            f"the reviewer returned no content (stop_reason={resp.stop_reason},"
            f" {resp.output_tokens} output tokens{channel})"
        )
    return "unparseable reviewer output"


def structured_review(
    provider: Provider, ctx: _panel.ReviewContext, *, seat: str, model: str, max_tokens: int = 1500
) -> _panel.ReviewVerdict:
    """Run one diff-tier seat.

    Args:
        provider: The seat's provider.
        ctx: The review context, with the seat's persona.
        seat: The seat's name.
        model: The model name, for the verdict.
        max_tokens: The output cap of the call.

    Returns:
        The seat's verdict; a provider error or junk output abstains with `error` set.
    """
    system = review.REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": _build_user_message(ctx)}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        return _panel.ReviewVerdict(
            seat=seat, model=model, verdict="pass", error=f"provider: {exc}"
        )
    obj = _llm_json.extract_json(resp.text, prefer=("verdict", "findings"))
    if obj is None:
        return _panel.ReviewVerdict(
            seat=seat, model=model, verdict="pass", error=_no_verdict_error(resp)
        )
    return _verdict_from_obj(obj, seat, model)


def _verdict_from_obj(obj: dict[str, Any], seat: str, model: str) -> _panel.ReviewVerdict:
    raw_verdict = obj.get("verdict")
    if not isinstance(raw_verdict, str) or raw_verdict.lower() not in ("pass", "block"):
        return _panel.ReviewVerdict(
            seat=seat, model=model, verdict="pass", error="invalid reviewer verdict"
        )
    findings = _coerce_findings(obj.get("findings"))
    verdict = "block" if raw_verdict.lower() == "block" else "pass"
    return _panel.ReviewVerdict(
        seat=seat,
        model=model,
        verdict=verdict,
        findings=findings,
        summary=str(obj.get("summary", "")).strip()[:300],
    )


def explore_review(
    provider: Provider,
    ctx: _panel.ReviewContext,
    *,
    seat: str,
    model: str,
    tools: list[ToolDefinition],
    dispatch: ReviewDispatch,
    max_iters: int = 6,
    max_tokens: int = 2000,
    deadline_s: float = 90.0,
) -> _panel.ReviewVerdict:
    """Run one explore-tier seat: a bounded loop of read-only tool calls, then a verdict.

    Args:
        provider: The seat's provider.
        ctx: The review context, with the seat's persona.
        seat: The seat's name.
        model: The model name, for the verdict.
        tools: The read-only tools offered.
        dispatch: The dispatch that refuses every other tool.
        max_iters: The provider calls allowed.
        max_tokens: The output cap of each call.
        deadline_s: The wall-clock budget.

    Returns:
        The seat's verdict; a provider error, the deadline or no verdict in time abstains.
    """
    system = review.EXPLORE_REVIEW_SYSTEM_PROMPT.format(
        persona=ctx.persona or "general correctness"
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_message(ctx)}]
    start = time.monotonic()
    for i in range(max_iters):
        if time.monotonic() - start > deadline_s:
            return _panel.ReviewVerdict(
                seat=seat, model=model, verdict="pass", error="explore: deadline exceeded"
            )
        try:
            resp = provider.call(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except ProviderError as exc:
            return _panel.ReviewVerdict(
                seat=seat, model=model, verdict="pass", error=f"provider: {exc}"
            )
        messages.append({"role": "assistant", "content": resp.raw.get("content") or []})
        if not resp.tool_uses:
            obj = _llm_json.extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is None:
                return _panel.ReviewVerdict(
                    seat=seat, model=model, verdict="pass", error=_no_verdict_error(resp)
                )
            return _verdict_from_obj(obj, seat, model)
        # On the last iteration a verdict beside tool calls counts; without one no tool runs.
        if i == max_iters - 1:
            obj = _llm_json.extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is not None and ("verdict" in obj or "findings" in obj):
                return _verdict_from_obj(obj, seat, model)
            break
        tool_results: list[dict[str, Any]] = []
        for tu in resp.tool_uses:
            name = tu.get("name", "")
            tu_id = tu.get("id", "")
            try:
                out = dispatch(name, tu.get("input", {}) or {})
                content = json.dumps(out.to_wire(), ensure_ascii=False)[:8000]
            except Exception as exc:
                content = f"error: {exc}"[:2000]
            tool_results.append({"type": "tool_result", "tool_use_id": tu_id, "content": content})
        messages.append({"role": "user", "content": tool_results})
    return _panel.ReviewVerdict(
        seat=seat, model=model, verdict="pass", error="explore: no verdict within max_iters"
    )


def run_panel(
    seats: Sequence[ReviewSeat],
    ctx: _panel.ReviewContext,
    *,
    decision: _panel.ReviewDecision,
    quorum: int,
    panel_id: str,
    concurrency: int = 1,
    tools: list[ToolDefinition] | None = None,
    dispatch: ReviewDispatch | None = None,
) -> _panel.PanelResult:
    """Run every seat over the same context and aggregate the verdicts in seat order.

    Args:
        seats: The seats.
        ctx: The review context; each seat gets its own persona substituted.
        decision: The decision mode.
        quorum: The blocks a `quorum` rejection needs.
        panel_id: The panel's id, for the result.
        concurrency: The seat calls run at once.
        tools: The read-only tools for explore seats.
        dispatch: The read-only dispatch for explore seats.

    Returns:
        The aggregated panel result.
    """

    def _run(s: ReviewSeat) -> _panel.ReviewVerdict:
        seat_ctx = dataclasses.replace(ctx, persona=s.persona)
        if s.tier == "explore" and tools is not None and dispatch is not None:
            return explore_review(
                s.provider, seat_ctx, seat=s.persona, model=s.model, tools=tools, dispatch=dispatch
            )
        return structured_review(s.provider, seat_ctx, seat=s.persona, model=s.model)

    if concurrency > 1 and len(seats) > 1:
        verdicts = _run_seats_concurrently(seats, _run, concurrency)
    else:
        verdicts = [_run(s) for s in seats]
    return _panel.aggregate_verdicts(
        verdicts, ctx, decision=decision, quorum=quorum, panel_id=panel_id
    )


def _run_seats_concurrently(
    seats: Sequence[ReviewSeat],
    run_seat: Callable[[ReviewSeat], _panel.ReviewVerdict],
    concurrency: int,
) -> list[_panel.ReviewVerdict]:
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
    slots: list[_panel.ReviewVerdict | None] = [None] * len(seats)
    errors: list[BaseException] = []
    gate = threading.Semaphore(min(concurrency, len(seats)))
    done = threading.Semaphore(0)

    def work(i: int, seat: ReviewSeat) -> None:
        with gate:
            try:
                slots[i] = run_seat(seat)
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


__all__ = [
    "ReviewDispatch",
    "ReviewSeat",
    "explore_review",
    "run_panel",
    "structured_review",
]


# The before-finish panel's rejection by the ending it rejected; the findings follow.
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
        review_tools: Builds the read-only tools and dispatch an explore seat gets.
        budget_remaining: The fraction of the budget left, or None without a tracker.
        log: The run's text logger.
        emit: The run's event emitter.
    """

    settings: ReviewSettings
    chain: _chain.RunChain
    review_tools: Callable[[], tuple[list[ToolDefinition], ReviewDispatch]]
    budget_remaining: Callable[[], float | None]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def triggers(self, state: _loop_state.LoopState, turn: _loop_state.TurnState) -> None:
        """Sit the observe-only panels: after a verify failure, or every `period` iterations.

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
            critique = self.critique(state, trigger="verify_failed", iteration=turn.iteration)
            if critique is not None:
                turn.review_text = critique.text
        elif (
            self.settings.trigger == "periodic"
            and self.available()
            and turn.iteration % max(1, self.settings.period) == 0
        ):
            critique = self.critique(state, trigger="periodic", iteration=turn.iteration)
            if critique is not None:
                turn.review_text = critique.text

    def end_rejected(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, *, ending: str
    ) -> bool:
        """Sit the before-finish panel over an end, once per turn.

        After `max_consecutive_rejections` back-to-back rejections the end goes through with the
        findings injected. A turn that declares two ends gets one verdict for both.

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
        critique = self.critique(state, trigger="before_finish", iteration=turn.iteration)
        if critique is None:
            return False
        cap = self.settings.max_consecutive_rejections
        cap_reached = cap > 0 and state.gates.review_consecutive >= cap
        if not critique.satisfied and not cap_reached:
            self.log(f"  review rejected {ending} at iter {turn.iteration}")
            self.emit("loop.review.rejected_finish", iteration=turn.iteration, ending=ending)
            state.gates.review_consecutive += 1
            turn.review_text = REVIEW_REJECTED[ending] + critique.text
            return True
        if not critique.satisfied:
            self.log(
                f"  review rejected {ending} at iter {turn.iteration} but"
                f" rejection cap ({cap}) reached - letting the end through"
            )
            self.emit(
                "loop.review.rejection_cap_reached",
                iteration=turn.iteration,
                rejections=state.gates.review_consecutive,
            )
            turn.review_text = (
                "The review panel flagged issues but the rejection cap was"
                " reached; the end stands. Findings:\n\n" + critique.text
            )
        else:
            self.log(f"  review approved {ending}")
        state.gates.review_consecutive = 0
        return False

    def available(self) -> bool:
        """Return whether the panel has seats; every in-loop trigger gates on it."""
        return bool(self.settings.seats)

    def critique(
        self, state: _loop_state.LoopState, *, trigger: str, iteration: int
    ) -> CritiqueResult | None:
        """Run the panel over the run diff.

        The before-finish rejection counter decays on a pass and disarms the gate at its cap, so
        a gating panel cannot stall the run.

        Args:
            state: The execution's state.
            trigger: What called the panel, for the events.
            iteration: The current turn.

        Returns:
            The critique, `satisfied` False only when the panel blocks and the gate is armed;
            None when the panel was skipped (no diff, a scarce budget, a spent budget).
        """
        diff = self.chain.diff_since_base()
        if not diff.strip():
            self.emit("loop.review.skipped", iteration=iteration, trigger=trigger, reason="no_diff")
            return None
        # A skipped panel approves: the gate blocks only on an explicit unsatisfied critique.
        remaining = self.budget_remaining()
        if remaining is not None and remaining < self.settings.budget_fraction:
            self.emit(
                "loop.review.skipped",
                iteration=iteration,
                trigger=trigger,
                reason="budget_fraction",
                remaining=round(remaining, 3),
            )
            return None
        # Only the before-finish panel gates.
        decision: _panel.ReviewDecision = (
            self.settings.decision if trigger == "before_finish" else "advisory"
        )
        ctx = _panel.ReviewContext(
            task=state.original_task,
            agents_md=_context.agents_md_text(self.chain.root),
            diff=diff,
            verify_ok=state.verify.last_ok,
            verify_output=state.verify.last_tail,
        )
        self.emit(
            "loop.review.start",
            iteration=iteration,
            trigger=trigger,
            seats=len(self.settings.seats),
        )
        tools: list[ToolDefinition] | None = None
        dispatch: ReviewDispatch | None = None
        if any(s.tier == "explore" for s in self.settings.seats):
            tools, dispatch = self.review_tools()
        try:
            result = run_panel(
                self.settings.seats,
                ctx,
                decision=decision,
                quorum=self.settings.quorum,
                panel_id=f"{trigger}-{iteration}",
                concurrency=self.settings.concurrency,
                tools=tools,
                dispatch=dispatch,
            )
        except budget.BudgetExceededError:
            self.emit("loop.review.skipped", iteration=iteration, reason="budget")
            return None
        for v in result.per_seat:
            self.emit(
                "loop.review.seat",
                iteration=iteration,
                seat=v.seat,
                model=v.model,
                verdict="abstain" if v.error else v.verdict,
                findings=len(v.findings),
            )
        disarmed = state.gates.review_total >= self.settings.max_total_rejections
        effective_blocked = result.blocked and not disarmed
        self.emit(
            "loop.review.panel",
            iteration=iteration,
            trigger=trigger,
            decision=decision,
            blocked=effective_blocked,
            raw_blocked=result.blocked,
            disarmed=disarmed,
            n_block=result.n_block,
            n_abstain=result.n_abstain,
        )
        if trigger == "before_finish":
            if effective_blocked:
                state.gates.review_total += 1
            else:
                state.gates.review_total = max(0, state.gates.review_total - 1)
        # An all-abstain panel reviewed nothing and says so; it still lets the end through.
        if _panel.panel_is_inconclusive(result):
            text = _panel.inconclusive_note(result)
        else:
            text = _panel.render_findings(result.merged_findings) or "No blocking findings."
        return CritiqueResult(text=text, satisfied=not effective_blocked)
