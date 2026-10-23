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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from agent6 import budget
from agent6.config import ReviewScope
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
        satisfied: False only when a blocking block_on mode rejects.
        capped: The panel blocked but the run's rejected finishes reached the cap, so the end
            stands with the findings.
    """

    text: str
    satisfied: bool
    capped: bool = False


# A read-only dispatch callable for repo-scope seats: (tool_name, input) -> result.
ReviewDispatch = Callable[[str, dict[str, Any]], results.ToolResult]


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
    provider: Provider,
    ctx: _panel.ReviewContext,
    *,
    seat: str,
    model: str,
    max_tokens: int = 1500,
    temperature: float | None = 0.0,
) -> _panel.ReviewVerdict:
    """Run one diff-scope seat.

    Args:
        provider: The seat's provider.
        ctx: The review context, with the seat's focus.
        seat: The seat's name.
        model: The model name, for the verdict.
        max_tokens: The output cap of the call.
        temperature: The sampling temperature.

    Returns:
        The seat's verdict; a provider error or junk output abstains with `error` set.
    """
    system = review.REVIEW_SYSTEM_PROMPT.format(focus=ctx.focus or "general correctness")
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": _build_user_message(ctx)}],
            max_tokens=max_tokens,
            temperature=temperature,
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


def repo_review(
    provider: Provider,
    ctx: _panel.ReviewContext,
    *,
    seat: str,
    model: str,
    tools: list[ToolDefinition],
    dispatch: ReviewDispatch,
    max_turns: int,
    max_tokens: int = 2000,
    temperature: float | None = 0.0,
) -> _panel.ReviewVerdict:
    """Run one repo-scope seat with at most `max_turns` provider calls.

    Args:
        provider: The seat's provider.
        ctx: The review context, with the seat's focus.
        seat: The seat's name.
        model: The model name, for the verdict.
        tools: The read-only tools offered.
        dispatch: The dispatch that refuses every other tool.
        max_turns: The provider calls allowed.
        max_tokens: The output cap of each call.
        temperature: The sampling temperature on every call.

    Returns:
        The seat's verdict; a provider error or no valid verdict abstains with `error` set.
    """
    system = review.REPO_REVIEW_SYSTEM_PROMPT.format(
        focus=ctx.focus or "general correctness", max_turns=max_turns
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_message(ctx)}]
    for i in range(max_turns):
        try:
            resp = provider.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
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
        # On the last turn a verdict beside tool calls counts; without one no tool runs.
        if i == max_turns - 1:
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
        seat=seat, model=model, verdict="pass", error=f"repo: no verdict within {max_turns} calls"
    )


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
    dispatch: ReviewDispatch | None = None,
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

    def _run(s: ReviewSeat) -> _panel.ReviewVerdict:
        seat_ctx = dataclasses.replace(ctx, focus=s.focus)
        if s.scope == "repo" and tools is not None and dispatch is not None:
            return repo_review(
                s.provider,
                seat_ctx,
                seat=s.focus,
                model=s.model,
                tools=tools,
                dispatch=dispatch,
                max_turns=repo_max_turns,
                temperature=s.temperature,
            )
        return structured_review(
            s.provider, seat_ctx, seat=s.focus, model=s.model, temperature=s.temperature
        )

    if concurrency > 1 and len(seats) > 1:
        verdicts = _run_seats_concurrently(seats, _run, concurrency)
    else:
        verdicts = [_run(s) for s in seats]
    return _panel.aggregate_verdicts(
        verdicts, ctx, block_on=block_on, min_blocking_models=min_blocking_models, panel_id=panel_id
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
    "repo_review",
    "run_panel",
    "structured_review",
]


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
    review_tools: Callable[[], tuple[list[ToolDefinition], ReviewDispatch]]
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
        diff = self.chain.diff_since_base()
        if not diff.strip():
            self.emit("loop.review.skipped", turn=turn, trigger=trigger, reason="no_diff")
            return None
        # A skipped panel approves: the gate blocks only on an explicit unsatisfied critique.
        remaining = self.budget_remaining()
        if remaining is not None and remaining < self.settings.min_remaining_budget_fraction:
            self.emit(
                "loop.review.skipped",
                turn=turn,
                trigger=trigger,
                reason="min_remaining_budget_fraction",
                remaining=round(remaining, 3),
            )
            return None
        # Only the before-finish panel gates.
        block_on: _panel.ReviewBlockOn = (
            self.settings.block_on if trigger == "before_finish" else "none"
        )
        ctx = _panel.ReviewContext(
            task=state.original_task,
            agents_md=_context.agents_md_text(self.chain.root, self.hide),
            diff=diff,
            verify_ok=state.verify.last_ok,
            verify_output=state.verify.last_tail,
            prior_findings=state.review_findings,
        )
        self.emit(
            "loop.review.start",
            turn=turn,
            trigger=trigger,
            seats=len(self.settings.reviewers),
        )
        tools: list[ToolDefinition] | None = None
        dispatch: ReviewDispatch | None = None
        if any(s.scope == "repo" for s in self.settings.reviewers):
            tools, dispatch = self.review_tools()
        try:
            result = run_panel(
                self.settings.reviewers,
                ctx,
                block_on=block_on,
                min_blocking_models=self.settings.min_blocking_models,
                panel_id=f"{trigger}-{turn}",
                repo_max_turns=self.settings.repo_max_turns,
                concurrency=self.settings.concurrency,
                tools=tools,
                dispatch=dispatch,
            )
        except budget.BudgetExceededError:
            self.emit("loop.review.skipped", turn=turn, reason="budget")
            return None
        for v in result.per_seat:
            self.emit(
                "loop.review.seat",
                turn=turn,
                seat=v.seat,
                model=v.model,
                verdict="abstain" if v.error else v.verdict,
                findings=len(v.findings),
            )
        disarmed = state.gates.review_rejections >= self.settings.max_review_rejections
        effective_blocked = result.blocked and not disarmed
        self.emit(
            "loop.review.panel",
            turn=turn,
            trigger=trigger,
            block_on=block_on,
            blocked=effective_blocked,
            raw_blocked=result.blocked,
            disarmed=disarmed,
            n_block=result.n_block,
            n_abstain=result.n_abstain,
        )
        # An all-abstain panel reviewed nothing and says so; it still lets the end through.
        if _panel.panel_is_inconclusive(result):
            text = _panel.inconclusive_note(result)
        else:
            # Only a finding the worker is shown counts as injected; a cut one comes back.
            shown = _panel.findings_within(
                result.merged_findings, _panel.REVIEW_NOTICE_BYTES - _NOTICE_ROOM
            )
            state.review_findings += shown
            text = _panel.render_findings(shown) or "No blocking findings."
            if cut := len(result.merged_findings) - len(shown):
                text += f"\n[{cut} more finding{'s' if cut > 1 else ''} not shown]"
        return CritiqueResult(
            text=text,
            satisfied=not effective_blocked,
            capped=result.blocked and disarmed and trigger == "before_finish",
        )
