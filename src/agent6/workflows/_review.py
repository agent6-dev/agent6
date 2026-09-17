# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review-panel seat call + sequential orchestration.

A *seat* is one adversarial reviewer: a single grounded LLM call over the diff
(+ verify result) that returns a structured `ReviewVerdict`. `run_panel` runs
the seats and folds them with the pure `aggregate_verdicts` (in `_panel`).
The grounding that prevents false blocks is enforced in the aggregator; this
module asks each model for findings in a parseable shape.

Network calls live here (each seat takes an injected `Provider`); the pure
grounding/aggregation stays in `_panel` so it is testable without the network.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from agent6.budget import BudgetExceeded
from agent6.config import ReviewTier
from agent6.prompts.review import EXPLORE_REVIEW_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT
from agent6.providers import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    output_cap_truncated,
)
from agent6.tools.results import ToolResult
from agent6.workflows._chain import RunChain
from agent6.workflows._context import agents_md_text
from agent6.workflows._llm_json import extract_json
from agent6.workflows._panel import (
    ALL_CATEGORIES,
    Finding,
    PanelResult,
    ReviewContext,
    ReviewDecision,
    ReviewVerdict,
    aggregate_verdicts,
    inconclusive_note,
    panel_is_inconclusive,
    render_findings,
)

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState, TurnState


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    """The in-loop panel's verdict the trigger logic consumes: the findings
    text injected for the worker, and whether the panel is satisfied
    (`satisfied=False` only when a blocking decision mode rejects)."""

    text: str
    satisfied: bool


# A read-only dispatch callable for explore-tier seats: (tool_name, input) -> result.
ReviewDispatch = Callable[[str, dict[str, Any]], ToolResult]


@dataclass(frozen=True, slots=True)
class ReviewSeat:
    """One panel seat: a persona stance bound to a provider/model.

    `tier` is "diff" (a single grounded call over the diff) or "explore" (a
    read-only tool-using mini-loop that investigates the broader repo first);
    typed as the config's `ReviewTier` Literal, the vocabulary's one owner."""

    persona: str
    model: str
    provider: Provider
    tier: ReviewTier = "diff"


@dataclass(frozen=True, slots=True)
class ReviewSettings:
    """The in-loop review panel, as the run configures it (`[review]`). The
    panel runs at `trigger` (on a verify failure, before a finish, or every
    `period` iterations; `off` never) over the run diff, with one call per
    seat, and its findings return to the model on the next user turn.
    `decision` gates only for veto/quorum; `advisory` just injects the
    findings. `max_total_rejections` blocks disarm the gate to advisory for
    the rest of the run, and after `max_consecutive_rejections` back-to-back
    before-finish rejections the next finish is accepted (with the review
    still injected), so neither can stall the run; 0 disables the latter."""

    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    period: int = 10
    seats: Sequence[ReviewSeat] = ()
    decision: ReviewDecision = "advisory"
    quorum: int = 2
    max_total_rejections: int = 4
    budget_fraction: float = 0.25
    concurrency: int = 1
    max_consecutive_rejections: int = 2


def _build_user_message(ctx: ReviewContext) -> str:
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


def _coerce_findings(raw: object) -> tuple[Finding, ...]:
    out: list[Finding] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "other"))
        if category not in ALL_CATEGORIES:
            category = "other"
        severity = str(item.get("severity", "warn"))
        if severity not in ("block", "warn", "nit"):
            severity = "warn"
        out.append(
            Finding(
                category=category,
                severity=severity,  # type: ignore[arg-type]
                file_line=" ".join(str(item.get("file_line", "")).split()),
                title=" ".join(str(item.get("title", "")).split())[:200],
                detail=" ".join(str(item.get("detail", "")).split())[:1000],
            )
        )
    return tuple(out)


def _no_verdict_error(resp: ProviderResponse) -> str:
    """Why a seat produced no verdict JSON, in the reviewer's own terms: the
    output cap ate the answer, the reviewer returned nothing to parse, or what
    it returned would not parse: a generic "unparseable reviewer output" over
    the first two would blame the parser for the provider's own truncation or
    error."""
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
        # No content at all: an upstream error, or a reasoning model that spent
        # its whole budget in the reasoning channel. Naming the reasoning it
        # DID produce separates the two, and a seat that only ever thinks is
        # the operator's to re-route.
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
    provider: Provider, ctx: ReviewContext, *, seat: str, model: str, max_tokens: int = 1500
) -> ReviewVerdict:
    """Run one seat. Returns a ReviewVerdict; any failure (provider error, junk
    output) yields an ABSTAINING verdict (`error` set) -- never a false pass."""
    system = REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": _build_user_message(ctx)}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        return ReviewVerdict(seat=seat, model=model, verdict="pass", error=f"provider: {exc}")
    obj = extract_json(resp.text, prefer=("verdict", "findings"))
    if obj is None:
        return ReviewVerdict(seat=seat, model=model, verdict="pass", error=_no_verdict_error(resp))
    return _verdict_from_obj(obj, seat, model)


def _verdict_from_obj(obj: dict[str, Any], seat: str, model: str) -> ReviewVerdict:
    raw_verdict = obj.get("verdict")
    if not isinstance(raw_verdict, str) or raw_verdict.lower() not in ("pass", "block"):
        return ReviewVerdict(
            seat=seat, model=model, verdict="pass", error="invalid reviewer verdict"
        )
    findings = _coerce_findings(obj.get("findings"))
    verdict = "block" if raw_verdict.lower() == "block" else "pass"
    return ReviewVerdict(
        seat=seat,
        model=model,
        verdict=verdict,
        findings=findings,
        summary=str(obj.get("summary", "")).strip()[:300],
    )


def explore_review(
    provider: Provider,
    ctx: ReviewContext,
    *,
    seat: str,
    model: str,
    tools: list[ToolDefinition],
    dispatch: ReviewDispatch,
    max_iters: int = 6,
    max_tokens: int = 2000,
    deadline_s: float = 90.0,
) -> ReviewVerdict:
    """A read-only tool-using reviewer: a bounded mini-loop where the seat may
    call read-only tools to investigate the repo, then emits a ReviewVerdict.
    Tools are an explicit read-only allowlist enforced by the caller's dispatch;
    any failure (provider error, deadline, no verdict within max_iters) ABSTAINS."""
    system = EXPLORE_REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_message(ctx)}]
    start = time.monotonic()
    for i in range(max_iters):
        if time.monotonic() - start > deadline_s:
            return ReviewVerdict(
                seat=seat, model=model, verdict="pass", error="explore: deadline exceeded"
            )
        try:
            resp = provider.call(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except ProviderError as exc:
            return ReviewVerdict(seat=seat, model=model, verdict="pass", error=f"provider: {exc}")
        messages.append({"role": "assistant", "content": resp.raw.get("content") or []})
        if not resp.tool_uses:
            obj = extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is None:
                return ReviewVerdict(
                    seat=seat, model=model, verdict="pass", error=_no_verdict_error(resp)
                )
            return _verdict_from_obj(obj, seat, model)
        # On the last allowed iteration, a verdict emitted ALONGSIDE tool calls
        # still counts (don't waste the investigation by abstaining). With no
        # verdict, skip the dispatches: no model call follows to consume their
        # results, so executing them only spends tool time on an abstention.
        if i == max_iters - 1:
            obj = extract_json(resp.text, prefer=("verdict", "findings"))
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
    return ReviewVerdict(
        seat=seat, model=model, verdict="pass", error="explore: no verdict within max_iters"
    )


def run_panel(
    seats: Sequence[ReviewSeat],
    ctx: ReviewContext,
    *,
    decision: ReviewDecision,
    quorum: int,
    panel_id: str,
    concurrency: int = 1,
    tools: list[ToolDefinition] | None = None,
    dispatch: ReviewDispatch | None = None,
) -> PanelResult:
    """Run every seat and aggregate. Each seat sees the same context with its own
    persona substituted. With `concurrency > 1` the seat calls run on a thread
    pool (the shared budget tracker + transcript sink are both lock-protected, and
    each seat has its own provider); results stay in seat order, so the merged
    verdict is deterministic regardless of how the calls interleave."""

    def _run(s: ReviewSeat) -> ReviewVerdict:
        seat_ctx = replace(ctx, persona=s.persona)
        if s.tier == "explore" and tools is not None and dispatch is not None:
            return explore_review(
                s.provider, seat_ctx, seat=s.persona, model=s.model, tools=tools, dispatch=dispatch
            )
        return structured_review(s.provider, seat_ctx, seat=s.persona, model=s.model)

    if concurrency > 1 and len(seats) > 1:
        verdicts = _run_seats_concurrently(seats, _run, concurrency)
    else:
        verdicts = [_run(s) for s in seats]
    return aggregate_verdicts(verdicts, ctx, decision=decision, quorum=quorum, panel_id=panel_id)


def _run_seats_concurrently(
    seats: Sequence[ReviewSeat],
    run_seat: Callable[[ReviewSeat], ReviewVerdict],
    concurrency: int,
) -> list[ReviewVerdict]:
    """Run the seat calls on daemon threads; results stay in seat order.

    Deliberately not a ThreadPoolExecutor: its workers are non-daemon and
    joined at interpreter exit, and an in-flight seat call is a non-streaming
    provider POST with no abort hook -- Ctrl-C on `agent6 review` would hang
    until every in-flight AND queued seat finished.
    Daemon threads die with the process, and the timeout-polling wait lets
    KeyboardInterrupt land promptly on the main thread."""
    slots: list[ReviewVerdict | None] = [None] * len(seats)
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


# The before-finish panel's rejection, by the ending it rejected; the
# findings follow.
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


@dataclass(frozen=True, slots=True)
class Reviewer:
    """The in-loop review panel for one run: its settings, the run's chain
    (the diff it grounds on, the AGENTS.md it reads), the read-only tools an
    explore seat gets, and the run's budget, log and event callables.
    `critique` runs the panel over the run diff; `triggers` is the
    observe-only schedule; `end_reviewed` the before-finish verdict over an
    end."""

    settings: ReviewSettings
    chain: RunChain
    review_tools: Callable[[], tuple[list[ToolDefinition], ReviewDispatch]]
    budget_remaining: Callable[[], float | None]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def triggers(self, state: LoopState, turn: TurnState) -> None:
        """The observe-only review triggers (before_finish, which can revoke a
        finish, is `end_reviewed`):

          on_verify_fail - the verify just failed; surface a critique
                           alongside the failure so the worker has a second
                           opinion before its next edit.
          periodic       - every ReviewSettings.period iterations.
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

    def end_reviewed(self, state: LoopState, turn: TurnState, *, ending: str) -> bool:
        """The before-finish panel over an end (`finish_session`, a silent
        finish, or the settled stop or metric plateau the harness declares):
        True when the panel rejected it
        and the run carries on with the findings injected. After
        `ReviewSettings.max_consecutive_rejections` back-to-back rejections the end
        goes through (findings still injected) so the worker can't bounce
        indefinitely. False when there is no panel or it approved. One turn
        can declare two ends (a finish a gate revokes, then the plateau or
        settled stop): the panel sits once and its verdict covers both."""
        if turn.end_reviewed is None:
            turn.end_reviewed = self._judge_end(state, turn, ending=ending)
        return turn.end_reviewed

    def _judge_end(self, state: LoopState, turn: TurnState, *, ending: str) -> bool:
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
        """A second opinion is available: the review panel has seats. Gates
        every in-loop review trigger."""
        return bool(self.settings.seats)

    def critique(self, state: LoopState, *, trigger: str, iteration: int) -> CritiqueResult | None:
        """Run the grounded review panel over the run diff. Returns a
        `CritiqueResult` (`satisfied=False` only when the panel BLOCKS and
        the gate is still armed). Per-seat + panel events are emitted in seat
        order; the per-run rejection counter decays on a pass and disarms the gate
        once it hits the cap so a gating panel can never stall the run."""
        diff = self.chain.diff_since_base()
        if not diff.strip():
            # No diff to ground against (nothing changed, or base_sha missing on a
            # pre-field resume). Can't review -> approve, but make the skip visible
            # so a "gate didn't run" is never silent.
            self.emit("loop.review.skipped", iteration=iteration, trigger=trigger, reason="no_diff")
            return None
        # Skip the panel once the run's remaining token budget falls below
        # ReviewSettings.budget_fraction: reviewing is most expensive (esp. explore-tier
        # seats) exactly when budget is scarcest, and a skipped panel is
        # approve-and-proceed (the before_finish gate only blocks on an explicit
        # unsatisfied critique, so returning None here lets finish through). This
        # is the sole read site for ReviewSettings.budget_fraction.
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
        # on_verify_fail/periodic never gate (advisory text only); only
        # before_finish consumes .satisfied + the rejection counter.
        decision: ReviewDecision = (
            self.settings.decision if trigger == "before_finish" else "advisory"
        )
        ctx = ReviewContext(
            task=state.original_task,
            # The same text the run prompt injects (repo root's file included on
            # a subdirectory start), so review and worker see one set of conventions.
            agents_md=agents_md_text(self.chain.root),
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
        except BudgetExceeded:
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
        # An all-abstain panel reviewed nothing: name that in the critique text
        # (the model reads it) instead of "No blocking findings.". The gate still
        # lets the finish through -- a panel must never deadlock a run -- so
        # `satisfied` is unchanged.
        if panel_is_inconclusive(result):
            text = inconclusive_note(result)
        else:
            text = render_findings(result.merged_findings) or "No blocking findings."
        return CritiqueResult(text=text, satisfied=not effective_blocked)
