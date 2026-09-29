# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The finish gates: what an end must satisfy, what it is called, its refusals.

The loop runs a gate list in order over the turn that declared an end and
applies the first `Refusal`: `FINISH_GATES` over a finish_session,
`END_GATES` over an end the harness declares (settled, a plateau),
`SILENT_END_GATES` over a silent finish. `turn.ending` names the end judged.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from agent6.harness import _advice, _metric, _nudges, _snapshot, _verify_gate, _verify_verdict
from agent6.tools import schema

if TYPE_CHECKING:
    from agent6.harness import _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class FinishCall:
    """A dispatched finish call as the model sent it; the finish gates may revoke it.

    Attributes:
        kind: finish_session or finish_planning.
        summary: The model's summary.
        payload: finish_session's `result`.
        stale_gate: finish_session's claim that the configured gate is stale.
        plan_markdown: finish_planning's plan.
        plan_salvaged: The summary was folded under a title-only plan.
    """

    kind: Literal["finish_session", "finish_planning"]
    summary: str
    payload: dict[str, Any] | None = None
    stale_gate: str = ""
    plan_markdown: str = ""
    plan_salvaged: bool = False

    @classmethod
    def parse(cls, name: str, tool_input: Any) -> FinishCall | None:
        """Return the finish a dispatched tool call declares.

        The raw tool_input is what the model sent, so a malformed call still
        parses.

        Args:
            name: The tool's name.
            tool_input: The model's input to the tool.

        Returns:
            The call, or None for any other tool.
        """
        fields = tool_input if isinstance(tool_input, dict) else {}
        summary = str(fields.get("summary", "(no summary)"))
        if name == schema.FinishSessionInput.TOOL_NAME:
            raw_result = fields.get("result")
            if isinstance(raw_result, str):
                # A stringified result parses here; schema validation stays strict.
                try:
                    raw_result = json.loads(raw_result)
                except ValueError:
                    raw_result = None
            return cls(
                "finish_session",
                summary,
                payload=raw_result if isinstance(raw_result, dict) else None,
                stale_gate=str(fields.get("stale_gate", "")).strip(),
            )
        if name == schema.FinishPlanningInput.TOOL_NAME:
            plan_md = str(fields.get("plan_markdown", ""))
            # A plan left in `summary` under a bare title folds under it; the review
            # gate judged the content.
            salvaged = _plan_is_title_only(plan_md) and len(summary) > len(plan_md)
            if salvaged:
                title = next((ln for ln in plan_md.splitlines() if ln.strip()), "# Plan")
                plan_md = f"{title}\n\n{summary}"
            return cls("finish_planning", summary, plan_markdown=plan_md, plan_salvaged=salvaged)
        return None


def _plan_is_title_only(plan_md: str) -> bool:
    """Return whether the plan has no body: only heading lines and blanks.

    Args:
        plan_md: The plan markdown.

    Returns:
        True when no line is prose.
    """
    return not any(
        line.strip() and not line.lstrip().startswith("#") for line in plan_md.splitlines()
    )


@dataclasses.dataclass(slots=True)
class FinishGates:
    """The finish gates' counters.

    Attributes:
        task_nudges_used: Open-task refusals sent; `TASK_FINISH_PATIENCE` caps them.
        verify_retries_used: Red finish certifications returned; `verify_retries`
            caps them.
        review_consecutive: The before-finish panel's consecutive rejections; its
            cap lets the end through.
        review_total: The panel's run-total, persisted; past
            `max_total_rejections` the gate disarms to advisory.
    """

    task_nudges_used: int = 0
    verify_retries_used: int = 0
    review_consecutive: int = 0
    review_total: int = 0


def task_finish_nudge(open_tasks: Sequence[tuple[str, str]], gates: FinishGates) -> str | None:
    """Return the nudge to re-prompt with while the worker's own subtasks are open.

    After `TASK_FINISH_PATIENCE` refusals the end goes through and its receipt
    names the open tasks, so a worker that neither closes nor retires a task
    cannot bounce the loop for the whole budget.

    Args:
        open_tasks: The (id, title) pairs still open.
        gates: The counters; the nudge count advances here.

    Returns:
        The nudge, or None to let the end through.
    """
    if not open_tasks:
        return None
    if gates.task_nudges_used >= _nudges.TASK_FINISH_PATIENCE:
        return None  # cap reached: the end goes through, the receipt names them
    gates.task_nudges_used += 1
    listing = "\n".join(f"- {tid}: {title}" for tid, title in open_tasks)
    return (
        f"[harness] finish_session deferred: {len(open_tasks)} task(s) are"
        f" pending or in_progress:\n{listing}\n"
        "update_task marks one skipped or obsolete; the run ends once the"
        f" list is clear, or on the {_nudges.TASK_FINISH_PATIENCE + 1}th call."
    )


def red_gate_returns(
    verify_when: Literal["finish", "step", "never"],
    verify_retries: int,
    verify: _verify_verdict.VerifyVerdict,
    gates: FinishGates,
    *,
    gate_present: bool,
) -> bool:
    """Return whether a red gate is the model's to fix, so an end over it goes back.

    One answer for finish_session and the ends the harness declares, so neither
    hands back a gate the model cannot run.

    Args:
        verify_when: `[harness].verify_when`.
        verify_retries: `[harness].verify_retries`.
        verify: The run's verify verdict.
        gates: The finish gates' counters.
        gate_present: A gate exists and is the harness's to run.

    Returns:
        True when the gate exists, was not red before the run touched anything
        (or this run has since made it green), and returns are left.
    """
    return (
        verify_when != "never"
        and gate_present
        and (verify.baseline_ok is not False or verify.ever_passed)
        and gates.verify_retries_used < verify_retries
    )


def contract_refusal(problems: Sequence[str]) -> str:
    """Return the notice a finish_session with a `result` off the schema returns with.

    Args:
        problems: The validator's findings.

    Returns:
        The refusal text.
    """
    return (
        "finish_session refused: "
        + "; ".join(problems)
        + ". Call finish_session again with a `result` that satisfies the schema."
    )


def finish_reason(
    kind: _snapshot.SessionEndReason,
    *,
    stale_gate: str,
    tree_green: bool | None,
    verify: _verify_verdict.VerifyVerdict,
) -> _snapshot.SessionEndReason:
    """Return what a finish is called.

    `gate_stale` needs a gate that is red: green means it passed, and a gateless
    run has none to be stale. `gate_red_at_base` outranks a plain finish over
    red, since the gate was failing before this run touched anything; it comes
    from an observation, never a guess.

    Args:
        kind: The end as declared.
        stale_gate: The model's claim that the gate is stale, "" for none.
        tree_green: The gate's verdict on the tree, None on a gateless run.
        verify: The run's verify verdict.

    Returns:
        The end reason.
    """
    if kind == "finish_session" and tree_green is False:
        if stale_gate:
            return "gate_stale"
        if verify.baseline_ok is False and not verify.ever_passed:
            return "gate_red_at_base"
    return kind


def finish_contract(
    turn: _loop_state.TurnState, state: _loop_state.LoopState, ctx: _advice.TurnContext
) -> _advice.Refusal | None:
    """Return a finish_session whose `result` is off the machine state's schema.

    The retry happens in-execution instead of the execution ending failed over
    correct work. Uncapped: the budget and iteration backstops end a model that
    never conforms.

    Args:
        turn: The turn that declared the end.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The refusal with the problems, or None.
    """
    if ctx.finish_validator is None:
        return None
    problems = ctx.finish_validator(turn.finish.payload if turn.finish is not None else None)
    if not problems:
        return None
    return _advice.Refusal(
        contract_refusal(problems),
        event="loop.finish_contract.refused",
        fields={"iteration": turn.iteration, "problems": problems},
        log=f"  finish_session returned: result violates the contract at iter {turn.iteration}",
    )


def review_finish(
    turn: _loop_state.TurnState, state: _loop_state.LoopState, ctx: _advice.TurnContext
) -> _advice.Refusal | None:
    """Sit the before-finish panel over the turn's end.

    Args:
        turn: The turn that declared the end.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        An empty refusal on a rejection, since the findings reach the model
        with the turn's notices; else None.
    """
    return _advice.Refusal("") if ctx.end_rejected(turn, turn.ending or "finish_session") else None


def open_tasks_finish(
    turn: _loop_state.TurnState, state: _loop_state.LoopState, ctx: _advice.TurnContext
) -> _advice.Refusal | None:
    """Hold a run's end while the worker's own subtasks are open, capped.

    Args:
        turn: The turn that declared the end.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The refusal naming the open tasks, or None.
    """
    if ctx.mode != "run":
        return None
    nudge = task_finish_nudge(ctx.open_subtasks(), state.gates)
    if nudge is None:
        return None
    ending = turn.ending or "finish_session"
    return _advice.Refusal(
        nudge,
        event="loop.task_finish.gated",
        fields={
            "iteration": turn.iteration,
            "nudges_used": state.gates.task_nudges_used,
            "trigger": ending,
        },
        log=(
            f"  {ending} gated: open subtasks remain (nudge"
            f" #{state.gates.task_nudges_used}) at iter {turn.iteration}"
        ),
    )


def verify_finish(
    turn: _loop_state.TurnState, state: _loop_state.LoopState, ctx: _advice.TurnContext
) -> _advice.Refusal | None:
    """Return a run's end over a tree the gate did not certify.

    The end returns `verify_retries` times, then stands: reported finished,
    never passed. A gate that was red before the run touched anything is not
    the model's to fix, so it is never returned.

    Args:
        turn: The turn that declared the end.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The refusal with the red notice, or None.
    """
    if not (
        ctx.mode == "run"
        and ctx.tree_green() is False
        and red_gate_returns(
            ctx.verify_when,
            ctx.verify_retries,
            state.verify,
            state.gates,
            gate_present=ctx.gate_present(),
        )
    ):
        return None
    state.gates.verify_retries_used += 1
    used = state.gates.verify_retries_used
    return _advice.Refusal(
        _verify_gate.finish_red_notice(used=used, retries=ctx.verify_retries),
        event="loop.verify_finish.gated",
        fields={"iteration": turn.iteration, "nudges_used": used},
        log=(
            f"  finish_session returned: verify not green (return #{used} of"
            f" {ctx.verify_retries}) at iter {turn.iteration}"
        ),
    )


def memory_finish(
    turn: _loop_state.TurnState, state: _loop_state.LoopState, ctx: _advice.TurnContext
) -> _advice.Refusal | None:
    """Defer once a run's first finish after a red-to-green recovery with no memory write.

    The nudge asks for the root cause or an immediate re-finish; `_nudges` holds
    the measurement behind it.

    Args:
        turn: The turn that declared the end.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The refusal, or None.
    """
    if not (
        ctx.mode == "run"
        and ctx.memory_wired
        and state.verify.ever_failed
        and state.verify.last_ok is True
        and not state.memory.written
        and not state.memory.finish_nudged
    ):
        return None
    state.memory.finish_nudged = True
    return _advice.Refusal(
        _nudges.MEMORY_FINISH_NUDGE,
        event="loop.memory_finish.gated",
        fields={"iteration": turn.iteration},
        log=f"  finish_session deferred once: memory backstop at iter {turn.iteration}",
    )


def standing_finish(
    turn: _loop_state.TurnState, state: _loop_state.LoopState, ctx: _advice.TurnContext
) -> _advice.Refusal | None:
    """Re-enter a ready standing task instead of ending the run.

    Uncapped, since the goal is deliberate; the absorb refuses on spent budget
    or a spin, so the finish then stands.

    Args:
        turn: The turn that declared the end.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The re-entry nudge as a refusal, or None.
    """
    if ctx.mode != "run":
        return None
    nudge = ctx.standing_absorb("finish_session", turn.iteration)
    return None if nudge is None else _advice.Refusal(nudge)


# One precedence for every end: a red tree returns the end before the panel sits.
FINISH_GATES: tuple[_advice.Gate, ...] = (
    finish_contract,
    verify_finish,
    review_finish,
    _metric.metric_early_finish,
    open_tasks_finish,
    memory_finish,
    standing_finish,
)
# A harness-declared end has no payload; its memory and standing rules sit at the stop.
END_GATES: tuple[_advice.Gate, ...] = (verify_finish, review_finish, open_tasks_finish)
# A silent finish is a finish the model wrote in prose: the metric rule applies.
SILENT_END_GATES: tuple[_advice.Gate, ...] = (
    verify_finish,
    review_finish,
    _metric.metric_early_finish,
    open_tasks_finish,
)
