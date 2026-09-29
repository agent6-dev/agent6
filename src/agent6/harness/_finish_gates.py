# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The finish gates: what a finish_session must satisfy before the loop
honours it, what an end is called, and the words each refusal carries. The
loop runs a gate list in order over the turn that declared an end and
applies the first `Refusal` (`Harness._refuse`): `FINISH_GATES` over a
finish_session, `END_GATES` over an end the harness declares (settled, a
plateau), `SILENT_END_GATES` over a silent finish; `turn.ending` names the
end the gates judge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from agent6.harness._advice import Gate, Refusal, TurnContext
from agent6.harness._metric import metric_early_finish
from agent6.harness._nudges import MEMORY_FINISH_NUDGE, TASK_FINISH_PATIENCE
from agent6.harness._session_state import SessionEndReason
from agent6.harness._verify_gate import finish_red_notice
from agent6.harness._verify_verdict import VerifyVerdict
from agent6.tools.schema import FinishPlanningInput, FinishSessionInput

if TYPE_CHECKING:
    from agent6.harness._loop_state import LoopState, TurnState


@dataclass(frozen=True, slots=True)
class FinishCall:
    """A dispatched finish_session or finish_planning call, as the model
    sent it; the finish gates may revoke it. `payload` is finish_session's
    `result`, `stale_gate` its claim that the configured gate is stale, and
    `plan_markdown` finish_planning's plan, with the summary folded under a
    title-only one (`plan_salvaged`)."""

    kind: Literal["finish_session", "finish_planning"]
    summary: str
    payload: dict[str, Any] | None = None
    stale_gate: str = ""
    plan_markdown: str = ""
    plan_salvaged: bool = False

    @classmethod
    def parse(cls, name: str, tool_input: Any) -> FinishCall | None:
        """The finish a dispatched tool call declares; None for any other
        tool. Schema validation guaranteed the fields when the dispatcher
        dispatched the call, but the raw tool_input is what the model sent,
        so a malformed call still parses."""
        fields = tool_input if isinstance(tool_input, dict) else {}
        summary = str(fields.get("summary", "(no summary)"))
        if name == FinishSessionInput.TOOL_NAME:
            raw_result = fields.get("result")
            if isinstance(raw_result, str):
                # Weak models routinely STRINGIFY the structured result; one
                # tolerant parse here, schema validation downstream stays
                # strict about content.
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
        if name == FinishPlanningInput.TOOL_NAME:
            plan_md = str(fields.get("plan_markdown", ""))
            # Weak models leave plan_markdown a bare title and put the plan in
            # `summary`, a stub `--from` would have to re-derive: the summary
            # folds under the title. The review gate judged the content; this
            # only rescues field misuse.
            salvaged = _plan_is_title_only(plan_md) and len(summary) > len(plan_md)
            if salvaged:
                title = next((ln for ln in plan_md.splitlines() if ln.strip()), "# Plan")
                plan_md = f"{title}\n\n{summary}"
            return cls("finish_planning", summary, plan_markdown=plan_md, plan_salvaged=salvaged)
        return None


def _plan_is_title_only(plan_md: str) -> bool:
    """True when plan_markdown has no body: only heading lines (`# ...`) and
    blanks."""
    return not any(
        line.strip() and not line.lstrip().startswith("#") for line in plan_md.splitlines()
    )


@dataclass(slots=True)
class FinishGates:
    """The finish gates' counters: the open-task refusals sent
    (`TASK_FINISH_PATIENCE` caps them), the red finish certifications
    returned (`verify_retries` caps them), the before-finish panel's
    consecutive rejections (its cap lets the end through) and its run-total
    (persisted; past `max_total_rejections` the gate disarms to advisory)."""

    task_nudges_used: int = 0
    verify_retries_used: int = 0
    review_consecutive: int = 0
    review_total: int = 0


def task_finish_nudge(open_tasks: Sequence[tuple[str, str]], gates: FinishGates) -> str | None:
    """The nudge to re-prompt with instead of finishing while the worker's
    own subtasks are open; None lets the end through. Capped by
    `TASK_FINISH_PATIENCE`, as the review gate is: after that many refusals
    the end goes through and its receipt names the open tasks
    (`with_open_tasks`), so a worker that neither closes nor retires a task
    cannot bounce the loop for the whole budget."""
    if not open_tasks:
        return None
    if gates.task_nudges_used >= TASK_FINISH_PATIENCE:
        return None  # cap reached: the end goes through, the receipt names them
    gates.task_nudges_used += 1
    listing = "\n".join(f"- {tid}: {title}" for tid, title in open_tasks)
    return (
        f"[harness] finish_session deferred: {len(open_tasks)} task(s) are"
        f" pending or in_progress:\n{listing}\n"
        "update_task marks one skipped or obsolete; the run ends once the"
        f" list is clear, or on the {TASK_FINISH_PATIENCE + 1}th call."
    )


def red_gate_returns(
    verify_when: Literal["finish", "step", "never"],
    verify_retries: int,
    verify: VerifyVerdict,
    gates: FinishGates,
    *,
    gate_present: bool,
) -> bool:
    """Whether a red gate is the model's to fix, so an end over it goes back:
    a gate exists and is the harness's to run, was not red before the run
    touched anything (or this run has since made it green), was not denied
    or withheld by the operator, and returns are left. One answer for
    finish_session and the ends the harness declares, so neither can hand
    back a gate the model cannot run."""
    return (
        verify_when != "never"
        and gate_present
        and (verify.baseline_ok is not False or verify.ever_passed)
        and gates.verify_retries_used < verify_retries
    )


def contract_refusal(problems: Sequence[str]) -> str:
    """The notice a finish_session whose `result` violates the machine
    state's output_schema returns with."""
    return (
        "finish_session refused: "
        + "; ".join(problems)
        + ". Call finish_session again with a `result` that satisfies the schema."
    )


def finish_reason(
    kind: SessionEndReason, *, stale_gate: str, tree_green: bool | None, verify: VerifyVerdict
) -> SessionEndReason:
    """What a finish is called. `gate_stale` needs a gate that is actually
    RED: green means it passed, and `tree_green` is None on a gateless run,
    where no gate can be stale. `gate_red_at_base` outranks a plain finish
    over red: the gate was failing before this run touched anything, so a
    red end is not this run's failure; only ever from an observation, never
    a guess."""
    if kind == "finish_session" and tree_green is False:
        if stale_gate:
            return "gate_stale"
        if verify.baseline_ok is False and not verify.ever_passed:
            return "gate_red_at_base"
    return kind


def finish_contract(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """A finish_session whose `result` does not satisfy the machine state's
    output_schema returns to the model with the problems, so the retry
    happens in-leg instead of the leg ending failed over correct work.
    Unbounded on purpose: the budget and iteration backstops end a model
    that never conforms, and the engine records that truthfully."""
    if ctx.finish_validator is None:
        return None
    problems = ctx.finish_validator(turn.finish.payload if turn.finish is not None else None)
    if not problems:
        return None
    return Refusal(
        contract_refusal(problems),
        event="loop.finish_contract.refused",
        fields={"iteration": turn.iteration, "problems": problems},
        log=f"  finish_session returned: result violates the contract at iter {turn.iteration}",
    )


def review_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """The before-finish panel over the turn's end: a rejection revokes it,
    the findings reaching the model with the turn's notices."""
    return Refusal("") if ctx.end_rejected(turn, turn.ending or "finish_session") else None


def open_tasks_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """A run's finish_session waits while the worker's own subtasks are open
    (`task_finish_nudge`, capped)."""
    if ctx.mode != "run":
        return None
    nudge = task_finish_nudge(ctx.open_subtasks(), state.gates)
    if nudge is None:
        return None
    ending = turn.ending or "finish_session"
    return Refusal(
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


def verify_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """A run's finish over a tree the gate did not certify returns to the
    model `verify_retries` times (`red_gate_returns`), then stands: reported
    finished, never passed. A gate that was red before the run touched
    anything is not the model's to fix, so it is never returned."""
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
    return Refusal(
        finish_red_notice(used=used, retries=ctx.verify_retries),
        event="loop.verify_finish.gated",
        fields={"iteration": turn.iteration, "nudges_used": used},
        log=(
            f"  finish_session returned: verify not green (return #{used} of"
            f" {ctx.verify_retries}) at iter {turn.iteration}"
        ),
    )


def memory_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """The memory write-side backstop: a run's first finish_session after a
    recovery from a red verify to green, with nothing recorded in the memory
    store, is deferred once; the nudge asks for the root cause or an
    immediate re-finish (see `_nudges` for the measurement behind it)."""
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
    return Refusal(
        MEMORY_FINISH_NUDGE,
        event="loop.memory_finish.gated",
        fields={"iteration": turn.iteration},
        log=f"  finish_session deferred once: memory backstop at iter {turn.iteration}",
    )


def standing_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """While a ready standing task exists, a run's finish_session re-enters
    it instead of ending the run (uncapped: the goal is deliberate; the
    absorb refuses on spent budget or a spin, so the finish then stands)."""
    if ctx.mode != "run":
        return None
    nudge = ctx.standing_absorb("finish_session", turn.iteration)
    return None if nudge is None else Refusal(nudge)


# The gates over a finish_session, in precedence order.
# One precedence for every end: the contract, then the verify certification
# (a red tree returns the end before the panel sits), the panel, the metric
# rule, the open tasks, the memory backstop, the standing goal.
FINISH_GATES: tuple[Gate, ...] = (
    finish_contract,
    verify_finish,
    review_finish,
    metric_early_finish,
    open_tasks_finish,
    memory_finish,
    standing_finish,
)
# An end the harness declares has no payload to check; its memory backstop
# and standing re-entry are judged where the end is decided.
END_GATES: tuple[Gate, ...] = (verify_finish, review_finish, open_tasks_finish)
# A silent finish is a finish the model wrote in prose: the metric rule applies.
SILENT_END_GATES: tuple[Gate, ...] = (
    verify_finish,
    review_finish,
    metric_early_finish,
    open_tasks_finish,
)
