# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One reducer for a session's lifecycle and usage, under both journal folds.

`state.apply_event` (the viewers' fold) and `session_log.scan_session_log` (the listings' scan)
read the same `logs.jsonl`. The facts they share, where the session stands and what it spent,
fold here once; each keeps its own projection, the view's rich one and the listing's light one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from agent6 import budget, kinds
from agent6.sessions import journal
from agent6.viewmodel import events, state_types
from agent6.viewmodel.session import status as session_status

# The families that move the core; the listing parses these rows and no other.
_CORE_TYPES = frozenset(
    {
        "budget.update",
        "environment.acquired",
        "environment.starting",
        "execution.environment",
        "pin.added",
        "pin.restored",
        "question.answer",
        "model.result",
        "session.resume",
        "session.start",
        "session.stop",
        "session.stranded",
        "session.undone",
        "verify.end",
        "verify.start",
    }
)


def concerns(raw: Mapping[str, Any]) -> bool:
    """Return whether a raw row can move the core."""
    if raw.get("type") == "tool.call":
        return raw.get("name") in events.FINISH_TOOLS
    return raw.get("type") in _CORE_TYPES


def apply(  # noqa: PLR0911, PLR0912  # one branch per family
    core: state_types.SessionCore, event: events.Event, raw: Mapping[str, Any]
) -> state_types.SessionCore:
    """Fold one record into the core; a row of any other family returns the core itself.

    Args:
        core: The core so far.
        event: The row's record.
        raw: The row, for its envelope and the keys a record leaves out.

    Returns:
        The core after the row.
    """
    life = core.lifecycle
    match event:
        case journal.SessionStart(user_task=task, execution_id=eid):
            # The ask REPL re-runs on one log: a second start clears the prior end without
            # banking, since the REPL's one tracker is already cumulative.
            label = life.task_label.start(task)
            return dataclasses.replace(core, lifecycle=_begun(life, label, raw, eid))
        case journal.SessionResume(user_task=task, steer=steer, execution_id=eid):
            begun = _begun(life, life.task_label.resume(task, steer), raw, eid)
            begun = dataclasses.replace(
                begun, executions=life.executions + 1 if life.started else life.executions
            )
            # The session's usage keeps the finished execution's; the account's last reading
            # stays, the run is plan-metered before its next call too.
            usage = dataclasses.replace(core.budget, execution=state_types.ExecutionUsage())
            return state_types.SessionCore(lifecycle=begun, budget=usage)
        case journal.SessionStop(
            verification=verification, reason=reason, summary=summary, scoped=scoped
        ):
            ended = dataclasses.replace(
                life,
                finished=True,
                verification=events.verification_word(verification),
                stop_reason=reason,
                end_summary=session_status.end_text(reason, summary or "", life.finish_summary),
                verify_scoped=bool(scoped),
            )
            return dataclasses.replace(core, lifecycle=ended)
        case journal.SessionStranded():
            return dataclasses.replace(core, lifecycle=dataclasses.replace(life, stranded=True))
        case journal.SessionUndone(new_session_id=new_id):
            return dataclasses.replace(core, lifecycle=dataclasses.replace(life, undone_to=new_id))
        case journal.ToolCall(name=name, args_preview=args) if name in events.FINISH_TOOLS:
            summary = str(args.get("summary", "")).strip() or life.finish_summary
            return dataclasses.replace(
                core, lifecycle=dataclasses.replace(life, finish_summary=summary)
            )
        case journal.VerifyStart():
            # The verdict before it described an older tree.
            return dataclasses.replace(
                core, lifecycle=dataclasses.replace(life, last_verify_rc=None, verify_timeout_s=0.0)
            )
        case journal.VerifyEnd(exit_code=code, outcome=outcome, timeout_s=limit):
            return dataclasses.replace(
                core,
                lifecycle=dataclasses.replace(
                    life,
                    last_verify_rc=code,
                    verify_timeout_s=events.verify_limit_hit(outcome, limit),
                ),
            )
        case journal.ModelResult(ok=False, role=role, error=error) if not kinds.is_side_role(role):
            # A side call's failure never ends the run, so it is not the reason.
            return dataclasses.replace(
                core, lifecycle=dataclasses.replace(life, provider_error=error or "")
            )
        case journal.QuestionAnswer(unseen=True):
            # Counted per event: prompt ids restart on every execution.
            return dataclasses.replace(
                core,
                lifecycle=dataclasses.replace(
                    life, unattended_questions=life.unattended_questions + 1
                ),
            )
        case journal.PinAdded(text=text):
            return dataclasses.replace(
                core, lifecycle=dataclasses.replace(life, pins=(*life.pins, text))
            )
        case journal.PinRestored(pins=pins):
            return dataclasses.replace(core, lifecycle=dataclasses.replace(life, pins=tuple(pins)))
        case journal.EnvironmentStarting() | journal.EnvironmentAcquired():
            view = held_environment(event)
            return dataclasses.replace(core, lifecycle=dataclasses.replace(life, environment=view))
        case journal.ExecutionEnvironment(outcome=outcome):
            view = dataclasses.replace(
                life.environment or state_types.EnvironmentView(), outcome=outcome
            )
            return dataclasses.replace(core, lifecycle=dataclasses.replace(life, environment=view))
        case journal.BudgetUpdate():
            return dataclasses.replace(core, budget=_usage(core.budget, event, raw))
        case _:
            return core


def _begun(
    life: state_types.Lifecycle,
    label: session_status.TaskLabel,
    raw: Mapping[str, Any],
    execution_id: str,
) -> state_types.Lifecycle:
    """Return the lifecycle at an execution's start: the end cleared, the counters reset."""
    return dataclasses.replace(
        life,
        started=True,
        finished=False,
        mode=str(raw.get("mode", life.mode)),
        execution_id=execution_id,
        task_label=label,
        stop_reason="",
        verification=None,
        verify_scoped=False,
        stranded=False,
        end_summary="",
        finish_summary="",
        last_verify_rc=None,
        verify_timeout_s=0.0,
        unattended_questions=0,
        # The lease rows precede the start: a view bound to this execution stays.
        environment=(
            life.environment
            if life.environment is not None and life.environment.execution_id == execution_id
            else None
        ),
    )


def held_environment(
    row: journal.EnvironmentStarting | journal.EnvironmentAcquired, outcome: str = ""
) -> state_types.EnvironmentView:
    """Return the container view a lease row describes.

    Args:
        row: The start in progress, or the lease the run holds.
        outcome: The lifecycle's end when already known.

    Returns:
        The view both folds show.
    """
    if isinstance(row, journal.EnvironmentStarting):
        return state_types.EnvironmentView(
            execution_id=row.execution_id,
            starting=tuple(str(c) for c in row.command),
            outcome=outcome,
        )
    return state_types.EnvironmentView(
        execution_id=row.execution_id,
        role=row.role,
        reason=row.reason,
        started=row.started,
        start_s=row.start_s,
        outcome=outcome,
    )


def _usage(
    prior: state_types.BudgetView, row: journal.BudgetUpdate, raw: Mapping[str, Any]
) -> state_types.BudgetView:
    """Fold a budget row: the tracker's lifetime is the execution's, the session banks the rest.

    Args:
        prior: The usage so far.
        row: The budget record.
        raw: The row, for its envelope.

    Returns:
        The usage after the row.
    """
    banked_usd = prior.session.usd - prior.execution.usd
    banked_points = prior.session.plan_observed_points - prior.execution.plan_observed_points
    execution = state_types.ExecutionUsage(
        input_tokens=row.input_total,
        output_tokens=row.output_total,
        cache_read_tokens=row.cache_read_total,
        cache_write_tokens=row.cache_write_total,
        usd=row.usd_total,
        usd_partial=bool(row.usd_partial),
        unmetered_tokens=row.tokens_unmetered or 0,
        usd_running_lanes=row.usd_running_lanes,
        tokens_running_lanes=row.tokens_running_lanes,
        plan_observed_points=row.plan_observed_points or 0.0,
        usd_cap=row.usd_cap,
        tokens_fallback_cap=row.tokens_fallback_cap,
        plan_cap=row.plan_cap,
    )
    session = state_types.SessionUsage(
        settled_lanes=tuple(sorted(set(prior.session.settled_lanes).union(row.settled_lanes))),
        unmetered_tokens=(
            prior.session.unmetered_tokens
            - prior.execution.unmetered_tokens
            + execution.unmetered_tokens
        ),
        usd=banked_usd + row.usd_total,
        usd_partial=bool(row.usd_partial) or prior.session.usd_partial,
        plan_observed_points=banked_points + (row.plan_observed_points or 0.0),
    )
    reading = prior.account
    if row.account is not None:
        reading = state_types.AccountReading(
            provider=row.account.provider,
            used_percent=row.account.used_percent,
            resets_at=row.account.resets_at,
            windows=tuple(
                budget.PlanWindow(w.name, w.used_percent, w.window_minutes, w.resets_at)
                for w in row.account.windows
            ),
            credits_has=row.account.credits_has,
            credits_unlimited=row.account.credits_unlimited,
            credits_balance=row.account.credits_balance,
            observed_at=events.event_epoch(raw.get("ts")) or 0.0,
        )
    return state_types.BudgetView(session=session, execution=execution, account=reading)
