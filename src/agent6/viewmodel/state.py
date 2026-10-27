# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's events into a `SessionState`.

The fold does no I/O: `apply_event` returns a new frozen state, so an unchanged
identity means nothing changed.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Iterable
from typing import Any

from agent6 import terminal
from agent6.graph import models
from agent6.sessions import layout
from agent6.viewmodel import _session_status, _state_types, events, format, log_line, review


def initial_state() -> _state_types.SessionState:
    """Return the state before any event."""
    return _state_types.SessionState()


_MAX_TOOL_HISTORY = 50
_MAX_DIFF_HISTORY = 30
# The inline log widget caps to this so it stays a gapless window.
MAX_LOG_TAIL = 400
# Only the tail of an in-flight stream travels; the transcript keeps the full turn.
_STREAM_TAIL = 6000

# Live-view only: a reasoning model would flood the log with contentless delta lines.
STREAM_DELTA_EVENTS = frozenset({"role.thinking_delta", "role.text_delta"})
# Loop-side mirrors of events the log already shows.
LOG_NOISE_EVENTS = frozenset({"loop.tool.call", "loop.budget"})


def _answered_only(
    prompts: tuple[_state_types.ApprovalPrompt, ...],
) -> tuple[_state_types.ApprovalPrompt, ...]:
    """Drop the unanswered approvals at an execution boundary.

    They belong to the execution that died holding them; the new one re-asks with
    restarted ids.

    Args:
        prompts: The approvals folded so far.

    Returns:
        The answered ones.
    """
    return tuple(p for p in prompts if p.answered)


def _carried(
    prompts: tuple[_state_types.QuestionPrompt, ...],
) -> tuple[_state_types.QuestionPrompt, ...]:
    """Carry every question past an execution boundary.

    Args:
        prompts: The questions folded so far.

    Returns:
        The same questions, none pausing the new execution.
    """
    return tuple(dataclasses.replace(q, blocking=False) for q in prompts)


def apply_event(  # noqa: C901, PLR0911, PLR0912, PLR0915  # one branch per event type
    state: _state_types.SessionState, event: dict[str, Any]
) -> _state_types.SessionState:
    """Fold one event into the session state.

    The event is parsed once into a typed family; the log line and the session id
    peek read the raw dict, since they render every event. An unknown or telemetry
    type changes no state.

    Args:
        state: The state so far.
        event: The raw event dict.

    Returns:
        The new state, or `state` itself when nothing changed.
    """
    etype = event.get("type", "")
    review_view = review.apply_event(state.review, event)
    if review_view is not state.review:
        state = dataclasses.replace(state, review=review_view)
    if not state.session_id and event.get("session_id"):
        state = dataclasses.replace(state, session_id=str(event["session_id"]))
    # The event's own ts, so replayed history measures idle time from when the run last spoke.
    if (ep := events.event_epoch(event.get("ts"))) is not None:
        state = dataclasses.replace(state, last_event_ep=ep)
    if etype not in STREAM_DELTA_EVENTS and etype not in LOG_NOISE_EVENTS:
        # cursor_task_id is the focus task: graph.update lands before a turn's calls.
        entry = _state_types.LogLine(log_line.format_log_line(event), state.cursor_task_id)
        new_log = _push_bounded(state.log_tail, entry, MAX_LOG_TAIL)
        state = dataclasses.replace(state, log_tail=new_log, log_count=state.log_count + 1)

    match events.parse_event(event):
        case events.SessionStart(user_task=task):
            # The ask REPL re-runs on one log, so a second start clears the prior end.
            # No banking, unlike ResumeStart: the REPL's one tracker is already cumulative.
            return dataclasses.replace(
                state,
                task_label=state.task_label.start(task),
                mode=str(event.get("mode", state.mode)),
                started=True,
                finished=False,
                end_reason="",
                stranded=False,
                end_summary="",
                last_role=None,
                finish_summary="",
                last_verify=None,
                pending_approvals=_answered_only(state.pending_approvals),
                pending_questions=_carried(state.pending_questions),
                unattended_questions=0,
                tool_count=0,
            )

        case events.ResumeStart(user_task=task, steer=steer):
            # The new execution's counters start fresh: bank the spend, zero the rest.
            return dataclasses.replace(
                state,
                task_label=state.task_label.resume(task, steer),
                mode=str(event.get("mode", state.mode)),
                started=True,
                finished=False,
                end_reason="",
                stranded=False,
                end_summary="",
                last_role=None,
                finish_summary="",
                last_verify=None,
                pending_approvals=_answered_only(state.pending_approvals),
                pending_questions=_carried(state.pending_questions),
                unattended_questions=0,
                tool_count=0,
                budget=dataclasses.replace(
                    state.budget,
                    usd_prior_executions=state.budget.usd_total,
                    input_total=0,
                    output_total=0,
                    usd_cap=0.0,
                    tokens_unmetered=0,
                    tokens_fallback_cap=0,
                    plan_used_percent=0.0,
                    plan_prior_executions=state.budget.plan_consumed,
                    plan_cap=0.0,
                    plan_resets_at=0.0,
                ),
            )

        case events.GraphUpdate(nodes=nodes, cursor=cursor):
            return dataclasses.replace(
                state,
                tasks=task_tree_views(nodes, cursor),
                cursor_task_id=cursor,
            )

        case events.AutoCommit(turn=turn, sha=sha, subject=subject):
            if not sha:
                return state
            step = _state_types.CommitStep(turn=turn, sha=sha, subject=subject)
            return dataclasses.replace(state, steps=(*state.steps, step))

        case events.DiffUpdated(patch=patch, sha=sha):
            entry = _state_types.DiffView(patch=patch, task_id=state.cursor_task_id, sha=sha)
            return dataclasses.replace(
                state,
                latest_diff=patch,
                recent_diffs=_push_bounded(state.recent_diffs, entry, _MAX_DIFF_HISTORY),
            )

        case events.RoleCall(role=role, model=model, provider=provider):
            prior = state.last_role
            return dataclasses.replace(
                state,
                last_role=_state_types.RoleCall(
                    role=role,
                    model=model,
                    in_flight=True,
                    provider=provider,
                    # The last known context size stays until this call's result lands.
                    ctx_tokens=prior.ctx_tokens if prior is not None else 0,
                    streamed_text="",
                    streamed_thinking="",
                ),
            )

        case events.RoleTextDelta(text=piece):
            # Scrub the concatenation: an escape sequence can arrive split across deltas.
            last = state.last_role
            if last is None or not last.in_flight or not piece:
                return state
            joined = terminal.scrub_terminal_controls(last.streamed_text + piece)
            return dataclasses.replace(
                state,
                last_role=dataclasses.replace(last, streamed_text=joined[-_STREAM_TAIL:]),
            )

        case events.RoleThinkingDelta(text=piece):
            last = state.last_role
            if last is None or not last.in_flight or not piece:
                return state
            joined = terminal.scrub_terminal_controls(last.streamed_thinking + piece)
            return dataclasses.replace(
                state,
                last_role=dataclasses.replace(last, streamed_thinking=joined[-_STREAM_TAIL:]),
            )

        case events.RoleResult(tokens_in=tin, cache_read=cr, cache_creation=cc):
            last = state.last_role
            if last is None:
                return state
            ctx = tin + cr + cc
            return dataclasses.replace(
                state,
                last_role=dataclasses.replace(
                    last, in_flight=False, ctx_tokens=ctx if ctx > 0 else last.ctx_tokens
                ),
            )

        case events.ToolCall(name=name, args=raw_args, call_id=cid):
            tc = _state_types.ToolCallView(
                name=name,
                args_preview=log_line.render_args(raw_args),
                args_full=log_line.render_args(raw_args, max_value=4000),
                ok=None,
                task_id=state.cursor_task_id,
                call_id=cid,
            )
            finish_summary = state.finish_summary
            finish = name in events.FINISH_TOOLS
            if finish and isinstance(raw_args, dict):
                finish_summary = str(raw_args.get("summary", "")).strip() or finish_summary
            return dataclasses.replace(
                state,
                tool_calls=_push_bounded(state.tool_calls, tc, _MAX_TOOL_HISTORY),
                tool_count=state.tool_count + (0 if finish else 1),
                finish_summary=finish_summary,
            )

        case events.ToolResult(name=name, ok=ok, summary=summary, call_id=cid):
            if not state.tool_calls:
                return state
            if cid is not None:
                # Concurrent seats interleave events, so the matching call may not be the last.
                for i in range(len(state.tool_calls) - 1, -1, -1):
                    if state.tool_calls[i].call_id == cid:
                        updated = dataclasses.replace(
                            state.tool_calls[i], ok=ok, result_summary=summary
                        )
                        return dataclasses.replace(
                            state,
                            tool_calls=(
                                *state.tool_calls[:i],
                                updated,
                                *state.tool_calls[i + 1 :],
                            ),
                        )
                return state
            last = state.tool_calls[-1]
            if last.name != name:
                return state
            updated_last = dataclasses.replace(last, ok=ok, result_summary=summary)
            return dataclasses.replace(
                state,
                tool_calls=(*state.tool_calls[:-1], updated_last),
            )

        case events.VerifyStart(cmd=cmd):
            return dataclasses.replace(state, last_verify=_state_types.VerifyView(cmd=cmd))

        case events.VerifyEnd(
            cmd=cmd,
            exit_code=code,
            duration_s=dur,
            stdout_tail=out,
            stderr_tail=err,
            timeout_s=limit,
            outcome=outcome,
            reason=reason,
        ):
            return dataclasses.replace(
                state,
                last_verify=_state_types.VerifyView(
                    cmd=cmd,
                    exit_code=code,
                    duration_s=dur,
                    stdout_tail=out,
                    stderr_tail=err,
                    timeout_s=limit,
                    outcome=outcome,
                    reason=reason,
                ),
            )

        case events.BudgetUpdate(
            input_total=it,
            output_total=ot,
            cache_read_total=cr,
            cache_creation_total=cc,
            usd_total=usd,
            usd_partial=partial,
            usd_cap=ucap,
            tokens_unmetered=unmet,
            tokens_fallback_cap=fcap,
            plan_used_percent=plan_pct,
            plan_consumed=plan_used,
            plan_cap=plan_cap,
            plan_resets_at=plan_resets,
        ):
            # The event's usd_total and plan_consumed are this execution's; the view's are
            # cumulative.
            return dataclasses.replace(
                state,
                budget=_state_types.BudgetView(
                    input_total=it,
                    output_total=ot,
                    cache_read_total=cr,
                    cache_creation_total=cc,
                    usd_total=state.budget.usd_prior_executions + usd,
                    usd_prior_executions=state.budget.usd_prior_executions,
                    usd_partial=partial or state.budget.usd_partial,
                    usd_cap=ucap,
                    tokens_unmetered=unmet,
                    tokens_fallback_cap=fcap,
                    plan_used_percent=plan_pct,
                    plan_consumed=state.budget.plan_prior_executions + plan_used,
                    plan_prior_executions=state.budget.plan_prior_executions,
                    plan_cap=plan_cap,
                    plan_resets_at=plan_resets,
                ),
            )

        case events.ApprovalPrompt(id=aid, prompt=prompt, standing=standing, asked_ep=asked_ep):
            ap = _state_types.ApprovalPrompt(
                id=aid, prompt=prompt, standing=standing, asked_ep=asked_ep
            )
            return dataclasses.replace(state, pending_approvals=(*state.pending_approvals, ap))

        case events.ApprovalAnswer(id=wanted_id, approved=approved):
            new = tuple(
                dataclasses.replace(a, answered=True, approved=approved) if a.id == wanted_id else a
                for a in state.pending_approvals
            )
            return dataclasses.replace(state, pending_approvals=new)

        case events.QuestionPrompt(id=qid, questions=qs, asked_ep=asked_ep, wait=wait):
            questions = tuple(
                _state_types.Question(question=q.question, options=q.options) for q in qs
            )
            qp = _state_types.QuestionPrompt(
                id=qid,
                questions=questions,
                from_harness=not state.started or state.finished,
                asked_ep=asked_ep,
                wait=wait,
                blocking=wait,
            )
            return dataclasses.replace(state, pending_questions=(*state.pending_questions, qp))

        case events.QuestionAnswer(id=wanted, answers=answers, unseen=unseen, source=source):
            new_q = tuple(
                dataclasses.replace(q, answered=True, answers=answers, source=source)
                if q.id == wanted
                else q
                for q in state.pending_questions
            )
            # Counted per event: prompt ids restart on every execution.
            return dataclasses.replace(
                state,
                pending_questions=new_q,
                unattended_questions=state.unattended_questions + (1 if unseen else 0),
            )

        case events.PinAdded(text=text):
            return dataclasses.replace(state, pins=(*state.pins, text))

        case events.PinsRestored(pins=pins):
            return dataclasses.replace(state, pins=pins)

        case events.CompactRestored(elided=elided, gists=gists):
            return dataclasses.replace(state, compact_elided=elided, compact_gists_live=gists)

        case events.CompactDropped(n=n):
            return dataclasses.replace(state, compact_elided=state.compact_elided + n)

        case events.CompactGists(gisted=gisted, demoted=demoted):
            live = max(0, state.compact_gists_live + gisted - demoted)
            return dataclasses.replace(state, compact_gists_live=live)

        case events.CompactSummarised():
            return dataclasses.replace(state, compact_elided=0, compact_gists_live=0)

        case events.SteerRequested():
            return dataclasses.replace(state, steer_requests=state.steer_requests + 1)

        case events.SessionEnd(
            all_passed=all_passed, reason=reason, summary=summary, scoped=scoped
        ):
            return dataclasses.replace(
                state,
                finished=True,
                all_passed=all_passed,
                end_reason=reason,
                end_summary=_session_status.end_text(reason, summary, state.finish_summary),
                verify_scoped=scoped,
                # The worker is on no task after the end: each draws its own status.
                tasks=tuple(dataclasses.replace(t, is_cursor=False, glyph="") for t in state.tasks),
            )

        case events.SessionStranded():
            return dataclasses.replace(state, stranded=True)

        case events.SessionUndone(new_session_id=new_id, undone_text=text):
            return dataclasses.replace(state, undone_to=new_id, undone_text=text)

        case events.RawEvent():
            return state


def task_tree_views(
    nodes: dict[str, Any], cursor: str | None
) -> tuple[_state_types.TaskNodeView, ...]:
    """Flatten the task node map into DFS pre-order with depths, for an indented tree.

    Args:
        nodes: The `graph.update` node map.
        cursor: The id of the task in progress.

    Returns:
        The views: roots first in sorted id order, each followed by its children in
        recorded order; a node whose parent is missing follows the roots; cycles and
        duplicates are visited once.
    """
    out: list[_state_types.TaskNodeView] = []
    seen: set[str] = set()

    def visit(nid: str, depth: int) -> None:
        """Append the node and its subtree, skipping a malformed or already-seen node."""
        node = nodes.get(nid)
        if not isinstance(node, dict) or nid in seen:
            return
        seen.add(nid)
        created_by, standing = str(node.get("created_by", "")), bool(node.get("standing", False))
        parent_id = node.get("parent_id")
        out.append(
            _state_types.TaskNodeView(
                id=nid,
                title=str(node.get("title", "")),
                status=node.get("status", "pending"),
                depth=depth,
                is_cursor=(nid == cursor),
                created_by=created_by,
                standing=standing,
                note=models.owner_note(
                    created_by=created_by,
                    parent_id=str(parent_id) if isinstance(parent_id, str) else None,
                    standing=standing,
                ),
                short_id=format.short_task_id(nid),
            )
        )
        children = node.get("children", ())
        if not isinstance(children, (list, tuple)):
            return
        for child in children:
            visit(str(child), depth + 1)

    # Sorted like `tree_order`, so the roots read in the same order on every surface.
    roots = [
        nid
        for nid in sorted(nodes)
        if not isinstance((n := nodes[nid]), dict) or n.get("parent_id") is None
    ]
    for nid in roots:
        visit(nid, 0)
    for nid in sorted(nodes):
        visit(nid, 0)
    return tuple(out)


def _push_bounded[T](existing: tuple[T, ...], item: T, cap: int) -> tuple[T, ...]:
    """Return the tuple with the item appended, the oldest dropped past the cap."""
    new = (*existing, item)
    if len(new) > cap:
        return new[-cap:]
    return new


def fold_session(events: Iterable[dict[str, Any]]) -> _state_types.SessionState:
    """Fold a session's whole event stream into one state.

    Args:
        events: The raw events, in order.

    Returns:
        The state after the last event.
    """
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return state


def open_approval_of(
    state: _state_types.SessionState, *, taken: Callable[[str], bool] = lambda _aid: False
) -> _state_types.ApprovalPrompt | None:
    """Return the approval a surface answers now.

    The one rule behind the server's answer route, the run views' docked row and the
    web's box, so no surface offers an approval the run will refuse.

    Args:
        state: The folded state.
        taken: Whether the surface already took the approval with that id.

    Returns:
        The oldest unanswered approval not yet taken, or None.
    """
    return next(
        (ap for ap in state.pending_approvals if not ap.answered and not taken(ap.id)), None
    )


def open_approval(session_dir: pathlib.Path) -> _state_types.ApprovalPrompt | None:
    """Return the run's open approval from its journal, or None when none is open."""
    from agent6.viewmodel import tail  # noqa: PLC0415  # cycle at import time  # noqa: PLC0415  # cycle at import time

    return open_approval_of(
        fold_session(tail.tail_events(session_dir / layout.LOGS_NAME, follow=False))
    )


def open_question(session_dir: pathlib.Path) -> _state_types.QuestionPrompt | None:
    """Return the run's oldest unanswered `ask_user` prompt, or None when none is open.

    Every surface that writes an answer file checks against it, so an answer list of
    the wrong length is refused rather than thrown away by the asking side.

    Args:
        session_dir: The session's state dir.

    Returns:
        The prompt, None when none is open.
    """
    from agent6.viewmodel import tail  # noqa: PLC0415  # cycle at import time  # noqa: PLC0415  # cycle at import time

    state = fold_session(tail.tail_events(session_dir / layout.LOGS_NAME, follow=False))
    return next((q for q in state.pending_questions if not q.answered), None)


def fold_until_commit(
    events: Iterable[dict[str, Any]], sha: str
) -> _state_types.SessionState | None:
    """Fold the state as of one of the run's commits.

    Args:
        events: The raw events, in order.
        sha: The commit's full sha or a prefix of at least 7 hex digits; the first
            commit it matches wins.

    Returns:
        The state after that commit's `loop.auto_commit`, or None when no commit has it.
    """
    if len(sha) < 7:
        return None
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
        if state.steps and state.steps[-1].sha.startswith(sha):
            return state
    return None
