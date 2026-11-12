# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render session status, context usage, and the wire snapshot.

`session_state_as_dict` is the contract every external viewer reads (`attach --json`,
the web page). With a session dir, the wire form also reads the dir's status probes
and manifest. The generated Formats reference owns its serialized fields.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import pathlib
import time
from typing import Any

from agent6.models import registry
from agent6.sessions import ipc, process, shells
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import format, listing, policy, review, session_status, state_types
from agent6.viewmodel import state as session_fold

PUBLIC_STATE_FIELDS = (
    "session_id",
    "mode",
    "execution_id",
    "tasks",
    "cursor_task_id",
    "last_model_call",
    "tool_calls",
    "last_verify",
    "review",
    "environment",
    "budget",
    "approvals",
    "questions",
    "log_tail",
    "log_count",
    "tool_count",
    "recent_diffs",
    "started",
    "finished",
    "verification",
    "verify_scoped",
    "stop_reason",
    "stranded",
    "end_summary",
    "unattended_questions",
    "undone_to",
    "undone_text",
    "finish_summary",
    "latest_diff",
    "commits",
    "steer_requests",
    "compact_elided",
    "compact_gists_live",
    "pins",
    "last_event_at",
)


def _wire_value(value: Any) -> Any:
    """Return a declared state field in wire form."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, tuple):
        return tuple(_wire_value(item) for item in value)
    return value


def status_facts(state: state_types.SessionState) -> session_status.StatusFacts:
    """Return the fold's answers to the status questions.

    The typed twin of `LogScan.status_facts`; the two agree on the same log.

    Args:
        state: The folded state.

    Returns:
        The facts `status_for_session_dir` reads.
    """
    pending: list[tuple[str, float | None]] = [
        ("approval", a.asked_at) for a in state.approvals if not a.answered
    ] + [("question", q.asked_at) for q in state.questions if not q.answered and q.blocking]
    oldest = min(pending, key=lambda p: p[1] if p[1] is not None else float("inf"), default=None)
    life = state.core.lifecycle
    return session_status.StatusFacts(
        started=state.started,
        finished=state.finished,
        verification=state.verification,
        verify_scoped=state.verify_scoped,
        gate_red=life.last_verify_rc is not None and life.last_verify_rc != 0,
        gate_timed_out=life.verify_timeout_s > 0,
        stop_reason=state.stop_reason,
        operator_blocked=bool(pending),
        blocked_kind=oldest[0] if oldest else "",
        blocked_since_ep=oldest[1] if oldest else None,
        unattended_questions=state.unattended_questions,
        pending_questions=sum(1 for q in state.questions if not q.answered and not q.from_harness),
        stranded=state.stranded,
    )


@functools.lru_cache(maxsize=64)
def cached_context_window(provider: str, model: str) -> int | None:
    """Return the model's context window, memoised since every surface asks per heartbeat."""
    return registry.context_window(provider, model)


def context_fill(state: state_types.SessionState) -> int | None:
    """Return the context-window fill in percent at the last completed model call.

    The one rule behind every surface's `ctx N%` readout.

    Args:
        state: The folded state.

    Returns:
        The call's full prompt tokens over the model's window, capped at 100; None
        until both sides are known.
    """
    call = state.last_model_call
    if call is None or call.context_tokens <= 0 or not call.model:
        return None
    window = cached_context_window(call.provider, call.model)
    if not window:
        return None
    return min(100, round(100 * call.context_tokens / window))


def budget_dict(budget: state_types.BudgetView) -> dict[str, Any]:
    """Return the wire form of a `BudgetView`, the one `attach --json` and `sessions show` share.

    Attributes:
        usd_text: The rendered session spend, then this execution's against its cap.
        plan_text: The account's reading: the binding window's percent, its reset and
            when it was read; "" without a reading.

    Args:
        budget: The usage.

    Returns:
        The view's fields plus the rendered `usd_text` and `plan_text`.
    """
    d = dataclasses.asdict(budget)
    account = budget.account
    d["usd_text"] = format.budget_usd_text(
        budget.usd_so_far,
        partial=budget.session.usd_partial,
        execution_usd=budget.execution.usd + budget.execution.usd_running_lanes,
        usd_cap=budget.execution.usd_cap,
        cell=session_status.usage_cell(
            budget.usd_so_far,
            partial=budget.session.usd_partial,
            plan_observed_points=budget.session.plan_observed_points,
            plan_used_percent=account.used_percent if account else 0.0,
        ),
    )
    d["plan_text"] = (
        format.plan_readout(
            account.used_percent,
            account.resets_at,
            now=time.time(),
            observed_at=account.observed_at,
        )
        if account is not None and account.used_percent > 0
        else ""
    )
    return d


def session_state_as_dict(
    state: state_types.SessionState,
    session_dir: pathlib.Path | None = None,
    *,
    unmerged: bool = False,
) -> dict[str, Any]:
    """Return the wire form of a `SessionState`, what `attach --json` and the web serialize.

    Attributes:
        review: The shared review evidence with computed merge facts.
        context_used_percent: The last model call's context fill percentage, or null when unknown.
        needs_new_work: Whether a resume requires a new task rather than continuing work.
        budget: The BudgetView with its rendered `usd_text` and `plan_text`.
        open_approval: The current approval identifier, or null when none is open.
        status: The lifecycle status word.
        status_label: The rendered status with mode and unmerged marks when available.
        task: The operator's words, whole.
        continued_task: The latest resume steer, whole; empty without a continuation.
        task_line: The task clipped to 200 characters, then a newline and `continued: <steer>`,
            clipped the same way, when a resume steer exists.
        dead_state: The explanation for a worker that died without an end, or an empty string.
        operator_blocked: Whether an unanswered blocking prompt is open.
        log_tail: The retained log lines as plain strings rather than LogLine records.
        live: Whether steer or stop can reach the session, or null without a directory.
        session_id: The journaled session identifier with the directory name as fallback.
        mode: The session mode with the manifest as fallback.
        policy: The rendered launch policy when a session directory is supplied.
        ports: The worker's listening ports when a session directory is supplied.
        shells: The background shell roster when a session directory is supplied.
        plan_md: A plan session's deliverable when readable.
        head: The question part added to each pending approval.
        payload: The command part added to each pending approval.
        label: The picker label added to each commit.

    Args:
        state: The folded state.
        unmerged: The manifest header's fact that the run holds unmerged work.
        session_dir: The session's state dir; with it `status` is the dir-aware word,
            `live` says whether a steer or stop would reach anything, `ports` and
            `shells` are live probes, a plan's `plan_md` is its deliverable, and the
            dir fills the identity the fold left empty. Without it `live` is None,
            right only for a dir-less stream (the machine reasoning snapshot).

    Returns:
        The state's fields with tuples as lists, plus the computed `context_used_percent`,
        `needs_new_work`, `open_approval`, `status`, `status_label`, the full `task`, `task_line`,
        `dead_state`, `operator_blocked`, the rendered budget text, approval parts and
        commit labels, and `log_tail` as plain strings.
        Journal event fields keep their stored names.
    """
    d = {name: _wire_value(getattr(state, name)) for name in PUBLIC_STATE_FIELDS}
    d["review"] = review.as_dict(state.review)
    d["context_used_percent"] = context_fill(state)
    d["needs_new_work"] = listing.needs_new_work(
        finished=state.finished,
        stop_reason=state.stop_reason,
        verification=state.verification,
        saved_answers=len(ipc.saved_question_answers(session_dir)) if session_dir else 0,
    )
    d["budget"] = budget_dict(state.budget)
    for ap, row in zip(state.approvals, d["approvals"], strict=True):
        row["head"], row["payload"] = state_types.approval_parts(ap.prompt)
    for commit, row in zip(state.commits, d["commits"], strict=True):
        row["label"] = commit.label
    current = session_fold.open_approval_of(state)
    d["open_approval"] = None if current is None else current.id
    fallback_task = ""
    fallback_continued = ""
    if session_dir is not None:
        word, reason = session_status.status_for_session_dir(session_dir, status_facts(state))
        d["live"] = word in session_status.LIVE_STATUS_WORDS
        # A forked execution's log opens at session.resume and folds its identity empty.
        d["session_id"] = d["session_id"] or session_dir.name
        d["mode"] = d.get("mode") or ""
        manifest = None
        with contextlib.suppress(sessions_manifest.ManifestError):
            manifest = sessions_manifest.read_manifest(session_dir)
            if not state.task_label.task:
                fallback_task = manifest.user_task
                fallback_continued = manifest.continued_task
            d["mode"] = d["mode"] or manifest.mode
        d["policy"] = policy.manifest_policy(
            manifest, target=state.targets.current, environment=state.environment
        ).line()
        d["ports"] = process.listening_ports(session_dir)
        # On every frame: the run view streams from this dict.
        d["shells"] = shells.roster_from_dir(session_dir / shells.SHELLS_DIR)
        if d["mode"] == "plan":
            with contextlib.suppress(OSError):
                d["plan_md"] = (session_dir / "plan.md").read_text(encoding="utf-8")
    else:
        d["live"] = None
        facts = status_facts(state)
        word, reason = session_status.status_word(
            finished=state.finished,
            verification=state.verification,
            stop_reason=state.stop_reason,
            scoped=state.verify_scoped,
            gate_red=facts.gate_red,
            gate_timed_out=facts.gate_timed_out,
            stranded=facts.stranded,
        )
    d["status"] = word
    d["status_label"] = (
        format.listing_status_label(d["mode"], word, reason, unmerged=unmerged)
        if session_dir is not None
        else format.status_label(word, reason)
    )
    label = state.task_label.fallback(fallback_task, fallback_continued)
    d["task"] = label.task
    d["continued_task"] = label.continued_task
    d["task_line"] = label.line()
    d["dead_state"] = format.dead_run_note(word, reason)[0]
    # From the fold, so a dir-less consumer still gets the "blocked, not working" signal.
    d["operator_blocked"] = status_facts(state).operator_blocked
    d["log_tail"] = [line.text for line in state.log_tail]
    return d
