# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render session status, context usage, and the wire snapshot.

`session_state_as_dict` is the contract every external viewer reads (`attach --json`,
the web page): the state's fields plus `status`, `status_label`, `dead_state`,
`shells`, `live` and `operator_blocked`, with `log_tail` as plain strings; its keys
stay stable. With a session dir, the wire form also reads the dir's status probes
and manifest.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import pathlib
from typing import Any

from agent6.models import registry
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.tools import background
from agent6.viewmodel import _state_types, events, format, listing, policy
from agent6.viewmodel import state as session_fold


def status_facts(state: _state_types.SessionState) -> listing.StatusFacts:
    """Return the fold's answers to the status questions.

    The typed twin of `LogScan.status_facts`; the two agree on the same log.

    Args:
        state: The folded state.

    Returns:
        The facts `status_for_session_dir` reads.
    """
    pending: list[tuple[str, float | None]] = [
        ("approval", a.asked_ep) for a in state.pending_approvals if not a.answered
    ] + [("question", q.asked_ep) for q in state.pending_questions if not q.answered]
    oldest = min(pending, key=lambda p: p[1] if p[1] is not None else float("inf"), default=None)
    gate = state.last_verify
    return listing.StatusFacts(
        started=state.started,
        finished=state.finished,
        all_passed=state.all_passed,
        verify_scoped=state.verify_scoped,
        gate_red=gate is not None and gate.exit_code is not None and gate.exit_code != 0,
        gate_timed_out=gate is not None
        and events.verify_limit_hit(gate.exit_code, gate.timeout_s) > 0,
        end_reason=state.end_reason,
        operator_blocked=bool(pending),
        blocked_kind=oldest[0] if oldest else "",
        blocked_since_ep=oldest[1] if oldest else None,
        unattended_questions=state.unattended_questions,
        stranded=state.stranded,
    )


@functools.lru_cache(maxsize=64)
def cached_context_window(provider: str, model: str) -> int | None:
    """Return the model's context window, memoised since every surface asks per heartbeat."""
    return registry.context_window(provider, model)


def context_fill(state: _state_types.SessionState) -> int | None:
    """Return the context-window fill in percent at the last completed model call.

    The one rule behind every surface's `ctx N%` readout.

    Args:
        state: The folded state.

    Returns:
        The call's full prompt tokens over the model's window, capped at 100; None
        until both sides are known.
    """
    role = state.last_role
    if role is None or role.ctx_tokens <= 0 or not role.model:
        return None
    window = cached_context_window(role.provider, role.model)
    if not window:
        return None
    return min(100, round(100 * role.ctx_tokens / window))


def session_state_as_dict(
    state: _state_types.SessionState,
    session_dir: pathlib.Path | None = None,
    *,
    unmerged: bool = False,
) -> dict[str, Any]:
    """Return the wire form of a `SessionState`, what `attach --json` and the web serialize.

    Args:
        state: The folded state.
        unmerged: The manifest header's fact that the run holds unmerged work.
        session_dir: The session's state dir; with it `status` is the dir-aware word,
            `live` says whether a steer or stop would reach anything, `ports` and
            `shells` are live probes, a plan's `plan_md` is its deliverable, and the
            dir fills the identity the fold left empty. Without it `live` is None,
            right only for a dir-less stream (the machine reasoning snapshot).

    Returns:
        The state's fields with tuples as lists, plus the computed `context_pct`,
        `needs_new_work`, `open_approval`, `status`, `status_label`, `task_line`,
        `dead_state`, `operator_blocked`, the rendered budget text, approval parts and
        step labels, and `log_tail` as plain strings.
    """
    d = dataclasses.asdict(state)
    d["context_pct"] = context_fill(state)
    d["needs_new_work"] = listing.needs_new_work(
        finished=state.finished, end_reason=state.end_reason, all_passed=state.all_passed
    )
    d["budget"]["usd_text"] = format.budget_usd_text(
        state.budget.usd_total,
        partial=state.budget.usd_partial,
        usd_cap=state.budget.usd_cap,
        usd_prior_executions=state.budget.usd_prior_executions,
        cell=listing.cost_cell(
            state.budget.usd_total,
            partial=state.budget.usd_partial,
            plan_consumed=state.budget.plan_consumed,
            plan_used_percent=state.budget.plan_used_percent,
        ),
    )
    for ap, row in zip(state.pending_approvals, d["pending_approvals"], strict=True):
        row["head"], row["payload"] = _state_types.approval_parts(ap.prompt)
    for step, row in zip(state.steps, d["steps"], strict=True):
        row["label"] = step.label
    current = session_fold.open_approval_of(state)
    d["open_approval"] = None if current is None else current.id
    fallback_task = ""
    if session_dir is not None:
        word, reason = listing.status_for_session_dir(session_dir, status_facts(state))
        d["live"] = word in listing.LIVE_STATUS_WORDS
        # A forked execution's log opens at loop.resume.start and folds its identity empty.
        d["session_id"] = d["session_id"] or session_dir.name
        d["mode"] = d.get("mode") or ""
        manifest = None
        with contextlib.suppress(sessions_manifest.ManifestError):
            manifest = sessions_manifest.read_manifest(session_dir)
            if not state.task_label.task:
                fallback_task = manifest.user_task
            d["mode"] = d["mode"] or manifest.mode
        d["policy"] = policy.manifest_policy(manifest).line()
        d["ports"] = process.listening_ports(session_dir)
        # On every frame: the run view streams from this dict.
        d["shells"] = background.roster_from_dir(session_dir / background.SHELLS_DIR)
        if d["mode"] == "plan":
            with contextlib.suppress(OSError):
                d["plan_md"] = (session_dir / "plan.md").read_text(encoding="utf-8")
    else:
        d["live"] = None
        facts = status_facts(state)
        word, reason = listing.status_word(
            finished=state.finished,
            all_passed=state.all_passed,
            end_reason=state.end_reason,
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
    label = state.task_label.fallback(fallback_task)
    d.pop("task_label")
    d["user_task"] = label.task
    d["continued_task"] = label.continued_task
    d["task_line"] = label.line()
    d["dead_state"] = format.dead_run_note(word, reason)[0]
    # From the fold, so a dir-less consumer still gets the "blocked, not working" signal.
    d["operator_blocked"] = status_facts(state).operator_blocked
    d["log_tail"] = [line.text for line in state.log_tail]
    return d
