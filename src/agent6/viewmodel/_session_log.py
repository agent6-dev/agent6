# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold session journals tolerantly for listings and session inspection."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
from collections.abc import Mapping

from agent6 import budget, evidence, kinds, terminal
from agent6.viewmodel import _session_status, events, format, review


@dataclasses.dataclass(frozen=True, slots=True)
class LogScan:
    """One tolerant pass over a session's journal, behind the hub listing and `sessions show`.

    One owner, so the resume rules (bank the cost of past executions, un-finish) and
    the torn-line tolerances cannot drift between consumers. Token counters are the
    current execution's; `cost_usd` and `plan_consumed` are cumulative, matching the typed
    fold's BudgetView.

    Attributes:
        saw_start: A session.start or loop.resume.start was seen; neither means unstarted.
        mode: The session's mode, "?" until a start event names it.
        task_label: The first task and the latest resume steer from the journal.
        finished: A session.end was seen and no later execution began.
        all_passed: The verify tri-state; None when no verify command gated the end.
        verify_scoped: The judging gate ran scoped.
        end_reason: The session.end reason word.
        stranded: The latest execution finished with its edits uncommitted and no commit landed.
        setup_output: The workspace's setup command and retained output, or its failure diagnostic.
        end_summary: The journaled end sentence, "" while an execution runs or none was recorded.
        gate_swapped: The verify gate changed between executions, so the first baseline is
            another gate's verdict.
        baseline_ok: The start gate's verdict on the tree before the run's first turn; None
            when none was recorded.
        verified: The gate's word for the final tree, as the last session.end carried it:
            `passed`, `failed`, `unverified` or `not_applicable`; "" while a later execution
            runs, and for an end that carried none.
        end_tree: The worktree's tree id that end recorded; "" when it recorded none or a
            later execution began.
        undone_to: The fork `/undo` continues the session as; "" when it was never undone.
        provider_error: The last failed driving call's error, the reason behind a
            `provider_error` end; "" when none failed.
        cost_usd: The spend across every execution; None when no budget.update was seen.
        usd_partial: Some execution had unpriced spend, so `cost_usd` is a lower bound.
        executions: One plus the completed resume executions.
        input_tokens: This execution's input tokens, None until a budget.update carries them.
        output_tokens: This execution's output tokens.
        cache_read_tokens: The cached side of the input, None when the event lacked it.
        cache_creation_tokens: Cache-creation tokens, None when the event lacked it.
        plan_consumed: The plan points consumed across every execution; 0.0 until a plan
            call runs.
        plan_prior_executions: The points banked by completed executions.
        plan_cap: `[budget].max_plan_points` for this execution, 0.0 until a plan call runs.
        plan_used_percent: The account's last reading, 0.0 until a plan call runs.
        plan_usage: That reading's windows, None until a plan call reports them.
        plan_usage_at: When the reading was taken, as epoch seconds; 0.0 without one.
        turn: The last event's turn.
        start_ep: session.start's epoch, else the first event's (a fork's log has none).
        last_ep: The last event's epoch.
        last_type: The last event's type.
        operator_blocked: A prompt is still unanswered on this execution.
        blocked_kind: The oldest unanswered prompt's kind, "approval" or "question".
        blocked_since_ep: That prompt's asked-at epoch.
        last_verify_rc: This execution's last verify.end exit code.
        verify_timeout_s: The limit that verify ran out, 0.0 when it ended on its own.
        pins: The operator's pinned instructions in force.
        unattended_questions: This execution's questions answered empty because nobody was
            attached.
        left_out: The (path, reason) of each edit the commits left out, across every
            execution; the reason is `ignored` or `untracked` (`loop.edit.left_out`).
        review: The shared fold of panel evidence and finding history.
        end_ep: The latest execution's end time as epoch seconds; None while it runs.
        execution_id: The current execution's recorded id, if present.
        source_tip: The most recent diff checkpoint's commit.
        verifications: Every recorded verification, including baseline observations.
        final_verify_id: The end event's verification id, absent if evidence is damaged.
        evidence_notes: Missing or unreadable verification evidence.
    """

    saw_start: bool = False
    mode: str = "?"
    task_label: _session_status.TaskLabel = dataclasses.field(
        default_factory=_session_status.TaskLabel
    )
    finished: bool = False
    all_passed: bool | None = False
    verify_scoped: bool = False
    end_reason: str = ""
    stranded: bool = False
    setup_output: str = ""
    end_summary: str = ""
    baseline_ok: bool | None = None
    gate_swapped: bool = False
    verified: str = ""
    end_tree: str = ""
    undone_to: str = ""
    provider_error: str = ""
    cost_usd: float | None = None
    usd_partial: bool = False
    executions: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    plan_consumed: float = 0.0
    plan_prior_executions: float = 0.0
    plan_cap: float = 0.0
    plan_used_percent: float = 0.0
    plan_usage: budget.PlanUsage | None = None
    plan_usage_at: float = 0.0
    turn: int | None = None
    start_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    operator_blocked: bool = False
    blocked_kind: str = ""
    blocked_since_ep: float | None = None
    last_verify_rc: int | None = None
    verify_timeout_s: float = 0.0
    pins: tuple[str, ...] = ()
    unattended_questions: int = 0
    left_out: tuple[tuple[str, str], ...] = ()
    review: review.ReviewView = dataclasses.field(default_factory=review.ReviewView)

    execution_id: str | None = None
    source_tip: str | None = None
    verifications: tuple[evidence.VerificationEvidence, ...] = ()
    final_verify_id: str | None = None
    end_ep: float | None = None
    evidence_notes: tuple[str, ...] = ()

    def verify_verdict(self) -> bool | None:
        """Return the gate verdict for judging candidates.

        The status word cannot answer it: a finish over a red gate folds to "finished".

        Returns:
            True when the run ended all-passed, False when this execution's last verify
            failed, None when nothing observed the final tree (gateless, no verify this
            execution, a green made stale by later edits, or a mode other than run).
        """
        if self.mode != "run":
            return None
        if self.finished and self.all_passed:
            return True
        if self.last_verify_rc is not None and self.last_verify_rc != 0:
            return False
        return None

    def gate_word(self) -> str:
        """Return `verified`, with `timed_out` for a red that is the gate running out its limit.

        Returns:
            The gate's word for the final tree.
        """
        return "timed_out" if self.verified == "failed" and self.verify_timeout_s else self.verified

    @property
    def cost_cell(self) -> str:
        """The session's cost as `cost_cell` spells it; "" before any budget.update."""
        if self.cost_usd is None:
            return ""
        return _session_status.cost_cell(
            self.cost_usd,
            partial=self.usd_partial,
            plan_consumed=self.plan_consumed,
            plan_used_percent=self.plan_used_percent,
        )

    def status_facts(self) -> _session_status.StatusFacts:
        """Return this scan's answers to the status questions.

        The typed fold's `_state_view.status_facts` agrees on the same log.

        Returns:
            The facts `status_for_session_dir` reads.
        """
        return _session_status.StatusFacts(
            started=self.saw_start,
            finished=self.finished,
            all_passed=self.all_passed,
            verify_scoped=self.verify_scoped,
            gate_red=self.last_verify_rc is not None and self.last_verify_rc != 0,
            gate_timed_out=self.verify_timeout_s > 0,
            end_reason=self.end_reason,
            operator_blocked=self.operator_blocked,
            blocked_kind=self.blocked_kind,
            blocked_since_ep=self.blocked_since_ep,
            unattended_questions=self.unattended_questions,
            stranded=self.stranded,
        )


def read_plan_usage(ev: Mapping[str, object]) -> budget.PlanUsage | None:
    """Return the plan windows a budget.update carries, or None when it carries none."""
    raw = ev.get("plan_windows")
    windows: list[budget.PlanWindow] = []
    for w in raw if isinstance(raw, list) else ():
        if not isinstance(w, dict):
            continue
        try:
            windows.append(
                budget.PlanWindow(
                    str(w["name"]),
                    float(w["used_percent"]),
                    int(w["window_minutes"]),
                    float(w["resets_at"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return budget.PlanUsage(windows=tuple(windows)) if windows else None


def figure(ev: Mapping[str, object], key: str, last_good: float) -> float:
    """Read a budget.update figure as the typed fold does.

    Args:
        ev: The event.
        key: The figure's key.
        last_good: The figure to keep when the value is unusable.

    Returns:
        0.0 for an absent key (an aggregate event carries only what it summed), else
        the tolerant float.
    """
    return tolerant_float(ev[key], last_good) if key in ev else 0.0


def tolerant_float(raw: object, last_good: float) -> float:
    """Read a figure that a torn or adversarial line may have mangled.

    Args:
        raw: The raw value.
        last_good: The figure to keep when the value is unusable; falsy junk keeps it
            too, where an `or 0.0` fallback would reset it.

    Returns:
        The float of a real number or numeric string, else `last_good`.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return float(raw)
    return last_good


def scan_session_log(logs: pathlib.Path) -> LogScan:  # noqa: C901, PLR0912, PLR0915  # one branch per event type
    """Fold a session's journal into a `LogScan`.

    A live writer can leave a torn multibyte tail, so the file is decoded with
    replacement and the mangled line fails `json.loads` and is skipped.

    Args:
        logs: The journal path.

    Returns:
        The scan; an empty one when the file cannot be read.
    """
    mode, task = "?", _session_status.TaskLabel("")
    finished, end_reason, end_summary, verified = False, "", "", ""
    stranded = False
    end_tree = ""
    baseline_ok: bool | None = None
    gate_swapped = False
    undone_to = ""
    setup_output = ""
    provider_error = ""
    all_passed: bool | None = False
    verify_scoped = False
    saw_start = False
    usd_execution = 0.0
    usd_prior_executions = 0.0
    saw_budget = False
    usd_partial = False
    executions = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    plan_consumed = plan_prior_executions = plan_cap = plan_used_percent = 0.0
    plan_usage: budget.PlanUsage | None = None
    plan_usage_at: float = 0.0
    turn: int | None = None
    start_ep: float | None = None
    first_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    # Only an answer or an execution boundary clears a prompt: a steer request leaves it waiting.
    pending_prompts: dict[str, tuple[str, float | None]] = {}
    unattended_questions = 0
    last_verify_rc: int | None = None
    verify_timeout_s = 0.0
    pins: list[str] = []
    left_out: dict[str, str] = {}
    review_view = review.ReviewView()
    execution_id: str | None = None
    source_tip: str | None = None
    final_verify_id: str | None = None
    end_ep: float | None = None
    verifications: list[evidence.VerificationEvidence] = []
    pending_verifications: set[str] = set()
    evidence_notes: list[str] = []
    try:
        with logs.open(encoding="utf-8", errors="replace") as fh:
            for line_number, line in enumerate(fh, 1):
                try:
                    ev = json.loads(line)
                except ValueError as exc:
                    evidence_notes.append(f"journal line {line_number}: {exc}")
                    review_view = dataclasses.replace(
                        review_view,
                        read_failure=review_view.read_failure
                        or f"journal line {line_number}: {exc}",
                        completion="unknown",
                        freshness="unknown",
                    )
                    continue
                if not isinstance(ev, dict):
                    evidence_notes.append(f"journal line {line_number}: expected an event object")
                    continue
                review_view = review.apply_event(review_view, ev)
                etype = ev.get("type")
                if etype in ("session.start", "loop.resume.start"):
                    execution_id = ev.get("execution_id")
                    final_verify_id = None
                elif etype == "diff.updated":
                    source_tip = ev.get("sha") or source_tip
                ep = events.event_epoch(ev.get("ts"))
                if ep is not None:
                    last_ep = ep
                    if first_ep is None:
                        first_ep = ep
                if isinstance(etype, str):
                    last_type = etype
                # Journals written before the turn rename stamp `iteration` and `iterations`.
                if isinstance(ev_turn := ev.get("turn", ev.get("iteration")), int):
                    turn = ev_turn
                if etype in _session_status.OPERATOR_PROMPT_EVENTS:
                    # Keyed by str(id) like the answer side, so an int id still clears.
                    if (pid := ev.get("id")) is not None:
                        kind = "approval" if etype == "approval.prompt" else "question"
                        pending_prompts[str(pid)] = (kind, ep)
                elif etype in _session_status.OPERATOR_ANSWER_EVENTS:
                    pending_prompts.pop(str(ev.get("id")), None)
                    if etype == "question.answer" and ev.get("unseen") is True:
                        unattended_questions += 1
                if etype == "session.start":
                    task = task.start(str(ev.get("user_task", "")))
                    saw_start = True
                    finished, stranded, verified = False, False, ""
                    end_ep = None
                    end_summary = ""
                    end_tree = ""
                    mode = str(ev.get("mode", mode))
                    # A new execution re-asks with restarted ids; a held-over prompt waits forever.
                    pending_prompts.clear()
                    unattended_questions = 0
                    last_verify_rc = None
                    verify_timeout_s = 0.0
                    if start_ep is None:
                        start_ep = ep
                elif etype == "session.end":
                    finished = True
                    if isinstance(end_turns := ev.get("turns", ev.get("iterations")), int):
                        turn = end_turns
                    # An explicit null is the ungated tri-state; an absent key folds False.
                    raw_ap = ev.get("all_passed", False)
                    all_passed = None if raw_ap is None else bool(raw_ap)
                    verify_scoped = bool(ev.get("scoped", False))
                    end_reason = str(ev.get("reason", ""))
                    end_summary = str(ev.get("summary", ""))
                    if end_reason == "setup_command_failed" and ev.get("summary"):
                        setup_output = terminal.scrub_terminal_controls(str(ev["summary"]))
                    verified = str(ev.get("verified") or "")
                    end_tree = str(ev.get("tree") or "")
                    final_verify_id = ev.get("final_verify_id")
                    end_ep = ep
                elif etype == "loop.resume.start":
                    task = task.resume(str(ev.get("user_task", "")), str(ev.get("steer", "")))
                    if saw_start:
                        # Each execution's budget starts at 0: bank the finished one's total.
                        usd_prior_executions += usd_execution
                        usd_execution = 0.0
                        input_tokens = output_tokens = None
                        cache_read_tokens = cache_creation_tokens = None
                        # The account's reading stays: the run is plan-metered before its
                        # next call too.
                        plan_prior_executions += plan_consumed
                        plan_consumed = plan_cap = 0.0
                        last_verify_rc = None
                        verify_timeout_s = 0.0
                        executions += 1
                    saw_start = True
                    mode = str(ev.get("mode", mode))
                    finished, stranded, verified = False, False, ""
                    end_ep = None
                    end_summary = ""
                    end_tree = ""
                    pending_prompts.clear()
                    unattended_questions = 0
                elif etype == "session.stranded":
                    stranded = True
                elif etype == "session.undone":
                    undone_to = str(ev.get("new_session_id", ""))
                elif etype == "workspace.setup":
                    setup_output = format.setup_output(ev)
                elif etype == "verify.start":
                    if ev.get("verify_id"):
                        pending_verifications.add(str(ev["verify_id"]))
                    # The verdict before it described an older tree; the run page drops it too.
                    last_verify_rc = None
                    verify_timeout_s = 0.0
                elif etype == "verify.end":
                    try:
                        record = evidence.from_event(ev)
                    except (ValueError, TypeError) as exc:
                        evidence_notes.append(f"journal line {line_number}: {exc}")
                    else:
                        verifications.append(record)
                        if record.verify_id:
                            pending_verifications.discard(record.verify_id)
                    rc = ev.get("exit_code")
                    last_verify_rc = (
                        rc if isinstance(rc, int) and not isinstance(rc, bool) else None
                    )
                    verify_timeout_s = events.verify_limit_hit(rc, ev.get("timeout_s"))
                elif etype == "role.result":
                    # A side call's failure never ends the run, so it is not the reason.
                    if ev.get("ok") is False and not kinds.is_side_role(str(ev.get("role", ""))):
                        provider_error = str(ev.get("error", ""))
                elif etype == "loop.baseline":
                    if baseline_ok is None:
                        baseline_ok = bool(ev.get("ok"))
                    for i, record in enumerate(verifications):
                        matches = bool(record.verify_id) and record.verify_id == ev.get("verify_id")
                        legacy = not ev.get("verify_id") and i == len(verifications) - 1
                        if matches or legacy:
                            verifications[i] = dataclasses.replace(record, baseline=True)
                elif etype == "loop.verify_swapped":
                    gate_swapped = True
                elif etype == "loop.edit.left_out":
                    for reason in ("ignored", "untracked"):
                        raw_paths = ev.get(reason)
                        for p in raw_paths if isinstance(raw_paths, list) else []:
                            left_out[str(p)] = reason
                elif etype == "loop.pin.added":
                    pins.append(str(ev.get("text", "")))
                elif etype == "loop.pin.restored":
                    raw_pins = ev.get("pins")
                    pins = [str(p) for p in raw_pins] if isinstance(raw_pins, list) else []
                elif etype == "budget.update":
                    saw_budget = True
                    usd_execution = figure(ev, "usd_total", usd_execution)
                    usd_partial = bool(ev.get("usd_partial")) or usd_partial
                    ti, to = ev.get("input_total"), ev.get("output_total")
                    if isinstance(ti, int):
                        # The four counters travel as one group: no stale cached side from earlier.
                        cr, cc = ev.get("cache_read_total"), ev.get("cache_creation_total")
                        input_tokens = ti
                        cache_read_tokens = cr if isinstance(cr, int) else None
                        cache_creation_tokens = cc if isinstance(cc, int) else None
                    output_tokens = to if isinstance(to, int) else output_tokens
                    plan_consumed = figure(ev, "plan_consumed", plan_consumed)
                    plan_cap = figure(ev, "plan_cap", plan_cap)
                    plan_used_percent = figure(ev, "plan_used_percent", plan_used_percent)
                    if (reading := read_plan_usage(ev)) is not None:
                        plan_usage = reading
                        plan_usage_at = events.event_epoch(ev.get("ts")) or 0.0
    except FileNotFoundError:
        evidence_notes.append("Journal missing.")
    except OSError as exc:
        evidence_notes.append(f"Journal unreadable: {exc}")
        review_view = dataclasses.replace(review_view, read_failure=str(exc))
    return LogScan(
        saw_start=saw_start,
        review=review_view,
        execution_id=execution_id,
        source_tip=source_tip,
        verifications=tuple(verifications),
        final_verify_id=None if evidence_notes else final_verify_id,
        end_ep=end_ep,
        evidence_notes=tuple(evidence_notes)
        + tuple(f"Verification {v} has no recorded result." for v in sorted(pending_verifications)),
        mode=mode,
        task_label=task,
        finished=finished,
        all_passed=all_passed,
        verify_scoped=verify_scoped,
        end_reason=end_reason,
        stranded=stranded,
        end_summary=end_summary,
        baseline_ok=baseline_ok,
        gate_swapped=gate_swapped,
        verified=verified,
        setup_output=setup_output,
        end_tree=end_tree,
        undone_to=undone_to,
        provider_error=provider_error,
        cost_usd=(usd_prior_executions + usd_execution) if saw_budget else None,
        usd_partial=usd_partial,
        executions=executions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        plan_consumed=plan_prior_executions + plan_consumed,
        plan_prior_executions=plan_prior_executions,
        plan_cap=plan_cap,
        plan_used_percent=plan_used_percent,
        plan_usage=plan_usage,
        plan_usage_at=plan_usage_at,
        unattended_questions=unattended_questions,
        turn=turn,
        start_ep=start_ep if start_ep is not None else first_ep,
        last_ep=last_ep,
        last_type=last_type,
        operator_blocked=bool(pending_prompts),
        blocked_kind=(
            oldest[0]
            if (
                oldest := min(
                    pending_prompts.values(),
                    key=lambda kv: kv[1] if kv[1] is not None else float("inf"),
                    default=None,
                )
            )
            else ""
        ),
        blocked_since_ep=oldest[1] if oldest else None,
        last_verify_rc=last_verify_rc,
        verify_timeout_s=verify_timeout_s,
        pins=tuple(pins),
        left_out=tuple(left_out.items()),
    )
