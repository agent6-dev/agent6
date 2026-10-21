# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Share session labels and status decisions across journal readers and listings."""

from __future__ import annotations

import dataclasses
import pathlib
import time

from agent6 import task_text
from agent6.sessions import ipc, process
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import format


@dataclasses.dataclass(frozen=True, slots=True)
class TaskLabel:
    """The original task and the latest continuation, before display clipping."""

    task: str = ""
    continued_task: str = ""

    def start(self, task: str) -> TaskLabel:
        """Return the first non-empty task; REPL questions do not rename the session.

        Args:
            task: The task an execution starts with.

        Returns:
            This label when it holds a task, else a label holding `task`.
        """
        return self if self.task else TaskLabel(task)

    def resume(self, task: str, steer: str) -> TaskLabel:
        """Return the original task and latest steer; an unsteered resume clears the prior steer.

        Args:
            task: The original task the resume carries.
            steer: The resume's steer, "" for an unsteered resume.

        Returns:
            This label with `steer` as its continuation when it holds a task, else a label
            holding `steer`, or `task` when `steer` is empty.
        """
        if not self.task:
            return self.start(steer or task)
        return TaskLabel(self.task, steer)

    def fallback(self, task: str) -> TaskLabel:
        """Return manifest text only when the journal holds no task.

        Args:
            task: The manifest's task text.

        Returns:
            This label when it holds a task, else a label holding `task` in the journal's
            display form.
        """
        return self if self.task else TaskLabel(task_text.task_display_text(task))

    def line(self, max_chars: int | None = None) -> str:
        """Return each task clipped once at the caller's width.

        Args:
            max_chars: The caller's width; None leaves each line whole.

        Returns:
            The original task's one-line summary, then a `continued: ...` line for the
            latest continuation when there is one.
        """
        line = task_snippet(self.task, max_chars=max_chars)
        if self.continued_task:
            line += "\n" + task_snippet(f"continued: {self.continued_task}", max_chars=max_chars)
        return line


def task_snippet(text: str, max_chars: int | None = None) -> str:
    """Return a task's one-line summary for a listing.

    Args:
        text: The task text or ask transcript.
        max_chars: The width to clip to, with an ellipsis; None leaves it whole.

    Returns:
        The task's headline, else its first non-blank line.
    """
    snip = task_text.task_headline(text) or next(
        (ln.strip() for ln in text.splitlines() if ln.strip()), ""
    )
    if max_chars is not None and len(snip) > max_chars:
        snip = snip[: max_chars - 1] + "…"
    return snip


def cost_cell(
    cost_usd: float, *, partial: bool, plan_consumed: float, plan_used_percent: float
) -> str:
    """Return a session's cost, the one spelling the listings, show and the reports print.

    Args:
        cost_usd: The dollars spent across executions.
        partial: The dollar figure is a lower bound.
        plan_consumed: The plan points consumed across executions.
        plan_used_percent: The account's last plan reading.

    Returns:
        The points once a subscription plan answered a call, the dollars when any were
        spent, both for a run that spent both; "" for a clean $0.
    """
    metered = plan_used_percent > 0 or plan_consumed > 0
    return format.format_cost_cell(
        cost_usd, partial=partial, plan_points=plan_consumed if metered else None
    )


# The operator's own stops: each reads "stopped" and exits 130.
OPERATOR_STOPS = frozenset({"interactive_stop", "interrupted", "steer_abort", "steer_exit"})


def end_text(reason: str, summary: str = "", finish_summary: str = "") -> str:
    """Return the sentence a session's end carries, "" when it carries none.

    Args:
        reason: The session.end reason token.
        summary: The journaled end sentence.
        finish_summary: The finish tool's summary; only a clean end may use it.

    Returns:
        The journaled sentence, else a clean end's finish summary; a journal with neither
        has the reason word its label already shows.
    """
    if summary.strip():
        return summary.strip()
    if reason in ("", "finish_session", "finish_planning"):
        return finish_summary.strip()
    return ""


def status_word(
    *,
    finished: bool,
    all_passed: bool | None,
    end_reason: str,
    scoped: bool = False,
    gate_red: bool = False,
    gate_timed_out: bool = False,
    stranded: bool = False,
) -> tuple[str, str]:
    """Map an end state to the status word and its detail.

    The one place that decides how a run's outcome reads; headers and listings share
    it. "stopped" and "undone" are the operator's own acts, "planned" and "answered"
    the clean exits that verified nothing, "passed" a finish_session over a green gate,
    "finished" a deliberate finish without all-passed or an end the harness declared,
    and anything else "failed" with the reason.

    Args:
        finished: A session.end was seen.
        all_passed: The verify tri-state: True when the final tree was observed
            verify-green, False when not (red, stale or an error end), None when no
            verify command gated it.
        end_reason: The session.end reason word.
        scoped: The certifying gate ran scoped to the tests nearest the run's diff,
            so a pass reads "passed · scoped gate".
        gate_red: This execution's last verify ran and failed; False also covers a
            stale green and a gate nothing ran.
        gate_timed_out: That verify ended on its timeout, so it gave no verdict.
        stranded: The run finished with its edits uncommitted and no commit landed.

    Returns:
        The `(word, detail)` pair.
    """
    if not finished:
        return "running", ""
    not_green = "gate timed out" if gate_timed_out else "gate red" if gate_red else "unverified"
    if end_reason in OPERATOR_STOPS:
        return "stopped", ""
    # An end with its own word, never "passed" and never "failed", whatever the gate said.
    own_word = {
        "finish_planning": ("planned", ""),
        "answered": ("answered", ""),
        "undone": ("undone", ""),
        "silent_finish": ("finished", "no finish call" if all_passed is not False else not_green),
        "metric_plateau": ("finished", "metric plateau" if all_passed is not False else not_green),
        # The harness ended the run over a green gate; the worker called no finish_session.
        "verify_settled": ("finished", "settled"),
        "settled": ("finished", not_green),
        # A verify against the unmodified tree proved the gate red before the run touched it.
        "gate_red_at_base": ("finished", "gate was already red"),
        # A fan-out's own end: its lanes carry the gate verdicts, and no model finished it.
        "lanes_ranked": ("finished", "lanes ranked"),
    }
    if end_reason in own_word:
        word, detail = own_word[end_reason]
        # A harness end over edits no commit holds exits 5; the label says so too.
        if stranded and end_reason in ("silent_finish", "metric_plateau", "verify_settled"):
            detail = f"{detail} · not committed"
        return word, detail
    # A clean finish whose work no commit holds says so before anything else.
    uncommitted = "not committed" if stranded else ""
    if all_passed:
        return "passed", uncommitted or ("scoped gate" if scoped else "")
    # Only an observed not-green words "failed"; a deliberate finish over one is "finished".
    if all_passed is False and end_reason:
        clean_end = end_reason == "finish_session"
        # A stale-gate claim is its own detail, as the raw reason so a surface names it once.
        word = "finished" if clean_end or end_reason == "gate_stale" else "failed"
        return word, not_green if clean_end else end_reason
    return "finished", uncommitted


# The prompt events that mean "alive but blocked on the operator".
OPERATOR_PROMPT_EVENTS = frozenset({"approval.prompt", "question.prompt"})
OPERATOR_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


# A parked submission's word; the detail beside it is the manifest's short cause.
PARKED_WORD = "parked"


@dataclasses.dataclass(frozen=True, slots=True)
class StatusFacts:
    """The event-derived inputs to `status_for_session_dir`.

    Both event readers produce them, the tolerant scanner behind listings and the
    typed fold behind the live views, so every surface feeds the one status decision
    the same answers for the same log.

    Attributes:
        started: A session.start was seen; a parked or created run has none.
        finished: A session.end was seen and no later execution began.
        all_passed: The verify tri-state; None when no verify command gated the end.
        verify_scoped: The judging gate ran scoped, which qualifies a pass.
        gate_red: This execution's last verify ran and failed.
        gate_timed_out: That verify ended on its timeout.
        end_reason: The session.end reason word.
        operator_blocked: An approval or question is still unanswered.
        blocked_kind: The oldest unanswered prompt's kind, "approval" or "question".
        blocked_since_ep: That prompt's asked-at epoch.
        unattended_questions: Questions this execution's harness answered empty because nobody
            was attached.
        stranded: This execution finished with its edits uncommitted and no commit landed.
    """

    started: bool = False
    finished: bool = False
    all_passed: bool | None = False
    verify_scoped: bool = False
    gate_red: bool = False
    gate_timed_out: bool = False
    end_reason: str = ""
    operator_blocked: bool = False
    blocked_kind: str = ""
    blocked_since_ep: float | None = None
    unattended_questions: int = 0
    stranded: bool = False


def status_for_session_dir(session_dir: pathlib.Path, facts: StatusFacts) -> tuple[str, str]:
    """Decide the status word and detail for a session that has a dir on disk.

    The dir supplies what events cannot: a parked submission (the manifest), a pending
    stop (the marker) and worker liveness (the pid file). A started session is live iff
    its worker is. The pid is written before session.start, so no pid file means the
    worker cleared it on the way out. Log silence cannot stand in for this: a `kill -9`
    leaves the pid file while an abnormal exit through the finally clears it.

    Args:
        session_dir: The session's state dir.
        facts: The event-derived facts from either reader.

    Returns:
        The `(word, detail)` pair.
    """
    if facts.finished:
        word, reason = status_word(
            finished=True,
            all_passed=facts.all_passed,
            end_reason=facts.end_reason,
            scoped=facts.verify_scoped,
            gate_red=facts.gate_red,
            gate_timed_out=facts.gate_timed_out,
            stranded=facts.stranded,
        )
        if n := facts.unattended_questions:
            note = f"{n} question{'s' if n != 1 else ''} unanswered"
            reason = f"{reason} · {note}" if reason else note
        return word, reason
    live = process.worker_is_alive(session_dir)
    if facts.operator_blocked and live:
        # Before session.start too: a run asks about uncommitted changes before it starts.
        word = "waiting"
        if facts.blocked_kind and facts.blocked_since_ep is not None:
            age = format.format_age(time.time() - facts.blocked_since_ep)
            reason = f"{facts.blocked_kind} {age}"
        else:
            reason = "needs answer"
    elif not facts.started:
        word, reason = unstarted_status(session_dir)
    elif not live:
        return "stale", ""
    else:
        word, reason = "running", ""
    if (
        word in LIVE_STATUS_WORDS
        and ipc.stop_request_pending(session_dir)
        and not ipc.stop_request_is_now(session_dir)
    ):
        note = "stopping after this step"
        reason = f"{reason} · {note}" if reason else note
    return word, reason


def unstarted_status(session_dir: pathlib.Path) -> tuple[str, str]:
    """Decide the status of a session before any session.start.

    A live worker is still in preflight ("starting"). Without one, the dir is a parked
    submission, a manifest that fails to parse ("unreadable"), a worker that died
    launching (its pid file survives the kill: "stale", since a killed preflight spent
    real dollars) or a never-started dir ("created").

    Args:
        session_dir: The session's state dir.

    Returns:
        The `(word, detail)` pair.
    """
    if process.worker_is_alive(session_dir):
        return "starting", ""
    if (session_dir / sessions_manifest.MANIFEST_NAME).is_file():
        try:
            manifest = sessions_manifest.read_manifest(session_dir)
        except sessions_manifest.ManifestError as exc:
            return "unreadable", format.clip_cell(str(exc), 60)
        if manifest.parked_task:
            return PARKED_WORD, manifest.parked_reason
    if process.read_worker_pid(session_dir) is not None:
        return "stale", "died launching"
    return "created", ""


# Terminal without a session.end: the fan-out's await accepts them so it cannot hang.
DIED_WITHOUT_END = frozenset({"stale", "created", "parked", "unreadable", "?"})


def died_without_end(status: str) -> bool:
    """Return whether the status word means the session never reached its own end."""
    return status in DIED_WITHOUT_END


# A positive set: a new status word is not a result a fan-out may rank until it is listed.
RESULT_WORDS = frozenset({"passed", "finished", "stopped", "planned", "answered"})


def produced_result(status: str) -> bool:
    """Return whether the status word means a deliberate end that left mergeable work.

    Only such a lane is a fan-out compare candidate or joins a coordinator's branch.

    Args:
        status: The session's status word.

    Returns:
        True for a word in `RESULT_WORDS`.
    """
    return status in RESULT_WORDS


# A run in any other state reads no steer or answer marker: a surface offers resume instead.
LIVE_STATUS_WORDS = frozenset({"running", "starting", "waiting"})
