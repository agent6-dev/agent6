# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Scan session dirs into the listing rows every front-end's hub shows.

The last-activity time, the status word and the task snippet live only here, so the
CLI, TUI and web listings cannot disagree.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import time
from collections.abc import Container, Iterable, Mapping, Sequence

from agent6 import git_ops, task_text
from agent6.sessions import ipc, layout
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import events, format


def session_mtime(session_dir: pathlib.Path) -> float:
    """Return a session's last-activity time as epoch seconds.

    The journal's mtime, else the manifest's, else the dir's; never the dir's first,
    since a viewer's `frontends/` claim bumps it and would float a viewed run to newest.

    Args:
        session_dir: The session's state dir.

    Returns:
        The mtime, 0.0 when none of the three can be read.
    """
    for candidate in (
        session_dir / layout.LOGS_NAME,
        session_dir / sessions_manifest.MANIFEST_NAME,
        session_dir,
    ):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def session_dirs(
    state_dir: pathlib.Path, buckets: Iterable[str] = layout.HUB_BUCKETS
) -> list[pathlib.Path]:
    """Return every session dir a listing shows, newest first by last activity.

    Args:
        state_dir: The repo's state dir.
        buckets: The bucket names under `sessions/` to walk; the hub's by default.

    Returns:
        The session dirs, husks skipped.
    """
    dirs: list[pathlib.Path] = []
    for name in buckets:
        bucket = layout.bucket_dir(state_dir, name)
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir() and not is_session_husk(p))
    dirs.sort(key=session_mtime, reverse=True)
    return dirs


def newest_session_dir(buckets: Iterable[pathlib.Path]) -> pathlib.Path | None:
    """Return the most recently active session dir across the given bucket dirs.

    Husks are skipped: a crash-orphaned dir is newer than the real runs, and returning
    it would point a bare `attach` at a phantom no listing shows.

    Args:
        buckets: The bucket dirs in scope; a missing one is skipped.

    Returns:
        The newest session dir by last activity, or None when no bucket holds one.
    """
    runs: list[pathlib.Path] = []
    for bucket in buckets:
        if bucket.is_dir():
            runs.extend(p for p in bucket.iterdir() if p.is_dir() and not is_session_husk(p))
    dirs = sorted(runs, key=session_mtime, reverse=True)
    return dirs[0] if dirs else None


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


def is_session_husk(session_dir: pathlib.Path) -> bool:
    """Return whether a session dir never started: no manifest, no log, no live worker.

    A dir with a live worker is a just-launched run in its pre-manifest preflight
    window and stays listed as "starting". Listings skip husks, and an id lookup must
    not let one shadow a real run of the same id in another bucket.
    """
    return not layout.session_has_record(session_dir) and not ipc.worker_is_alive(session_dir)


def session_compare(session_dir: pathlib.Path) -> sessions_manifest.CompareStamp | None:
    """Return the fan-out compare stamp on a lane's manifest.

    The event fold does not carry it, so every run view reads it from here.

    Args:
        session_dir: The session's state dir.

    Returns:
        The stamp, or None for a run outside a compared fan-out or with an unreadable
        manifest.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return None
    return manifest.compare


def is_winner(session_dir: pathlib.Path) -> bool:
    """Return whether a run is its fan-out's compare winner."""
    compare = session_compare(session_dir)
    return compare is not None and compare.winner


@dataclasses.dataclass(frozen=True, slots=True)
class SessionSummary:
    """One listing row: everything a hub or `sessions list` needs, uncolored.

    Attributes:
        session_id: The session's id.
        mode: "run", "plan", "ask" or "?".
        task: The raw task text; callers snippet it for their layout.
        status: The status word (created, starting, running, waiting, stale, passed,
            answered, planned, finished, stopped, undone, failed).
        reason: The detail beside the word: the end reason when failed, what it waits
            on when waiting, else "".
        cost_usd: The cumulative spend across executions.
        usd_partial: `cost_usd` is a lower bound: some execution had unpriced spend.
        mtime: The last-activity time as epoch seconds.
        unmerged: The run's chain or branch holds commits its base does not, per the
            merge stamp and the caller's branch-tips snapshot; False without a
            snapshot, and for an undone run (its /undo child carries the mark).
        verify_ok: The gate verdict from the gate facts, not the status word: the
            compare table and the judge read it.
        coordinator: The session that dispatched this lane, the row it nests under.
        lane: The lane number within its fan-out.
        plan_consumed: The plan points this execution consumed on a plan-metered provider.
        plan_cap: The plan's points cap, `[budget].max_percent`.
        plan_used_percent: The account's reading; 0 unless a plan provider answered.
        model: The manifest's provider/model route for this execution.
        model_from_flag: The route came from a `--model` flag rather than config.
    """

    session_id: str
    mode: str
    task: str
    status: str
    reason: str
    cost_usd: float
    usd_partial: bool
    mtime: float
    unmerged: bool = False
    verify_ok: bool | None = None
    coordinator: str = ""
    lane: int | None = None
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_used_percent: float = 0.0
    model: str = ""
    model_from_flag: bool = False

    @property
    def cost_cell(self) -> str:
        """The cost cell: plan points once a subscription plan answered a call, else USD."""
        metered = self.plan_used_percent > 0 or self.plan_consumed > 0
        points = self.plan_consumed if metered else None
        return format.format_cost_cell(self.cost_usd, partial=self.usd_partial, plan_points=points)


@dataclasses.dataclass(frozen=True, slots=True)
class ListingRow:
    """One listing row: a session and the fan-out lanes nested under it, rows themselves."""

    summary: SessionSummary
    lanes: tuple[ListingRow, ...] = ()

    @property
    def mtime(self) -> float:
        """The row's last activity: its own or a lane's, whichever is later."""
        return max((self.summary.mtime, *(lane.mtime for lane in self.lanes)))


def nested_rows(summaries: Iterable[SessionSummary]) -> list[ListingRow]:
    """Nest each lane under the listed session that dispatched it.

    Args:
        summaries: The sessions to list.

    Returns:
        The rows, newest first, every session appearing once: a lane whose coordinator
        is not listed stays a row of its own.
    """
    items = list(summaries)
    by_id = {s.session_id: s for s in items}
    children: dict[str, list[SessionSummary]] = {}
    for s in items:
        if s.coordinator and s.coordinator in by_id and s.coordinator != s.session_id:
            children.setdefault(s.coordinator, []).append(s)
    placed: set[str] = set()

    def row(s: SessionSummary) -> ListingRow:
        placed.add(s.session_id)
        lanes = sorted(children.get(s.session_id, ()), key=_lane_order)
        return ListingRow(s, tuple(row(ln) for ln in lanes if ln.session_id not in placed))

    roots = [s for s in items if not (s.coordinator in by_id and s.coordinator != s.session_id)]
    rows = [row(s) for s in roots]
    # A coordinator chain that loops reaches no root: its sessions are rows.
    rows.extend(row(s) for s in items if s.session_id not in placed)
    rows.sort(key=lambda r: r.mtime, reverse=True)
    return rows


def _lane_order(lane: SessionSummary) -> tuple[int, str]:
    """Return the sort key placing lanes in lane order, then by id."""
    return (lane.lane or 0, lane.session_id)


def lanes_of(
    state_dir: pathlib.Path, coordinator: str, *, branch_tips: Mapping[str, str] | None = None
) -> list[SessionSummary]:
    """Return the lanes whose manifests name a coordinator, in lane order.

    Args:
        state_dir: The repo's state dir.
        coordinator: The dispatching session's id.
        branch_tips: The caller's branch-tips snapshot; with it each lane carries its
            unmerged mark.

    Returns:
        The lanes' summaries.
    """
    lanes = [
        s
        for d in session_dirs(state_dir)
        if (s := summarize_session_dir(d, branch_tips=branch_tips)).coordinator == coordinator
    ]
    return sorted(lanes, key=_lane_order)


def row_json(row: ListingRow, *, winners: Container[str]) -> dict[str, object]:
    """Return a listing row and its nested lanes as JSON.

    Args:
        row: The row.
        winners: The ids of the fan-out compare winners.

    Returns:
        The `summary_row` dict with `mtime` and `when` set to the group's latest activity.
    """
    out = summary_row(
        row.summary,
        winner=row.summary.session_id in winners,
        lanes=[row_json(ln, winners=winners) for ln in row.lanes],
    )
    out["mtime"] = row.mtime
    out["when"] = format.format_when(row.mtime) if row.mtime else ""
    return out


def summary_row(
    s: SessionSummary,
    *,
    winner: bool = False,
    lanes: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Return one listing row as JSON, the shape `sessions list --json` and `/api/hub` share.

    Args:
        s: The row's summary.
        winner: The row is its fan-out's compare winner.
        lanes: The nested lane rows, as `row_json` builds them.

    Returns:
        The row, with the status cell rendered as `label` and `level` so a client
        needs no copy of the status maps, and the one-line `task_line` beside the task.
    """
    return {
        "session_id": s.session_id,
        "mode": s.mode,
        "task": s.task,
        "task_line": task_snippet(s.task),
        "status": s.status,
        "reason": s.reason,
        "label": format.listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged),
        "level": format.status_level(s.status),
        "mtime": s.mtime,
        "when": format.format_when(s.mtime) if s.mtime else "",
        "cost_usd": s.cost_usd,
        "usd_partial": s.usd_partial,
        "plan_consumed": s.plan_consumed,
        "plan_cap": s.plan_cap,
        "cost": s.cost_cell,
        "model": s.model,
        "model_from_flag": s.model_from_flag,
        "id_cell": format.winner_id(s.session_id, winner=winner),
        "unmerged": s.unmerged,
        "verify_ok": s.verify_ok,
        "winner": winner,
        "lane": s.lane,
        "coordinator": s.coordinator,
        "lanes": list(lanes),
    }


def status_word(
    *,
    finished: bool,
    all_passed: bool | None,
    end_reason: str,
    scoped: bool = False,
    gate_red: bool = False,
) -> tuple[str, str]:
    """Map an end state to the status word and its detail.

    The one place that decides how a run's outcome reads; headers and listings share
    it. "stopped" and "undone" are the operator's own acts, "planned" and "answered"
    the clean exits that verified nothing, "passed" every gate green, "finished" a
    deliberate finish without all-passed, and anything else "failed" with the reason.

    Args:
        finished: A session.end was seen.
        all_passed: The verify tri-state: True when the final tree was observed
            verify-green, False when not (red, stale or an error end), None when no
            verify command gated it; None reads "finished" whatever the reason.
        end_reason: The session.end reason word.
        scoped: The certifying gate ran scoped to the tests nearest the run's diff,
            so a pass reads "passed · scoped gate".
        gate_red: This execution's last verify ran and failed; False also covers a
            stale green and a gate nothing ran.

    Returns:
        The `(word, detail)` pair.
    """
    if not finished:
        return "running", ""
    if end_reason in ("steer_abort", "steer_exit", "interrupted", "interactive_stop"):
        return "stopped", ""
    # A clean exit that verified nothing gets its own word, never "passed" and never "failed".
    not_green = {
        "finish_planning": ("planned", ""),
        "answered": ("answered", ""),
        "undone": ("undone", ""),
        "settled": ("finished", "gate red" if gate_red else "unverified"),
        # A verify against the unmodified tree proved the gate red before the run touched it.
        "gate_red_at_base": ("finished", "gate was already red"),
    }
    if end_reason in not_green:
        return not_green[end_reason]
    if all_passed:
        return "passed", "scoped gate" if scoped else ""
    # Only an observed not-green words "failed"; a deliberate finish over one is "finished".
    if all_passed is False and end_reason:
        detail = "gate red" if gate_red else "unverified"
        clean_end = end_reason in ("finish_session", "silent_finish", "metric_plateau")
        return ("finished", detail) if clean_end else ("failed", end_reason)
    return "finished", ""


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
        end_reason: The session.end reason word.
        operator_blocked: An approval or question is still unanswered.
        blocked_kind: The oldest unanswered prompt's kind, "approval" or "question".
        blocked_since_ep: That prompt's asked-at epoch.
        unattended_questions: Questions the harness answered empty because nobody was attached.
    """

    started: bool = False
    finished: bool = False
    all_passed: bool | None = False
    verify_scoped: bool = False
    gate_red: bool = False
    end_reason: str = ""
    operator_blocked: bool = False
    blocked_kind: str = ""
    blocked_since_ep: float | None = None
    unattended_questions: int = 0


def status_for_session_dir(session_dir: pathlib.Path, facts: StatusFacts) -> tuple[str, str]:
    """Decide the status word and detail for a session that has a dir on disk.

    The dir supplies what events cannot: a parked submission (the manifest) and
    worker liveness (the pid file). A started session is live iff its worker is: the
    pid is written before session.start, so no pid file means the worker cleared it
    on the way out. Log silence cannot stand in for this: a `kill -9` leaves the pid
    file while an abnormal exit through the finally clears it.

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
        )
        if not reason and (n := facts.unattended_questions):
            reason = f"{n} question{'s' if n != 1 else ''} unanswered"
        return word, reason
    if facts.operator_blocked and ipc.worker_is_alive(session_dir):
        # Before session.start too: a run asks about uncommitted changes before it starts.
        if facts.blocked_kind and facts.blocked_since_ep is not None:
            age = format.format_age(time.time() - facts.blocked_since_ep)
            return "waiting", f"{facts.blocked_kind} {age}"
        return "waiting", "needs answer"
    if not facts.started:
        return _unstarted_status(session_dir)
    if not ipc.worker_is_alive(session_dir):
        return "stale", ""
    return "running", ""


def _unstarted_status(session_dir: pathlib.Path) -> tuple[str, str]:
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
    if ipc.worker_is_alive(session_dir):
        return "starting", ""
    if (session_dir / sessions_manifest.MANIFEST_NAME).is_file():
        try:
            manifest = sessions_manifest.read_manifest(session_dir)
        except sessions_manifest.ManifestError as exc:
            return "unreadable", format.clip_cell(str(exc), 60)
        if manifest.parked_task:
            return PARKED_WORD, manifest.parked_reason
    if ipc.read_worker_pid(session_dir) is not None:
        return "stale", "died launching"
    return "created", ""


# Terminal without a session.end: the fan-out's await accepts them so it cannot hang.
_DIED_WITHOUT_END = frozenset({"stale", "created", "parked", "unreadable", "?"})


def died_without_end(status: str) -> bool:
    """Return whether the status word means the session never reached its own end."""
    return status in _DIED_WITHOUT_END


# A positive set: a new status word is not a result a fan-out may rank until it is listed.
_RESULT_WORDS = frozenset({"passed", "finished", "stopped", "planned", "answered"})


def produced_result(status: str) -> bool:
    """Return whether the status word means a deliberate end that left mergeable work.

    Only such a lane is a fan-out compare candidate or joins a coordinator's branch.
    """
    return status in _RESULT_WORDS


# A run in any other state reads no steer or answer marker: a surface offers resume instead.
LIVE_STATUS_WORDS = frozenset({"running", "starting", "waiting"})


def session_is_live(session_dir: pathlib.Path) -> bool:
    """Return whether anything will read what the operator writes to this session.

    Derived from the status word, so a surface cannot disagree with the label it shows.
    """
    logs = session_dir / layout.LOGS_NAME
    facts = scan_session_log(logs).status_facts() if logs.is_file() else StatusFacts()
    return status_for_session_dir(session_dir, facts)[0] in LIVE_STATUS_WORDS


@dataclasses.dataclass(frozen=True, slots=True)
class LogScan:
    """One tolerant pass over a session's journal, behind the hub listing and `sessions show`.

    One owner, so the resume rules (bank the cost of past executions, un-finish) and
    the torn-line tolerances cannot drift between consumers. Token counters are the
    current execution's; `cost_usd` is cumulative, matching the typed fold's BudgetView.

    Attributes:
        saw_start: A session.start or loop.resume.start was seen; neither means unstarted.
        mode: The session's mode, "?" until a start event names it.
        task: The task text as session.start carried it.
        finished: A session.end was seen and no later execution began.
        all_passed: The verify tri-state; None when no verify command gated the end.
        verify_scoped: The judging gate ran scoped.
        end_reason: The session.end reason word.
        cost_usd: The spend across every execution; None when no budget.update was seen.
        usd_partial: Some execution had unpriced spend, so `cost_usd` is a lower bound.
        executions: One plus the completed resume executions.
        input_tokens: This execution's input tokens, None until a budget.update carries them.
        output_tokens: This execution's output tokens.
        cache_read_tokens: The cached side of the input, None when the event lacked it.
        cache_creation_tokens: Cache-creation tokens, None when the event lacked it.
        plan_consumed: The plan points this execution consumed; 0.0 until a plan call runs.
        plan_cap: `[budget].max_percent`, 0.0 until a plan call runs.
        plan_used_percent: The account's reading, 0.0 until a plan call runs.
        iteration: The last event's iteration.
        start_ep: session.start's epoch, else the first event's (a fork's log has none).
        last_ep: The last event's epoch.
        last_type: The last event's type.
        operator_blocked: A prompt is still unanswered on this execution.
        blocked_kind: The oldest unanswered prompt's kind, "approval" or "question".
        blocked_since_ep: That prompt's asked-at epoch.
        last_verify_rc: This execution's last verify.end exit code.
        pins: The operator's pinned instructions in force.
        unattended_questions: Questions answered empty because nobody was attached.
    """

    saw_start: bool = False
    mode: str = "?"
    task: str = ""
    finished: bool = False
    all_passed: bool | None = False
    verify_scoped: bool = False
    end_reason: str = ""
    cost_usd: float | None = None
    usd_partial: bool = False
    executions: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_used_percent: float = 0.0
    iteration: int | None = None
    start_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    operator_blocked: bool = False
    blocked_kind: str = ""
    blocked_since_ep: float | None = None
    last_verify_rc: int | None = None
    pins: tuple[str, ...] = ()
    unattended_questions: int = 0

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

    def status_facts(self) -> StatusFacts:
        """Return this scan's answers to the status questions.

        The typed fold's `state.status_facts` agrees on the same log.
        """
        return StatusFacts(
            started=self.saw_start,
            finished=self.finished,
            all_passed=self.all_passed,
            verify_scoped=self.verify_scoped,
            gate_red=self.last_verify_rc is not None and self.last_verify_rc != 0,
            end_reason=self.end_reason,
            operator_blocked=self.operator_blocked,
            blocked_kind=self.blocked_kind,
            blocked_since_ep=self.blocked_since_ep,
            unattended_questions=self.unattended_questions,
        )


def _figure(ev: Mapping[str, object], key: str, last_good: float) -> float:
    """Read a budget.update figure as the typed fold does.

    Args:
        ev: The event.
        key: The figure's key.
        last_good: The figure to keep when the value is unusable.

    Returns:
        0.0 for an absent key (an aggregate event carries only what it summed), else
        the tolerant float.
    """
    return _tolerant_float(ev[key], last_good) if key in ev else 0.0


def _tolerant_float(raw: object, last_good: float) -> float:
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


def needs_new_work(*, finished: bool, end_reason: str, all_passed: bool | None) -> bool:
    """Return whether a bare resume of a run in this state would have nothing to do.

    The one predicate behind the resume refusal, the web composer's hint and the
    wire's `needs_new_work`. A bare resume of such a run spends a call, answers in
    prose, records a silent_finish and leaves a passed run reading as failed.

    Args:
        finished: A session.end was seen.
        end_reason: The session.end reason word.
        all_passed: The verify tri-state.

    Returns:
        True only for a `finish_session` end over a tree the gate certified green or
        that no gate judged; every other end is what resume is for.
    """
    return finished and end_reason == "finish_session" and all_passed is not False


def finished_needs_new_work(session_dir: pathlib.Path) -> bool:
    """Return `needs_new_work` over the run's log, for a verb that holds only the dir."""
    scan = scan_session_log(session_dir / layout.LOGS_NAME)
    return needs_new_work(
        finished=scan.finished, end_reason=scan.end_reason, all_passed=scan.all_passed
    )


def needs_new_work_refusal(session_id: str) -> str:
    """Return the refusal `agent6 resume` and the web composer share for a finished run."""
    return (
        f"run {session_id!r} already finished (the agent called finish_session)."
        " Give it new work with:\n"
        f'    agent6 resume {session_id} --steer "<what to do next>"'
    )


def scan_session_log(logs: pathlib.Path) -> LogScan:  # noqa: C901, PLR0912, PLR0915  # one branch per event type
    """Fold a session's journal into a `LogScan`.

    A live writer can leave a torn multibyte tail, so the file is decoded with
    replacement and the mangled line fails `json.loads` and is skipped.

    Args:
        logs: The journal path.

    Returns:
        The scan; an empty one when the file cannot be read.
    """
    mode, task = "?", ""
    finished, end_reason = False, ""
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
    plan_consumed = plan_cap = plan_used_percent = 0.0
    iteration: int | None = None
    start_ep: float | None = None
    first_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    # Only an answer or an execution boundary clears a prompt: a steer request leaves it waiting.
    pending_prompts: dict[str, tuple[str, float | None]] = {}
    unattended_questions = 0
    last_verify_rc: int | None = None
    pins: list[str] = []
    try:
        with logs.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                etype = ev.get("type")
                ep = events.event_epoch(ev.get("ts"))
                if ep is not None:
                    last_ep = ep
                    if first_ep is None:
                        first_ep = ep
                if isinstance(etype, str):
                    last_type = etype
                if isinstance(ev.get("iteration"), int):
                    iteration = ev["iteration"]
                if etype in OPERATOR_PROMPT_EVENTS:
                    # Keyed by str(id) like the answer side, so an int id still clears.
                    if (pid := ev.get("id")) is not None:
                        kind = "approval" if etype == "approval.prompt" else "question"
                        pending_prompts[str(pid)] = (kind, ep)
                elif etype in OPERATOR_ANSWER_EVENTS:
                    pending_prompts.pop(str(ev.get("id")), None)
                    if etype == "question.answer" and ev.get("unseen") is True:
                        unattended_questions += 1
                if etype == "session.start":
                    saw_start = True
                    finished = False
                    mode = str(ev.get("mode", mode))
                    task = str(ev.get("user_task", ""))
                    # A new execution re-asks with restarted ids; a held-over prompt waits forever.
                    pending_prompts.clear()
                    last_verify_rc = None
                    if start_ep is None:
                        start_ep = ep
                elif etype == "session.end":
                    finished = True
                    if isinstance(ev.get("iterations"), int):
                        iteration = ev["iterations"]
                    # An explicit null is the ungated tri-state; an absent key folds False.
                    raw_ap = ev.get("all_passed", False)
                    all_passed = None if raw_ap is None else bool(raw_ap)
                    verify_scoped = bool(ev.get("scoped", False))
                    end_reason = str(ev.get("reason", ""))
                elif etype == "loop.resume.start":
                    if saw_start:
                        # Each execution's budget starts at 0: bank the finished one's total.
                        usd_prior_executions += usd_execution
                        usd_execution = 0.0
                        input_tokens = output_tokens = None
                        cache_read_tokens = cache_creation_tokens = None
                        plan_consumed = plan_cap = plan_used_percent = 0.0
                        last_verify_rc = None
                        executions += 1
                    saw_start = True
                    mode = str(ev.get("mode", mode))
                    finished = False
                    pending_prompts.clear()
                elif etype == "verify.end":
                    rc = ev.get("exit_code")
                    if isinstance(rc, int) and not isinstance(rc, bool):
                        last_verify_rc = rc
                elif etype == "loop.pin.added":
                    pins.append(str(ev.get("text", "")))
                elif etype == "loop.pin.restored":
                    raw_pins = ev.get("pins")
                    pins = [str(p) for p in raw_pins] if isinstance(raw_pins, list) else []
                elif etype == "budget.update":
                    saw_budget = True
                    usd_execution = _figure(ev, "usd_total", usd_execution)
                    usd_partial = bool(ev.get("usd_partial")) or usd_partial
                    ti, to = ev.get("input_total"), ev.get("output_total")
                    if isinstance(ti, int):
                        # The four counters travel as one group: no stale cached side from earlier.
                        cr, cc = ev.get("cache_read_total"), ev.get("cache_creation_total")
                        input_tokens = ti
                        cache_read_tokens = cr if isinstance(cr, int) else None
                        cache_creation_tokens = cc if isinstance(cc, int) else None
                    output_tokens = to if isinstance(to, int) else output_tokens
                    plan_consumed = _figure(ev, "plan_consumed", plan_consumed)
                    plan_cap = _figure(ev, "plan_cap", plan_cap)
                    plan_used_percent = _figure(ev, "plan_used_percent", plan_used_percent)
    except OSError:
        pass
    return LogScan(
        saw_start=saw_start,
        mode=mode,
        task=task,
        finished=finished,
        all_passed=all_passed,
        verify_scoped=verify_scoped,
        end_reason=end_reason,
        cost_usd=(usd_prior_executions + usd_execution) if saw_budget else None,
        usd_partial=usd_partial,
        executions=executions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        plan_consumed=plan_consumed,
        plan_cap=plan_cap,
        plan_used_percent=plan_used_percent,
        unattended_questions=unattended_questions,
        iteration=iteration,
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
        pins=tuple(pins),
    )


def summarize_session_dir(
    session_dir: pathlib.Path, *, branch_tips: Mapping[str, str] | None = None
) -> SessionSummary:
    """Fold a session dir's journal and manifest into one listing row.

    The manifest owns the task (the event clips it to 200 chars); an ask's task is
    the first line under its transcript's first heading, which shows what was asked.

    Args:
        session_dir: The session's state dir.
        branch_tips: The caller's `git_ops.run_ref_tips` snapshot; with it the row says
            whether the chain or branch still holds commits no merge stamped, without
            it `unmerged` stays False (no mark, never a wrong one).

    Returns:
        The row.
    """
    logs = session_dir / layout.LOGS_NAME
    scan = scan_session_log(logs) if logs.is_file() else LogScan()
    manifest: sessions_manifest.SessionManifest | None = None
    with contextlib.suppress(sessions_manifest.ManifestError):
        manifest = sessions_manifest.read_manifest(session_dir)
    mode, task = scan.mode, scan.task
    if manifest is not None:
        # A log with no session.start yet (preflight, `fork --no-run`, a fork) names no mode.
        task = manifest.user_task or task
        if mode == "?":
            mode = manifest.mode or mode
    if mode == "?" and not task:
        task = "(no logs)"
    word, reason = status_for_session_dir(session_dir, scan.status_facts())
    if mode == "ask":
        with contextlib.suppress(OSError):
            transcript = (session_dir / "transcript.md").read_text(
                encoding="utf-8", errors="replace"
            )
            # The first non-heading line would be the answer whenever the question begins with `#`.
            lines = transcript.splitlines()
            heading = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), None)
            body = lines[heading + 1 :] if heading is not None else []
            asked = next((ln.strip() for ln in body if ln.strip()), "")
            task = asked[:200] or transcript.strip()[:200]
    lineage = manifest.parallel if manifest is not None else None
    unmerged = False
    if branch_tips is not None and manifest is not None and word != "undone":
        tips = {
            tip
            for tip in (
                branch_tips.get(git_ops.chain_ref_for(manifest.session_id)),
                branch_tips.get(manifest.run_branch or ""),
            )
            if tip
        }
        # The branch may carry the operator's own commits; a merge stamps whichever tip it merged.
        unmerged = bool(tips - {manifest.base_sha}) and (
            manifest.merged is None or manifest.merged.tip not in tips
        )
    return SessionSummary(
        session_id=session_dir.name,
        mode=mode,
        task=task,
        status=word,
        reason=reason,
        unmerged=unmerged,
        cost_usd=scan.cost_usd or 0.0,
        usd_partial=scan.usd_partial,
        mtime=session_mtime(session_dir),
        verify_ok=scan.verify_verdict(),
        coordinator=lineage.coordinator if lineage is not None else "",
        lane=lineage.lane if lineage is not None else None,
        plan_consumed=scan.plan_consumed,
        plan_cap=scan.plan_cap,
        plan_used_percent=scan.plan_used_percent,
        model=format.format_model_route(manifest.models.driver) if manifest is not None else "",
        model_from_flag=manifest.models.driver_from_flag if manifest is not None else False,
    )
