"""Classify a session for the review queue: derived from its journal, never persisted."""

from __future__ import annotations

import dataclasses
from typing import Literal

from agent6.viewmodel import _session_log, review

Group = Literal["needs_answer", "blocked", "unverified", "awaiting_merge", "other"]
# The queue's order: questions nobody answered, then open blocks, then ended runs the gate did
# not certify, then clean runs waiting for a merge, then the rest with its reason.
GROUPS: tuple[Group, ...] = ("needs_answer", "blocked", "unverified", "awaiting_merge", "other")
_OTHER_REASONS = {
    "unreadable": "review evidence unreadable",
    "legacy": "legacy review counts",
    "stale": "review stale",
    "inconclusive": "review inconclusive",
    "skipped": "review skipped",
    "abstained": "a seat abstained",
    "running": "review running",
    "incomplete": "review incomplete",
    "freeform": "freeform review, no verdict",
}


@dataclasses.dataclass(frozen=True, slots=True)
class QueueFacts:
    """What the queue needs beyond the journal, read where the listing row is built.

    Attributes:
        status: The lifecycle status word.
        mode: The session's mode.
        unmerged: Whether the run holds work outside its merge stamp.
        merged: Whether a merge stamp records a landing.
        alive: Whether a worker process holds the session.
        fanout: Whether the session is a fan-out coordinator, a container for its lanes.
        saved_answers: The answers written to its questions that wait on disk for the run.
        tip_matches: Whether every ref holding the run's work, its chain and its branch,
            sits at the run's last recorded checkpoint, so the journal's end describes the
            source as it is now; None without a tips snapshot, a checkpoint or a ref.
    """

    status: str
    mode: str
    unmerged: bool = False
    merged: bool = False
    alive: bool = False
    fanout: bool = False
    tip_matches: bool | None = None
    saved_answers: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewQueueEntry:
    """Where a session sits in the review queue.

    Lifecycle status stays apart: a running session with an open block reads running and
    sits in the blocked group.

    Attributes:
        group: The queue group.
        reason: The words beside the group: the open block count, the gate word, the review
            state, or what else keeps the session off the clean list.
        open_blocks: Open grounded block findings.
        open_warnings: Open grounded warn findings.
        open_nits: Open grounded nit findings.
        review_state: The latest panel's state against the session's end tree: unreadable,
            legacy, unreviewed, running, incomplete, skipped, inconclusive, freeform,
            abstained, stale or complete.
        verification_state: The gate word for the final tree (passed, failed, timed_out,
            unverified, not_applicable), "passed scoped" for a pass over selected paths only,
            "passed, no evidence" for an end whose pass left no verification record.
        oldest_open_at: When the pending work began, as epoch seconds: the oldest unanswered
            question's ask, the oldest open block's first observation, else the run's end;
            None when nothing is pending.
        source_tip: The run's last diff checkpoint commit.
        merge_allowed: Whether the open-block merge guard lets this session merge.
    """

    group: Group = "other"
    reason: str = ""
    open_blocks: int = 0
    open_warnings: int = 0
    open_nits: int = 0
    review_state: str = "unreviewed"
    verification_state: str = "not_applicable"
    oldest_open_at: float | None = None
    source_tip: str | None = None
    merge_allowed: bool = True

    @property
    def rank(self) -> int:
        """The group's place in the queue, 0 first."""
        return GROUPS.index(self.group)


def review_state(view: review.ReviewView, *, live: bool) -> str:
    """Return the one word the queue shows for a session's review evidence.

    Args:
        view: The session's review fold.
        live: Whether a worker holds the session; a panel started without its completion
            reads running under one and incomplete without.

    Returns:
        The word.
    """
    if view.read_failure:
        return "unreadable"
    if view.legacy_notice:
        return "legacy"
    if not view.panels:
        return "unreviewed"
    return _panel_state(view.panels[-1], stale=view.freshness == "stale", live=live)


def _panel_state(panel: review.PanelView, *, stale: bool, live: bool) -> str:
    """Return the queue's word for the session's latest panel."""
    if panel.completion == "running":
        return "running" if live else "incomplete"
    if panel.completion != "complete":
        return panel.completion
    if panel.result is None:
        return "freeform"
    if panel.result.n_abstain:
        return "abstained"
    return "stale" if stale else "complete"


def _verification(scan: _session_log.LogScan, mode: str) -> tuple[str, bool]:
    """Return the gate word the queue shows and whether the final tree passed whole."""
    if mode != "run":
        return "not_applicable", False
    word = scan.gate_word() or "unverified"
    if scan.verified != "passed":
        return word, False
    final = next(
        (v for v in scan.verifications if v.verify_id and v.verify_id == scan.final_verify_id),
        None,
    )
    if final is None or not final.certifies(scan.end_tree or None):
        return "passed, no evidence", False
    if final.scoped or scan.verify_scoped:
        return "passed scoped", False
    return "passed", True


def classify(scan: _session_log.LogScan, facts: QueueFacts) -> ReviewQueueEntry:
    """Place a session in the review queue from its journal and the listing's facts.

    Args:
        scan: The session's journal fold.
        facts: The lifecycle, merge and liveness facts the listing row holds.

    Returns:
        The entry, with the reason the operator reads beside it.
    """
    saved = min(facts.saved_answers, scan.pending_questions)
    entry = _classify(scan, facts, open_questions=scan.pending_questions - saved)
    if not saved:
        return entry
    note = f"{saved} answer{'s' if saved != 1 else ''} saved"
    if entry.group != "needs_answer":
        note += f"; the resume delivers {'them' if saved != 1 else 'it'}"
    return dataclasses.replace(entry, reason=f"{entry.reason} · {note}" if entry.reason else note)


def _classify(
    scan: _session_log.LogScan, facts: QueueFacts, *, open_questions: int
) -> ReviewQueueEntry:
    """Place the session by its findings, gate and merge state, after the open questions.

    Args:
        scan: The session's journal fold.
        facts: The listing row's facts.
        open_questions: The questions no answer has reached, on disk or in the journal.

    Returns:
        The entry.
    """
    view = scan.review
    if scan.end_tree and view.panels and view.panels[-1].target is not None:
        target = view.panels[-1].target
        view = view.for_comparison(dataclasses.replace(target, head_tree=scan.end_tree, paths=()))
    counts = view.open_counts
    state = review_state(view, live=facts.alive)
    gate, full_pass = _verification(scan, facts.mode)
    ended_run = facts.mode == "run" and scan.finished and not facts.alive
    guard = view.merge_guard
    entry = ReviewQueueEntry(
        open_blocks=counts.block,
        open_warnings=counts.warn,
        open_nits=counts.nit,
        review_state=state,
        verification_state=gate,
        source_tip=scan.source_tip,
        merge_allowed=not guard.open_block_ids and not guard.read_failure,
    )
    if open_questions:
        plural = "s" if open_questions != 1 else ""
        reason = f"{open_questions} question{plural} pending"
        if counts.block:
            reason += f" · {counts.block} open block finding{'s' if counts.block != 1 else ''}"
        return dataclasses.replace(
            entry,
            group="needs_answer",
            reason=reason,
            oldest_open_at=scan.oldest_pending_question_ep,
        )
    if counts.block:
        opened = [
            opened_at
            for finding in view.findings
            if finding.status == "open"
            and finding.observations[-1].finding.severity == "block"
            and (opened_at := finding.opened_at) is not None
        ]
        plural = "s" if counts.block != 1 else ""
        return dataclasses.replace(
            entry,
            group="blocked",
            reason=f"{counts.block} open block finding{plural}",
            oldest_open_at=min(opened, default=scan.end_ep),
        )
    if ended_run and not facts.fanout and not full_pass:
        return dataclasses.replace(
            entry, group="unverified", reason=f"gate {gate}", oldest_open_at=scan.end_ep
        )
    clean = (
        ended_run
        and full_pass
        and counts == review.SeverityCounts()
        and state == "complete"
        and facts.unmerged
        and facts.tip_matches is True
        and not scan.stranded
        and facts.status != "undone"
        and not scan.undone_to
    )
    if clean:
        return dataclasses.replace(
            entry,
            group="awaiting_merge",
            reason="verified and reviewed, unmerged",
            oldest_open_at=scan.end_ep,
        )
    return dataclasses.replace(entry, reason=_other_reason(scan, facts, state, counts))


def _other_reason(
    scan: _session_log.LogScan, facts: QueueFacts, state: str, counts: review.SeverityCounts
) -> str:
    """Return why a session is review work without a place in the first three groups."""
    open_words = ", ".join(
        f"{n} {word}" for n, word in ((counts.warn, "warn"), (counts.nit, "nit")) if n
    )
    reasons = (
        (state in _OTHER_REASONS, _OTHER_REASONS.get(state, "")),
        (bool(open_words), f"{open_words} open"),
        (facts.mode not in ("run", "?"), facts.mode),
        (facts.alive or not scan.finished, facts.status),
        (facts.fanout, "fan-out"),
        (bool(scan.stranded or facts.status == "undone" or scan.undone_to), facts.status),
        (facts.merged and not facts.unmerged, "merged"),
        (state == "unreviewed", "unreviewed"),
        (facts.tip_matches is False, "commits after the run"),
    )
    fallback = "no recorded checkpoint" if facts.unmerged else "nothing to merge"
    return next((reason for holds, reason in reasons if holds), fallback)
