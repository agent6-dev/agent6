"""The review queue: a session's group and reason from its journal."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
from typing import Any

import pytest

from agent6 import review_records
from agent6.viewmodel import _session_log, _session_status, listing, review_queue

_TS = "2026-10-25T10:00:05+00:00"
_TS_EPOCH = datetime.datetime.fromisoformat(_TS).timestamp()


def _panel_rows(
    *, block: int = 0, warn: int = 0, abstain: int = 0, head: str = "ht"
) -> list[dict[str, Any]]:
    """Return a complete panel's start and completion rows with the given open findings."""
    target = review_records.ReviewTarget("b", "bt", "h", head, "h", (), "revisions")
    observations = []
    for index, severity in enumerate(["block"] * block + ["warn"] * warn):
        finding = review_records.Finding(
            category="correctness",
            severity=severity,  # type: ignore[arg-type]
            title=f"f{index}",
            location=review_records.ReviewLocation(
                path="a.py", side="head", start_line=1, end_line=1
            ),
        )
        observations.append(review_records.FindingObservation(f"f{index}", "p", finding, (0,)))
    record = review_records.ReviewPanel(
        panel_id="p",
        execution_id="e",
        turn=1,
        trigger="before_finish",
        target=target,
        findings=tuple(observations),
        injected_finding_ids=(),
        block_on="any",
        raw_blocked=block > 0,
        blocked=block > 0,
        disarmed=False,
        n_block=block,
        n_abstain=abstain,
        completion="complete",
    )
    return [
        {
            "type": "loop.review.start",
            "ts": "2026-10-25T10:00:00+00:00",
            "panel_id": "p",
            "execution_id": "e",
            "target": dataclasses.asdict(target),
            "seats": [],
        },
        {"type": "loop.review.panel", "ts": _TS, **record.model_dump(mode="json")},
    ]


def _verify_rows(
    *, exit_code: int = 0, scoped: bool = False, tree: str = "ht"
) -> list[dict[str, Any]]:
    """Return a verification attempt that observed `tree` on both sides."""
    return [
        {"type": "verify.start", "verify_id": "v", "cmd": ["pytest"], "tree_before": tree},
        {
            "type": "verify.end",
            "verify_id": "v",
            "cmd": ["pytest"],
            "exit_code": exit_code,
            "tree_before": tree,
            "tree_after": tree,
            "scoped": scoped,
            "timeout_s": 5.0 if exit_code == 124 else 0.0,
        },
    ]


_START: dict[str, Any] = {
    "type": "session.start",
    "ts": "2026-10-25T09:00:00+00:00",
    "mode": "run",
    "user_task": "t",
}
_END: dict[str, Any] = {
    "type": "session.end",
    "ts": "2026-10-25T11:00:00+00:00",
    "reason": "finish_session",
    "all_passed": True,
    "verified": "passed",
    "tree": "ht",
    "final_verify_id": "v",
}
_END_EPOCH = datetime.datetime.fromisoformat("2026-10-25T11:00:00+00:00").timestamp()


def _scan(rows: list[dict[str, Any]], tmp_path: pathlib.Path) -> _session_log.LogScan:
    log = tmp_path / "logs.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return _session_log.scan_session_log(log)


def _facts(
    status: str = "passed",
    mode: str = "run",
    *,
    unmerged: bool = True,
    merged: bool = False,
    alive: bool = False,
    fanout: bool = False,
    tip_matches: bool | None = True,
) -> review_queue.QueueFacts:
    return review_queue.QueueFacts(
        status=status,
        mode=mode,
        unmerged=unmerged,
        merged=merged,
        alive=alive,
        fanout=fanout,
        tip_matches=tip_matches,
    )


CASES = [
    (
        "open block",
        [_START, *_verify_rows(), *_panel_rows(block=1), _END],
        _facts(),
        "blocked",
        "1 open block finding",
        "complete",
        "passed",
    ),
    (
        "running with a block",
        [_START, *_panel_rows(block=2)],
        _facts("running", alive=True),
        "blocked",
        "2 open block findings",
        "complete",
        "unverified",
    ),
    (
        "no gate",
        [_START, {**_END, "all_passed": None, "verified": "unverified", "final_verify_id": None}],
        _facts("finished"),
        "unverified",
        "gate unverified",
        "unreviewed",
        "unverified",
    ),
    (
        "failed gate",
        [_START, *_verify_rows(exit_code=1), {**_END, "all_passed": False, "verified": "failed"}],
        _facts("finished"),
        "unverified",
        "gate failed",
        "unreviewed",
        "failed",
    ),
    (
        "timed out gate",
        [_START, *_verify_rows(exit_code=124), {**_END, "all_passed": False, "verified": "failed"}],
        _facts("finished"),
        "unverified",
        "gate timed_out",
        "unreviewed",
        "timed_out",
    ),
    (
        "scoped green",
        [_START, *_verify_rows(scoped=True), *_panel_rows(), {**_END, "scoped": True}],
        _facts(),
        "unverified",
        "gate passed scoped",
        "complete",
        "passed scoped",
    ),
    (
        "pass without evidence",
        [_START, *_panel_rows(), {**_END, "final_verify_id": None}],
        _facts(),
        "unverified",
        "gate passed, no evidence",
        "complete",
        "passed, no evidence",
    ),
    (
        "edited after green",
        [
            _START,
            *_verify_rows(tree="older"),
            *_panel_rows(),
            {**_END, "all_passed": False, "verified": "unverified", "reason": "settled"},
        ],
        _facts("finished"),
        "unverified",
        "gate unverified",
        "complete",
        "unverified",
    ),
    (
        "clean and unmerged",
        [_START, *_verify_rows(), *_panel_rows(), _END],
        _facts(),
        "awaiting_merge",
        "verified and reviewed, unmerged",
        "complete",
        "passed",
    ),
    (
        "clean but commits after the run",
        [_START, *_verify_rows(), *_panel_rows(), _END],
        _facts(tip_matches=False),
        "other",
        "commits after the run",
        "complete",
        "passed",
    ),
    (
        "clean and merged",
        [_START, *_verify_rows(), *_panel_rows(), _END],
        _facts(unmerged=False, merged=True),
        "other",
        "merged",
        "complete",
        "passed",
    ),
    (
        "warnings open",
        [_START, *_verify_rows(), *_panel_rows(warn=2), _END],
        _facts(),
        "other",
        "2 warn open",
        "complete",
        "passed",
    ),
    (
        "stale review",
        [_START, *_verify_rows(), *_panel_rows(head="older"), _END],
        _facts(),
        "other",
        "review stale",
        "stale",
        "passed",
    ),
    (
        "all abstained",
        [_START, *_verify_rows(), *_panel_rows(abstain=3), _END],
        _facts(),
        "other",
        "a seat abstained",
        "abstained",
        "passed",
    ),
    (
        "unreviewed",
        [_START, *_verify_rows(), _END],
        _facts(),
        "other",
        "unreviewed",
        "unreviewed",
        "passed",
    ),
    (
        "legacy counts",
        [_START, *_verify_rows(), {"type": "loop.review.panel", "blocked": False}, _END],
        _facts(),
        "other",
        "legacy review counts",
        "legacy",
        "passed",
    ),
    (
        "fan-out container",
        [
            _START,
            {
                **_END,
                "reason": "lanes_ranked",
                "all_passed": None,
                "verified": "unverified",
                "final_verify_id": None,
            },
        ],
        _facts("finished", fanout=True),
        "other",
        "fan-out",
        "unreviewed",
        "unverified",
    ),
    (
        "review incomplete after the end",
        [_START, *_verify_rows(), _panel_rows()[0], _END],
        _facts("finished"),
        "other",
        "review incomplete",
        "incomplete",
        "passed",
    ),
    (
        "mode from the manifest",
        [*_verify_rows(), *_panel_rows(), _END],
        _facts(),
        "awaiting_merge",
        "verified and reviewed, unmerged",
        "complete",
        "passed",
    ),
    (
        "no recorded checkpoint",
        [_START, *_verify_rows(), *_panel_rows(), _END],
        _facts(tip_matches=None),
        "other",
        "no recorded checkpoint",
        "complete",
        "passed",
    ),
    (
        "no logs",
        [],
        _facts("created", mode="?", unmerged=False, tip_matches=None),
        "other",
        "created",
        "unreviewed",
        "not_applicable",
    ),
    (
        "ask session",
        [{**_START, "mode": "ask"}, {"type": "session.end", "reason": "answered"}],
        _facts("answered", mode="ask", unmerged=False),
        "other",
        "ask",
        "unreviewed",
        "not_applicable",
    ),
]


@pytest.mark.parametrize(
    "case,rows,facts,group,reason,state,gate", CASES, ids=[c[0] for c in CASES]
)
def test_the_classifier_places_each_journal_shape(
    tmp_path: pathlib.Path,
    case: str,
    rows: list[dict[str, Any]],
    facts: review_queue.QueueFacts,
    group: str,
    reason: str,
    state: str,
    gate: str,
) -> None:
    entry = review_queue.classify(_scan(rows, tmp_path), facts)
    assert (entry.group, entry.reason, entry.review_state, entry.verification_state) == (
        group,
        reason,
        state,
        gate,
    ), case


def test_a_panel_without_its_completion_reads_running_only_while_the_worker_lives(
    tmp_path: pathlib.Path,
) -> None:
    """A started panel is running under a live worker; once the run ended it is incomplete."""
    scan = _scan([_START, _panel_rows()[0]], tmp_path)
    assert review_queue.review_state(scan.review, live=True) == "running"
    assert review_queue.review_state(scan.review, live=False) == "incomplete"
    assert review_queue.classify(scan, _facts("running", alive=True)).reason == "review running"


def test_every_entry_carries_a_reason(tmp_path: pathlib.Path) -> None:
    """No queue entry reads blank: each row of the listing says why it sits where it does."""
    for _case, rows, facts, *_rest in CASES:
        assert review_queue.classify(_scan(rows, tmp_path), facts).reason, _case


def test_a_block_dates_the_queue_entry_from_its_first_observation(tmp_path: pathlib.Path) -> None:
    rows = [_START, *_panel_rows(block=1), _END]
    entry = review_queue.classify(_scan(rows, tmp_path), _facts())
    assert entry.oldest_open_at == pytest.approx(_TS_EPOCH)
    assert entry.merge_allowed is False and entry.open_blocks == 1


def test_an_ended_run_dates_its_entry_from_its_end(tmp_path: pathlib.Path) -> None:
    rows = [_START, *_verify_rows(), *_panel_rows(), _END]
    entry = review_queue.classify(_scan(rows, tmp_path), _facts())
    assert entry.group == "awaiting_merge" and entry.oldest_open_at == pytest.approx(_END_EPOCH)
    resumed = [
        *rows,
        {"type": "loop.resume.start", "ts": "2026-10-25T12:00:00+00:00", "steer": "more"},
    ]
    scan = _scan(resumed, tmp_path)
    assert scan.end_ep is None
    assert review_queue.classify(scan, _facts("running", alive=True)).group == "other"


def test_lifecycle_status_stays_apart_from_the_queue(tmp_path: pathlib.Path) -> None:
    """A failed run reads failed and sits in the unverified group: two words, neither replaced."""
    rows = [
        _START,
        *_verify_rows(exit_code=1),
        {**_END, "reason": "no_progress", "all_passed": False, "verified": "failed"},
    ]
    entry = review_queue.classify(_scan(rows, tmp_path), _facts("failed"))
    assert entry.group == "unverified" and entry.reason == "gate failed"


def _summary(
    session_id: str,
    group: review_queue.Group,
    *,
    mtime: float,
    pending: float | None = None,
    coordinator: str = "",
    lane: int | None = None,
) -> listing.SessionSummary:
    return listing.SessionSummary(
        session_id=session_id,
        mode="run",
        task_label=_session_status.TaskLabel(session_id),
        status="finished",
        reason="",
        cost_usd=0.0,
        usd_partial=False,
        mtime=mtime,
        coordinator=coordinator,
        lane=lane,
        review_queue=review_queue.ReviewQueueEntry(group=group, oldest_open_at=pending),
    )


def test_review_order_groups_rows_and_dates_pending_work_oldest_first() -> None:
    """Rows sort by queue group, the oldest pending work first; nothing pending keeps activity."""
    summaries = [
        _summary("merged", "other", mtime=50.0),
        _summary("clean", "awaiting_merge", mtime=40.0, pending=400.0),
        _summary("block-new", "blocked", mtime=60.0, pending=600.0),
        _summary("unverified", "unverified", mtime=30.0, pending=300.0),
        _summary("block-old", "blocked", mtime=20.0, pending=200.0),
        _summary("warn", "other", mtime=10.0),
    ]
    by_activity = [r.summary.session_id for r in listing.nested_rows(summaries)]
    assert by_activity == ["block-new", "merged", "clean", "unverified", "block-old", "warn"]
    by_review = [r.summary.session_id for r in listing.nested_rows(summaries, sort="review")]
    assert by_review == ["block-old", "block-new", "unverified", "clean", "merged", "warn"]


def test_a_coordinator_takes_its_lanes_place_and_keeps_its_own_verification() -> None:
    """A fan-out whose lane holds an open block sorts first; the lane counts ride the row."""
    summaries = [
        _summary("clean", "awaiting_merge", mtime=90.0, pending=900.0),
        _summary("fan", "other", mtime=10.0),
        _summary("lane-1", "unverified", mtime=11.0, pending=110.0, coordinator="fan", lane=1),
        _summary("lane-2", "blocked", mtime=12.0, pending=120.0, coordinator="fan", lane=2),
        _summary("orphan", "blocked", mtime=5.0, pending=50.0, coordinator="gone", lane=1),
    ]
    rows = listing.nested_rows(summaries, sort="review")
    assert [r.summary.session_id for r in rows] == ["orphan", "fan", "clean"]
    fan = rows[1]
    assert [lane.summary.session_id for lane in fan.lanes] == ["lane-1", "lane-2"]
    assert listing.row_group(fan) == "blocked" and fan.summary.review_queue.group == "other"
    assert listing.pending_since(fan) == 120.0
    row = listing.row_json(fan, winners=set())
    assert row["queue_group"] == "blocked" and row["lane_groups"] == {"blocked": 1, "unverified": 1}
    entry = row["review_queue"]
    assert isinstance(entry, dict) and entry["verification_state"] == "not_applicable"


def test_a_nested_lanes_pending_work_dates_its_coordinator() -> None:
    """A lane's own lane places and dates the coordinator, as the group does."""
    summaries = [
        _summary("fan", "other", mtime=10.0),
        _summary("lane-1", "other", mtime=11.0, coordinator="fan", lane=1),
        _summary("sub-1", "blocked", mtime=12.0, pending=5.0, coordinator="lane-1", lane=1),
    ]
    (fan,) = listing.nested_rows(summaries, sort="review")
    assert listing.row_group(fan) == "blocked" and listing.pending_since(fan) == 5.0
    assert listing.review_sort_key(fan) == (0, 0, 5.0, "fan")
