"""The review queue: a session's group and reason from its journal."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
from typing import Any

import pytest

from agent6 import review_records
from agent6.viewmodel import listing, review_queue, session_log, session_status

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
        trigger="finish",
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
            "type": "review.start",
            "ts": "2026-10-25T10:00:00+00:00",
            "panel_id": "p",
            "execution_id": "e",
            "target": dataclasses.asdict(target),
            "seats": [],
        },
        {"type": "review.panel", "ts": _TS, **record.model_dump(mode="json")},
    ]


def _verify_rows(
    *, exit_code: int = 0, scoped: bool = False, tree: str = "ht"
) -> list[dict[str, Any]]:
    """Return a verification attempt that observed `tree` on both sides."""
    return [
        {"type": "verify.start", "verify_id": "v", "command": ["pytest"], "tree_before": tree},
        {
            "type": "verify.end",
            "verify_id": "v",
            "command": ["pytest"],
            "exit_code": exit_code,
            "tree_before": tree,
            "tree_after": tree,
            "scoped": scoped,
            "timeout_s": 5.0 if exit_code == 124 else 0.0,
            "timed_out": exit_code == 124,
        },
    ]


_START: dict[str, Any] = {
    "type": "session.start",
    "ts": "2026-10-25T09:00:00+00:00",
    "mode": "run",
    "user_task": "t",
}
_END: dict[str, Any] = {
    "type": "session.stop",
    "ts": "2026-10-25T11:00:00+00:00",
    "reason": "finish_session",
    "verification": "passed",
    "tree": "ht",
    "final_verify_id": "v",
}
_END_EPOCH = datetime.datetime.fromisoformat("2026-10-25T11:00:00+00:00").timestamp()


def _scan(rows: list[dict[str, Any]], tmp_path: pathlib.Path) -> session_log.LogScan:
    log = tmp_path / "logs.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return session_log.scan_session_log(log)


def _facts(
    status: str = "passed",
    mode: str = "run",
    *,
    unmerged: bool = True,
    merged: bool = False,
    alive: bool = False,
    fanout: bool = False,
    tip_matches: bool | None = True,
    saved_answers: int = 0,
) -> review_queue.QueueFacts:
    return review_queue.QueueFacts(
        status=status,
        mode=mode,
        saved_answers=saved_answers,
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
        "gate never ran",
        [_START, {**_END, "verification": "unverified", "final_verify_id": None}],
        _facts("finished"),
        "unverified",
        "gate never ran",
        "unreviewed",
        "unverified",
    ),
    (
        "gate not configured",
        [_START, {**_END, "verification": "not_applicable"}],
        _facts("finished"),
        "unverified",
        "gate not configured",
        "unreviewed",
        "not_applicable",
    ),
    (
        "gate red at base",
        [
            _START,
            *_verify_rows(exit_code=1),
            {**_END, "verification": "failed", "reason": "gate_red_at_base"},
        ],
        _facts("finished"),
        "unverified",
        "gate red at base",
        "unreviewed",
        "failed",
    ),
    (
        "failed gate",
        [_START, *_verify_rows(exit_code=1), {**_END, "verification": "failed"}],
        _facts("finished"),
        "unverified",
        "gate failed",
        "unreviewed",
        "failed",
    ),
    (
        "timed out gate",
        [_START, *_verify_rows(exit_code=124), {**_END, "verification": "failed"}],
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
            {**_END, "verification": "unverified", "reason": "settled"},
        ],
        _facts("finished"),
        "unverified",
        "",
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
        [_START, *_verify_rows(), {"type": "review.panel", "blocked": False}, _END],
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
                "verification": "unverified",
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
        [{**_START, "mode": "ask"}, {"type": "session.stop", "reason": "answered"}],
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
        {"type": "session.resume", "ts": "2026-10-25T12:00:00+00:00", "steer": "more"},
    ]
    scan = _scan(resumed, tmp_path)
    assert scan.end_ep is None
    assert review_queue.classify(scan, _facts("running", alive=True)).group == "other"


def test_lifecycle_status_stays_apart_from_the_queue(tmp_path: pathlib.Path) -> None:
    """A failed run reads failed and sits in the unverified group: two words, neither replaced."""
    rows = [
        _START,
        *_verify_rows(exit_code=1),
        {**_END, "reason": "no_progress", "verification": "failed"},
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
        task_label=session_status.TaskLabel(session_id),
        status="finished",
        detail="",
        cost_usd=0.0,
        usd_partial=False,
        last_activity=mtime,
        coordinator=coordinator,
        lane=lane,
        review_queue=review_queue.ReviewQueueEntry(group=group, oldest_open_at=pending),
    )


def test_awaiting_merge_runs_that_changed_the_same_paths_flag_each_other() -> None:
    """Two clean runs over one path name each other; a run touching other paths stays clean."""
    paths = {
        "one": frozenset({"src/a.py", "src/b.py"}),
        "two": frozenset({"src/b.py", "docs/x.md"}),
        "three": frozenset({"README.md"}),
        "lane-1": frozenset({"src/a.py"}),
    }
    summaries = [
        _summary("one", "awaiting_merge", mtime=90.0, pending=900.0),
        _summary("two", "awaiting_merge", mtime=80.0, pending=800.0),
        _summary("three", "awaiting_merge", mtime=70.0, pending=700.0),
        _summary("fan", "other", mtime=10.0),
        _summary("lane-1", "awaiting_merge", mtime=11.0, pending=110.0, coordinator="fan", lane=1),
        _summary("busy", "blocked", mtime=5.0, pending=50.0),
    ]
    rows = listing.flag_overlaps(listing.nested_rows(summaries), lambda s: paths[s.session_id])
    by_id = {r.summary.session_id: r for r in rows}
    assert by_id["one"].summary.review_queue.overlaps == (
        review_queue.Overlap("two", ("src/b.py",)),
        review_queue.Overlap("lane-1", ("src/a.py",)),
    )
    assert by_id["two"].summary.review_queue.overlaps == (
        review_queue.Overlap("one", ("src/b.py",)),
    )
    assert by_id["three"].summary.review_queue.overlaps == ()
    (lane,) = by_id["fan"].lanes
    assert lane.summary.review_queue.overlaps == (review_queue.Overlap("one", ("src/a.py",)),)
    assert by_id["busy"].summary.review_queue.overlaps == ()
    wire = json.loads(json.dumps(listing.row_json(by_id["one"], winners=set())))
    assert wire["review_queue"]["overlaps"] == [
        {"session_id": "two", "paths": ["src/b.py"]},
        {"session_id": "lane-1", "paths": ["src/a.py"]},
    ]


def test_a_coordinator_takes_its_lanes_place_and_keeps_its_own_verification() -> None:
    """A fan-out takes its lanes' earliest review state; the lane counts ride the row."""
    summaries = [
        _summary("clean", "awaiting_merge", mtime=90.0, pending=900.0),
        _summary("fan", "other", mtime=10.0),
        _summary("lane-1", "unverified", mtime=11.0, pending=110.0, coordinator="fan", lane=1),
        _summary("lane-2", "blocked", mtime=12.0, pending=120.0, coordinator="fan", lane=2),
        _summary("orphan", "blocked", mtime=5.0, pending=50.0, coordinator="gone", lane=1),
    ]
    rows = listing.nested_rows(summaries)
    assert [r.summary.session_id for r in rows] == ["clean", "fan", "orphan"]  # newest first
    fan = rows[1]
    assert [lane.summary.session_id for lane in fan.lanes] == ["lane-1", "lane-2"]
    assert listing.row_group(fan) == "blocked" and fan.summary.review_queue.group == "other"
    row = listing.row_json(fan, winners=set())
    assert row.get("queue_group") == "blocked" and row.get("lane_groups") == {
        "blocked": 1,
        "unverified": 1,
    }
    entry = row["review_queue"]
    assert isinstance(entry, dict) and entry["verification_state"] == "not_applicable"


def test_a_pending_question_leads_the_queue(tmp_path: pathlib.Path) -> None:
    """A session with a question nobody answered sits first, dated from the oldest open one."""
    import datetime

    def row(kind: str, second: int, **fields: Any) -> dict[str, Any]:
        return {"type": kind, "ts": f"2026-07-14T10:00:0{second}+00:00", **fields}

    rows = [
        row("session.start", 0, mode="run", user_task="t"),
        row("question.prompt", 1, id="question-1", questions=[{"question": "a?"}], wait=False),
        row("question.prompt", 2, id="question-2", questions=[{"question": "b?"}], wait=False),
        row("question.answer", 3, id="question-1", answers=["x"], source="late"),
        row("session.stop", 4, reason="finish_session", verification="passed"),
    ]
    entry = review_queue.classify(_scan(rows, tmp_path), _facts())
    assert review_queue.GROUPS[0] == "needs_answer"
    assert (entry.group, entry.reason) == ("needs_answer", "1 question pending")
    asked = datetime.datetime(2026, 7, 14, 10, 0, 2, tzinfo=datetime.UTC).timestamp()
    assert entry.oldest_open_at == asked


def test_a_saved_answer_leaves_the_needs_answer_group(tmp_path: pathlib.Path) -> None:
    """A question whose answer is saved for the resume needs nothing from the operator."""
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {
            "type": "question.prompt",
            "id": "question-1",
            "questions": [{"question": "a?"}],
            "wait": False,
        },
        {
            "type": "question.prompt",
            "id": "question-2",
            "questions": [{"question": "b?"}],
            "wait": False,
        },
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    scan = _scan(rows, tmp_path)
    one = review_queue.classify(scan, _facts(saved_answers=1))
    assert (one.group, one.reason) == ("needs_answer", "1 question pending · 1 answer saved")
    both = review_queue.classify(scan, _facts(saved_answers=2))
    assert both.group != "needs_answer"
    assert both.reason.endswith("2 answers saved; the resume delivers them")


@pytest.mark.parametrize("alive", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_saved_answer_delivery_note_uses_worker_liveness(
    tmp_path: pathlib.Path, alive: bool, count: int
) -> None:
    """A live run reads saved answers next turn, without requiring a resume."""
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        *[
            {
                "type": "question.prompt",
                "id": f"question-{i}",
                "questions": [{"question": "a?"}],
                "wait": False,
            }
            for i in range(count)
        ],
    ]
    entry = review_queue.classify(
        _scan(rows, tmp_path),
        _facts(status="running" if alive else "stale", alive=alive, saved_answers=count),
    )
    pronoun = "it" if count == 1 else "them"
    expected = (
        f"the run reads {pronoun} at its next turn" if alive else f"the resume delivers {pronoun}"
    )
    assert entry.reason.endswith(f"{count} answer{'s' if count != 1 else ''} saved; {expected}")
