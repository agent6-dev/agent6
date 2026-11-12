"""A listing row's open findings, its same-path runs and the one note every hub shows for them."""

from __future__ import annotations

import json
import pathlib

from agent6 import paths
from agent6.sessions import layout
from agent6.viewmodel import listing, review, session_status
from tests import review_helpers


def _summary(
    session_id: str,
    *,
    mtime: float,
    status: str = "passed",
    unmerged: bool = True,
    coordinator: str = "",
    lane: int | None = None,
    block: int = 0,
) -> listing.SessionSummary:
    return listing.SessionSummary(
        session_id=session_id,
        mode="run",
        task_label=session_status.TaskLabel(session_id),
        status=status,
        detail="",
        cost_usd=0.0,
        usd_partial=False,
        last_activity=mtime,
        unmerged=unmerged,
        coordinator=coordinator,
        lane=lane,
        open_findings=review.SeverityCounts(block=block),
    )


def test_ended_unmerged_runs_that_changed_the_same_paths_flag_each_other() -> None:
    """Ended, unmerged runs over one path name each other, whatever their gate or review said.

    A live run and a merged run name nobody and are named by nobody.
    """
    changed = {
        "one": frozenset({"src/a.py", "src/b.py"}),
        "two": frozenset({"src/b.py", "docs/x.md"}),
        "three": frozenset({"README.md"}),
        "lane-1": frozenset({"src/a.py"}),
        "live": frozenset({"src/a.py"}),
        "landed": frozenset({"src/b.py"}),
    }
    summaries = [
        _summary("one", mtime=90.0),
        _summary("two", mtime=80.0, status="failed", block=1),
        _summary("three", mtime=70.0),
        _summary("fan", mtime=10.0, status="finished", unmerged=False),
        _summary("lane-1", mtime=11.0, coordinator="fan", lane=1),
        _summary("live", mtime=5.0, status="running"),
        _summary("landed", mtime=4.0, unmerged=False),
    ]
    rows = listing.flag_overlaps(listing.nested_rows(summaries), lambda s: changed[s.session_id])
    by_id = {r.summary.session_id: r for r in rows}
    assert by_id["one"].summary.overlaps == (
        listing.Overlap("two", ("src/b.py",)),
        listing.Overlap("lane-1", ("src/a.py",)),
    )
    assert by_id["two"].summary.overlaps == (listing.Overlap("one", ("src/b.py",)),)
    assert by_id["three"].summary.overlaps == ()
    (lane,) = by_id["fan"].lanes
    assert lane.summary.overlaps == (listing.Overlap("one", ("src/a.py",)),)
    assert by_id["live"].summary.overlaps == () and by_id["landed"].summary.overlaps == ()

    assert by_id["one"].summary.merge_note() == "same paths as two and 1 more"
    assert by_id["two"].summary.merge_note() == "open findings: 1 block · same paths as one"
    wire = json.loads(json.dumps(listing.row_json(by_id["one"], winners=set())))
    assert wire["overlaps"] == [
        {"session_id": "two", "paths": ["src/b.py"]},
        {"session_id": "lane-1", "paths": ["src/a.py"]},
    ]
    assert wire["merge_note"] == "same paths as two and 1 more"


def test_a_row_counts_its_open_findings_from_the_journal(tmp_path: pathlib.Path) -> None:
    """A row's open findings are the journal's review fold, the count a merge refuses on."""
    session_dir = layout.bucket_dir(paths.state_dir(tmp_path), "runs") / "run-block"
    session_dir.mkdir(parents=True)
    rows = [
        review_helpers.START,
        *review_helpers.panel_rows(block=2, warn=1),
        review_helpers.END,
    ]
    (session_dir / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = listing.summarize_session_dir(session_dir)
    assert summary.open_findings == review.SeverityCounts(block=2, warn=1)
    assert summary.merge_note() == "open findings: 2 block, 1 warn"
    row = listing.summary_row(summary)
    assert row["open_findings"] == {"block": 2, "warn": 1, "nit": 0}
    assert row["merge_note"] == "open findings: 2 block, 1 warn" and row["overlaps"] == []

    clean = listing.summary_row(_summary("clean", mtime=1.0))
    assert clean["merge_note"] == "" and clean["open_findings"] == {
        "block": 0,
        "warn": 0,
        "nit": 0,
    }
