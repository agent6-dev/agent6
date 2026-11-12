"""A listing row's open findings and the one note every hub shows for them."""

from __future__ import annotations

import dataclasses
import json
import pathlib

from agent6 import paths
from agent6.sessions import layout
from agent6.viewmodel import listing, review
from tests import review_helpers


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
    assert row["merge_note"] == "open findings: 2 block, 1 warn"

    clean = listing.summary_row(dataclasses.replace(summary, open_findings=review.SeverityCounts()))
    assert clean["merge_note"] == "" and clean["open_findings"] == {
        "block": 0,
        "warn": 0,
        "nit": 0,
    }
