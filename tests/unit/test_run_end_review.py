from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import budget, paths
from agent6.app import finalize, reporter
from agent6.harness import loop_snapshot
from agent6.sessions import layout
from tests import review_helpers


def _end(repo: pathlib.Path, rows: list[dict[str, Any]], *, streaming: bool = False) -> str:
    session = layout.SessionLayout(paths.state_dir(repo), "review-end")
    session.ensure()
    events = [{**review_helpers.START, "execution_id": "e"}, *rows, review_helpers.END]
    session.logs_path.write_text("".join(json.dumps(row) + "\n" for row in events))
    lines: list[str] = []
    finalize.print_session_end(
        loop_snapshot.SessionResult(True, "finish_session", "done", 2, 1, verification="passed"),
        layout=session,
        cwd=repo,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
        console_stream=streaming,
        reporter=reporter.Reporter(out=lines.append, err=reporter.STDIO_REPORTER.err),
    )
    return "\n".join(lines)


def _panel(*votes: str, blocks: int = 0, warnings: int = 0) -> list[dict[str, Any]]:
    start, result = review_helpers.panel_rows(
        block=blocks, warn=warnings, abstain=votes.count("abstain")
    )
    roster = [
        dict(reviewer_index=i, focus=f"check {i}", model_route="local/reviewer")
        for i in range(len(votes))
    ]
    start["reviewers"] = roster
    result.update(
        roster=roster,
        block_on="none",
        blocked=False,
        raw_blocked=False,
        n_block=int(bool(blocks)),
        completion="inconclusive" if all(v == "abstain" for v in votes) else "complete",
    )
    verdicts = [
        {"type": "review.verdict", "panel_id": "p", "execution_id": "e", **seat, "verdict": vote}
        for seat, vote in zip(roster, votes, strict=True)
    ]
    return [start, *verdicts, result]


@pytest.mark.parametrize("streaming", [False, True])
def test_advisory_panel_votes_and_still_open_findings_reach_the_end(
    tmp_path: pathlib.Path, streaming: bool
) -> None:
    """Two block votes on one route stay two votes even after a finding is resolved."""
    rows = _panel("block", "block", "pass", "abstain", blocks=2, warnings=1)
    rows.append(
        {
            "type": "review.finding.resolved",
            "finding_id": "f0",
            "request_id": "resolve",
            "source": "cli",
            "version": len(rows) + 1,
            "resolution": {"disposition": "fixed", "reason": "Corrected the input check."},
        }
    )
    out = _end(tmp_path, rows, streaming=streaming)
    assert (
        out.splitlines().count("  last review panel: 2 voted block, 1 voted pass, 1 abstained") == 1
    )
    assert out.splitlines().count("  open findings: 1 block, 1 warn") == 1
    assert out.count("agent6 sessions receipt review-end") == 1


@pytest.mark.parametrize(
    ("vote", "outcome"), [("pass", "1 voted pass"), ("abstain", "inconclusive (1 abstained)")]
)
def test_review_without_findings_still_reports_what_returned(
    tmp_path: pathlib.Path, vote: str, outcome: str
) -> None:
    """A panel with no findings can have passed or reviewed nothing."""
    out = _end(tmp_path, _panel(vote))
    assert f"last review panel: {outcome}" in out
    assert "open findings:" not in out


def test_resume_without_a_fresh_panel_keeps_findings_but_not_votes(tmp_path: pathlib.Path) -> None:
    """Reusing a review on resume does not cast its votes again."""
    rows = _panel("block", blocks=1)
    rows.extend(
        [
            review_helpers.END,
            {"type": "session.resume", "execution_id": "next"},
            {
                "type": "review.reused",
                "panel_id": "reuse",
                "execution_id": "next",
                "reused_panel_id": "p",
                "target": rows[-1]["target"],
            },
        ]
    )
    out = _end(tmp_path, rows)
    assert "open findings: 1 block" in out
    assert "last review panel:" not in out


def test_skip_alone_adds_no_review_to_the_end(tmp_path: pathlib.Path) -> None:
    """A skipped panel cast no votes."""
    out = _end(
        tmp_path,
        [{"type": "review.skipped", "panel_id": "p", "execution_id": "e", "reason": "no_diff"}],
    )
    assert "last review panel:" not in out and "open findings:" not in out
