from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest

from agent6 import review_records
from agent6.viewmodel import _session_log, _state_view, state


def _target(**changes: Any) -> review_records.ReviewTarget:
    return dataclasses.replace(
        review_records.ReviewTarget("base", "bt", "head", "ht", "head", (), "revisions"),
        **changes,
    )


def _panel(
    panel_id: str = "p1", *, findings: bool = True, completion: str = "complete"
) -> dict[str, Any]:
    finding = review_records.Finding(
        category="security",
        severity="block",
        title="leak",
        detail="a secret",
        location=review_records.ReviewLocation(path="a.py", side="head", start_line=1, end_line=1),
    )
    record = review_records.ReviewPanel.model_validate(
        dict(
            panel_id=panel_id,
            execution_id="exec",
            turn=1,
            trigger="before_finish",
            target=_target(),
            findings=[review_records.FindingObservation("f1", panel_id, finding, (0, 1))]
            if findings
            else [],
            injected_finding_ids=["f1"] if findings else [],
            block_on="none",
            raw_blocked=False,
            blocked=False,
            disarmed=False,
            n_block=2 if findings else 0,
            n_abstain=2 if completion == "inconclusive" else 0,
            completion=completion,
        )
    )
    return {"type": "loop.review.panel", "event_id": panel_id, **record.model_dump(mode="json")}


def _action(
    kind: str, *, request_id: str = "request", version: int = 2, **fields: Any
) -> dict[str, Any]:
    return dict(
        type=f"loop.review.{kind}",
        event_id=kind + request_id,
        finding_id="f1",
        request_id=request_id,
        version=version,
        source="cli",
        **fields,
    )


@pytest.mark.parametrize(
    "later",
    [
        {"type": "loop.resume.start", "execution_id": "next"},
        {
            "type": "loop.review.skipped",
            "panel_id": "p2",
            "execution_id": "exec",
            "reason": "budget",
        },
        {"type": "loop.review.rejection_cap_reached", "rejections": 4},
        _panel("p2", findings=False),
        _panel("p2", findings=False, completion="inconclusive"),
    ],
)
def test_open_findings_survive_later_events(later: dict[str, Any]) -> None:
    """Keep an open block after a pass, budget skip, resume, abstention or rejection cap."""
    result = state.fold_session([_panel(), later])
    assert result.review.open_counts.block == 1
    assert result.review.findings[0].status == "open"
    assert result.review.merge_guard.open_block_ids == ("f1",)


def test_repeated_observation_preserves_resolution_and_history(tmp_path: pathlib.Path) -> None:
    """Fold append order identically for a live viewer and a journal scan."""
    resolution = dict(disposition="fixed", reason="reproduced and corrected")
    rows = [
        _panel(),
        _action("finding.resolved", resolution=resolution, ts="2030-01-01T00:00:00Z"),
        _panel("p2"),
        _action(
            "finding.reopened",
            request_id="again",
            version=4,
            reason="still reproducible",
            ts="2020-01-01T00:00:00Z",
        ),
    ]
    live = state.fold_session(rows).review
    log = tmp_path / "logs.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert _session_log.scan_session_log(log).review == live
    assert live.version == 4
    finding = live.findings[0]
    assert len(finding.observations) == 2
    assert finding.status == "open"
    assert [a.reason for a in finding.actions] == [resolution["reason"], "still reproducible"]
    resolved = state.fold_session(rows[:3]).review.findings[0]
    assert resolved.status == "resolved"
    assert resolved.resolution is not None and resolved.resolution.verify_id is None
    assert state.fold_session([*rows, rows[-1]]).review == live


def test_line_origin_follows_the_latest_observation_without_changing_merge_policy() -> None:
    first = _panel()
    first["findings"][0]["line_origin"] = dict(
        kind="pre_existing",
        reason="The cited lines exist at the base.",
        base_location=dict(path="old.py", side="base", start_line=7, end_line=7),
        commits=["a" * 40],
    )
    result = state.fold_session([first])
    assert result.review.findings[0].line_origin.kind == "pre_existing"
    assert result.review.merge_guard.open_block_ids == ("f1",)
    wire = _state_view.session_state_as_dict(result)["review"]
    assert wire["findings"][0]["line_origin"] == first["findings"][0]["line_origin"]
    later = _panel("p2")
    later["findings"][0]["line_origin"] = dict(kind="changed", reason="Changed lines.")
    finding = state.fold_session([first, later]).review.findings[0]
    assert finding.line_origin.kind == "changed"
    assert finding.observations[0].line_origin.kind == "pre_existing"
    legacy = _panel()
    legacy["findings"][0].pop("line_origin", None)
    finding = state.fold_session([legacy]).review.findings[0]
    assert finding.line_origin.kind == "unknown"
    assert "historical" in finding.line_origin.reason.lower()


def test_fix_delivery_is_unknown_until_submission_and_never_resolves() -> None:
    """Keep a requested Fix open and show an unconfirmed delivery as unknown."""
    requested = _action("fix.requested", target=dataclasses.asdict(_target()), action="live-steer")
    result = state.fold_session([_panel(), requested]).review
    assert result.findings[0].fix_state == "delivery_unknown"
    submitted = _action("fix.submitted", execution_id="next-exec")
    result = state.fold_session([_panel(), requested, submitted]).review
    assert result.findings[0].fix_state == "submitted"
    assert result.findings[0].status == "open"
    failed = _action("fix.failed", error="worker exited")
    assert (
        state.fold_session([_panel(), requested, failed]).review.findings[0].fix_state == "failed"
    )


def test_freshness_compares_both_trees_and_coverage() -> None:
    """Do not certify a different base or a whole run from a filtered review."""
    view = state.fold_session([_panel()]).review
    assert view.freshness == "unknown"
    assert view.for_comparison(_target()).freshness == "current"
    assert view.for_comparison(_target(base_tree="other")).freshness == "stale"
    changed = view.for_comparison(_target(head_tree="other"))
    assert changed.freshness == "stale"
    assert changed.findings[0].anchor_state == "outdated"
    assert changed.findings[0].status == "open"
    filtered = _panel()
    filtered["target"]["paths"] = ["a.py"]
    view = state.fold_session([filtered]).review
    assert view.for_comparison(_target()).freshness == "stale"
    assert view.for_comparison(_target(paths=("a.py",))).freshness == "current"


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"type": "loop.review.panel", "blocked": False}],
        [{"type": "loop.review.seat", "findings": 0, "verdict": "pass"}],
    ],
)
def test_legacy_and_missing_evidence_are_not_clean(rows: list[dict[str, Any]]) -> None:
    view = state.fold_session(rows).review
    assert view.completion == "unknown"
    assert view.freshness == "unknown"
    assert view.findings == ()
    assert bool(view.legacy_notice) == bool(rows)


def test_corrupt_new_evidence_keeps_its_reason(tmp_path: pathlib.Path) -> None:
    row = _panel()
    row["findings"][0]["finding"]["location"]["path"] = "../escape"
    log = tmp_path / "logs.jsonl"
    log.write_text(json.dumps(row) + "\n")
    result = state.fold_session([row]).review
    assert result.merge_guard.read_failure
    assert "repository" in result.merge_guard.read_failure
    assert _session_log.scan_session_log(log).review == result
    assert state.fold_session(
        [row, _panel("clean", findings=False)]
    ).review.merge_guard.read_failure


def test_unreadable_journal_tail_cannot_certify_a_review(tmp_path: pathlib.Path) -> None:
    """Keep a torn finding record from making an earlier clean panel look authoritative."""
    log = tmp_path / "logs.jsonl"
    log.write_text(json.dumps(_panel(findings=False)) + '\n{"type":"loop.review.panel",')
    view = _session_log.scan_session_log(log).review
    assert view.read_failure
    assert view.for_comparison(_target()).freshness == "unknown"


def test_a_panel_row_with_an_unknown_field_is_unreadable_evidence(tmp_path: pathlib.Path) -> None:
    """Unknown fields on a panel row were ignored; the row is refused and both readers say so."""
    from agent6.harness import _review_history

    enveloped = {**_panel(), "ts": "2026-10-25T00:00:00+00:00"}
    assert not state.fold_session([enveloped]).review.read_failure
    unknown = {**enveloped, "unexpected": 1}
    failure = state.fold_session([unknown]).review.read_failure
    assert failure and "unexpected" in failure
    log = tmp_path / "logs.jsonl"
    log.write_text(json.dumps(enveloped) + "\n")
    assert len(_review_history.load_panels(log)) == 1
    log.write_text(json.dumps(unknown) + "\n")
    with pytest.raises(_review_history.ReviewHistoryError, match="unexpected"):
        _review_history.load_panels(log)


def test_wire_contains_structured_findings_and_counts() -> None:
    result = state.fold_session([_panel()])
    wire = json.loads(json.dumps(_state_view.session_state_as_dict(result)))
    review = wire.get("review")
    assert review is not None
    assert review["open_counts"]["block"] == 1
    assert review["findings"][0]["observations"][0]["finding"]["location"]["path"] == "a.py"


def test_replayed_resolution_request_does_not_change_history() -> None:
    resolved = _action(
        "finding.resolved", resolution=dict(disposition="intentional", reason="needed")
    )
    view = state.fold_session([_panel(), resolved]).review
    replay = {**resolved, "event_id": "replayed"}
    assert state.fold_session([_panel(), resolved, replay]).review == view


def test_legacy_tail_cannot_certify_an_earlier_review() -> None:
    view = state.fold_session([_panel(), {"type": "loop.review.panel", "blocked": False}]).review
    assert view.for_comparison(_target()).freshness == "unknown"


@pytest.mark.parametrize(
    "change",
    [
        {"resolution": {"disposition": "fixed", "reason": " "}},
        {"resolution": {"disposition": "duplicate", "reason": "same", "duplicate_of": "elsewhere"}},
        {"version": 99},
        {"finding_id": "missing"},
    ],
)
def test_bad_disposition_preserves_the_open_finding(change: dict[str, Any]) -> None:
    action = _action("finding.resolved", resolution=dict(disposition="fixed", reason="corrected"))
    view = state.fold_session([_panel(), {**action, **change}]).review
    assert view.read_failure
    assert view.open_counts.block == 1
    assert view.findings[0].actions == ()


def test_start_and_seat_keep_the_roster_and_full_verdict() -> None:
    panel = _panel()
    start = {
        "type": "loop.review.start",
        "panel_id": "p1",
        "execution_id": "exec",
        "target": panel["target"],
        "seats": [{"seat_index": 0, "model_route": "p/m", "focus": "security"}],
    }
    seat = {
        "type": "loop.review.seat",
        "panel_id": "p1",
        "execution_id": "exec",
        "seat_index": 0,
        "model_route": "p/m",
        "focus": "security",
        "verdict": "block",
        "summary": "leak",
        "error": None,
        "findings": [panel["findings"][0]["finding"]],
    }
    running = state.fold_session([start, seat]).review
    assert running.completion == "running"
    assert running.panels[0].roster[0].model_route == "p/m"
    finished = state.fold_session([start, seat, panel]).review
    assert finished.panels[0].seats[0].summary == "leak"
    assert finished.panels[0].seats[0].findings[0].title == "leak"
