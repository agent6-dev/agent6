from __future__ import annotations

import dataclasses
from typing import Any

import pydantic
import pytest

from agent6 import review_records


def _finding() -> review_records.Finding:
    return review_records.Finding(
        category="security",
        severity="block",
        title="Secret in source",
        detail="Remove the credential.",
        location=review_records.ReviewLocation(
            path="old/name.py", side="base", start_line=11, end_line=13
        ),
    )


@pytest.mark.parametrize("side", ["base", "head"])
@pytest.mark.parametrize("span", [(None, None), (1, 1), (2, 5)])
def test_locations_round_trip(side: str, span: tuple[int | None, int | None]) -> None:
    location = review_records.ReviewLocation.model_validate(
        {"path": "src/a:b.py", "side": side, "start_line": span[0], "end_line": span[1]}
    )
    assert review_records.ReviewLocation.model_validate_json(location.model_dump_json()) == location


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "..",
        "../secret",
        "src/../../secret",
        "/etc/passwd",
        "C:/secret",
        "C:secret",
        "\\\\server\\share",
        "src\\..\\secret",
        "x\x00y",
        "x\ny",
    ],
)
def test_locations_refuse_paths_outside_the_repository(path: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        review_records.ReviewLocation(path=path, side="head")


def test_locations_normalize_relative_paths_without_losing_real_prefixes() -> None:
    location = review_records.ReviewLocation(path="./a//file.py", side="head")
    assert location.path == "a/file.py"


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 1), (1, 0), (-1, 1), (3, 2), (1, None), (None, 2), (True, 2), (1.5, 2)],
)
def test_locations_require_positive_complete_inclusive_ranges(start: Any, end: Any) -> None:
    with pytest.raises(pydantic.ValidationError):
        review_records.ReviewLocation(path="a.py", side="head", start_line=start, end_line=end)


@pytest.mark.parametrize(
    ("start", "end", "text"),
    [(None, None, "a.py"), (11, 11, "a.py:11"), (11, 13, "a.py:11-13")],
)
def test_locations_render_without_an_ambiguous_stored_citation(
    start: int | None, end: int | None, text: str
) -> None:
    location = review_records.ReviewLocation(
        path="a.py", side="head", start_line=start, end_line=end
    )
    assert str(location) == text


def test_findings_and_seat_replies_round_trip() -> None:
    finding = _finding()
    reply = review_records.SeatReply(
        verdict="block", summary="Credential leak", findings=(finding,)
    )
    restored = review_records.SeatReply.model_validate_json(reply.model_dump_json())
    assert restored == reply
    assert restored.findings[0].location.side == "base"
    assert set(finding.model_dump()) == {"category", "severity", "title", "detail", "location"}
    with pytest.raises(pydantic.ValidationError):
        review_records.Finding.model_validate({**finding.model_dump(), "file_line": "a.py:1"})


@pytest.mark.parametrize(("field", "limit"), [("title", 200), ("detail", 1000)])
def test_findings_keep_the_text_limits(field: str, limit: int) -> None:
    payload = _finding().model_dump()
    payload[field] = "x" * limit
    assert getattr(review_records.Finding.model_validate(payload), field) == "x" * limit
    payload[field] += "x"
    with pytest.raises(pydantic.ValidationError):
        review_records.Finding.model_validate(payload)


def test_frozen_records_round_trip() -> None:
    target = review_records.ReviewTarget(
        base_commit="base",
        base_tree="base-tree",
        head_commit=None,
        head_tree="head-tree",
        snapshot_commit="snapshot",
        paths=("src/a.py",),
        source="worktree",
    )
    verdict = review_records.ReviewVerdict(
        seat_index=2,
        focus="security",
        model_route="provider/model",
        verdict="block",
        findings=(_finding(),),
        summary="Leak",
        error=None,
    )
    observation = review_records.FindingObservation(
        finding_id="finding-1", panel_id="panel-1", finding=_finding(), seat_indexes=(0, 2)
    )
    for record in (target, verdict, observation):
        adapter = pydantic.TypeAdapter(type(record))
        assert adapter.validate_json(adapter.dump_json(record)) == record
        assert not hasattr(record, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, dataclasses.fields(record)[0].name, "different")


def test_abstentions_are_distinct_from_passes() -> None:
    verdict = review_records.ReviewVerdict(
        seat_index=0,
        focus="security",
        model_route="provider/model",
        verdict="abstain",
        findings=(),
        summary="",
        error="provider unavailable",
    )
    adapter = pydantic.TypeAdapter(review_records.ReviewVerdict)
    assert adapter.validate_json(adapter.dump_json(verdict)) == verdict
    with pytest.raises(pydantic.ValidationError):
        review_records.SeatReply.model_validate(
            {"verdict": "abstain", "summary": "", "findings": []}
        )


@pytest.mark.parametrize(
    "disposition",
    ["fixed", "false_positive", "intentional", "wont_fix", "duplicate", "pre_existing"],
)
def test_resolutions_round_trip_without_inventing_verification(disposition: str) -> None:
    resolution = review_records.FindingResolution.model_validate(
        {
            "disposition": disposition,
            "reason": "Checked the report.",
            "duplicate_of": "finding-2" if disposition == "duplicate" else None,
        }
    )
    assert resolution.commit is None and resolution.verify_id is None
    assert (
        review_records.FindingResolution.model_validate_json(resolution.model_dump_json())
        == resolution
    )


@pytest.mark.parametrize("duplicate_of", [None, "", " \n"])
def test_duplicate_resolutions_require_a_finding_id(duplicate_of: str | None) -> None:
    with pytest.raises(pydantic.ValidationError):
        review_records.FindingResolution(
            disposition="duplicate", reason="Same defect.", duplicate_of=duplicate_of
        )


@pytest.mark.parametrize("reason", ["", " \t\n", "x" * 2001])
def test_resolution_reasons_are_nonblank_and_bounded(reason: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        review_records.FindingResolution(disposition="fixed", reason=reason)


@pytest.mark.parametrize("action", ["fix", "resolve", "reopen"])
def test_actions_round_trip(action: str) -> None:
    payload: dict[str, Any] = {"request_id": "request-1", "expected_version": 3, "action": action}
    if action == "resolve":
        payload["resolution"] = {"disposition": "fixed", "reason": "Reproduced and corrected."}
    elif action == "reopen":
        payload["reason"] = "The defect remains."
    request = review_records.FindingAction.model_validate(payload)
    assert request.resume is False
    assert review_records.FindingAction.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize("reason", [None, "", " \t\n", "x" * 2001])
def test_reopen_requires_a_nonblank_bounded_reason(reason: str | None) -> None:
    with pytest.raises(pydantic.ValidationError):
        review_records.FindingAction(
            request_id="request-1", expected_version=0, action="reopen", reason=reason
        )


def test_resolve_requires_a_resolution() -> None:
    with pytest.raises(pydantic.ValidationError):
        review_records.FindingAction(request_id="request-1", expected_version=0, action="resolve")


@pytest.mark.parametrize(
    "field,value", [("request_id", ""), ("request_id", " "), ("expected_version", -1)]
)
def test_actions_require_an_id_and_a_nonnegative_version(field: str, value: object) -> None:
    payload = {"request_id": "request-1", "expected_version": 0, "action": "fix", field: value}
    with pytest.raises(pydantic.ValidationError):
        review_records.FindingAction.model_validate(payload)


def test_boundary_models_are_frozen_and_reject_unknown_fields() -> None:
    models = (
        _finding(),
        _finding().location,
        review_records.SeatReply(verdict="pass", summary="", findings=()),
        review_records.FindingResolution(disposition="fixed", reason="Corrected."),
        review_records.FindingAction(request_id="request-1", expected_version=0, action="fix"),
    )
    for model in models:
        with pytest.raises(pydantic.ValidationError):
            type(model).model_validate({**model.model_dump(), "unexpected": True})
        with pytest.raises(pydantic.ValidationError):
            setattr(model, next(iter(type(model).model_fields)), "different")
