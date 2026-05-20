# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for tool/LLM JSON-Schema generation."""

from __future__ import annotations

from agent6.models import Edit, FileEdit, OpenQuestion, Plan, RefinedSpec, Review, Step, Summary
from agent6.tools.schema import ALL_TOOLS, schemas_as_provider_tools


def test_schemas_as_provider_tools_shape() -> None:
    out = schemas_as_provider_tools()
    assert len(out) == len(ALL_TOOLS)
    for entry in out:
        assert set(entry) == {"name", "description", "input_schema"}
        assert isinstance(entry["name"], str) and entry["name"]
        assert entry["input_schema"].get("type") == "object"


def test_refined_spec_roundtrip() -> None:
    rs = RefinedSpec(
        refined_task="do x",
        open_questions=(OpenQuestion(question="a?", suggestions=("x", "y")),),
    )
    assert RefinedSpec.model_validate_json(rs.model_dump_json()) == rs


def test_plan_min_one_step() -> None:
    import pytest

    with pytest.raises(Exception):
        Plan(summary="s", steps=())


def test_edit_and_step_roundtrip() -> None:
    e = Edit(
        notes="",
        edits=(FileEdit(path="a.py", kind="create", old_string="", new_string="x=1"),),
    )
    assert Edit.model_validate_json(e.model_dump_json()) == e
    s = Step(title="t", rationale="r", relevant_paths=("a.py",), acceptance="ok")
    assert Step.model_validate_json(s.model_dump_json()) == s


def test_review_literal_enforced() -> None:
    import pytest
    from pydantic import ValidationError

    Review(verdict="pass")
    Review(verdict="fail")
    with pytest.raises(ValidationError):
        Review(verdict="maybe")  # type: ignore[arg-type]


def test_review_proposed_followup_optional() -> None:
    # Default empty; round-trips when set.
    r = Review(
        verdict="fail",
        comments="missing test",
        proposed_followup="add test_x to tests/y.py",
    )
    assert r.proposed_followup == "add test_x to tests/y.py"
    assert Review(verdict="pass").proposed_followup == ""


def test_summary_requires_text() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Summary(summary="")
