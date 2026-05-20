# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the planner_revise sub-agent.

The Anthropic call is monkeypatched; we verify the prompt assembled by
`planner_revise` contains the previous plan and feedback, and that the
returned value is the validated `Plan` from `call_for_model`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Import the submodule via sys.modules; `agent6.agents.planner_revise` as an
# attribute of the parent package is shadowed by the re-exported function in
# `agent6.agents.__init__`, so direct `import ... as pr_module` resolves to
# the function rather than the module.
import agent6.agents.planner_revise  # noqa: F401  # pyright: ignore[reportUnusedImport]
from agent6.agents.planner_revise import planner_revise
from agent6.models import Plan, Step
from agent6.types import RepoSummary

pr_module = sys.modules["agent6.agents.planner_revise"]


def _plan(*titles: str) -> Plan:
    return Plan(
        summary="prev summary",
        steps=tuple(
            Step(title=t, rationale="why-" + t, acceptance="acc-" + t, relevant_paths=(t + ".py",))
            for t in titles
        ),
    )


def test_planner_revise_passes_previous_plan_and_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_call_for_model(
        provider: Any,
        *,
        system: str,
        user: str,
        output_model: type[Plan],
        max_tokens: int,
    ) -> Plan:
        captured["system"] = system
        captured["user"] = user
        captured["model"] = output_model
        captured["max_tokens"] = max_tokens
        return _plan("revised-step")

    monkeypatch.setattr(pr_module, "call_for_model", fake_call_for_model)

    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=3,
        top_level=("src/",),
        agents_md="my AGENTS doc",
        recent_log="",
    )
    out = planner_revise(
        MagicMock(),
        previous_plan=_plan("alpha", "beta"),
        user_feedback="drop beta and rename alpha to gamma",
        repo=repo,
    )
    assert out.steps[0].title == "revised-step"
    assert "alpha" in captured["user"]
    assert "beta" in captured["user"]
    assert "drop beta" in captured["user"]
    assert "my AGENTS doc" in captured["user"]
    assert captured["model"] is Plan
    assert captured["max_tokens"] == 4096
    # No steering instruction provided → STEERING block must be absent.
    assert "STEERING INSTRUCTION" not in captured["user"]


def test_planner_revise_includes_steering_instruction_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_call_for_model(
        provider: Any,
        *,
        system: str,
        user: str,
        output_model: type[Plan],
        max_tokens: int,
    ) -> Plan:
        captured["user"] = user
        return _plan("steered-step")

    monkeypatch.setattr(pr_module, "call_for_model", fake_call_for_model)
    repo = RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=1,
        top_level=("src/",),
        agents_md="",
        recent_log="",
    )
    planner_revise(
        MagicMock(),
        previous_plan=_plan("a"),
        user_feedback="mid-run change",
        repo=repo,
        steer_instruction="skip tests and focus on docs",
    )
    assert "STEERING INSTRUCTION" in captured["user"]
    assert "skip tests and focus on docs" in captured["user"]
