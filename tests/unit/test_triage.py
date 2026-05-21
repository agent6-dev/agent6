# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the triage sub-agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent6.agents.triage import TaskClass, TaskClassification, triage_classify
from agent6.providers import ProviderResponse
from agent6.types import RepoSummary


class _FakeProvider:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        return ProviderResponse(
            text=json.dumps(self._payload),
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            tool_uses=(),
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def _repo(tmp_path: Path) -> RepoSummary:
    return RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=42,
        top_level=("src", "tests"),
        agents_md="# AGENTS.md\n\nverify with pytest.",
        recent_log="",
    )


def test_triage_returns_validated_classification(tmp_path: Path) -> None:
    provider = _FakeProvider(
        {
            "task_class": "trivial",
            "reasoning": "single-line typo fix",
            "confidence": 0.95,
        }
    )
    out = triage_classify(
        provider,  # type: ignore[arg-type]
        user_task="fix typo in README",
        agents_md="# AGENTS.md",
        repo=_repo(tmp_path),
    )
    assert isinstance(out, TaskClassification)
    assert out.task_class is TaskClass.TRIVIAL
    assert 0.0 <= out.confidence <= 1.0
    assert provider.calls, "provider was not called"


def test_triage_accepts_all_four_classes(tmp_path: Path) -> None:
    for label, expected in (
        ("trivial", TaskClass.TRIVIAL),
        ("single", TaskClass.SINGLE_STEP),
        ("multi", TaskClass.MULTI_STEP),
        ("exploration", TaskClass.EXPLORATION),
    ):
        provider = _FakeProvider({"task_class": label, "reasoning": "x", "confidence": 0.7})
        out = triage_classify(
            provider,  # type: ignore[arg-type]
            user_task="t",
            agents_md="",
            repo=_repo(tmp_path),
        )
        assert out.task_class is expected


def test_triage_rejects_invalid_class(tmp_path: Path) -> None:
    provider = _FakeProvider({"task_class": "gigantic", "reasoning": "x", "confidence": 0.5})
    from agent6.agents._common import SubAgentError  # local: avoid public re-export pressure

    with pytest.raises(SubAgentError):
        triage_classify(
            provider,  # type: ignore[arg-type]
            user_task="t",
            agents_md="",
            repo=_repo(tmp_path),
        )


def test_triage_rejects_confidence_out_of_range(tmp_path: Path) -> None:
    provider = _FakeProvider({"task_class": "trivial", "reasoning": "x", "confidence": 1.5})
    from agent6.agents._common import SubAgentError

    with pytest.raises(SubAgentError):
        triage_classify(
            provider,  # type: ignore[arg-type]
            user_task="t",
            agents_md="",
            repo=_repo(tmp_path),
        )
