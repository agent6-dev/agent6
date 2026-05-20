# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the alignment-guard sub-agent.

The Anthropic call is monkeypatched; we verify the prompt assembled by
`alignment_check` contains the original task, parent path, current node,
proposed action, and (when supplied) the formatted resume diff.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

import pytest

import agent6.agents.alignment  # noqa: F401  # pyright: ignore[reportUnusedImport]
from agent6.agents.alignment import alignment_check
from agent6.graph.models import CommittedDelta, ResumeDiff, TaskNode, UncommittedFileDiff
from agent6.models import AlignmentVerdict
from agent6.providers import AnthropicProvider

al_module = sys.modules["agent6.agents.alignment"]


def _node(title: str, *, id_: str = "0" * 26) -> TaskNode:
    now = datetime.now(tz=UTC)
    return TaskNode(
        id=id_,
        parent_id=None,
        title=title,
        rationale="why-" + title,
        acceptance="acc-" + title,
        relevant_paths=("src/x.py",),
        depends_on=(),
        children=(),
        status="pending",
        created_at=now,
        updated_at=now,
        created_by="planner",
        commit_sha="",
        notes="",
    )


def _fake_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="test", model="claude-test", prompt_caching=False)


def test_alignment_check_assembles_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake(
        provider: Any,
        *,
        system: str,
        user: str,
        output_model: type[AlignmentVerdict],
        max_tokens: int,
    ) -> AlignmentVerdict:
        captured["system"] = system
        captured["user"] = user
        captured["model"] = output_model
        captured["max_tokens"] = max_tokens
        return AlignmentVerdict(verdict="proceed", reasoning="ok")

    monkeypatch.setattr(al_module, "call_for_model", fake)

    verdict = alignment_check(
        _fake_provider(),
        node=_node("do the thing"),
        parent_path=(_node("root task", id_="1" * 26),),
        original_task="ship the feature",
        proposed_action="execute",
    )
    assert verdict.verdict == "proceed"
    user = captured["user"]
    assert "ship the feature" in user
    assert "do the thing" in user
    assert "root task" in user
    assert "execute" in user
    assert "RESUME DIFF" not in user
    assert captured["model"] is AlignmentVerdict
    assert captured["max_tokens"] == 1024


def test_alignment_check_includes_resume_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake(
        provider: Any,
        *,
        system: str,
        user: str,
        output_model: type[AlignmentVerdict],
        max_tokens: int,
    ) -> AlignmentVerdict:
        captured["user"] = user
        return AlignmentVerdict(verdict="ask", reasoning="diff large; ask user")

    monkeypatch.setattr(al_module, "call_for_model", fake)

    diff = ResumeDiff(
        run_id="r1",
        snapshot_head="a" * 40,
        current_head="b" * 40,
        committed_delta=CommittedDelta(
            from_sha="a" * 40, to_sha="b" * 40, files=("src/changed.py",)
        ),
        uncommitted_diff=(
            UncommittedFileDiff(
                path="src/dirty.py",
                expected_sha256="0" * 64,
                actual_sha256="1" * 64,
                note="hash mismatch",
            ),
        ),
        snapshot_missing=False,
        guard_summary="1 committed delta, 1 dirty file",
    )

    verdict = alignment_check(
        _fake_provider(),
        node=_node("do the thing"),
        parent_path=(),
        original_task="ship",
        proposed_action="resume",
        resume_diff=diff,
    )
    assert verdict.verdict == "ask"
    assert "RESUME DIFF" in captured["user"]
    assert "src/changed.py" in captured["user"]
    assert "src/dirty.py" in captured["user"]
    assert "1 committed delta, 1 dirty file" in captured["user"]
