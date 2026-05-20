# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Verify worker and reviewer thread AGENTS.md + sibling/parent context into prompts."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.agents.reviewer import reviewer_review
from agent6.agents.worker import worker_edit
from agent6.models import Step
from agent6.providers import ProviderResponse
from agent6.types import FileContext


class _CapturingProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_system: str = ""
        self.last_user: str = ""

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> ProviderResponse:
        self.last_system = system
        self.last_user = messages[0]["content"]
        return ProviderResponse(
            text=self.reply,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            stop_reason="end_turn",
            tool_uses=(),
        )


def test_worker_includes_agents_md_parent_acceptance_and_siblings() -> None:
    step = Step(
        title="add foo",
        rationale="needed",
        acceptance="foo exists",
        relevant_paths=("src/foo.py",),
    )
    ctx = FileContext(files=((Path("src/foo.py"), "x = 1\n"),))
    reply = json.dumps({"edits": [], "notes": "noop"})
    prov = _CapturingProvider(reply)

    worker_edit(
        prov,  # type: ignore[arg-type]
        step=step,
        file_context=ctx,
        agents_md="LINE-LENGTH=100. Use ruff.",
        parent_acceptance="ship the foo feature end-to-end",
        sibling_commits=(("abc1234def", "create stub"), ("9876543210", "add tests")),
    )

    user = prov.last_user
    assert "LINE-LENGTH=100" in user
    assert "AGENTS.md" in user
    assert "ship the foo feature" in user
    assert "PARENT TASK ACCEPTANCE" in user
    assert "abc1234" in user
    assert "create stub" in user
    assert "COMPLETED SIBLING STEPS" in user
    # AGENTS.md is referenced from the system prompt (so worker is told to follow it).
    assert "AGENTS.md" in prov.last_system


def test_worker_empty_context_is_clean() -> None:
    step = Step(title="t", rationale="r", acceptance="a", relevant_paths=())
    ctx = FileContext(files=())
    prov = _CapturingProvider(json.dumps({"edits": [], "notes": ""}))
    worker_edit(prov, step=step, file_context=ctx)  # type: ignore[arg-type]
    # No empty PARENT TASK / SIBLING blocks when not provided.
    assert "PARENT TASK ACCEPTANCE" not in prov.last_user
    assert "COMPLETED SIBLING STEPS" not in prov.last_user
    # AGENTS.md placeholder still emitted so the system prompt's reference is valid.
    assert "AGENTS.md" in prov.last_user
    assert "(empty)" in prov.last_user


def test_reviewer_includes_agents_md() -> None:
    step = Step(title="t", rationale="r", acceptance="a", relevant_paths=())
    prov = _CapturingProvider(json.dumps({"verdict": "pass", "comments": "ok"}))
    reviewer_review(
        prov,  # type: ignore[arg-type]
        step=step,
        diff="--- a\n+++ b\n",
        verify_output="all green",
        verify_ok=True,
        agents_md="No bare except.",
    )
    assert "No bare except." in prov.last_user
    assert "AGENTS.md" in prov.last_user
