# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Test the critic's deterministic fallback when AGENTS.md is missing.

The model call is monkeypatched. We don't trust the model to always emit
the recommended open_question; the critic appends it post-hoc when
agents_md is empty.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent6.agents import critic as critic_module
from agent6.agents.critic import critic_refine
from agent6.models import OpenQuestion, RefinedSpec
from agent6.providers import AnthropicProvider


def _fake_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="x", model="m", prompt_caching=False)


def test_missing_agents_md_appends_open_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(provider: Any, **kwargs: Any) -> RefinedSpec:
        return RefinedSpec(refined_task="do the thing", open_questions=())

    monkeypatch.setattr(critic_module, "call_for_model", fake)
    out = critic_refine(_fake_provider(), user_task="t", agents_md="")
    assert any("AGENTS.md" in q.question for q in out.open_questions)


def test_present_agents_md_leaves_open_questions_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = OpenQuestion(question="real q", suggestions=("yes", "no"))

    def fake(provider: Any, **kwargs: Any) -> RefinedSpec:
        return RefinedSpec(refined_task="do the thing", open_questions=(q,))

    monkeypatch.setattr(critic_module, "call_for_model", fake)
    out = critic_refine(_fake_provider(), user_task="t", agents_md="# AGENTS\n\nverify: pytest\n")
    assert out.open_questions == (q,)


def test_critic_does_not_duplicate_if_model_already_mentioned_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(provider: Any, **kwargs: Any) -> RefinedSpec:
        return RefinedSpec(
            refined_task="t",
            open_questions=(OpenQuestion(question="There is no AGENTS.md; please add one."),),
        )

    monkeypatch.setattr(critic_module, "call_for_model", fake)
    out = critic_refine(_fake_provider(), user_task="t", agents_md="")
    assert len(out.open_questions) == 1
