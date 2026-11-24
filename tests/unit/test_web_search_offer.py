# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Who is offered the hosted web search: the worker's own turns, and nothing else."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from agent6 import budget as agent6_budget
from agent6.app import role_providers
from agent6.app.review import recap as review_recap
from agent6.config import model as config_model
from agent6.harness import loop, loop_conversation, provider_call, run_chain
from agent6.harness.compaction import calls as compaction_calls
from agent6.harness.compaction import window as compaction_window
from agent6.harness.prompt import blocks as prompt_blocks
from agent6.harness.prompt import repo_context
from agent6.prompts import loop as loop_prompts
from agent6.providers import provider_types
from tests.unit import test_loop_wire_golden as golden


class _Recorder:
    """A provider that records each call's keyword arguments and answers in prose."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call(self, **kwargs: Any) -> provider_types.ProviderResponse:
        self.calls.append(kwargs)
        return golden._resp(text="The answer.")


def _drop(_event: object) -> None:
    """Take a journal event and write nothing."""


def _caller(provider: _Recorder) -> provider_call.ProviderCaller:
    return provider_call.ProviderCaller(
        provider=provider,
        retry_count=0,
        retry_delay_s=0.0,
        retry_max_delay_s=0.0,
        temperature=None,
        should_abort=lambda: False,
        should_interrupt=lambda: False,
        log=lambda _msg: None,
        emit=_drop,
    )


@pytest.mark.parametrize("offered", [False, True])
def test_a_worker_turn_carries_the_offer_to_the_provider(
    tmp_path: pathlib.Path, offered: bool
) -> None:
    """The turn's call names the search only when the run offers it, through every wrapper."""
    inner = _Recorder()
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    worker = role_providers.InstrumentedProvider(
        inner=inner, actor="worker", model="m", provider_name="p", events=None, budget=budget
    )
    wf = loop.Harness(
        chain=run_chain.RunChain(tmp_path),
        config=golden._config(),
        provider=worker,
        dispatcher=golden._Dispatcher([False]),  # pyrefly: ignore[bad-argument-type]
        logger=lambda _msg: None,
        mode="ask",
        web_search=offered,
    )
    conversation = loop_conversation.Conversation()
    conversation.notice("TASK:\nwhat changed?")
    result = wf._drive_loop(
        system="s",
        conversation=conversation,
        tool_calls=0,
        start_turn=1,
        root_task_id=None,
        original_task="what changed?",
    )
    assert result.reason == "answered"
    assert [call.get("web_search") for call in inner.calls] == [True if offered else None]
    assert ("web_search" in inner.calls[0]) is offered


def test_a_summary_call_and_a_reviewer_are_never_offered_search() -> None:
    """A summary reuses the worker's provider and tools, and still names no search."""
    worker = _Recorder()
    conversation = loop_conversation.Conversation()
    conversation.notice("TASK:\nt")
    calls = compaction_calls.Calls(
        worker=worker,
        caller=_caller(worker),
        settings=compaction_window.CompactionSettings(),
        conversation=conversation,
        system="s",
        tools=[],
        temperature=None,
        execution_id="e",
        turn_identity=None,
        check_budget=lambda: None,
        should_abort=lambda: False,
        should_interrupt=lambda: False,
    )
    calls.call_worker("Summarize.", "c1")
    calls.call_endpoint("Summarize.", "c2", 1)
    calls.call_helper("s", "Summarize.", "c3")
    review_recap.review_digest(worker, digest="the record")
    assert len(worker.calls) == 4
    assert not any("web_search" in call for call in worker.calls)


def test_the_prompt_states_the_search_only_where_it_is_offered(tmp_path: pathlib.Path) -> None:
    """One sentence in the run and ask prompts of a worker entry that sets the key."""
    providers: dict[str, Any] = {"p": {"backend": "anthropic", "web_search": True}}
    on = config_model.Config.model_validate(
        {"providers": providers, "models": {"default": {"model": "p/m"}}}
    )
    off = config_model.Config.model_validate(
        {"providers": {"p": {"backend": "anthropic"}}, "models": {"default": {"model": "p/m"}}}
    )
    repo = repo_context.load_repo_summary(tmp_path, ())
    line = loop_prompts.WEB_SEARCH_LINE
    assert line == "The provider's web search is available.\n"

    def prompt(cfg: config_model.Config, mode: Any) -> str:
        return prompt_blocks.build_system_prompt(
            config=cfg, repo=repo, mode=mode, skills=None, web_search=cfg.offers_web_search(mode)
        )

    for mode in ("run", "ask"):
        assert prompt(on, mode).count(line) == 1
        assert prompt(on, mode).replace(line + "\n", "") == prompt(off, mode)
    for mode in ("plan", "read_only"):
        assert line not in prompt(on, mode)
