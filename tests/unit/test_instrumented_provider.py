# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The cli ``_InstrumentedProvider`` wrapper must forward every
provider.call kwarg to the inner provider. A missing passthrough is
invisible to unit tests that call providers directly but crashes every
real run (regression: ``reasoning_effort`` was added to the providers
and the loop but not the wrapper, so the perf bench died with
``TypeError: ... got an unexpected keyword argument 'reasoning_effort'``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agent6.budget import BudgetTracker
from agent6.cli.providers import _InstrumentedProvider  # pyright: ignore[reportPrivateUsage]
from agent6.providers import ProviderResponse


def _resp() -> ProviderResponse:
    return ProviderResponse(
        text="ok",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": "ok"}]},
    )


def _wrap(inner: MagicMock) -> _InstrumentedProvider:
    return _InstrumentedProvider(
        inner=inner,
        role="worker",
        model="moonshotai/kimi-k2.6",
        provider_name="openai",
        events=MagicMock(),
        budget=BudgetTracker(max_input_tokens=1000, max_output_tokens=1000),
    )


def test_instrumented_provider_forwards_reasoning_effort() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    wrapper.call(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="off",
    )

    kwargs: dict[str, Any] = inner.call.call_args.kwargs
    assert kwargs["reasoning_effort"] == "off"


def test_instrumented_provider_forwards_should_abort() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    def _abort() -> bool:
        return True

    wrapper.call(system="s", messages=[{"role": "user", "content": "hi"}], should_abort=_abort)
    assert inner.call.call_args.kwargs["should_abort"] is _abort


def test_instrumented_provider_defaults_reasoning_effort_to_none() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    wrapper.call(system="s", messages=[{"role": "user", "content": "hi"}])

    assert inner.call.call_args.kwargs["reasoning_effort"] is None
