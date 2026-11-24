# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""call_for_text: the guarded drafting call both commit-message drafters share."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent6.providers import provider_calls, provider_types


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


def _prov(text: str) -> provider_calls.Provider:
    class P:
        def call(self, **kw: Any) -> Any:
            return _Resp(text)

    return cast(provider_calls.Provider, P())


def test_returns_the_stripped_reply() -> None:
    assert (
        provider_calls.call_for_text(_prov("  a subject\n"), system="s", user="u", max_tokens=10)
        == "a subject"
    )


def test_empty_reply_is_none() -> None:
    assert provider_calls.call_for_text(_prov("   \n"), system="s", user="u", max_tokens=10) is None


def test_a_failure_keeps_its_reason() -> None:
    """A failed call read as None, so the caller printed "the model returned nothing"."""

    class Boom:
        def call(self, **kw: Any) -> Any:
            raise RuntimeError("provider down")

    class Limit:
        def call(self, **kw: Any) -> Any:
            raise provider_types.ProviderError("You've reached your Fable limit.")

    with pytest.raises(provider_types.ProviderError, match="RuntimeError: provider down"):
        provider_calls.call_for_text(
            cast(provider_calls.Provider, Boom()), system="s", user="u", max_tokens=10
        )
    with pytest.raises(provider_types.ProviderError, match=r"^You've reached your Fable limit\.$"):
        provider_calls.call_for_text(
            cast(provider_calls.Provider, Limit()), system="s", user="u", max_tokens=10
        )
