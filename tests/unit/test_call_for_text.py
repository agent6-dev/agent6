# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""call_for_text: the guarded drafting call both commit-message drafters share."""

from __future__ import annotations

from typing import Any, cast

from agent6.providers import Provider, call_for_text


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


def _prov(text: str) -> Provider:
    class P:
        def call(self, **kw: Any) -> Any:
            return _Resp(text)

    return cast(Provider, P())


def test_returns_the_stripped_reply() -> None:
    assert call_for_text(_prov("  a subject\n"), system="s", user="u", max_tokens=10) == "a subject"


def test_empty_reply_is_none() -> None:
    assert call_for_text(_prov("   \n"), system="s", user="u", max_tokens=10) is None


def test_any_failure_is_none() -> None:
    class Boom:
        def call(self, **kw: Any) -> Any:
            raise RuntimeError("provider down")

    assert call_for_text(cast(Provider, Boom()), system="s", user="u", max_tokens=10) is None
