# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the compare judge (fake providers, no network)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent6.providers import Provider, ProviderError
from agent6.workflows.judge import (
    _DIFF_CAP,  # pyright: ignore[reportPrivateUsage]
    CandidateBrief,
    JudgeError,
    _build_user_message,  # pyright: ignore[reportPrivateUsage]
    compare,
    mechanical_ranking,
)

_WELL_FORMED = '{"ranking": ["run-a", "run-b"], "rationale": "a passed verify, b did not"}'
_UNKNOWN_RUN_ID = '{"ranking": ["run-a", "run-x"], "rationale": "oops"}'
_MISSING_RUN_ID = '{"ranking": ["run-a"], "rationale": "oops"}'


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeProvider:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def call(self, **kw: Any) -> Any:
        self.calls += 1
        return _Resp(self._texts.pop(0))


class _ErrProvider:
    def call(self, **kw: Any) -> Any:
        raise ProviderError("boom")


def _prov(*texts: str) -> Provider:
    return cast(Provider, _FakeProvider(list(texts)))


def _candidates() -> list[CandidateBrief]:
    return [
        CandidateBrief(
            run_id="run-a", task="add auth", diff="diff a", verify_ok=True, cost_usd=0.10
        ),
        CandidateBrief(
            run_id="run-b", task="add auth", diff="diff b", verify_ok=False, cost_usd=0.05
        ),
    ]


def test_compare_parses_well_formed_verdict() -> None:
    v = compare(_prov(_WELL_FORMED), "m1", _candidates())
    assert v.ranking == ("run-a", "run-b")
    assert "verify" in v.rationale


def test_compare_parses_fenced_json_with_prose() -> None:
    text = f"Here is my ranking:\n```json\n{_WELL_FORMED}\n```\nDone."
    v = compare(_prov(text), "m1", _candidates())
    assert v.ranking == ("run-a", "run-b")


def test_compare_retries_once_then_succeeds() -> None:
    provider = _FakeProvider(["not json at all", _WELL_FORMED])
    v = compare(cast(Provider, provider), "m1", _candidates())
    assert v.ranking == ("run-a", "run-b")
    assert provider.calls == 2


def test_compare_raises_after_two_malformed_attempts() -> None:
    provider = _FakeProvider(["nope", "still nope"])
    with pytest.raises(JudgeError):
        compare(cast(Provider, provider), "m1", _candidates())
    assert provider.calls == 2


def test_compare_raises_when_ranking_names_unknown_run_id() -> None:
    provider = _FakeProvider([_UNKNOWN_RUN_ID, _UNKNOWN_RUN_ID])
    with pytest.raises(JudgeError):
        compare(cast(Provider, provider), "m1", _candidates())


def test_compare_raises_when_ranking_missing_a_run_id() -> None:
    provider = _FakeProvider([_MISSING_RUN_ID, _MISSING_RUN_ID])
    with pytest.raises(JudgeError):
        compare(cast(Provider, provider), "m1", _candidates())


def test_compare_retries_on_provider_error_then_succeeds() -> None:
    provider = _FakeProvider([_WELL_FORMED])  # only consumed on the 2nd attempt

    class _FlakyOnceProvider:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, **kw: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("boom")
            return provider.call(**kw)

    flaky = _FlakyOnceProvider()
    v = compare(cast(Provider, flaky), "m1", _candidates())
    assert v.ranking == ("run-a", "run-b") and flaky.calls == 2


def test_compare_raises_after_repeated_provider_errors() -> None:
    with pytest.raises(JudgeError):
        compare(cast(Provider, _ErrProvider()), "m1", _candidates())


def test_compare_raises_on_empty_candidates() -> None:
    with pytest.raises(JudgeError):
        compare(_prov(_WELL_FORMED), "m1", [])


def test_mechanical_ranking_orders_verify_pass_first_then_cost() -> None:
    candidates = [
        CandidateBrief(run_id="fail", task="t", diff="", verify_ok=False, cost_usd=0.01),
        CandidateBrief(run_id="pass-expensive", task="t", diff="", verify_ok=True, cost_usd=1.0),
        CandidateBrief(run_id="pass-cheap", task="t", diff="", verify_ok=True, cost_usd=0.1),
        CandidateBrief(run_id="none", task="t", diff="", verify_ok=None, cost_usd=0.02),
    ]
    assert mechanical_ranking(candidates) == (
        "pass-cheap",
        "pass-expensive",
        "fail",
        "none",
    )


def test_build_user_message_marks_only_truncated_diffs() -> None:
    """A diff over the cap is shown truncated with a visible marker (the prompt
    says to read every diff); an under-cap diff is shown whole, no marker."""
    big = "x" * (_DIFF_CAP + 10_000)
    small = "y" * 10
    msg = _build_user_message(
        [
            CandidateBrief(run_id="big", task="t", diff=big, verify_ok=True, cost_usd=0.1),
            CandidateBrief(run_id="small", task="t", diff=small, verify_ok=True, cost_usd=0.1),
        ]
    )
    assert msg.count("[diff truncated]") == 1  # only the oversized diff is marked
    assert ("x" * _DIFF_CAP + "\n[diff truncated]") in msg
    assert "x" * (_DIFF_CAP + 1) not in msg  # the overflow bytes are not shown


def test_mechanical_ranking_stable_within_ties() -> None:
    candidates = [
        CandidateBrief(run_id="first", task="t", diff="", verify_ok=True, cost_usd=0.1),
        CandidateBrief(run_id="second", task="t", diff="", verify_ok=True, cost_usd=0.1),
    ]
    assert mechanical_ranking(candidates) == ("first", "second")
