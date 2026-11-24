# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A hosted web search's fee: the listed rate, the count per billed attempt, the unpriced mark."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import httpx2
import pytest

from agent6 import budget as agent6_budget
from agent6 import kinds
from agent6.app import preflight
from agent6.config import model as config_model
from agent6.config import provider_config
from agent6.models import cache, listing_store, pricing
from agent6.providers import anthropic, provider_types, responses
from tests.unit import test_web_search_messages as messages_reply
from tests.unit import test_web_search_responses as responses_reply

_ROUTE = kinds.ModelRoute("anthropic", "claude-x")
# The cache name of the default fallback price list.
_PRICE_LIST = "openrouter.ai"


def _cfg(worker: dict[str, Any] | None = None, **extra: Any) -> config_model.Config:
    """Return a config whose direct Anthropic worker is priced from the fallback price list."""
    entry = {"backend": "anthropic", "web_search": True} if worker is None else worker
    return config_model.Config.model_validate(
        {
            "providers": {"anthropic": entry},
            "models": {"default": {"model": "anthropic/claude-x"}},
            **extra,
        }
    )


def _list_prices(
    monkeypatch: pytest.MonkeyPatch, cfg: config_model.Config, web_search: object
) -> None:
    """Fetch the fallback price list, with one model and its `pricing.web_search` as given."""
    row: dict[str, Any] = {"prompt": "0.000003", "completion": "0.000015"}
    if web_search is not None:
        row["web_search"] = web_search
    payload = {"data": [{"id": "anthropic/claude-x", "pricing": row}]}

    def fake_request(method: str, url: str, **_kwargs: Any) -> httpx2.Response:
        return httpx2.Response(
            200, request=httpx2.Request(method, url), content=json.dumps(payload)
        )

    monkeypatch.setattr("agent6.providers.transport.http_request", fake_request)
    source = pricing.price_list_source(provider_config.DEFAULT_PRICE_LIST)
    assert cache.fetch_models_live(*source, None)


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def _tracker(cfg: config_model.Config) -> agent6_budget.BudgetTracker:
    return agent6_budget.BudgetTracker(
        providers=cfg.providers, max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1
    )


@pytest.mark.parametrize(
    ("listed", "rate"), [("0.01", 0.01), ("0", None), (None, None), (True, None)]
)
def test_the_listing_gives_the_per_search_rate(
    monkeypatch: pytest.MonkeyPatch, cache_home: pathlib.Path, listed: object, rate: float | None
) -> None:
    """A positive `pricing.web_search` is the rate; a zero or an absent one is no rate."""
    cfg = _cfg()
    _list_prices(monkeypatch, cfg, listed)
    stored = listing_store.read(listing_store.cache_path(_PRICE_LIST))
    assert stored.get("search_pricing") == ({"anthropic/claude-x": rate} if rate else None)
    quote = pricing.lookup_price(_ROUTE, cfg.providers)
    assert quote is not None and quote.web_search_usd == rate
    assert quote.rates.as_list() == [3.0, 15.0]


def test_a_listed_rate_prices_each_search(
    monkeypatch: pytest.MonkeyPatch, cache_home: pathlib.Path
) -> None:
    cfg = _cfg()
    _list_prices(monkeypatch, cfg, "0.01")
    tracker = _tracker(cfg)
    tracker.record(
        route=_ROUTE,
        input_tokens=1000,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        web_searches=2,
    )
    snap = tracker.snapshot()
    assert snap.web_searches == 2
    assert snap.usd_total == pytest.approx(0.003 + 0.02)
    assert not snap.usd_partial and not snap.search_fee_unpriced
    assert "calls=1 searches=2 $0.02" in tracker.format_summary()


def test_an_unpriced_search_marks_the_total_a_lower_bound(
    monkeypatch: pytest.MonkeyPatch, cache_home: pathlib.Path
) -> None:
    """Tokens stay priced; the fee no listing gives is said, and the total reads partial."""
    cfg = _cfg()
    _list_prices(monkeypatch, cfg, None)
    tracker = _tracker(cfg)
    tracker.record(
        route=_ROUTE, input_tokens=1000, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0
    )
    assert not tracker.snapshot().usd_partial
    tracker.record(
        route=_ROUTE,
        input_tokens=1000,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        web_searches=1,
    )
    snap = tracker.snapshot()
    assert (snap.web_searches, snap.usd_partial, snap.search_fee_unpriced) == (1, True, True)
    assert snap.usd_total == pytest.approx(0.006)
    assert tracker.estimate_usd() == (pytest.approx(0.006), True)
    summary = tracker.format_summary()
    assert "calls=2 searches=1 ~$0.0060 (price from openrouter.ai/anthropic/claude-x)" in summary
    assert "(search fee unpriced)" in summary
    assert "cost~$0.0060+ of unlimited (search fee unpriced; figure is a lower bound)" in summary


def test_a_reported_cost_already_holds_the_fee(cache_home: pathlib.Path) -> None:
    tracker = _tracker(_cfg())
    tracker.record(
        route=_ROUTE,
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.5,
        web_searches=1,
    )
    snap = tracker.snapshot()
    assert (snap.web_searches, snap.usd_total, snap.usd_partial) == (1, 0.5, False)


def _ignore(_piece: str) -> None:
    """Take a streamed piece, which opts the call into streaming."""


def _messages_provider(tracker: agent6_budget.BudgetTracker) -> anthropic.AnthropicProvider:
    return anthropic.AnthropicProvider(
        api_key="k", model="claude-x", provider_name="anthropic", budget=tracker
    )


def _serve(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    def fake_stream(method: str, url: str, **_kwargs: Any) -> Any:
        return messages_reply._FakeStream(lines)

    monkeypatch.setattr(httpx2, "stream", fake_stream)


def test_each_attempt_bills_its_own_search_once(
    monkeypatch: pytest.MonkeyPatch, cache_home: pathlib.Path
) -> None:
    """A cut stream's search is counted when it is cut, and the retry's search once more.

    The cut attempt never reports its final usage, so the result block it received counts.
    """
    cfg = _cfg()
    _list_prices(monkeypatch, cfg, "0.01")
    tracker = _tracker(cfg)
    lines = messages_reply._stream_of(messages_reply._CONTENT)
    cut = next(i for i, line in enumerate(lines) if '"index": 3' in line) - 1
    call: dict[str, Any] = {
        "system": "s",
        "messages": [{"role": "user", "content": "q"}],
        "text_delta_callback": _ignore,
        "web_search": True,
    }

    _serve(monkeypatch, lines[:cut])
    with pytest.raises(provider_types.ProviderError, match="no message_stop"):
        _messages_provider(tracker).call(**call)
    assert tracker.snapshot().web_searches == 1

    _serve(monkeypatch, lines)
    _messages_provider(tracker).call(**call)
    snap = tracker.snapshot()
    assert (snap.web_searches, snap.per_model[_ROUTE].calls) == (2, 2)
    fees = snap.usd_total - (2 * 30 * 3.0 + 12 * 15.0) / 1e6
    assert fees == pytest.approx(0.02)


@pytest.mark.parametrize("complete", [True, False])
def test_a_responses_stream_bills_its_distinct_search_calls(
    monkeypatch: pytest.MonkeyPatch, cache_home: pathlib.Path, complete: bool
) -> None:
    """A finished stream and one cut after its search item both count the search once."""
    cfg = _cfg({"backend": "responses", "base_url": "https://api.openai.com/v1"})
    tracker = _tracker(cfg)
    lines = responses_reply._stream_lines()
    _serve(monkeypatch, lines if complete else lines[:6])
    provider = responses.ResponsesProvider(
        api_key="k",
        model="claude-x",
        base_url="https://api.openai.com/v1",
        provider_name="anthropic",
        budget=tracker,
    )
    call: dict[str, Any] = {
        "system": "s",
        "messages": [{"role": "user", "content": "q"}],
        "text_delta_callback": _ignore,
        "web_search": True,
    }
    if complete:
        provider.call(**call)
    else:
        with pytest.raises(
            provider_types.ProviderError, match=r"ended without response\.completed"
        ):
            provider.call(**call)
    snap = tracker.snapshot()
    assert (snap.web_searches, snap.per_model[_ROUTE].calls) == (1, 1)


@pytest.mark.parametrize(
    ("worker", "mode", "listed", "said"),
    [
        (None, "run", None, True),
        (None, "ask", None, True),
        (None, "run", "0.01", False),
        (None, "plan", None, False),
        ({"backend": "anthropic"}, "run", None, False),
        ({"backend": "chatgpt", "web_search": True}, "run", None, False),
    ],
)
def test_the_start_says_once_when_the_search_fee_is_unpriced(
    monkeypatch: pytest.MonkeyPatch,
    cache_home: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    worker: dict[str, Any] | None,
    mode: str,
    listed: str | None,
    said: bool,
) -> None:
    """The notice names the key and the route; a rate, a plan or no offer keeps it silent."""
    cfg = _cfg(worker)
    _list_prices(monkeypatch, cfg, listed)
    preflight.note_unpriced_search(cfg, mode)
    noted = capsys.readouterr().err
    if said:
        assert noted.count("providers.anthropic.web_search: no per-search price") == 1
        assert "'anthropic/claude-x'" in noted and "lower bound" in noted
    else:
        assert noted == ""
