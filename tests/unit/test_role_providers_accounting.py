from __future__ import annotations

import json
import pathlib
from unittest import mock

import httpx2
import pytest

from agent6 import budget, kinds
from agent6.app import role_providers
from agent6.config import model as config_model
from agent6.providers import provider_types
from agent6.sessions import budget_events


def test_shared_model_id_keeps_each_routes_price(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building a cheaper reviewer cannot reprice the worker's earlier calls."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache = tmp_path / "agent6" / "models"
    cache.mkdir(parents=True)
    for name, price in (("expensive", 10.0), ("cheap", 1.0)):
        (cache / f"{name}.json").write_text(
            json.dumps({"models": ["shared"], "pricing": {"shared": [price, price]}}),
            encoding="utf-8",
        )
    cfg = config_model.Config.model_validate(
        {
            "providers": {
                "expensive": {"backend": "openai", "auth_style": "none"},
                "cheap": {"backend": "openai", "auth_style": "none"},
            },
            "models": {
                "default": {"model": "expensive/shared"},
                "reviewer": {"model": "cheap/shared"},
            },
        }
    )
    tracker = budget.BudgetTracker(max_usd=20, max_tokens_fallback=-1, max_plan_points=-1)
    sink = provider_types.TranscriptSink(
        tmp_path / "transcript.jsonl", objects_dir=tmp_path / "objects"
    )
    response = httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        },
    )
    with mock.patch("agent6.providers.transport.http_post", return_value=response):
        for role in ("worker", "reviewer"):
            provider = role_providers.build_role_provider(
                cfg, role, transcript_sink=sink, budget=tracker
            )
            provider.call(system="s", messages=[])
    assert tracker.estimate_usd() == (11.0, False)
    assert tracker.fraction_remaining() == pytest.approx(0.45)
    assert budget_events.budget_update_fields(tracker)["usd_total"] == 11.0
    assert set(tracker.snapshot().per_model) == {
        kinds.ModelRoute("expensive", "shared"),
        kinds.ModelRoute("cheap", "shared"),
    }
    summary = tracker.format_summary()
    assert "expensive/shared: in=1000000" in summary
    assert "cheap/shared: in=1000000" in summary
    assert "cost~$11.00" in summary


def test_shared_model_id_settles_both_provider_accounts() -> None:
    """A second account's reader cannot replace the first account's settlement."""
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    for provider, used in (("a", 20.0), ("b", 40.0)):
        tracker.record_plan_preflight(provider, budget.PlanUsage.single(used, 300, 2e9))
        reading = budget.PlanUsage.single(used + 3, 300, 2e9)
        tracker.register_plan_reader(provider, lambda reading=reading: reading)
    assert tracker.settle()
    snap = tracker.snapshot()
    assert {name: spend.observed_points for name, spend in snap.plans.items()} == {"a": 3, "b": 3}
    assert not tracker.settle()


def test_the_budget_row_names_the_account_it_read() -> None:
    """The account reading carries the provider entry that answered and every window."""
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    for provider, used in (("a", 20.0), ("b", 40.0)):
        tracker.record_plan_preflight(
            provider, budget.PlanUsage.single(used, 300, 2e9, secondary_used_percent=5.0)
        )
    fields = budget_events.budget_update_fields(tracker)
    account = fields["account"]
    assert account.provider == "b" and account.used_percent == 40.0
    assert [w.name for w in account.windows] == ["primary", "secondary"]
    assert "plan_used_percent" not in fields and "plan_windows" not in fields
