# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structured reviewers preserve routing and sampling settings at the provider boundary."""

from __future__ import annotations

import json
import tomllib
from typing import Any
from unittest import mock

import httpx2
import pydantic
import pytest

from agent6.app import providers
from agent6.config import Config, ReviewConfig, io, layer
from agent6.harness import _panel, _reviewer
from agent6.viewmodel import config_view


@pytest.mark.parametrize("route", [{"provider": "o"}, {"model": "m"}])
def test_reviewer_requires_provider_and_model_together(route: dict[str, str]) -> None:
    """A partial route is refused with both field names, not silently inherited."""
    with pytest.raises(pydantic.ValidationError, match=r"provider.*model.*together"):
        ReviewConfig.model_validate({"reviewers": [{"focus": "security", **route}]})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seats", ["security"]),
        ("decision", "advisory"),
        ("tier", "diff"),
        ("period", 10),
        ("budget_fraction", 0.25),
        ("quorum", 2),
    ],
)
def test_old_review_keys_are_unknown(key: str, value: object) -> None:
    with pytest.raises(pydantic.ValidationError) as caught:
        Config.model_validate({"review": {key: value}})
    assert any(
        err["loc"] == ("review", key) and err["type"] == "extra_forbidden"
        for err in caught.value.errors()
    )


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_model_quorum_checks_resolved_routes(role: str) -> None:
    """An inherited route and a pinned copy count once even with different effort."""
    with pytest.raises(pydantic.ValidationError, match=r"min_blocking_models=2.*distinct"):
        Config.model_validate(
            {
                "models": {role: {"provider": "o", "model": "m"}},
                "review": {
                    "block_on": "model_quorum",
                    "min_blocking_models": 2,
                    "reviewers": [
                        {"focus": "security"},
                        {"focus": "tests", "provider": "o", "model": "m", "effort": "high"},
                    ],
                },
            }
        )


def test_model_quorum_accepts_distinct_resolved_routes() -> None:
    cfg = Config.model_validate(
        {
            "models": {"reviewer": {"provider": "o", "model": "m"}},
            "review": {
                "block_on": "model_quorum",
                "reviewers": [
                    {"focus": "security"},
                    {"focus": "tests", "provider": "other", "model": "m"},
                ],
            },
        }
    )
    assert cfg.review.min_blocking_models == 2


def test_model_quorum_rejects_default_single_reviewer() -> None:
    with pytest.raises(pydantic.ValidationError, match="distinct"):
        Config.model_validate({"review": {"block_on": "model_quorum"}})


@pytest.mark.parametrize("entry", ["security", {"focus": " "}, {"focus": "tests", "unknown": True}])
def test_reviewers_refuse_strings_blank_focus_and_unknown_fields(entry: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        ReviewConfig.model_validate({"reviewers": [entry]})


@pytest.mark.parametrize("scope", ["diff", "repo"])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    ("overrides", "effort", "temperature"),
    [
        ({"effort": "high", "temperature": 0.4}, "high", 0.4),
        ({"effort": "off", "temperature": 0.0}, "off", 0.0),
        ({}, "medium", 0.7),
    ],
)
def test_reviewer_sampling_reaches_provider_call(
    scope: str, pinned: bool, overrides: dict[str, object], effort: str, temperature: float
) -> None:
    """Both review scopes send per-entry settings, including off and zero, to the provider."""
    route = {"provider": "o", "model": "moonshotai/kimi-k2"}
    cfg = Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"reviewer": {**route, "effort": "medium", "temperature": 0.7}},
            "review": {
                "scope": scope,
                "reviewers": [{"focus": "security", **(route if pinned else {}), **overrides}],
            },
        }
    )
    captured: dict[str, Any] = {}

    def post(*_args: Any, **kwargs: Any) -> httpx2.Response:
        captured.update(json.loads(kwargs["content"]))
        return httpx2.Response(
            200,
            request=httpx2.Request("POST", "https://x/v1/chat/completions"),
            json={
                "choices": [
                    {
                        "message": {"content": '{"verdict":"pass","findings":[]}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            },
        )

    seats = providers.build_review_seats(
        cfg, transcript_sink=mock.MagicMock(), budget=mock.MagicMock(), n=1
    )
    with mock.patch("agent6.providers._transport.http_post", side_effect=post):
        result = _reviewer.run_panel(
            seats,
            _panel.ReviewContext(task="review", diff=""),
            block_on="none",
            min_blocking_models=2,
            panel_id="test",
            tools=[],
            dispatch=mock.MagicMock(),
        )
    assert result.n_abstain == 0
    expected_effort = {"enabled": False} if effort == "off" else {"effort": effort}
    # Pinned routes use RoleModel defaults unless the entry sets sampling explicitly.
    if pinned and not overrides:
        expected_effort, temperature = {"effort": "low"}, 0.0
    assert captured["reasoning"] == expected_effort
    assert captured["temperature"] == temperature


def test_reviewers_are_one_serializable_config_leaf() -> None:
    """Config surfaces can render the roster as TOML and JSON without losing explicit off."""
    cfg = Config.model_validate({"review": {"reviewers": [{"focus": "security", "effort": "off"}]}})
    roster = layer.config_leaves(cfg)["review.reviewers"]
    assert json.loads(json.dumps(roster)) == [{"focus": "security", "effort": "off"}]
    assert tomllib.loads("reviewers = " + io.format_toml_value(roster)) == {
        "reviewers": [{"focus": "security", "effort": "off"}]
    }
    view = config_view.build_config_view(layer.EffectiveConfig(config=cfg, sources={}, layers=()))
    setting = next(s for s in view.settings if s.key == "review.reviewers")
    assert config_view.display_value(setting) == '[{"focus": "security", "effort": "off"}]'


def test_trigger_on_with_no_reviewers_builds_the_one_seat_roster() -> None:
    cfg = Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {
                "worker": {"provider": "o", "model": "m"},
                "reviewer": {"provider": "o", "model": "rm"},
            },
            "review": {"trigger": "before_finish"},
        }
    )
    seats = providers.build_review_seats(
        cfg, transcript_sink=mock.MagicMock(), budget=mock.MagicMock(), n=1
    )
    assert len(seats) == 1
    assert seats[0].model == "o/rm"
    assert seats[0].persona == "security"
