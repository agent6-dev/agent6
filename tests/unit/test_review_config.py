# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structured reviewers preserve routing and sampling settings at the provider boundary."""

from __future__ import annotations

import json
import pathlib
import tomllib
from typing import Any
from unittest import mock

import httpx2
import pydantic
import pytest

from agent6 import paths
from agent6.app import role_providers
from agent6.config import Config, ReviewConfig, io, layer_data
from agent6.harness.review import gate as review_gate
from agent6.harness.review import panel as review_panel
from agent6.ui.cli import cli_main
from agent6.viewmodel import config_view
from tests import review_lane_env
from tests.unit import test_run_lifecycle_bugfix


def test_max_turns_defaults_to_twenty_in_config_surfaces() -> None:
    """An unset seat budget is visible as 20 model calls."""
    assert layer_data.config_leaves(Config())["review.max_turns"] == 20


@pytest.mark.parametrize("value", [0, -1])
def test_max_turns_requires_a_positive_call_budget(value: int) -> None:
    """A seat needs at least one model call for its verdict."""
    with pytest.raises(pydantic.ValidationError) as caught:
        ReviewConfig.model_validate({"max_turns": value})
    assert any(
        err["loc"] == ("max_turns",) and err["type"] == "greater_than_equal"
        for err in caught.value.errors()
    )


def test_configured_call_budget_reaches_the_seat(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The in-loop panel stops a tool-only seat at the configured budget and names it."""
    repo = test_run_lifecycle_bugfix._setup(tmp_path, monkeypatch)
    config = tmp_path / "review.toml"
    config.write_text(
        "[review]\nmax_turns = 2\n"
        'when = "periodic"\ninterval_turns = 1\n'
        "[harness]\nmax_turns = 1\nverify.infer = false\n"
        '[harness.prompt]\ndecompose = "off"\n',
        encoding="utf-8",
    )
    tool = {"type": "tool_use", "id": "r1", "name": "read_file", "input": {"path": "a.py"}}
    seat = test_run_lifecycle_bugfix._Scripted([("", (tool,))] * 3)
    seat_call = mock.Mock(wraps=seat.call)
    monkeypatch.setattr(seat, "call", seat_call)

    def resolve_seats(_cfg: Config) -> list[review_gate.ReviewSeat]:
        return [review_lane_env.seat("security", "p/m", seat)]

    monkeypatch.setattr(role_providers, "resolve_review_seats", resolve_seats)
    monkeypatch.setattr(review_gate, "run_panel", review_lane_env.run_panel)
    aggregate = mock.Mock(wraps=review_panel.aggregate_verdicts)
    monkeypatch.setattr(review_panel, "aggregate_verdicts", aggregate)
    worker = test_run_lifecycle_bugfix._Scripted(
        [test_run_lifecycle_bugfix._edit("a.py", "x = 2\n")]
    )
    monkeypatch.setattr(
        role_providers, "build_role_provider", test_run_lifecycle_bugfix._use(worker)
    )
    argv = ["run", "--session-id", "review-TURNS1", "Update x"]

    rc = cli_main(["--config", str(config), *argv])

    captured = capsys.readouterr()
    assert rc == 1, captured.err
    assert seat.calls == 2
    (verdict,) = aggregate.call_args.args[0]
    assert verdict.error == "no verdict within 2 turns"
    assert "at most 2 replies" in seat_call.call_args_list[0].kwargs["system"]
    assert "x = 2" in json.dumps(seat_call.call_args_list[1].kwargs["messages"])
    session = paths.state_dir(repo) / "sessions" / "runs" / "review-TURNS1"
    snapshot = (session / "loop_state.json").read_text(encoding="utf-8")
    assert "review inconclusive: all 1 seats abstained" in snapshot


@pytest.mark.parametrize("route", [{"provider": "o"}, {"model": "m"}])
def test_reviewer_refuses_split_or_unqualified_routes(route: dict[str, str]) -> None:
    """A partial route is refused with both field names, not silently inherited."""
    with pytest.raises(
        pydantic.ValidationError, match=r"model.*provider/model|provider and model are one setting"
    ):
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
                "models": {role: {"model": "o/m"}},
                "review": {
                    "block_on": "model_quorum",
                    "min_blocking_models": 2,
                    "reviewers": [
                        {"focus": "security"},
                        {"focus": "tests", "model": "o/m", "effort": "high"},
                    ],
                },
            }
        )


def test_model_quorum_accepts_distinct_resolved_routes() -> None:
    cfg = Config.model_validate(
        {
            "models": {"default": {"model": "o/d"}, "reviewer": {"model": "o/m"}},
            "review": {
                "block_on": "model_quorum",
                "reviewers": [
                    {"focus": "security"},
                    {"focus": "tests", "model": "other/m"},
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
    pinned: bool, overrides: dict[str, object], effort: str, temperature: float
) -> None:
    """A seat sends its entry's settings, off and zero included; a pinned seat inherits too."""
    route = {"model": "o/moonshotai/kimi-k2"}
    cfg = Config.model_validate(
        {
            "providers": {"o": {"backend": "openai", "base_url": "https://x/v1"}},
            "models": {
                "default": {"model": "o/d"},
                "reviewer": {**route, "effort": "medium", "temperature": 0.7},
            },
            "review": {
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

    seats = role_providers.resolve_review_seats(cfg)
    wired = []
    for resolved in seats:
        provider = role_providers.build_role_provider(
            cfg.with_model_settings("worker", resolved.settings),
            "worker",
            transcript_sink=mock.MagicMock(),
            budget=mock.MagicMock(),
        )
        wired.append(
            review_lane_env.seat(
                resolved.focus,
                resolved.model,
                provider,
                temperature=resolved.temperature,
            )
        )
    with mock.patch("agent6.providers.transport.http_post", side_effect=post):
        result = review_lane_env.run_panel(
            wired,
            review_panel.ReviewContext(task="review", diff=""),
            block_on="none",
            min_blocking_models=2,
            panel_id="test",
            max_turns=cfg.review.max_turns,
            dispatch=mock.MagicMock(),
        )
    assert result.n_abstain == 0
    expected_effort = {"enabled": False} if effort == "off" else {"effort": effort}
    assert captured["reasoning"] == expected_effort
    assert captured["temperature"] == temperature


def test_reviewers_are_one_serializable_config_leaf() -> None:
    """Config surfaces can render the roster as TOML and JSON without losing explicit off."""
    cfg = Config.model_validate({"review": {"reviewers": [{"focus": "security", "effort": "off"}]}})
    roster = layer_data.config_leaves(cfg)["review.reviewers"]
    assert json.loads(json.dumps(roster)) == [{"focus": "security", "effort": "off"}]
    assert tomllib.loads("reviewers = " + io.format_toml_value(roster)) == {
        "reviewers": [{"focus": "security", "effort": "off"}]
    }
    view = config_view.build_config_view(
        layer_data.EffectiveConfig(config=cfg, sources={}, layers=())
    )
    setting = next(s for s in view.settings if s.key == "review.reviewers")
    assert config_view.display_value(setting) == '[{ focus = "security", effort = "off" }]'


def test_trigger_on_with_no_reviewers_builds_the_one_seat_roster() -> None:
    cfg = Config.model_validate(
        {
            "providers": {"o": {"backend": "openai", "base_url": "https://x/v1"}},
            "models": {
                "default": {"model": "o/m"},
                "reviewer": {"model": "o/rm"},
            },
            "review": {"when": "finish"},
        }
    )
    seats = role_providers.resolve_review_seats(cfg)
    assert len(seats) == 1
    assert seats[0].model == "o/rm"
    assert seats[0].focus == "security"
