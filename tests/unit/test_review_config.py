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
from agent6.app import providers
from agent6.config import Config, ReviewConfig, _layer_data, io
from agent6.harness import _panel, _reviewer
from agent6.ui.cli import cli_main
from agent6.viewmodel import config_view
from tests.unit import test_run_lifecycle_bugfix


def test_repo_max_turns_defaults_to_twenty_in_config_surfaces() -> None:
    """An unset repo seat budget is visible as 20 provider calls."""
    assert _layer_data.config_leaves(Config())["review.repo_max_turns"] == 20


@pytest.mark.parametrize("value", [0, -1])
def test_repo_max_turns_requires_a_positive_call_budget(value: int) -> None:
    """A repo seat needs at least one provider call for its verdict."""
    with pytest.raises(pydantic.ValidationError) as caught:
        ReviewConfig.model_validate({"repo_max_turns": value})
    assert any(
        err["loc"] == ("repo_max_turns",) and err["type"] == "greater_than_equal"
        for err in caught.value.errors()
    )


@pytest.mark.parametrize("command", ["run", "review"])
def test_configured_repo_call_budget_reaches_the_seat(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """Both panel paths stop a tool-only seat at the configured budget and name it."""
    repo = test_run_lifecycle_bugfix._setup(tmp_path, monkeypatch)
    config = tmp_path / "review.toml"
    config.write_text(
        '[review]\nrepo_max_turns = 2\nscope = "repo"\n'
        'trigger = "periodic"\ninterval_turns = 1\n'
        "[harness]\nmax_turns = 1\nverify_infer = false\n"
        '[prompt]\ndecompose = "off"\n',
        encoding="utf-8",
    )
    tool = {"type": "tool_use", "id": "r1", "name": "read_file", "input": {"path": "a.py"}}
    seat = test_run_lifecycle_bugfix._Scripted([("", (tool,))] * 3)
    seat_call = mock.Mock(wraps=seat.call)
    monkeypatch.setattr(seat, "call", seat_call)
    monkeypatch.setattr(providers, "_provider_from_entry", test_run_lifecycle_bugfix._use(seat))
    aggregate = mock.Mock(wraps=_panel.aggregate_verdicts)
    monkeypatch.setattr(_panel, "aggregate_verdicts", aggregate)
    if command == "run":
        worker = test_run_lifecycle_bugfix._Scripted(
            [test_run_lifecycle_bugfix._edit("a.py", "x = 2\n")]
        )
        monkeypatch.setattr(
            providers, "build_role_provider", test_run_lifecycle_bugfix._use(worker)
        )
        argv = ["run", "--session-id", "review-TURNS1", "Update x"]
    else:
        (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
        argv = ["review", "--reviewers", "1", "--scope", "repo"]

    rc = cli_main(["--config", str(config), *argv])

    captured = capsys.readouterr()
    assert rc == 1, captured.err
    assert seat.calls == 2
    (verdict,) = aggregate.call_args.args[0]
    assert verdict.error == "repo: no verdict within 2 calls"
    assert "at most 2 replies" in seat_call.call_args_list[0].kwargs["system"]
    assert "x = 2" in json.dumps(seat_call.call_args_list[1].kwargs["messages"])
    if command == "run":
        session = paths.state_dir(repo) / "sessions" / "runs" / "review-TURNS1"
        snapshot = (session / "loop_state.json").read_text(encoding="utf-8")
        assert "review inconclusive: all 1 seats abstained" in snapshot
    else:
        assert "INCONCLUSIVE" in captured.out
        assert "abstain: repo: no verdict within 2 calls" in captured.err


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
            repo_max_turns=cfg.review.repo_max_turns,
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
    roster = _layer_data.config_leaves(cfg)["review.reviewers"]
    assert json.loads(json.dumps(roster)) == [{"focus": "security", "effort": "off"}]
    assert tomllib.loads("reviewers = " + io.format_toml_value(roster)) == {
        "reviewers": [{"focus": "security", "effort": "off"}]
    }
    view = config_view.build_config_view(
        _layer_data.EffectiveConfig(config=cfg, sources={}, layers=())
    )
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
    assert seats[0].focus == "security"
