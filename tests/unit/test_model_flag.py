# SPDX-License-Identifier: Apache-2.0
"""`--model`: the session's route over every config layer, for the role its
mode runs; the picker lists every route the config can run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app._setup import load_session_config
from agent6.config import Config
from agent6.models.choices import route_choices, route_for

_GLOBAL = """\
[providers.anthropic]
api_format = "anthropic"

[providers.openrouter]
api_format = "openai"
base_url = "https://x/v1"

[models.worker]
provider = "anthropic"
model = "claude-x"

[models.reviewer]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
"""


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "xdg"
    (home / "config" / "agent6").mkdir(parents=True)
    (home / "config" / "agent6" / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def test_the_flag_routes_the_modes_role(repo: Path) -> None:
    """run and ask set the worker, plan the planner; the flag lands after the
    config layers and a preset, so it is what the session runs."""
    planned = load_session_config(repo, None, mode="plan", model="openrouter/m").config
    assert planned.models.planner is not None
    assert (planned.models.planner.provider, planned.models.planner.model) == ("openrouter", "m")
    assert planned.models.worker is not None and planned.models.worker.model == "claude-x"
    asked = load_session_config(repo, None, mode="ask", model="claude-y").config
    assert asked.models.worker is not None
    assert (asked.models.worker.provider, asked.models.worker.model) == ("anthropic", "claude-y")
    quick = load_session_config(repo, None, mode="run", preset="quick", model="claude-z").config
    assert quick.models.worker is not None and quick.models.worker.model == "claude-z"


def test_route_choices_are_the_cached_listings_plus_the_configured_routes(
    repo: Path, tmp_path: Path
) -> None:
    cache = tmp_path / "xdg" / "cache" / "agent6" / "models"
    cache.mkdir(parents=True)
    (cache / "anthropic.json").write_text(json.dumps({"models": ["claude-a", "claude-x"]}))
    cfg = load_session_config(repo, None, mode="ask").config
    assert route_choices(cfg) == [
        "anthropic/claude-a",
        "anthropic/claude-x",
        "openrouter/moonshotai/kimi-k2.6",
    ]


def test_route_for_applies_the_worker_fallback() -> None:
    cfg = Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": "m"}},
        }
    )
    assert route_for(cfg, "plan") == "o/m"
    assert route_for(cfg, "run") == "o/m"
    assert route_for(Config.model_validate({}), "run") == ""


def test_a_hubs_picker_follows_the_preset_and_lists_every_route(repo: Path, tmp_path: Path) -> None:
    """`default_route` is the mode's role under the preset (a preset that
    swaps the worker model moves the picker), `default_preset` the preset the
    config selects, `available_routes` every configured route; all degrade to
    nothing on a config error."""
    from agent6.models.choices import available_routes, default_preset, default_route

    config = tmp_path / "xdg" / "config" / "agent6" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[presets.fast.models.worker]\nmodel = "claude-fast"\n',
        encoding="utf-8",
    )
    assert default_route(repo, None, "run", "") == "anthropic/claude-x"
    assert default_route(repo, None, "run", "fast") == "anthropic/claude-fast"
    assert default_route(repo, None, "plan", "") == "anthropic/claude-x"
    assert available_routes(repo, None) == ["anthropic/claude-x", "openrouter/moonshotai/kimi-k2.6"]
    assert default_preset(repo, None) == ""
    config.write_text('preset = "fast"\n' + config.read_text(encoding="utf-8"), encoding="utf-8")
    assert default_preset(repo, None) == "fast"
    config.write_text("[models.worker]\nprovider = 1\n", encoding="utf-8")
    assert default_route(repo, None, "run", "") == ""
    assert available_routes(repo, None) == []
    config.write_text("preset = [\n", encoding="utf-8")
    assert default_preset(repo, None) == ""


def test_a_resume_rows_defaults_name_what_a_bare_resume_runs_under(
    repo: Path, tmp_path: Path
) -> None:
    """A preset or model the run set by flag is replayed, `as recorded`;
    anything else is what the config resolves now, the model's under a picked
    preset; an unreadable manifest names the config's."""
    from agent6.models.choices import resume_defaults

    config = tmp_path / "xdg" / "config" / "agent6" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[presets.fast.models.worker]\nmodel = "claude-fast"\n',
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()

    def manifest(**fields: object) -> None:
        body = {"session_id": "run", "mode": "run", **fields}
        (run / "manifest.json").write_text(json.dumps(body), encoding="utf-8")

    manifest(workflow={"preset": "quick"}, models={"driver": {"provider": "o", "model": "m"}})
    assert resume_defaults(repo, None, run) == (
        "none (config default)",
        "anthropic/claude-x (config default)",
    )
    assert resume_defaults(repo, None, run, preset="fast")[1] == (
        "anthropic/claude-fast (config default)"
    )
    manifest(
        workflow={"preset": "fast", "preset_from_flag": True},
        models={"driver": {"provider": "o", "model": "m"}, "driver_from_flag": True},
    )
    assert resume_defaults(repo, None, run, preset="quick") == (
        "fast (as recorded)",
        "o/m (as recorded)",
    )
    manifest(workflow={"preset": "fast", "preset_from_flag": True})
    assert resume_defaults(repo, None, run)[1] == "anthropic/claude-fast (config default)"
    (run / "manifest.json").unlink()
    assert resume_defaults(repo, None, run) == (
        "none (config default)",
        "anthropic/claude-x (config default)",
    )


def test_a_refused_flag_route_names_the_flag_not_the_config(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--model` typo refuses like a configured typo but says what the
    operator typed and which provider's listing lacks it; the config entry
    it never wrote is not the remedy."""
    from agent6.app import preflight
    from agent6.app.reporter import Reporter
    from agent6.models.validate import ModelValidation

    def _keys_ok(_cfg: object, extra: object = ()) -> None:
        return None

    def _refused(cfg: Config, role: str) -> ModelValidation:
        model = cfg.models.resolve(role).model  # type: ignore[union-attr]
        return ModelValidation(
            unknown=(model,), suggestions={model: ("claude-x",)}, can_validate=True
        )

    monkeypatch.setattr(preflight, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(preflight, "validate_configured_model", _refused)
    said: list[str] = []
    reporter = Reporter(out=said.append, err=said.append)
    flagged = load_session_config(repo, None, mode="run", model="claude-y").config
    assert (
        preflight.route_preflight(flagged, "worker", reporter=reporter, model_flag="claude-y")
        is False
    )
    assert said[-1].startswith("REFUSING: --model 'claude-y' is not in anthropic's model listing")
    assert "Closest: claude-x" in said[-1] and "config" not in said[-1]
    configured = load_session_config(repo, None, mode="run").config
    assert preflight.route_preflight(configured, "worker", reporter=reporter) is False
    assert "models.worker.model 'claude-x'" in said[-1]


def test_a_refused_flag_names_the_modes_provider_when_role_ids_collide(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A planner flag refusal names the planner provider whose listing was
    checked, even when the worker already uses the same model id elsewhere."""
    from agent6.app import preflight
    from agent6.app.reporter import Reporter
    from agent6.models.validate import ModelValidation

    config = tmp_path / "xdg" / "config" / "agent6" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[models.planner]\nprovider = "openrouter"\nmodel = "planner-old"\n',
        encoding="utf-8",
    )

    def _keys_ok(_cfg: object) -> None:
        return None

    def _refused(_cfg: Config, _role: str) -> ModelValidation:
        return ModelValidation(unknown=("claude-x",), suggestions={}, can_validate=True)

    monkeypatch.setattr(preflight, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(preflight, "validate_configured_model", _refused)
    said: list[str] = []
    cfg = load_session_config(repo, None, mode="plan", model="claude-x").config
    assert not preflight.route_preflight(
        cfg, "planner", reporter=Reporter(out=said.append, err=said.append), model_flag="claude-x"
    )
    assert "openrouter's model listing" in said[-1]


def test_a_refused_flag_names_a_provider_head_that_matches_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--model openrouterr/claude-x` (a typo'd provider) is read as one model
    id on the role's provider; the refusal says so, so the operator sees the
    typo and not a missing model."""
    from agent6.app import preflight
    from agent6.app.reporter import Reporter
    from agent6.models.validate import ModelValidation

    def _keys_ok(_cfg: object, extra: object = ()) -> None:
        return None

    def _refused(cfg: Config, role: str) -> ModelValidation:
        model = cfg.models.resolve(role).model  # type: ignore[union-attr]
        return ModelValidation(unknown=(model,), suggestions={}, can_validate=True)

    monkeypatch.setattr(preflight, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(preflight, "validate_configured_model", _refused)
    said: list[str] = []
    reporter = Reporter(out=said.append, err=said.append)
    cfg = load_session_config(repo, None, mode="run", model="openrouterr/claude-x").config
    assert cfg.models.worker is not None and cfg.models.worker.provider == "anthropic"
    assert not preflight.route_preflight(
        cfg, "worker", reporter=reporter, model_flag="openrouterr/claude-x"
    )
    assert "No provider is named 'openrouterr' (configured: anthropic, openrouter)" in said[-1]
    assert "read as a model id on anthropic" in said[-1]


def test_a_fresh_run_records_its_model_flag(repo: Path) -> None:
    """The manifest carries the `--model` a run started with, so a resume
    without the flag runs on it (as a flag-selected preset is replayed)."""
    from agent6.app.manifest import write_session_manifest
    from agent6.sessions.layout import SessionLayout
    from agent6.sessions.manifest import read_manifest

    cfg = load_session_config(repo, None, mode="run", model="claude-y").config
    layout = SessionLayout(state_dir=repo / "state", session_id="run-1")
    layout.session_dir.mkdir(parents=True)
    write_session_manifest(
        layout,
        session_id="run-1",
        user_task="t",
        base_sha="",
        base_branch="",
        run_branch=None,
        cfg=cfg,
        driver_from_flag=True,
    )
    stamp = read_manifest(layout.session_dir)
    assert stamp.models.driver_from_flag is True
    assert stamp.models.driver is not None and stamp.models.driver.model == "claude-y"
