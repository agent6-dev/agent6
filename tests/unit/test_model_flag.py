# SPDX-License-Identifier: Apache-2.0
"""`--model`: the session's route over every config layer, for the role its mode runs.

The picker lists every route the config can run.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6.app import _setup
from agent6.config import Config
from agent6.models import choices

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
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    home = tmp_path / "xdg"
    (home / "config" / "agent6").mkdir(parents=True)
    (home / "config" / "agent6" / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "cache"))
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def test_the_flag_routes_the_modes_role(repo: pathlib.Path) -> None:
    """The flag routes the mode's role.

    Run and ask set the worker, plan the planner; the flag lands after the config layers and a
    preset, so it is what the session runs.
    """
    planned = _setup.load_session_config(repo, None, mode="plan", model="openrouter/m").config
    assert planned.models.planner is not None
    assert (planned.models.planner.provider, planned.models.planner.model) == ("openrouter", "m")
    assert planned.models.worker is not None and planned.models.worker.model == "claude-x"
    asked = _setup.load_session_config(repo, None, mode="ask", model="claude-y").config
    assert asked.models.worker is not None
    assert (asked.models.worker.provider, asked.models.worker.model) == ("anthropic", "claude-y")
    quick = _setup.load_session_config(
        repo, None, mode="run", preset="quick", model="claude-z"
    ).config
    assert quick.models.worker is not None and quick.models.worker.model == "claude-z"


def test_route_choices_are_the_cached_listings_plus_the_configured_routes(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    cache = tmp_path / "xdg" / "cache" / "agent6" / "models"
    cache.mkdir(parents=True)
    (cache / "anthropic.json").write_text(json.dumps({"models": ["claude-a", "claude-x"]}))
    cfg = _setup.load_session_config(repo, None, mode="ask").config
    assert choices.route_choices(cfg) == [
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
    assert choices.route_for(cfg, "plan") == "o/m"
    assert choices.route_for(cfg, "run") == "o/m"
    assert choices.route_for(Config.model_validate({}), "run") == ""


def test_a_hubs_picker_follows_the_preset_and_lists_every_route(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A hub's picker follows the preset and lists every route.

    `default_route` is the mode's role under the preset (a preset that swaps the worker model moves
    the picker), `default_preset` the preset the config selects, `available_routes` every configured
    route; all degrade to nothing on a config error.
    """
    config = tmp_path / "xdg" / "config" / "agent6" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[presets.fast.models.worker]\nmodel = "claude-fast"\n',
        encoding="utf-8",
    )
    assert choices.default_route(repo, None, "run", "") == "anthropic/claude-x"
    assert choices.default_route(repo, None, "run", "fast") == "anthropic/claude-fast"
    assert choices.default_route(repo, None, "plan", "") == "anthropic/claude-x"
    assert choices.available_routes(repo, None) == [
        "anthropic/claude-x",
        "openrouter/moonshotai/kimi-k2.6",
    ]
    assert choices.default_preset(repo, None) == ""
    config.write_text('preset = "fast"\n' + config.read_text(encoding="utf-8"), encoding="utf-8")
    assert choices.default_preset(repo, None) == "fast"
    config.write_text("[models.worker]\nprovider = 1\n", encoding="utf-8")
    assert choices.default_route(repo, None, "run", "") == ""
    assert choices.available_routes(repo, None) == []
    config.write_text("preset = [\n", encoding="utf-8")
    assert choices.default_preset(repo, None) == ""


def test_a_resume_rows_defaults_name_what_a_bare_resume_runs_under(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A resume row's defaults name what a bare resume runs under.

    A preset or model the run set by flag is replayed, `as recorded`; anything else is what the
    current config resolves, the model's under a picked preset; an unreadable manifest names the
    config's.
    """
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

    manifest(harness={"preset": "quick"}, models={"driver": {"provider": "o", "model": "m"}})
    assert choices.resume_defaults(repo, None, run) == (
        "none (config default)",
        "anthropic/claude-x (config default)",
    )
    assert choices.resume_defaults(repo, None, run, preset="fast")[1] == (
        "anthropic/claude-fast (config default)"
    )
    manifest(
        harness={"preset": "fast", "preset_from_flag": True},
        models={"driver": {"provider": "o", "model": "m"}, "driver_from_flag": True},
    )
    assert choices.resume_defaults(repo, None, run, preset="quick") == (
        "fast (as recorded)",
        "o/m (as recorded)",
    )
    manifest(harness={"preset": "fast", "preset_from_flag": True})
    assert choices.resume_defaults(repo, None, run)[1] == "anthropic/claude-fast (config default)"
    (run / "manifest.json").unlink()
    assert choices.resume_defaults(repo, None, run) == (
        "none (config default)",
        "anthropic/claude-x (config default)",
    )


def test_a_refused_flag_route_names_the_flag_not_the_config(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused flag route names the flag, not the config.

    A `--model` typo refuses like a configured typo but says what the operator typed and which
    provider's listing lacks it; the config entry it never wrote is not the remedy.
    """
    from agent6.app import preflight
    from agent6.app import reporter as app_reporter
    from agent6.models import validate

    def _keys_ok(_cfg: object, extra: object = ()) -> None:
        return None

    def _refused(cfg: Config, role: str) -> validate.ModelValidation:
        model = cfg.models.resolve(role).model  # type: ignore[union-attr]
        return validate.ModelValidation(
            unknown=(model,), suggestions={model: ("claude-x",)}, can_validate=True
        )

    monkeypatch.setattr(preflight, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(preflight, "validate_configured_model", _refused)
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    flagged = _setup.load_session_config(repo, None, mode="run", model="claude-y").config
    assert (
        preflight.route_preflight(flagged, "worker", reporter=reporter, model_flag="claude-y")
        is False
    )
    assert said[-1].startswith("REFUSING: --model 'claude-y' is not in anthropic's model listing")
    assert "Closest: claude-x" in said[-1] and "config" not in said[-1]
    configured = _setup.load_session_config(repo, None, mode="run").config
    assert preflight.route_preflight(configured, "worker", reporter=reporter) is False
    assert "models.worker.model 'claude-x'" in said[-1]


def test_a_refused_flag_names_the_modes_provider_when_role_ids_collide(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A refused flag names the mode's provider when role ids collide.

    A planner flag refusal names the planner provider whose listing was checked, even when the
    worker already uses the same model id elsewhere.
    """
    from agent6.app import preflight
    from agent6.app import reporter as app_reporter
    from agent6.models import validate

    config = tmp_path / "xdg" / "config" / "agent6" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[models.planner]\nprovider = "openrouter"\nmodel = "planner-old"\n',
        encoding="utf-8",
    )

    def _keys_ok(_cfg: object) -> None:
        return None

    def _refused(_cfg: Config, _role: str) -> validate.ModelValidation:
        return validate.ModelValidation(unknown=("claude-x",), suggestions={}, can_validate=True)

    monkeypatch.setattr(preflight, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(preflight, "validate_configured_model", _refused)
    said: list[str] = []
    cfg = _setup.load_session_config(repo, None, mode="plan", model="claude-x").config
    assert not preflight.route_preflight(
        cfg,
        "planner",
        reporter=app_reporter.Reporter(out=said.append, err=said.append),
        model_flag="claude-x",
    )
    assert "openrouter's model listing" in said[-1]


def test_a_refused_flag_names_a_provider_head_that_matches_nothing(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused flag names a provider head that matches nothing.

    `--model openrouterr/claude-x` (a mistyped provider) is read as one model id on the role's
    provider; the refusal says so, so the operator sees the typo and not a missing model.
    """
    from agent6.app import preflight
    from agent6.app import reporter as app_reporter
    from agent6.models import validate

    def _keys_ok(_cfg: object, extra: object = ()) -> None:
        return None

    def _refused(cfg: Config, role: str) -> validate.ModelValidation:
        model = cfg.models.resolve(role).model  # type: ignore[union-attr]
        return validate.ModelValidation(unknown=(model,), suggestions={}, can_validate=True)

    monkeypatch.setattr(preflight, "check_provider_keys", _keys_ok)
    monkeypatch.setattr(preflight, "validate_configured_model", _refused)
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    cfg = _setup.load_session_config(repo, None, mode="run", model="openrouterr/claude-x").config
    assert cfg.models.worker is not None and cfg.models.worker.provider == "anthropic"
    assert not preflight.route_preflight(
        cfg, "worker", reporter=reporter, model_flag="openrouterr/claude-x"
    )
    assert "No provider is named 'openrouterr' (configured: anthropic, openrouter)" in said[-1]
    assert "read as a model id on anthropic" in said[-1]


def test_a_fresh_run_records_its_model_flag(repo: pathlib.Path) -> None:
    """A fresh run records its model flag.

    The manifest carries the `--model` a run started with, so a resume without the flag runs on it,
    as a flag-selected preset is replayed.
    """
    from agent6.app import manifest as app_manifest
    from agent6.sessions import layout as sessions_layout
    from agent6.sessions import manifest as sessions_manifest

    cfg = _setup.load_session_config(repo, None, mode="run", model="claude-y").config
    layout = sessions_layout.SessionLayout(state_dir=repo / "state", session_id="run-1")
    layout.session_dir.mkdir(parents=True)
    app_manifest.write_session_manifest(
        layout,
        session_id="run-1",
        user_task="t",
        base_sha="",
        base_branch="",
        run_branch=None,
        cfg=cfg,
        driver_from_flag=True,
    )
    stamp = sessions_manifest.read_manifest(layout.session_dir)
    assert stamp.models.driver_from_flag is True
    assert stamp.models.driver is not None and stamp.models.driver.model == "claude-y"
