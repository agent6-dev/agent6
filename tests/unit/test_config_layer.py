# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config.layer (layering, source map, show, fill)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import ConfigError, load_config
from agent6.config.layer import (
    load_effective,
    materialize,
    repo_config_path_for,
    set_config_value,
    unset_config_value,
)
from agent6.viewmodel.config_view import (
    ConfigSetting,
    ConfigView,
    build_config_view,
    render_show,
)

_GLOBAL = """\
[providers.anthropic]
api_format = "anthropic"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[sandbox]
run_commands = "ask"
"""

_REPO = """\
[workflow]
verify_command = ["pytest", "-q"]

[sandbox]
run_commands = "yes"
"""


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    rcfg = repo_config_path_for(repo_root)  # out of the workspace, under the state base
    rcfg.parent.mkdir(parents=True, exist_ok=True)
    rcfg.write_text(_REPO, encoding="utf-8")
    return repo_root


def test_layering_merges_global_and_repo(repo: Path) -> None:
    eff = load_effective(repo)
    cfg = eff.config
    # From global:
    assert cfg.models.worker is not None
    assert cfg.models.worker.model == "claude-sonnet-4-5"
    # From repo:
    assert cfg.workflow.verify_command == ("pytest", "-q")
    # Repo overrides global on the same field:
    assert cfg.sandbox.run_commands == "yes"


def test_source_map_attribution(repo: Path) -> None:
    eff = load_effective(repo)
    assert eff.sources["models.worker.model"] == "global"
    assert eff.sources["workflow.verify_command"] == "repo"
    assert eff.sources["sandbox.run_commands"] == "repo"  # repo wins
    # Untouched secure default:
    assert eff.sources["git.allow_push"] == "default"


def test_allow_urls_last_overlay_wins(repo: Path) -> None:
    # allow_urls is a list field: the most-specific tier that sets it replaces
    # it wholesale (no union across tiers), like every other list field.
    gpath = repo.parent / "g" / "config.toml"
    gpath.write_text(
        gpath.read_text(encoding="utf-8").replace(
            'run_commands = "ask"', 'run_commands = "ask"\nallow_urls = ["g.com"]'
        ),
        encoding="utf-8",
    )
    rpath = repo_config_path_for(repo)
    rpath.write_text(
        rpath.read_text(encoding="utf-8").replace(
            'run_commands = "yes"', 'run_commands = "yes"\nallow_urls = ["r.com"]'
        ),
        encoding="utf-8",
    )
    eff = load_effective(repo)
    assert eff.config.sandbox.allow_urls == ("r.com",)  # repo replaces global
    assert eff.sources["sandbox.allow_urls"] == "repo"


def test_render_show_marks_overrides(repo: Path) -> None:
    eff = load_effective(repo)
    text = render_show(eff)
    assert "global" in text and "repo" in text
    assert "* = overrides the built-in default" in text
    # A defaulted field is unmarked; an overridden one is marked.
    assert "* models.worker.model" in text


def test_render_show_json(repo: Path) -> None:
    eff = load_effective(repo)
    import json

    data = json.loads(render_show(eff, as_json=True))
    assert data["workflow.verify_command"]["source"] == "repo"


# --- the UI-agnostic config view-model (shared by config show / TUI / web) ---


def _by_key(view: ConfigView) -> dict[str, ConfigSetting]:
    return {s.key: s for s in view.settings}


def test_build_config_view_provenance_type_choices(repo: Path) -> None:
    settings = _by_key(build_config_view(load_effective(repo)))
    rc = settings["sandbox.run_commands"]
    assert rc.source == "repo" and rc.modified is True
    # enum field -> a dropdown's worth of choices, typed "choice"
    assert rc.py_type == "choice" and rc.choices is not None and "yes" in rc.choices
    ap = settings["git.allow_push"]
    assert ap.source == "default" and ap.modified is False
    assert ap.py_type == "bool" and ap.default is False


def test_build_config_view_adaptive_resolution(repo: Path) -> None:
    view = build_config_view(load_effective(repo), resolved={"context.drop_at_chars": 999_999})
    s = _by_key(view)["context.drop_at_chars"]
    assert s.value is None  # raw: unset -> adaptive
    assert s.effective_value == 999_999
    assert s.is_adaptive is True
    assert s.modified is False  # an adaptive default is not a user modification


def test_render_show_json_is_full_view(repo: Path) -> None:
    import json

    data = json.loads(render_show(load_effective(repo), as_json=True))
    entry = data["sandbox.run_commands"]
    assert set(entry) >= {
        "value",
        "effective",
        "default",
        "source",
        "modified",
        "adaptive",
        "type",
        "choices",
    }
    assert entry["type"] == "choice" and "yes" in entry["choices"]


def test_render_show_text_marks_adaptive(repo: Path) -> None:
    text = render_show(load_effective(repo), resolved={"context.drop_at_chars": 471859})
    assert "(adaptive)" in text and "471859" in text


# --- shared edit path (the CLI + TUI/web editors write through this) ---


def test_set_then_unset_config_value(repo: Path) -> None:
    # repo config starts with run_commands="yes"; global has "ask".
    err = set_config_value(repo, "sandbox.run_commands", "no", to_repo=True)
    assert err is None
    eff = load_effective(repo)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "repo"
    # unset removes the repo override -> falls through to the global "ask".
    assert unset_config_value(repo, "sandbox.run_commands", to_repo=True) is None
    assert load_effective(repo).config.sandbox.run_commands == "ask"


def test_set_config_value_invalid_rolls_back(repo: Path) -> None:
    err = set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True)
    assert err is not None  # invalid enum -> rejected
    # the repo file was rolled back to its prior contents (run_commands="yes").
    assert load_effective(repo).config.sandbox.run_commands == "yes"


def test_flag_layer_wins(repo: Path, tmp_path: Path) -> None:
    flag = tmp_path / "flag.toml"
    flag.write_text('[sandbox]\nrun_commands = "no"\n', encoding="utf-8")
    eff = load_effective(repo, flag)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "flag"


def test_overlay_is_highest_layer(repo: Path) -> None:
    from agent6.config.layer import load_effective_with_overlay

    overlay = {"sandbox": {"run_commands": "no"}, "review": {"trigger": "periodic"}}
    eff = load_effective_with_overlay(repo, overlay)
    # Overlay beats the repo value.
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "machine"
    # Overlay sets a brand-new value.
    assert eff.config.review.trigger == "periodic"
    assert eff.sources["review.trigger"] == "machine"
    # Lower layers still read through where the overlay is silent.
    assert eff.config.workflow.verify_command == ("pytest", "-q")


def test_empty_overlay_matches_load_effective(repo: Path) -> None:
    from agent6.config.layer import load_effective_with_overlay

    eff = load_effective_with_overlay(repo, {})
    assert eff.config.sandbox.run_commands == "yes"


def test_overlay_forbids_state_dir(repo: Path) -> None:
    from agent6.config.layer import load_effective_with_overlay

    overlay = {"agent6": {"state_dir": "/other"}}
    with pytest.raises(ConfigError, match="state_dir"):
        load_effective_with_overlay(repo, overlay)


@pytest.mark.parametrize("bad", ["relative/path", "also-relative", "."])
def test_global_state_dir_rejects_relative(repo: Path, bad: str) -> None:
    # The raw pre-model read of the GLOBAL config must reject a non-absolute
    # state_dir (it is the base the per-repo config + run state live under).
    from agent6.config.layer import _global_state_dir  # pyright: ignore[reportPrivateUsage]

    gpath = repo.parent / "g" / "config.toml"
    gpath.write_text(f'[agent6]\nstate_dir = "{bad}"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="state_dir"):
        _global_state_dir()


def test_global_state_dir_accepts_absolute(repo: Path) -> None:
    from agent6.config.layer import _global_state_dir  # pyright: ignore[reportPrivateUsage]

    gpath = repo.parent / "g" / "config.toml"
    gpath.write_text('[agent6]\nstate_dir = "/srv/agent6-state"\n', encoding="utf-8")
    assert _global_state_dir() == "/srv/agent6-state"


def test_deep_merge_replaces_provider_when_kind_changes() -> None:
    # A lower layer's kind-specific keys must not survive a kind change, or they
    # surface as a confusing extra_forbidden error under the new kind.
    from agent6.config.layer import _deep_merge  # pyright: ignore[reportPrivateUsage]

    base = {"providers": {"p": {"api_format": "anthropic", "api_key_env": "X"}}}
    override = {"providers": {"p": {"api_format": "openai", "base_url": "Y"}}}
    merged = _deep_merge(base, override)
    assert merged["providers"]["p"] == {"api_format": "openai", "base_url": "Y"}


def test_deep_merge_still_merges_when_kind_unchanged() -> None:
    from agent6.config.layer import _deep_merge  # pyright: ignore[reportPrivateUsage]

    base = {"providers": {"p": {"api_format": "openai", "base_url": "Y", "api_key_env": "X"}}}
    override = {"providers": {"p": {"base_url": "Z"}}}
    merged = _deep_merge(base, override)
    assert merged["providers"]["p"] == {"api_format": "openai", "base_url": "Z", "api_key_env": "X"}


def test_materialize_roundtrips(repo: Path, tmp_path: Path) -> None:
    eff = load_effective(repo)
    text = materialize(eff.config)
    out = tmp_path / "full.toml"
    out.write_text(text, encoding="utf-8")
    # The materialized file must be a complete, valid config on its own.
    reloaded = load_config(out)
    assert reloaded.workflow.verify_command == ("pytest", "-q")
    assert reloaded.sandbox.run_commands == "yes"
    assert reloaded.providers["anthropic"].api_format == "anthropic"


def test_missing_flag_file_errors(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_effective(repo, tmp_path / "does-not-exist.toml")
