# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config_layer (layering, source map, show, fill)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import ConfigError, load_config
from agent6.config_layer import (
    load_effective,
    materialize,
    render_show,
    repo_config_path_for,
)

_GLOBAL = """\
[providers.anthropic]
kind = "anthropic"

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


def test_flag_layer_wins(repo: Path, tmp_path: Path) -> None:
    flag = tmp_path / "flag.toml"
    flag.write_text('[sandbox]\nrun_commands = "no"\n', encoding="utf-8")
    eff = load_effective(repo, flag)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "flag"


def test_overlay_is_highest_layer(repo: Path) -> None:
    from agent6.config_layer import load_effective_with_overlay

    overlay = {"sandbox": {"run_commands": "no"}, "workflow": {"critic": "periodic"}}
    eff = load_effective_with_overlay(repo, overlay)
    # Overlay beats the repo value.
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "machine"
    # Overlay sets a brand-new value.
    assert eff.config.workflow.critic == "periodic"
    assert eff.sources["workflow.critic"] == "machine"
    # Lower layers still read through where the overlay is silent.
    assert eff.config.workflow.verify_command == ("pytest", "-q")


def test_empty_overlay_matches_load_effective(repo: Path) -> None:
    from agent6.config_layer import load_effective_with_overlay

    eff = load_effective_with_overlay(repo, {})
    assert eff.config.sandbox.run_commands == "yes"


def test_overlay_forbids_state_dir(repo: Path) -> None:
    from agent6.config_layer import load_effective_with_overlay

    overlay = {"agent6": {"state_dir": "/other"}}
    with pytest.raises(ConfigError, match="state_dir"):
        load_effective_with_overlay(repo, overlay)


@pytest.mark.parametrize("bad", ["relative/path", "also-relative", "."])
def test_global_state_dir_rejects_relative(repo: Path, bad: str) -> None:
    # The raw pre-model read of the GLOBAL config must reject a non-absolute
    # state_dir (it is the base the per-repo config + run state live under).
    from agent6.config_layer import _global_state_dir  # pyright: ignore[reportPrivateUsage]

    gpath = repo.parent / "g" / "config.toml"
    gpath.write_text(f'[agent6]\nstate_dir = "{bad}"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="state_dir"):
        _global_state_dir()


def test_global_state_dir_accepts_absolute(repo: Path) -> None:
    from agent6.config_layer import _global_state_dir  # pyright: ignore[reportPrivateUsage]

    gpath = repo.parent / "g" / "config.toml"
    gpath.write_text('[agent6]\nstate_dir = "/srv/agent6-state"\n', encoding="utf-8")
    assert _global_state_dir() == "/srv/agent6-state"


def test_deep_merge_replaces_provider_when_kind_changes() -> None:
    # A lower layer's kind-specific keys must not survive a kind change, or they
    # surface as a confusing extra_forbidden error under the new kind.
    from agent6.config_layer import _deep_merge  # pyright: ignore[reportPrivateUsage]

    base = {"providers": {"p": {"kind": "anthropic", "api_key_env": "X"}}}
    override = {"providers": {"p": {"kind": "openai", "base_url": "Y"}}}
    merged = _deep_merge(base, override)
    assert merged["providers"]["p"] == {"kind": "openai", "base_url": "Y"}


def test_deep_merge_still_merges_when_kind_unchanged() -> None:
    from agent6.config_layer import _deep_merge  # pyright: ignore[reportPrivateUsage]

    base = {"providers": {"p": {"kind": "openai", "base_url": "Y", "api_key_env": "X"}}}
    override = {"providers": {"p": {"base_url": "Z"}}}
    merged = _deep_merge(base, override)
    assert merged["providers"]["p"] == {"kind": "openai", "base_url": "Z", "api_key_env": "X"}


def test_materialize_roundtrips(repo: Path, tmp_path: Path) -> None:
    eff = load_effective(repo)
    text = materialize(eff.config)
    out = tmp_path / "full.toml"
    out.write_text(text, encoding="utf-8")
    # The materialized file must be a complete, valid config on its own.
    reloaded = load_config(out)
    assert reloaded.workflow.verify_command == ("pytest", "-q")
    assert reloaded.sandbox.run_commands == "yes"
    assert reloaded.providers["anthropic"].kind == "anthropic"


def test_missing_flag_file_errors(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_effective(repo, tmp_path / "does-not-exist.toml")
