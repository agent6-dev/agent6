# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config_layer (layering, source map, show, fill)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import ConfigError, load_config
from agent6.config_layer import load_effective, materialize, render_show

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
    (repo_root / ".agent6").mkdir(parents=True)
    (repo_root / ".agent6" / "config.toml").write_text(_REPO, encoding="utf-8")
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
