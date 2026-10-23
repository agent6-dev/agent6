from __future__ import annotations

import argparse
import json
import pathlib

import pytest

from agent6 import paths
from agent6.config import ConfigError, _layer_data, io, layer, notices, write
from agent6.ui.cli import cli_main, completers
from agent6.viewmodel import config_view


@pytest.fixture
def mapping_config(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    target = paths.global_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        '[providers.demo]\napi_format = "openai"\n'
        '[providers.demo.extra_body]\n"a.b" = "literal"\n'
        '[providers.demo.extra_body.a]\nb = "nested"\n'
        '[skills.state]\n"org.some.skill" = "enabled"\n',
        encoding="utf-8",
    )
    return target


def test_quoted_mapping_keys_have_distinct_rows_and_sources(
    mapping_config: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A literal dotted name cannot collide with the provenance of a nested path."""
    repo = paths.repo_config_path(tmp_path)
    repo.parent.mkdir(parents=True, exist_ok=True)
    repo.write_text('[providers.demo.extra_body.a]\nb = "repo"\n', encoding="utf-8")
    eff = layer.load_effective(tmp_path)
    literal = 'providers.demo.extra_body."a.b"'
    nested = "providers.demo.extra_body.a"
    leaves = _layer_data.config_leaves(eff.config)
    assert leaves[literal] == "literal"
    assert leaves[nested] == {"b": "repo"}
    assert eff.sources[literal] == "global"
    assert eff.sources[nested] == "repo"
    _, sources = _layer_data.merge_layers(list(eff.layers))
    assert sources[literal] == "global"
    assert sources[nested + ".b"] == "repo"
    assert _layer_data.flatten_leaves(io.read_toml_file(mapping_config))[literal] == "literal"
    assert notices.write_message(tmp_path, [literal], "Saved") == "Saved"
    assert nested in notices.write_message(tmp_path, [nested], "Saved")
    assert cli_main(["config", "show", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert literal in rows and nested in rows
    assert 'skills.state."org.some.skill"' in rows
    view = config_view.build_config_view(eff)
    skill = next(s for s in view.settings if s.key == 'skills.state."org.some.skill"')
    assert skill.choices == ("enabled", "disabled", "always")


@pytest.mark.parametrize("command", ["get", "set", "unset"])
@pytest.mark.parametrize("literal", [False, True])
def test_cli_addresses_only_the_selected_mapping_path(
    mapping_config: pathlib.Path,
    command: str,
    capsys: pytest.CaptureFixture[str],
    *,
    literal: bool,
) -> None:
    """The quoted entry and nested path take independent get, set and unset commands."""
    key = 'providers.demo.extra_body."a.b"' if literal else "providers.demo.extra_body.a.b"
    args = ["config", command, key, *(['"changed"'] if command == "set" else [])]
    assert cli_main(args) == 0, capsys.readouterr()
    if command == "get":
        assert f"{key} = " in capsys.readouterr().out
    expected: dict[str, object] = {"a.b": "literal", "a": {"b": "nested"}}
    selected = "a.b" if literal else "a"
    if command == "set":
        expected[selected] = "changed" if literal else {"b": "changed"}
    elif command == "unset":
        del expected[selected]
    assert io.read_toml_file(mapping_config)["providers"]["demo"]["extra_body"] == expected


@pytest.mark.parametrize(
    "key", ['skills.state."org.some.skill"', """"skills" . state . 'org.some.skill' """]
)
def test_cli_quoted_skill_round_trip(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str,
) -> None:
    """A dotted skill name survives the CLI write, read, show and removal paths."""
    monkeypatch.chdir(tmp_path)
    rendered = 'skills.state."org.some.skill"'
    assert cli_main(["config", "set", key, "enabled"]) == 0, capsys.readouterr()
    capsys.readouterr()
    assert cli_main(["config", "show"]) == 0
    assert rendered in capsys.readouterr().out
    assert cli_main(["config", "get", key]) == 0
    assert f"{rendered} = enabled  [global]" in capsys.readouterr().out
    assert cli_main(["config", "show", key]) == 0
    assert rendered in capsys.readouterr().out
    assert layer.load_effective(tmp_path).config.skills.state == {"org.some.skill": "enabled"}
    assert cli_main(["config", "unset", key]) == 0, capsys.readouterr()
    assert not layer.load_effective(tmp_path).config.skills.state


@pytest.mark.parametrize("command", ["get", "set", "show", "unset"])
def test_completion_offers_quoted_skill_keys(mapping_config: pathlib.Path, command: str) -> None:
    """Completion preserves component boundaries, including ancestor suggestions."""
    args = argparse.Namespace(config_command=command)
    keys = completers._complete_config_keys("skills.state", parsed_args=args)  # pyright: ignore[reportPrivateUsage]
    assert keys == ["skills.state", 'skills.state."org.some.skill"']


@pytest.mark.parametrize("command", ["get", "unset"])
def test_unquoted_skill_path_never_matches_a_literal_entry(
    mapping_config: pathlib.Path, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unquoted nested path cannot read or remove a literal dotted skill name."""
    before = mapping_config.read_text()
    assert cli_main(["config", command, "skills.state.org.some.skill"]) == 2
    assert "unknown config key" in capsys.readouterr().err
    assert mapping_config.read_text() == before


def test_quoted_entry_add_lookup_remove(tmp_path: pathlib.Path) -> None:
    """Entry forms use the plain name while the file and key use TOML quoting."""
    table = next(t for t in write.ENTRY_TABLES if t.path == "skills.state")
    assert write.add_entry(tmp_path, table, "org.some.skill", {"state": "always"}) is None
    key = 'skills.state."org.some.skill"'
    assert write.entry_of(key) == (table, "org.some.skill")
    assert layer.effective_leaf(layer.load_effective(tmp_path), key) is not None
    assert write.remove_entry(tmp_path, table, "org.some.skill") == write.UnsetResult(True)
    assert "org.some.skill" not in paths.global_config_path().read_text()


@pytest.mark.parametrize(
    "bad", ["skills.state.my skill", 'skills.state."unclosed', "a = 1 #", "a = 1\nb"]
)
def test_invalid_config_key_is_named_and_refused(tmp_path: pathlib.Path, bad: str) -> None:
    """A key must be one TOML dotted key, not an assignment or a second document line."""
    with pytest.raises(ConfigError) as error:
        io.upsert_toml_leaf(tmp_path / "config.toml", bad, "enabled")
    assert "config key" in str(error.value) and repr(bad) in str(error.value)
