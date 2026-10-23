from __future__ import annotations

import pathlib

import pydantic
import pytest

from agent6 import paths
from agent6.config import Config, ConfigError, io, write
from agent6.ui.cli import main


@pytest.mark.parametrize(
    "name", ["bad.name", "bad name", "bad/name", "bad]name", "bad\nname", "\u0430dmin", ""]
)
def test_schema_refuses_names_that_cannot_be_one_bare_key(name: str) -> None:
    """Provider names cannot create extra dotted components or require TOML quoting."""
    with pytest.raises(pydantic.ValidationError) as caught:
        Config.model_validate({"providers": {name: {"api_format": "openai"}}})
    expected = (
        (
            f"provider name {name!r} is not a valid TOML bare key"
            " (use only ASCII letters, digits, '-', '_')."
        )
        if name
        else "provider name is required"
    )
    assert caught.value.errors()[0]["msg"] == f"Value error, {expected}"


@pytest.mark.parametrize(
    "name", ["a", "OpenAI", "123", "-", "_", "team-provider", "team_provider", "two__parts"]
)
def test_bare_provider_names_roundtrip_through_the_editor(
    tmp_path: pathlib.Path, name: str
) -> None:
    """Bare-key names remain valid, including leading digits and double underscores."""
    table = next(t for t in write.ENTRY_TABLES if t.path == "providers")
    assert write.add_entry(tmp_path, table, name, {"api_format": "openai"}) is None
    assert (
        write.set_config_value(tmp_path, f"providers.{name}.base_url", "https://p.test/v1") is None
    )
    config = Config.model_validate(io.read_toml_file(paths.global_config_path()))
    assert config.providers[name].api_format == "openai"


@pytest.mark.parametrize("editor", ["add", "remove", "leaf", "table", "leaves", "whole", "preset"])
def test_provider_writers_use_the_schema_name_refusal(tmp_path: pathlib.Path, editor: str) -> None:
    """Every write path refuses an invalid name before TOML surgery can obscure the reason."""
    name = "bad name"
    expected = (
        "provider name 'bad name' is not a valid TOML bare key"
        " (use only ASCII letters, digits, '-', '_')."
    )
    table = next(t for t in write.ENTRY_TABLES if t.path == "providers")
    try:
        match editor:
            case "add":
                error = write.add_entry(tmp_path, table, name, {"api_format": "openai"})
            case "remove":
                error = write.remove_entry(tmp_path, table, name).error
            case "leaf":
                error = write.set_config_value(
                    tmp_path, f"providers.{io.toml_key(name)}.api_format", "openai"
                )
            case "table":
                error = write.set_config_table(
                    tmp_path, f"providers.{io.toml_key(name)}", {"api_format": "openai"}
                )
            case "leaves":
                error = write.set_config_leaves(
                    tmp_path, f"providers.{io.toml_key(name)}", {"api_format": "openai"}
                )
            case "whole":
                error = write.set_config_value(
                    tmp_path, "providers", '{ "bad name" = { api_format = "openai" } }'
                )
            case _:
                error = write.set_config_value(
                    tmp_path, f"presets.custom.providers.{io.toml_key(name)}.api_format", "openai"
                )
    except ConfigError as exc:
        error = str(exc)
    assert error == expected
    assert not paths.global_config_path().exists()


def test_whole_provider_table_refuses_a_dotted_name(tmp_path: pathlib.Path) -> None:
    """A provider map keeps name boundaries while validating keys before serialization."""
    error = write.set_config_value(
        tmp_path, "providers", '{ "bad.name" = { api_format = "openai" } }'
    )
    assert error == (
        "provider name 'bad.name' is not a valid TOML bare key"
        " (use only ASCII letters, digits, '-', '_')."
    )
    assert not paths.global_config_path().exists()


@pytest.mark.parametrize("command", ["connect", "set"])
def test_cli_provider_names_use_the_schema_refusal(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """Connect and config set refuse the same provider name without touching config or secrets."""
    monkeypatch.chdir(tmp_path)
    argv = (
        ["connect", "bad name"]
        if command == "connect"
        else ["config", "set", 'providers."bad name".api_format', "openai"]
    )
    assert main(argv) == 2
    assert (
        "provider name 'bad name' is not a valid TOML bare key"
        " (use only ASCII letters, digits, '-', '_')."
    ) in capsys.readouterr().err
    assert not paths.global_config_path().exists()
