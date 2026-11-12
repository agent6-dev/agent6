"""Machine help and config text describe the controls the implementation provides."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import machine
from agent6.app.machine import host_preflight
from agent6.config import Config, ConfigError, layer, sandbox_config, surface_config
from agent6.tools import schema
from agent6.ui.cli import main


def test_authoring_guide_attributes_commits_to_the_harness() -> None:
    """Run states have edit and verify tools, not a commit tool."""
    text = " ".join(machine.build_authoring_prompt("Write a machine", attempt=1).split())
    assert "full edit and verify tool set; the harness commits its edits." in text
    assert "commit" not in schema.mode_tools("run").names


def test_pass_env_text_names_explicit_config_layers(tmp_path: pathlib.Path) -> None:
    """The allowlist's description, refusal and docstring all name --config FILE."""
    state = machine.ToolState.model_validate(
        {"kind": "tool", "command": ["true"], "timeout_s": 1, "pass_env": ["TOKEN"], "on": {}}
    )
    refusal = host_preflight.machine_pass_env_refusal(Config(), {"fetch": state})
    assert refusal is not None
    description = surface_config.MachineConfig.model_fields["pass_env"].description or ""
    for text in (description, refusal, host_preflight.machine_pass_env_refusal.__doc__ or ""):
        assert "--config FILE" in text
    config = tmp_path / "operator.toml"
    config.write_text('[machine]\npass_env = ["TOKEN"]\n', encoding="utf-8")
    resolved = layer.load_effective(tmp_path, (config,)).config
    assert host_preflight.machine_pass_env_refusal(resolved, {"fetch": state}) is None


def test_snapshot_comment_agrees_with_the_field_description() -> None:
    """Snapshot guidance names status as the reader and the journal as recovery's source."""
    source = pathlib.Path(surface_config.__file__).read_text(encoding="utf-8")
    assert "recovery reads the latest" not in source
    assert "recovery folds the journal" in source
    text = surface_config.MachineConfig.model_fields["keep_snapshots"].description or ""
    assert "latest read by `machine status`" in text
    assert "journal folded for recovery" in text


def test_surface_model_docstring_includes_telemetry() -> None:
    """The module lists the telemetry table it defines."""
    assert "`[telemetry]`" in (surface_config.__doc__ or "")


@pytest.mark.parametrize("source", ["machine", "explicit"])
def test_preset_refusal_names_only_commands_that_accept_the_flag(
    tmp_path: pathlib.Path, source: str
) -> None:
    """A refused machine or explicit preset points to config, not a nonexistent machine flag."""
    with pytest.raises(ConfigError) as exc:
        if source == "machine":
            layer.load_effective_with_overlay(tmp_path, {"agent6": {"preset": "demo"}})
        else:
            config = tmp_path / "operator.toml"
            config.write_text('[agent6]\npreset = "demo"\n', encoding="utf-8")
            layer.load_effective(tmp_path, (config,))
    assert "set it in the global or repo config" in str(exc.value)
    assert "pass `--preset` to run, resume or plan" in str(exc.value)


def test_machine_test_help_names_recursive_script_discovery(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mock-test glob in help includes nested scripts."""
    with pytest.raises(SystemExit) as exc:
        main(["machine", "test", "--help"])
    assert exc.value.code == 0
    assert "scripts/**/*_test.py" in capsys.readouterr().out


def test_fetch_description_limits_the_mode_restriction_to_read_only_states() -> None:
    """Fetch belongs to run states' tool set, not read-only machine agent states'."""
    description = (
        sandbox_config.SandboxConfig.model_fields["fetch_approved_hosts"].description or ""
    )
    assert 'read-only machine agent states (`mode = "read_only"`)' in description
    assert "fetch" in schema.mode_tools("run").names
    assert "fetch" not in schema.mode_tools("read_only").names
    assert "fetch" not in schema.mode_tools("machine").names
