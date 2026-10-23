# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config show` renders TOML an operator can copy back out."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import paths
from agent6.app import _setup
from agent6.config import Config, _layer_data, layer
from agent6.models import registry
from agent6.ui.cli import main
from agent6.viewmodel import config_view


def test_a_top_level_scalar_is_not_dressed_as_a_table(tmp_path: pathlib.Path) -> None:
    """`preset` is a bare top-level key, not a `[preset]` table.

    `config fill` emits top-level scalars the same way.
    """
    out = config_view.render_show(layer.load_effective(tmp_path, preset="no_review"))

    assert "[preset]" not in out, "a scalar rendered as a TOML table header"
    assert "preset" in out, "the setting itself must still be shown"
    # It belongs above the tables, exactly where TOML requires it.
    assert out.index("preset") < out.index("["), "a top-level scalar must precede every section"


def test_config_presets_reads_the_explicit_config_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config presets` honours `--config FILE` like every other config subcommand."""
    cfg = tmp_path / "custom.toml"
    cfg.write_text('[presets.myfast.sandbox]\nrun_commands = "allow"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(cfg), "config", "presets"]) == 0
    assert "myfast" in capsys.readouterr().out, "presets ignored the explicit config file"


def test_a_filled_config_can_be_used_as_an_explicit_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fill` emits no `preset` selector, so its file loads as an explicit `--config`.

    A preset selects other leaves; once they are materialized the selector would apply twice.
    """
    filled = tmp_path / "filled.toml"
    filled.write_text(layer.materialize(layer.load_effective(tmp_path).config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # The point: this must not raise.
    reloaded = layer.load_effective(tmp_path, (filled,)).config
    assert reloaded.agent6.config_version == 1


def test_config_fill_keeps_the_presets_the_file_defines(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fill` keeps the `[presets.*]` tables the file defines.

    They are stripped before validation, so they are absent from the `Config` the snapshot is
    rendered from.
    """
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    (cfg_home / "agent6" / "config.toml").write_text(
        'preset = "myfast"\n\n[presets.myfast.sandbox]\nrun_commands = "allow"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["config", "fill", "--force"]) == 0

    after = layer.load_effective(tmp_path)
    assert after.config.sandbox.run_commands == "allow", "the preset stopped applying"
    text = (cfg_home / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[presets.myfast" in text, f"config fill deleted the operator's preset:\n{text}"
    # The selector survives and the preset's effect is not baked; the filled leaf is the default.
    assert 'preset = "myfast"' in text
    assert 'run_commands = "ask"' in text, f"the preset's effect was baked in:\n{text}"


@pytest.mark.parametrize("verb", ["get", "show"])
def test_mapping_setting_prints_its_contents(
    verb: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inspecting a mapping leaf shows its entries rather than an opaque placeholder."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "skills.state", '{debugging = "disabled"}']) == 0
    capsys.readouterr()

    assert main(["config", verb, "skills.state"]) == 0
    output = capsys.readouterr().out
    assert "skills.state.debugging" in output
    assert "disabled" in output
    assert "global" in output
    assert "{...}" not in output


def test_descriptions_mode_prints_the_meaning_under_each_row() -> None:
    """`--descriptions` adds each leaf's meaning; the default stays values-only."""
    eff = _layer_data.EffectiveConfig(config=Config(), sources={}, layers=())
    assert "Cap on the metered spend" not in config_view.render_show(eff)
    assert "Cap on the metered spend" in config_view.render_show(eff, descriptions=True)


def test_key_detail_always_carries_the_meaning() -> None:
    """`config show <key>` carries the meaning with no flag."""
    eff = _layer_data.EffectiveConfig(config=Config(), sources={}, layers=())
    detail = config_view.render_key_detail(eff, ["budget.max_usd"])
    assert "meaning: Cap on the metered spend" in detail


def test_key_detail_takes_several_keys_in_the_order_asked() -> None:
    """`config show a b` prints a's leaves then b's.

    A key matching nothing raises KeyError naming it.

    A section prefix expands to its leaves; a leaf named twice prints once.
    """
    eff = _layer_data.EffectiveConfig(config=Config(), sources={}, layers=())
    detail = config_view.render_key_detail(eff, ["sandbox.network", "budget", "sandbox.network"])
    heads = [line.strip() for line in detail.splitlines() if not line.startswith("    ")]
    assert heads[0] == "sandbox.network"
    assert "budget.max_usd" in heads
    assert heads.index("budget.max_usd") > 0
    assert heads.count("sandbox.network") == 1
    with pytest.raises(KeyError, match="nope"):
        config_view.render_key_detail(eff, ["budget.max_usd", "nope"])
    as_json = json.loads(config_view.render_key_detail(eff, ["sandbox.network"], as_json=True))
    assert list(as_json) == ["sandbox.network"]


def _effort_config(tmp_path: pathlib.Path, body: str) -> _layer_data.EffectiveConfig:
    cfg = tmp_path / "config.toml"
    cfg.write_text(body, encoding="utf-8")
    return layer.load_effective(tmp_path, (cfg,))


def test_an_unset_effort_shows_what_the_openai_wire_actually_sends(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset effort shows the `low` the OpenAI wire sends to a reasoning model, not `(unset)`."""
    monkeypatch.delenv("AGENT6_REASONING_EFFORT", raising=False)
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "moonshotai/kimi-k2.6"\n',
    )

    resolved = registry.resolved_adaptive_values(eff.config)

    assert resolved["models.worker.effort"] == "low"
    assert "low" in config_view.render_show(eff, resolved=resolved)


def test_a_model_that_takes_no_reasoning_knob_keeps_the_unset_row(tmp_path: pathlib.Path) -> None:
    """Only a resolution the wire really applies replaces `(unset)`."""
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "qwen/qwen3-coder"\n',
    )

    assert "models.worker.effort" not in registry.resolved_adaptive_values(eff.config)


def test_an_unset_anthropic_effort_resolves_to_off(tmp_path: pathlib.Path) -> None:
    """Anthropic sends no thinking at all when the role leaves effort unset."""
    eff = _effort_config(
        tmp_path,
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-opus-5"\n',
    )

    assert registry.resolved_adaptive_values(eff.config)["models.worker.effort"] == "off"


def test_a_configured_effort_is_not_marked_resolved(tmp_path: pathlib.Path) -> None:
    """The row shows the operator's own value, with its layer, not a default."""
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "moonshotai/kimi-k2.6"\n'
        'effort = "high"\n',
    )

    assert "models.worker.effort" not in registry.resolved_adaptive_values(eff.config)


def test_the_env_override_is_the_value_shown(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENT6_REASONING_EFFORT is the value shown.

    `config show` reads the resolver the request does.
    """
    monkeypatch.setenv("AGENT6_REASONING_EFFORT", "medium")
    eff = _effort_config(
        tmp_path,
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "moonshotai/kimi-k2.6"\n',
    )

    assert registry.resolved_adaptive_values(eff.config)["models.worker.effort"] == "medium"


def test_an_empty_string_default_renders_a_visible_token(tmp_path: pathlib.Path) -> None:
    """An empty string default renders a visible token, not a blank cell."""
    eff = _effort_config(tmp_path, "")
    rows = config_view.render_show(
        eff, resolved=registry.resolved_adaptive_values(eff.config)
    ).splitlines()
    prompt = next(line for line in rows if line.split()[:1] == ["prompt.system_prompt_file"])
    assert "(empty)" in prompt, prompt
    preset = next(line for line in rows if line.split()[:1] == ["preset"])
    assert preset.split()[1:] == ["none", "default"]


def test_the_auto_sandbox_leaves_show_what_this_host_resolves_them_to(
    tmp_path: pathlib.Path,
) -> None:
    """The `auto` sandbox leaves show what this host resolves them to on every surface."""
    from agent6.app import confine as app_confine

    eff = _effort_config(tmp_path, "")
    view = config_view.build_config_view(
        eff, resolved=app_confine.resolved_config_values(eff.config)
    )
    rows = {s.key: s for s in view.settings}
    assert rows["sandbox.isolation"].is_adaptive
    assert rows["sandbox.isolation"].effective_value in ("full", "partial", "none")
    assert rows["sandbox.network"].is_adaptive
    assert "(adaptive)" in config_view.render_show(
        eff, resolved=app_confine.resolved_config_values(eff.config)
    )

    explicit = _effort_config(tmp_path, '[sandbox]\nisolation = "none"\n')
    view = config_view.build_config_view(
        explicit, resolved=app_confine.resolved_config_values(explicit.config)
    )
    assert not {s.key: s for s in view.settings}["sandbox.isolation"].is_adaptive


def test_the_resolved_values_leave_the_sandbox_leaves_auto_without_a_jail_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a jail binary the config view keeps the sandbox leaves at `auto`.

    The probe raises JailBinaryError; a run here refuses, naming the binary.
    """
    from typing import NoReturn

    from agent6.app import confine
    from agent6.sandbox import jail

    def no_binary() -> NoReturn:
        raise jail.JailBinaryError("agent6-jail binary not found")

    monkeypatch.setattr(_setup, "detect_env", no_binary)
    cfg = Config()
    assert (cfg.sandbox.isolation, cfg.sandbox.network) == ("auto", "auto")
    resolved = confine.resolved_config_values(cfg)
    assert "sandbox.isolation" not in resolved and "sandbox.network" not in resolved
    assert resolved == registry.resolved_adaptive_values(cfg)


@pytest.mark.parametrize("entry", [False, True])
@pytest.mark.parametrize("verb", ["get", "show"])
@pytest.mark.parametrize("field", ["extra_body", "extra_headers", "extra_query"])
def test_mapping_credentials_are_masked(
    entry: bool,
    verb: str,
    field: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unmarked mapping fields reveal keys but never credential values."""
    cfg = tmp_path / "credentials.toml"
    cfg.write_text(
        f'[providers.demo]\napi_format = "openai"\n{field} = {{ token = "secret-value" }}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    key = f"providers.demo.{field}" + (".token" if entry else "")
    args = ["--config", str(cfg), "config", verb, key]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert f"providers.demo.{field}.token" in output
    assert "***" in output
    assert "secret-value" not in output
    if verb == "show":
        assert main([*args, "--json"]) == 0
        assert "secret-value" not in capsys.readouterr().out


def test_mapping_entries_show_each_origin(tmp_path: pathlib.Path) -> None:
    """Text and JSON attribute each skill state to the layer that set it."""
    global_config = paths.global_config_path()
    global_config.parent.mkdir(parents=True, exist_ok=True)
    global_config.write_text('[skills.state]\nbeta = "always"\n', encoding="utf-8")
    repo_config = paths.repo_config_path(tmp_path)
    repo_config.parent.mkdir(parents=True, exist_ok=True)
    repo_config.write_text('[skills.state]\nalpha = "enabled"\n', encoding="utf-8")
    eff = layer.load_effective(tmp_path)

    rows = [line.split() for line in config_view.render_show(eff).splitlines()]
    assert ["*", "skills.state.alpha", "enabled", "repo"] in rows
    assert ["*", "skills.state.beta", "always", "global"] in rows
    payload = json.loads(config_view.render_show(eff, as_json=True))
    assert "skills.state" not in payload
    for name, source in (("alpha", "repo"), ("beta", "global")):
        setting = payload[f"skills.state.{name}"]
        assert setting["source"] == source
        assert setting["type"] == "choice"
        assert setting["choices"] == ["enabled", "disabled", "always"]
        assert setting["description"]


def test_empty_mapping_keeps_its_field_row(tmp_path: pathlib.Path) -> None:
    """An empty mapping still appears in text and JSON with its default origin."""
    eff = layer.load_effective(tmp_path)
    rows = [line.split() for line in config_view.render_show(eff).splitlines()]
    assert ["skills.state", "{}", "default"] in rows
    payload = json.loads(config_view.render_show(eff, as_json=True))
    assert payload["skills.state"]["value"] == {}
    assert payload["skills.state"]["source"] == "default"


def test_long_mapping_entry_keeps_table_width() -> None:
    """The unfiltered table truncates long mapping keys instead of widening the key column."""
    cfg = Config.model_validate({"skills": {"state": {"x" * 200: "disabled"}}})
    eff = _layer_data.EffectiveConfig(config=cfg, sources={}, layers=())
    row = next(line for line in config_view.render_show(eff).splitlines() if "skills.state" in line)
    assert "…" in row
    assert "x" * 200 not in row
    assert "disabled" in row
    assert row.endswith("default")
    assert len(row) < 120


@pytest.mark.parametrize("display_safe", [False, True])
def test_json_and_text_mask_tables_inside_lists(display_safe: bool) -> None:
    """Text and JSON use the same masking policy for nested lists of tables."""
    value = [[{"token": "secret-value"}]]
    setting = config_view.ConfigSetting(
        key="example",
        section="example",
        value=value,
        effective_value=value,
        default=value,
        source="default",
        modified=False,
        is_adaptive=False,
        py_type="list",
        choices=None,
        description="",
        display_safe=display_safe,
    )
    expected = value if display_safe else [[{"token": "***"}]]
    assert json.loads(config_view.display_value(setting)) == expected
    payload = config_view._leaf_json(setting)
    for key in ("value", "effective", "default"):
        assert payload[key] == expected


def test_mapping_display_handles_non_json_values() -> None:
    """A display-safe mapping stringifies values JSON cannot encode."""
    assert config_view.format_value({"path": pathlib.Path("example")}, display_safe=True) == (
        '{"path": "example"}'
    )


def test_repeated_config_show_names_each_winning_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each displayed value names the command-line file that last set it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text("[review]\ninterval_turns = 7\n[budget]\nmax_usd = 2\n", encoding="utf-8")
    second.write_text("[budget]\nmax_usd = 3\n", encoding="utf-8")
    assert main(["--config", str(first), "--config", str(second), "config", "show"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any("review.interval_turns" in line and str(first) in line for line in lines)
    assert any("budget.max_usd" in line and str(second) in line for line in lines)
