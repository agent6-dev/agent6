# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config.layer (layering, source map, show, fill)."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import errors
from agent6 import paths as paths_mod
from agent6.config import (
    AnthropicProviderEntry,
    ConfigError,
    OpenAIProviderEntry,
    layer,
    load_config,
    write,
)
from agent6.viewmodel import config_view

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
[harness]
verify_command = ["pytest", "-q"]

[sandbox]
run_commands = "yes"
"""


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(_GLOBAL, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    rcfg = paths_mod.repo_config_path(repo_root)  # out of the workspace, under the state base
    rcfg.parent.mkdir(parents=True, exist_ok=True)
    rcfg.write_text(_REPO, encoding="utf-8")
    return repo_root


def test_layering_merges_global_and_repo(repo: pathlib.Path) -> None:
    eff = layer.load_effective(repo)
    cfg = eff.config
    # From global:
    assert cfg.models.worker is not None
    assert cfg.models.worker.model == "claude-sonnet-4-5"
    # From repo:
    assert cfg.harness.verify_command == ("pytest", "-q")
    # Repo overrides global on the same field:
    assert cfg.sandbox.run_commands == "yes"


def test_an_unknown_key_points_at_config_fix(repo: pathlib.Path) -> None:
    """An unknown key points at `config fix`, which drops it.

    A bad value still points at `config set`.
    """
    gcfg = pathlib.Path(repo).parent / "g" / "agent6" / "config.toml"
    gcfg.write_text('[sandbox]\nnonexistent_key = 1\nisolation = "srtict"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        layer.load_effective(repo)
    text = str(exc.value)
    assert "sandbox.nonexistent_key" in text and "fix: agent6 config fix" in text
    assert "fix: agent6 config set sandbox.isolation <value>" in text


def test_source_map_attribution(repo: pathlib.Path) -> None:
    eff = layer.load_effective(repo)
    assert eff.sources["models.worker.model"] == "global"
    assert eff.sources["harness.verify_command"] == "repo"
    assert eff.sources["sandbox.run_commands"] == "repo"  # repo wins
    # Untouched secure default:
    assert eff.sources["git.run_repo_hooks"] == "default"


def test_render_show_marks_overrides(repo: pathlib.Path) -> None:
    eff = layer.load_effective(repo)
    text = config_view.render_show(eff)
    assert "global" in text and "repo" in text
    assert "* = set by a config layer (see the source column)" in text
    # A defaulted field is unmarked; an overridden one is marked.
    assert "* models.worker.model" in text


def test_render_show_json(repo: pathlib.Path) -> None:
    eff = layer.load_effective(repo)
    import json

    data = json.loads(config_view.render_show(eff, as_json=True))
    assert data["harness.verify_command"]["source"] == "repo"


# --- the UI-agnostic config view-model (shared by config show / TUI / web) ---


def _by_key(view: config_view.ConfigView) -> dict[str, config_view.ConfigSetting]:
    return {s.key: s for s in view.settings}


def test_build_config_view_provenance_type_choices(repo: pathlib.Path) -> None:
    settings = _by_key(config_view.build_config_view(layer.load_effective(repo)))
    rc = settings["sandbox.run_commands"]
    assert rc.source == "repo" and rc.modified is True
    # enum field -> a dropdown's worth of choices, typed "choice"
    assert rc.py_type == "choice" and rc.choices is not None and "yes" in rc.choices
    ap = settings["git.run_repo_hooks"]
    assert ap.source == "default" and ap.modified is False
    assert ap.py_type == "bool" and ap.default is False


def test_build_config_view_unset_nested_section_is_typed_table(repo: pathlib.Path) -> None:
    """An unset optional nested section reads as py_type "table", never the pydantic class name."""
    s = _by_key(config_view.build_config_view(layer.load_effective(repo)))["models.reviewer"]
    assert s.value is None and s.source == "default"
    assert s.py_type == "table"


def test_build_config_view_adaptive_resolution(repo: pathlib.Path) -> None:
    view = config_view.build_config_view(
        layer.load_effective(repo), resolved={"context.drop_at_chars": 999_999}
    )
    s = _by_key(view)["context.drop_at_chars"]
    assert s.value is None  # raw: unset -> adaptive
    assert s.effective_value == 999_999
    assert s.is_adaptive is True
    assert s.modified is False  # an adaptive default is not a user modification


def test_render_show_json_is_full_view(repo: pathlib.Path) -> None:
    import json

    data = json.loads(config_view.render_show(layer.load_effective(repo), as_json=True))
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


def test_render_show_text_marks_adaptive(repo: pathlib.Path) -> None:
    text = config_view.render_show(
        layer.load_effective(repo), resolved={"context.drop_at_chars": 471859}
    )
    assert "(adaptive)" in text and "471859" in text


# --- shared edit path (the CLI + TUI/web editors write through this) ---


def test_config_write_keeps_the_edit_when_another_layer_was_already_invalid(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """An edit is rolled back only when it broke a valid config.

    `agent6 connect` saves the API key before writing the provider block, so a rollback on a stale
    value in another layer would strand the key.
    """
    # A pre-existing, unrelated error in the GLOBAL layer.
    (tmp_path / "g" / "agent6" / "config.toml").write_text('[cli]\ninput = "x"\n', encoding="utf-8")

    err = write.set_config_value(repo, "sandbox.run_commands", "no", to_repo=True)

    assert err is None, "a pre-existing error elsewhere must not refuse this edit"
    assert "run_commands" in paths_mod.repo_config_path(repo).read_text(encoding="utf-8")


def test_a_dynamic_leaf_in_an_unselected_preset_is_readable(repo: pathlib.Path) -> None:
    global_path = repo.parent / "g" / "agent6" / "config.toml"
    global_path.write_text(
        global_path.read_text(encoding="utf-8")
        + "\n[presets.demo.models.reviewer]\n"
        + 'provider = "anthropic"\n'
        + 'model = "claude-haiku"\n',
        encoding="utf-8",
    )

    assert layer.effective_leaf(
        layer.load_effective(repo), "presets.demo.models.reviewer.model"
    ) == (
        "claude-haiku",
        "preset demo (global)",
    )


def test_set_then_unset_config_value(repo: pathlib.Path) -> None:
    # repo config starts with run_commands="yes"; global has "ask".
    err = write.set_config_value(repo, "sandbox.run_commands", "no", to_repo=True)
    assert err is None
    eff = layer.load_effective(repo)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "repo"
    # unset removes the repo override -> falls through to the global "ask".
    res = write.unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert res.removed and res.error is None
    assert layer.load_effective(repo).config.sandbox.run_commands == "ask"


def test_unset_reports_whether_anything_was_removed(repo: pathlib.Path) -> None:
    """`config unset` says "nothing to unset" only when nothing was removed."""
    res = write.unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert res.removed and res.error is None
    again = write.unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert not again.removed and again.error is None


def test_unset_refuses_a_shape_the_surgery_cannot_carve(repo: pathlib.Path) -> None:
    """Unset refuses a dotted top-level key as an OperatorError, never a returned string."""
    rcfg = paths_mod.repo_config_path(repo)
    before = 'sandbox.run_commands = "yes"\n'
    rcfg.write_text(before, encoding="utf-8")
    with pytest.raises(errors.OperatorError):
        write.unset_config_value(repo, "sandbox.run_commands", to_repo=True)
    assert rcfg.read_text(encoding="utf-8") == before


def test_set_config_value_invalid_rolls_back(repo: pathlib.Path) -> None:
    err = write.set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True)
    assert err is not None  # invalid enum -> rejected
    # the repo file was rolled back to its prior contents (run_commands="yes").
    assert layer.load_effective(repo).config.sandbox.run_commands == "yes"


def test_set_config_value_rejects_a_value_masked_by_a_higher_layer(repo: pathlib.Path) -> None:
    """An engine writer rejects a value invalid on its own even when a higher layer masks it.

    The repo layer sets sandbox.run_commands="yes", so a bad global enum merges valid; only the
    standalone written-value check catches it, shared with the CLI's guard.
    """
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    before = gpath.read_text(encoding="utf-8")

    err = write.set_config_value(repo, "sandbox.run_commands", "garbage_not_an_enum", to_repo=False)

    assert err is not None and "sandbox.run_commands" in err
    assert gpath.read_text(encoding="utf-8") == before  # the masked bad value rolled back
    assert layer.load_effective(repo).config.sandbox.run_commands == "yes"  # repo layer intact


def test_set_config_value_rejects_a_masked_invalid_provider_base_url(repo: pathlib.Path) -> None:
    """A provider leaf a @field_validator rejects is caught on a masked write.

    The check validates the leaf against the provider model; a bare TypeAdapter of the annotation
    drops the validator.
    """
    paths_mod.repo_config_path(repo).write_text(
        '[providers.x]\napi_format = "openai"\nbase_url = "https://good.example/v1"\n',
        encoding="utf-8",
    )
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    before = gpath.read_text(encoding="utf-8")

    err = write.set_config_value(repo, "providers.x.base_url", "not a url", to_repo=False)

    assert err is not None and "base_url" in err
    assert gpath.read_text(encoding="utf-8") == before  # the masked bad base_url rolled back


def test_written_value_error_catches_an_invalid_container_element(tmp_path: pathlib.Path) -> None:
    """An error under the written key (`sandbox.fetch_hosts.0`) is the written value's own."""
    assert write.written_value_error("sandbox.fetch_hosts", [5], repo_root=tmp_path) is not None
    assert (
        write.written_value_error("providers.x.token_command", [1], repo_root=tmp_path) is not None
    )
    assert (
        write.written_value_error("sandbox.fetch_hosts", ["ok.example"], repo_root=tmp_path) is None
    )
    assert (
        write.written_value_error("providers.x.token_command", ["gcloud"], repo_root=tmp_path)
        is None
    )


def test_a_scalar_written_to_a_list_leaf_names_both_ways_to_write_one(
    tmp_path: pathlib.Path,
) -> None:
    """A scalar written to a list leaf names the array form and `config add`."""
    err = write.written_value_error(
        "harness.verify_command", "python -m pytest", repo_root=tmp_path
    )
    assert err is not None
    assert "expected a list" in err
    assert "config add harness.verify_command" in err


def test_setting_a_section_keeps_its_other_leaves_and_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`config set context '{ ... }'` keeps the table's other leaves and comments."""
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        "[context]\n# my tuning\nkeep_recent_chars = 50000\nsummary_max_tokens = 4096\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = write.set_config_value(
        repo_root, "context", "{ drop_at_chars = 200000, summarise_at_chars = 400000 }"
    )

    assert err is None, err
    text = (gdir / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert "# my tuning" in text
    assert "keep_recent_chars = 50000" in text
    assert "summary_max_tokens = 4096" in text
    # Both halves land under one revalidation, so the spanning rule sees its sibling.
    assert "drop_at_chars = 200000" in text and "summarise_at_chars = 400000" in text


def test_a_dict_typed_leaf_is_replaced_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`providers.<name>.extra_body` is one value, replaced whole."""
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        "[providers.openrouter]\n"
        'api_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        'extra_body = { provider = { sort = "throughput" } }  # prefer fast backends\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = write.set_config_value(repo_root, "providers.openrouter.extra_body", '{ order = ["x"] }')

    assert err is None, err
    text = (gdir / "agent6" / "config.toml").read_text(encoding="utf-8")
    assert 'extra_body = { order = ["x"] }' in text
    assert "sort" not in text, "the old value was merged into, not replaced"
    assert "# prefer fast backends" in text


def test_a_write_that_breaks_the_toml_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A write that breaks the TOML is rolled back even though the revalidation read raises."""
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    before = '[sandbox]\nnetwork = "none"\n'
    (gdir / "agent6" / "config.toml").write_text(before, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    err = write.set_config_value(repo_root, "providers.openai.extra_headers.X Title", "b")

    assert err is not None and "invalid TOML" in err
    assert (gdir / "agent6" / "config.toml").read_text(encoding="utf-8") == before


def test_written_value_error_catches_a_section_wide_rule(tmp_path: pathlib.Path) -> None:
    """A section-wide rule reported at the parent of the written key is the write's own error.

    The standalone dict holds only the written key, so a complaint about its section can only be
    about this write.
    """
    for key, value in (
        ("context.drop_at_chars", 200_000),  # pair: both or neither
        ("git.auto_stash_pop", True),  # needs auto_stash
        ("web.host", "0.0.0.0"),  # non-loopback needs the opt-in
    ):
        assert write.written_value_error(key, value, repo_root=tmp_path) is not None, (
            f"{key} slipped through"
        )
    # A provider filled in over several sets still validates field by field.
    assert (
        write.written_value_error("providers.x.base_url", "https://api.example", repo_root=tmp_path)
        is None
    )


def test_set_config_table_rejects_a_masked_invalid_leaf(repo: pathlib.Path) -> None:
    """set_config_table validates each leaf, not the table dict as one."""
    # The repo layer masks models.worker.effort, so only the per-leaf check catches the bad value.
    paths_mod.repo_config_path(repo).write_text(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude"\nthinking = "off"\n',
        encoding="utf-8",
    )
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    before = gpath.read_text(encoding="utf-8")

    err = write.set_config_table(
        repo,
        "models.worker",
        {"provider": "anthropic", "model": "claude", "effort": "garbage_level"},
        to_repo=False,
    )

    assert err is not None and "effort" in err
    assert gpath.read_text(encoding="utf-8") == before  # the masked bad leaf rolled back


def test_flag_layer_wins(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    flag = tmp_path / "flag.toml"
    flag.write_text('[sandbox]\nrun_commands = "no"\n', encoding="utf-8")
    eff = layer.load_effective(repo, flag)
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "flag"


def test_overlay_is_highest_layer(repo: pathlib.Path) -> None:
    overlay = {"sandbox": {"run_commands": "no"}, "review": {"trigger": "periodic"}}
    eff = layer.load_effective_with_overlay(repo, overlay)
    # Overlay beats the repo value.
    assert eff.config.sandbox.run_commands == "no"
    assert eff.sources["sandbox.run_commands"] == "machine"
    # Overlay sets a brand-new value.
    assert eff.config.review.trigger == "periodic"
    assert eff.sources["review.trigger"] == "machine"
    # Lower layers still read through where the overlay is silent.
    assert eff.config.harness.verify_command == ("pytest", "-q")


def test_empty_overlay_matches_load_effective(repo: pathlib.Path) -> None:
    eff = layer.load_effective_with_overlay(repo, {})
    assert eff.config.sandbox.run_commands == "yes"


def test_a_bad_leaf_from_a_machine_overlay_names_its_layer(repo: pathlib.Path) -> None:
    """A validator error names the layer holding the bad value.

    A machine overlay with no file path is named too.
    """
    with pytest.raises(ConfigError) as exc:
        layer.load_effective_with_overlay(repo, {"sandbox": {"isolation": "bogus"}})
    text = str(exc.value)
    assert "sandbox.isolation" in text
    assert "machine" in text.split("sandbox.isolation", 1)[1]


def test_deep_merge_replaces_provider_when_kind_changes() -> None:
    # A lower layer's kind-specific keys do not survive a kind change.

    base = {"providers": {"p": {"api_format": "anthropic", "api_key_env": "X"}}}
    override = {"providers": {"p": {"api_format": "openai", "base_url": "Y"}}}
    merged = layer._deep_merge(base, override)
    assert merged["providers"]["p"] == {"api_format": "openai", "base_url": "Y"}


def test_deep_merge_still_merges_when_kind_unchanged() -> None:
    base = {"providers": {"p": {"api_format": "openai", "base_url": "Y", "api_key_env": "X"}}}
    override = {"providers": {"p": {"base_url": "Z"}}}
    merged = layer._deep_merge(base, override)
    assert merged["providers"]["p"] == {"api_format": "openai", "base_url": "Z", "api_key_env": "X"}


def test_fix_finds_a_preset_definition_that_is_not_a_table(repo: pathlib.Path) -> None:
    global_path = repo.parent / "g" / "agent6" / "config.toml"
    global_path.write_text('[presets]\ndemo = "quick"\n', encoding="utf-8")

    diagnosis = layer.find_invalid_entries(repo)

    assert [(entry.leaf, entry.value, entry.path) for entry in diagnosis.removable] == [
        ("presets.demo", "quick", global_path)
    ]


def test_fix_finds_an_unknown_table_in_an_unselected_preset(repo: pathlib.Path) -> None:
    global_path = repo.parent / "g" / "agent6" / "config.toml"
    global_path.write_text('[presets.demo.cli]\ninput = "task"\n', encoding="utf-8")

    diagnosis = layer.find_invalid_entries(repo)

    assert [
        (entry.leaf, entry.layer, entry.path, entry.is_table) for entry in diagnosis.removable
    ] == [("presets.demo.cli", "global", global_path, True)]


def test_fix_finds_a_stale_value_masked_by_the_selected_preset(
    repo: pathlib.Path,
) -> None:
    global_path = repo.parent / "g" / "agent6" / "config.toml"
    global_path.write_text('preset = "quick"\n\n[review]\ntrigger = "banana"\n', encoding="utf-8")

    diagnosis = layer.find_invalid_entries(repo)

    assert [(entry.leaf, entry.layer, entry.path) for entry in diagnosis.removable] == [
        ("review.trigger", "global", global_path)
    ]


def test_materialize_roundtrips(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    eff = layer.load_effective(repo)
    text = layer.materialize(eff.config)
    out = tmp_path / "full.toml"
    out.write_text(text, encoding="utf-8")
    # The materialized file must be a complete, valid config on its own.
    reloaded = load_config(out)
    assert reloaded.harness.verify_command == ("pytest", "-q")
    assert reloaded.sandbox.run_commands == "yes"
    assert reloaded.providers["anthropic"].api_format == "anthropic"


def test_materialize_roundtrips_nested_objects_in_arrays(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Every JSON-shaped extra_body value survives materialize then parse.

    Dicts inside arrays included.
    """
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    gpath.write_text(
        gpath.read_text(encoding="utf-8")
        + "\n[providers.gw]\n"
        + 'api_format = "openai"\n'
        + 'base_url = "https://gw.example.com/v1"\n'
        + "[providers.gw.extra_body]\n"
        + 'models = [{name = "a", options = {weight = 2, tags = ["x"]}}, {name = "b"}]\n'
        + "mixed = [1, {flag = true}]\n",
        encoding="utf-8",
    )
    eff = layer.load_effective(repo)
    out = tmp_path / "full.toml"
    out.write_text(layer.materialize(eff.config), encoding="utf-8")
    reloaded = load_config(out)
    gw = reloaded.providers["gw"]
    assert isinstance(gw, OpenAIProviderEntry)
    body = gw.extra_body
    assert body["models"] == [
        {"name": "a", "options": {"weight": 2, "tags": ["x"]}},
        {"name": "b"},
    ]
    assert body["mixed"] == [1, {"flag": True}]


def test_missing_flag_file_errors(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        layer.load_effective(repo, tmp_path / "does-not-exist.toml")


def test_provenance_survives_a_format_changing_provider_replace(repo: pathlib.Path) -> None:
    """Provenance is stamped in the same walk as the merge.

    _deep_merge replaces a provider entry whole when api_format flips between layers, so a separate
    pass would keep the discarded layer's stale source entries.
    """
    gpath = repo.parent / "g" / "agent6" / "config.toml"
    gpath.write_text(
        gpath.read_text(encoding="utf-8")
        + "\n[providers.foo]\n"
        + 'api_format = "openai"\n'
        + 'base_url = "https://x.example/v1"\n'
        + "http_timeout_s = 30.0\n",
        encoding="utf-8",
    )
    rpath = paths_mod.repo_config_path(repo)
    rpath.write_text(
        rpath.read_text(encoding="utf-8") + '\n[providers.foo]\napi_format = "anthropic"\n',
        encoding="utf-8",
    )
    eff = layer.load_effective(repo)
    foo = eff.config.providers["foo"]
    assert foo.api_format == "anthropic"
    assert foo.base_url == "https://api.anthropic.com/v1"  # refilled default
    assert eff.sources["providers.foo.api_format"] == "repo"
    # The refilled defaults are DEFAULTS, not phantom global values.
    assert eff.sources["providers.foo.base_url"] == "default"
    assert eff.sources["providers.foo.http_timeout_s"] == "default"


def test_profile_key_is_rejected_in_flag_and_machine_layers(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The `preset` key is rejected in a --config file and a machine overlay.

    Neither layer can select a preset.
    """
    explicit = tmp_path / "ci.toml"
    explicit.write_text('preset = "ultra"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="preset"):
        layer.load_effective(repo, explicit)
    with pytest.raises(ConfigError, match="preset"):
        layer.load_effective_with_overlay(repo, {"preset": "ultra"})


def test_materialize_quotes_non_bare_keys(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Materialize quotes a provider name with a space or dot, so the header parses."""
    from agent6.config import Config, load_config

    cfg = Config.model_validate(
        {
            "providers": {
                "my provider": {"api_format": "openai", "base_url": "https://x.example/v1"},
                "openrouter.free": {"api_format": "openai", "base_url": "https://o.example/v1"},
            },
            "skills": {"state": {"org.some.skill": "enabled"}},
        }
    )
    out = tmp_path / "materialized.toml"
    out.write_text(layer.materialize(cfg), encoding="utf-8")
    reloaded = load_config(out)
    assert "my provider" in reloaded.providers
    assert "openrouter.free" in reloaded.providers  # not silently re-nested
    assert reloaded.skills.state == {"org.some.skill": "enabled"}


def test_materialize_escapes_control_chars_in_values(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A control char in a config string value serializes to valid TOML."""
    from agent6.config import Config, load_config

    cfg = Config.model_validate({"harness": {"verify_command": ["echo", "a\x01b\nc"]}})
    out = tmp_path / "materialized.toml"
    out.write_text(layer.materialize(cfg), encoding="utf-8")
    reloaded = load_config(out)
    assert list(reloaded.harness.verify_command) == ["echo", "a\x01b\nc"]


def test_concurrent_rollback_does_not_erase_a_valid_write(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole write, revalidate and rollback cycle holds locked_file.

    A concurrent valid write survives.
    """
    import threading
    import time

    from agent6.config import write as write_mod

    a_in_revalidate = threading.Event()
    b_attempted = threading.Event()
    real_load = write_mod.load_effective
    calls = {"n": 0}

    def gated_load(root: pathlib.Path, flag: pathlib.Path | None) -> object:
        calls["n"] += 1
        if calls["n"] == 1:  # A's revalidate: hold the transaction open
            a_in_revalidate.set()
            b_attempted.wait(timeout=5)
            time.sleep(0.4)  # window for B to (old code) land / (fixed) queue
        return real_load(root, flag)

    monkeypatch.setattr(write_mod, "load_effective", gated_load)
    results: dict[str, str | None] = {}

    def writer_a() -> None:
        results["a"] = write.set_config_value(repo, "sandbox.run_commands", "bogus", to_repo=True)

    def writer_b() -> None:
        a_in_revalidate.wait(timeout=5)
        b_attempted.set()
        results["b"] = write.set_config_value(repo, "git.dirty_tree", "stash", to_repo=True)

    ta = threading.Thread(target=writer_a, daemon=True)
    tb = threading.Thread(target=writer_b, daemon=True)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert results["a"] is not None  # the invalid write was rejected
    assert results["b"] is None  # ...without taking B's valid write down with it
    eff = layer.load_effective(repo)
    assert eff.config.git.dirty_tree == "stash"  # B's update survived A's rollback
    assert eff.config.sandbox.run_commands == "yes"  # A rolled back to the prior value


def test_prepare_write_target_hands_back_the_created_state_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A sudo config write on a fresh machine hands back the whole created state base.

    Chowning only the deepest dir would leave the base root-owned.
    """
    import os

    from agent6.config import write as write_mod

    # Through sudo the XDG vars are root's: the base is the real user's, two levels not yet there.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(
        paths_mod,
        "effective_user",
        lambda: paths_mod.RealUser(uid=1234, gid=1234, name="op", home=home, via_sudo=True),
    )
    base = home / ".local" / "state" / "agent6"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "1234")
    chowned: list[pathlib.Path] = []

    def _record(*a: object) -> None:
        chowned.append(pathlib.Path(str(a[0])))

    def _record_at(target: object, _uid: int, _gid: int, **kw: object) -> None:
        chowned.append(pathlib.Path(f"/proc/self/fd/{kw['dir_fd']}").readlink() / str(target))

    monkeypatch.setattr(os, "lchown", _record)
    monkeypatch.setattr(os, "chown", _record_at)
    target = write_mod._prepare_write_target(repo_root, to_repo=True)  # pyright: ignore[reportPrivateUsage]
    assert target.parent.is_dir()
    assert base in chowned  # the created base is handed back...
    assert home / ".local" in chowned  # ...and every created level above it


def test_config_write_hands_the_dir_over_before_writing(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under sudo the config dir is handed over before the write.

    Whether or not the write succeeds.
    """
    from agent6.config import write as write_mod

    handed: list[pathlib.Path] = []
    monkeypatch.setattr(write_mod, "mkdir_for_real_user", handed.append)

    def killed(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt  # stands in for the operator killing the writer

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", killed)
    with pytest.raises(KeyboardInterrupt):
        write.set_config_value(repo, "git.dirty_tree", "stash", to_repo=True)
    assert handed[0] == paths_mod.repo_config_path(repo).parent  # before the write, not after it


def test_config_write_hands_the_file_over_after_a_rejected_edit(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is handed over after a rejected edit too: its rollback republishes a new inode."""
    from agent6.config import write as write_mod

    handed: list[pathlib.Path] = []
    monkeypatch.setattr(write_mod, "chown_to_real_user", handed.append)
    assert (
        write.set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True)
        is not None
    )
    assert paths_mod.repo_config_path(repo) in handed


def test_engine_writers_refuse_a_write_into_an_unparseable_target(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Engine writers refuse a write into an unparseable target with the parse error.

    The CLI gives the same refusal.
    """
    gcfg = tmp_path / "g" / "agent6" / "config.toml"
    gcfg.write_text("[sandbox\nprotect_git = true\n", encoding="utf-8")  # missing ]
    before = gcfg.read_text(encoding="utf-8")

    with pytest.raises(ConfigError):
        write.set_config_value(repo, "sandbox.run_commands", "no")

    assert gcfg.read_text(encoding="utf-8") == before


def test_no_lock_rollback_keeps_the_write_and_says_so(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the lock, a failed revalidation keeps the write and says so.

    A whole-file restore could erase a concurrent writer's just-validated update.
    """
    import agent6.portable as portable_mod

    def _no_lock(_p: pathlib.Path) -> int | None:
        return None

    monkeypatch.setattr(portable_mod, "_acquire_lock", _no_lock)
    err = write.set_config_value(repo, "sandbox.run_commands", "bogus_value", to_repo=True)
    assert err is not None
    assert "kept as written" in err and "lock" in err
    # NOT restored: the invalid value is still in the file for the operator.
    text = paths_mod.repo_config_path(repo).read_text(encoding="utf-8")
    assert 'run_commands = "bogus_value"' in text


def test_an_optional_section_is_written_leaf_by_leaf(repo: pathlib.Path) -> None:
    """An optional `[table]` section (`models.worker`, `harness.metric`) is written leaf by leaf."""
    rcfg = paths_mod.repo_config_path(repo)
    rcfg.write_text(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-5"\n', encoding="utf-8"
    )

    assert (
        write.set_config_value(repo, "models.worker", '{ model = "gpt-y" }', to_repo=True) is None
    )

    text = rcfg.read_text(encoding="utf-8")
    assert "[models.worker]" in text
    assert 'model = "gpt-y"' in text
    assert 'provider = "anthropic"' in text, "a section's other leaves survive"


def test_a_name_keyed_table_is_written_entry_by_entry(repo: pathlib.Path) -> None:
    """`providers` and `mcp.servers` are written entry by entry, never replaced whole."""
    rcfg = paths_mod.repo_config_path(repo)
    rcfg.write_text(
        '[providers.anthropic]\napi_format = "anthropic"\napi_key_env = "A"\n', encoding="utf-8"
    )

    err = write.set_config_value(
        repo,
        "providers",
        '{ ollama = { api_format = "openai", base_url = "http://localhost:11434/v1" } }',
        to_repo=True,
    )

    assert err is None
    providers = layer.load_effective(repo).config.providers
    assert set(providers) >= {"anthropic", "ollama"}
    kept = providers["anthropic"]
    assert isinstance(kept, AnthropicProviderEntry)
    assert kept.api_key_env == "A"


def test_a_table_valued_leaf_replaces_the_block_it_already_has(repo: pathlib.Path) -> None:
    """A dict-typed leaf written inline replaces its own `[table.leaf]` block."""
    rcfg = paths_mod.repo_config_path(repo)
    rcfg.write_text('[skills.state]\nalpha = "enabled"\n', encoding="utf-8")

    assert (
        write.set_config_value(repo, "skills.state", '{ gamma = "always" }', to_repo=True) is None
    )

    text = rcfg.read_text(encoding="utf-8")
    assert "gamma" in text and "alpha" not in text, text
    assert layer.load_effective(repo).config.skills.state == {"gamma": "always"}


def test_an_invalid_value_is_refused_even_where_its_section_was_broken(repo: pathlib.Path) -> None:
    """An invalid value is refused even where a sibling had already broken its section."""
    rcfg = paths_mod.repo_config_path(repo)
    before = '[web]\nhost = "0.0.0.0"\n'  # already invalid: non-loopback, not opted in
    rcfg.write_text(before, encoding="utf-8")

    err = write.set_config_value(repo, "web.port", "abc", to_repo=True)

    assert err is not None and "valid integer" in err
    assert rcfg.read_text(encoding="utf-8") == before


def test_set_config_leaves_refuses_a_headerless_ancestor(
    repo: pathlib.Path,
) -> None:
    """set_config_leaves refuses a leaf under a header-less ancestor as an OperatorError.

    The file is untouched.
    """
    rcfg = paths_mod.repo_config_path(repo)
    before = '[providers]\nanthropic = { api_format = "anthropic" }\n'
    rcfg.write_text(before, encoding="utf-8")

    with pytest.raises(errors.OperatorError, match="not a plain"):
        write.set_config_leaves(
            repo, "providers.anthropic", {"base_url": "https://x/v1"}, to_repo=True
        )

    assert rcfg.read_text(encoding="utf-8") == before  # nothing partially written


def test_set_config_leaves_rolls_back_a_partial_multi_leaf_write(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One revalidate and rollback wraps all the leaf writes.

    A later leaf's error rolls back the earlier ones.
    """
    from agent6.config import write as write_mod

    rcfg = paths_mod.repo_config_path(repo)
    before = '[providers.anthropic]\napi_format = "anthropic"\n'
    rcfg.write_text(before, encoding="utf-8")

    real = write_mod.upsert_toml_leaf
    calls = {"n": 0}

    def _fail_second(path: pathlib.Path, key: str, value: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConfigError("second leaf refused")
        real(path, key, value)

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", _fail_second)

    with pytest.raises(errors.OperatorError, match="second leaf refused"):
        write.set_config_leaves(
            repo,
            "providers.anthropic",
            {"base_url": "https://x/v1", "api_key_env": "KEY"},
            to_repo=True,
        )

    assert rcfg.read_text(encoding="utf-8") == before  # the first leaf's write rolled back


def test_leaves_partial_write_without_the_lock_is_kept_and_says_so(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial multi-leaf write without the lock is kept, and the refusal says so."""
    import agent6.portable as portable_mod
    from agent6.config import write as write_mod

    def _no_lock(_p: pathlib.Path) -> int | None:
        return None

    monkeypatch.setattr(portable_mod, "_acquire_lock", _no_lock)
    rcfg = paths_mod.repo_config_path(repo)
    before = '[providers.anthropic]\napi_format = "anthropic"\n'
    rcfg.write_text(before, encoding="utf-8")
    real = write_mod.upsert_toml_leaf
    calls = {"n": 0}

    def _fail_second(path: pathlib.Path, key: str, value: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConfigError("second leaf refused")
        real(path, key, value)

    monkeypatch.setattr(write_mod, "upsert_toml_leaf", _fail_second)

    with pytest.raises(errors.OperatorError, match="kept as written"):
        write.set_config_leaves(
            repo,
            "providers.anthropic",
            {"base_url": "https://x/v1", "api_key_env": "KEY"},
            to_repo=True,
        )

    assert "base_url" in rcfg.read_text(encoding="utf-8")  # the landed leaf was kept


def test_load_config_wraps_an_unreadable_file(tmp_path: pathlib.Path) -> None:
    """The single-file loader wraps an unreadable file's OSError as its layered sibling does."""
    p = tmp_path / "c.toml"
    p.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    p.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match="cannot be read"):
            load_config(p)
    finally:
        p.chmod(0o600)


def test_provider_members_are_derived_from_the_union() -> None:
    """The provider member tuple is derived from the union, so a new entry type is validated."""
    from typing import get_args

    from agent6.config import ProviderEntry

    declared = get_args(get_args(ProviderEntry)[0])
    assert set(write.PROVIDER_MEMBERS) == set(declared)
    assert len(write.PROVIDER_MEMBERS) == len(declared) >= 2


def test_the_unknown_key_hint_reads_the_repo_root_not_the_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unknown-key hint reads the repo root the write chain holds, not the process cwd."""
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    repo.mkdir()
    elsewhere.mkdir()
    rcfg = paths_mod.repo_config_path(repo)
    rcfg.parent.mkdir(parents=True, exist_ok=True)
    rcfg.write_text('[mcp.servers.notes]\ncommand = ["notes-mcp"]\n', encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    assert "'mcp.servers.notes.command'" in write.unknown_key_error(
        "mcp.servers.notes.comand", repo
    )
