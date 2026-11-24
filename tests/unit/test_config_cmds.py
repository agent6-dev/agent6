# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`config set/add/remove --machine` re-validates the whole machine spec."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.config import ConfigError
from agent6.config import layer as config_layer
from agent6.config import model as config_model
from agent6.models import choices
from agent6.ui.cli import config_cmds as cc


def _noop_overlay(*_a: object, **_k: object) -> None:
    # load_effective_with_overlay is stubbed to isolate the machine-spec validation.
    return None


def test_extra_body_value_completer_offers_routing_presets() -> None:
    # TAB after `config set providers.<name>.extra_body` suggests the routing presets by suffix.
    import argparse

    from agent6.ui.cli import completers as cli_completers

    args = argparse.Namespace(key="providers.openrouter.extra_body")
    out = cli_completers._complete_config_values("", args)
    assert '{ provider = { sort = "throughput" } }' in out
    # a non-extra_body key is unaffected
    enum_args = argparse.Namespace(key="sandbox.isolation")
    assert cli_completers._complete_config_values("", enum_args) == [
        "auto",
        "full",
        "partial",
        "none",
        "container",
    ]


def test_profile_value_completer_offers_profile_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # TAB after `config set preset` offers built-ins plus user [presets.*] tables.
    import argparse

    from agent6.ui.cli import completers as cli_completers

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text("[presets.myteam.review]\npanel_size = 2\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(key="agent6.preset")
    out = cli_completers._complete_config_values("", args)
    assert "review" in out and "myteam" in out
    assert cli_completers._complete_config_values("re", args) == ["review"]


def test_config_key_completer_offers_user_profile_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # `config set presets.<TAB>` completes user presets only; a built-in name would replace it.
    from agent6.ui.cli import completers as cli_completers

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text("[presets.myteam.review]\npanel_size = 2\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)
    out = cli_completers._complete_config_keys("presets.")
    assert any(k.startswith("presets.myteam.review.") for k in out)
    assert not any(k.startswith("presets.review") for k in out)
    # the selector is offered under its own table, never inside a preset
    assert "agent6.preset" in cli_completers._complete_config_keys("agent6.")
    assert not any(".agent6." in k for k in out)
    # a bare TAB (empty prefix) is not flooded with the generated paths
    assert not any(k.startswith("presets.") for k in cli_completers._complete_config_keys(""))


@pytest.mark.parametrize("command", ["get", "set", "show", "unset"])
def test_mapping_key_completion_offers_the_table_and_its_entries(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """A populated mapping keeps whole-table completion alongside its entry keys."""
    import argparse

    from agent6.config import write
    from agent6.ui.cli import completers

    monkeypatch.chdir(tmp_path)
    assert write.set_config_value(tmp_path, "skills.state.alpha", '"enabled"') is None
    args = argparse.Namespace(config_command=command)
    keys = completers._complete_config_keys("skills.state", parsed_args=args)
    assert keys == ["skills.state", "skills.state.alpha"]


def test_parallel_models_completer_completes_after_last_comma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TAB on `run --parallel` completes routes, only the entry after the last comma.
    from agent6.config import Config
    from agent6.ui.cli import completers

    class _Eff:
        config = Config()

    def _eff(*_a: object, **_k: object) -> _Eff:
        return _Eff()

    def _routes(_cfg: object) -> list[str]:
        return ["s/gpt-sibling", "w/gpt-5", "w/gpt-5-mini", "w/opus"]

    monkeypatch.setattr(config_layer, "load_effective", _eff)
    monkeypatch.setattr(choices, "route_choices", _routes)
    assert completers._complete_parallel_models("w/gpt") == [
        "w/gpt-5",
        "w/gpt-5-mini",
    ]
    assert completers._complete_parallel_models("w/opus,w/gpt") == [
        "w/opus,w/gpt-5",
        "w/opus,w/gpt-5-mini",
    ]


_GOOD = (
    'machine = "m"\nversion = 1\ninitial = "s"\n'
    "[budget]\nmax_usd = 1.0\nmax_transitions = 10\n"
    '[states.s]\nkind = "terminal"\nstatus = "ok"\nreason = "done"\n'
)
# Same machine but with an unknown state kind -> a complete-but-invalid spec.
_BAD = (
    'machine = "m"\nversion = 1\ninitial = "s"\n'
    "[budget]\nmax_usd = 1.0\nmax_transitions = 10\n"
    '[states.s]\nkind = "bogus"\n'
)


def test_config_set_names_the_inline_table_a_leaf_lives_in(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config set` on a leaf inside an inline table refuses naming the value that owns the leaf.

    The leaf surgery knows only [table] headers, so a header for such a leaf would collide with the
    inline parent.
    """
    from agent6.ui.cli import entry

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    cfg = gdir / "agent6" / "config.toml"
    cfg.write_text(
        "[providers.openrouter]\n"
        'backend = "openai"\n'
        'base_url = "https://o.example/v1"\n'
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    before = cfg.read_text(encoding="utf-8")
    # The whole layer stack must read this same file, not just the write path.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)

    rc = entry.cli_main(["config", "set", "providers.openrouter.extra_body.provider.sort", "price"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "providers.openrouter.extra_body" in err
    assert "Cannot declare" not in err, "the raw TOML error is not an explanation"
    assert cfg.read_text(encoding="utf-8") == before


def test_config_set_refuses_a_target_that_does_not_parse(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config set` refuses a target that does not parse instead of appending to it."""
    from agent6.ui.cli import entry

    cfg = tmp_path / "config.toml"
    cfg.write_text('[sandbox\nprotect_git = "on"\n', encoding="utf-8")  # missing ]

    def _global_path(*_a: object, **_k: object) -> pathlib.Path:
        return cfg

    monkeypatch.setattr(paths, "global_config_path", _global_path)
    monkeypatch.setattr(paths, "global_config_path", _global_path)

    rc = entry.cli_main(["config", "set", "sandbox.run_commands", "allow"])
    out = capsys.readouterr()
    assert rc == 2, "a target that does not parse must not report success"
    assert "Set sandbox.run_commands" not in out.out
    # And the file is untouched: no surgery appended into a file we cannot read.
    assert cfg.read_text(encoding="utf-8") == '[sandbox\nprotect_git = "on"\n'


def test_reject_machine_protected_covers_every_spec_forbidden_key(tmp_path: pathlib.Path) -> None:
    """The machine-file write guard refuses every key the MachineSpec validator forbids.

    The compensating load re-check is skipped while the file is a `machine create` draft.
    """
    m = tmp_path / "m.asm.toml"
    for key in (
        "providers.openai.base_url",
        '"providers".openai.base_url',
        "presets.review.sandbox.run_commands",
        """'presets' . review.sandbox.run_commands""",
        "mcp.servers",
        "notify.on_event",
        "git.run_repo_hooks",
    ):
        assert cc._reject_machine_protected(key, m) is not None, key
    # Benign overlay keys stay writable (the forbid is surgical); a sandbox key is bounded by
    # the operator's config when the machine is checked or run, not refused at the write.
    assert cc._reject_machine_protected("sandbox.run_commands", m) is None
    assert cc._reject_machine_protected("git.commit.name", m) is None
    assert cc._reject_machine_protected("review.panel_size", m) is None


def test_write_machine_refuses_an_invalid_spec_and_writes_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cwd-dependent [config]-overlay validation is stubbed out.
    monkeypatch.setattr(config_layer, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_GOOD, encoding="utf-8")

    err = cc._write_machine(target, _GOOD, _BAD)

    assert err is not None  # the invalid machine was caught (not silently left)
    assert target.read_text(encoding="utf-8") == _GOOD  # and never reached the file


def test_write_machine_writes_a_valid_spec(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "m.asm.toml"
    target.write_text(_GOOD, encoding="utf-8")
    edited = _GOOD + "# edited\n"

    # prior_text=_GOOD makes _machine_is_valid true, so the edited spec is loaded.
    assert cc._write_machine(target, _GOOD, edited) is None
    assert target.read_text(encoding="utf-8") == edited


def test_config_show_and_get_share_the_unknown_key_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    expected = "ERROR: unknown config key 'nope.nope' (see `agent6 config show`)\n"
    for verb in ("show", "get"):
        assert entry.main(["config", verb, "nope.nope"]) == 2
        assert capsys.readouterr().err == expected


def test_config_get_suggests_a_key_from_the_layers_it_was_given(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config get`'s did-you-mean pool comes from the layers it was given, `--config` included."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    overlay = tmp_path / "overlay.toml"
    overlay.write_text(
        '[providers.zz]\nbackend = "openai"\nbase_url = "https://zz.test/v1"\n',
        encoding="utf-8",
    )
    assert entry.main(["--config", str(overlay), "config", "get", "providers.zz.base_ur"]) == 2
    assert "Did you mean 'providers.zz.base_url'" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["show", "get", "set"])
def test_config_refuses_the_old_settled_stop_key(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
) -> None:
    """The removed key is refused by name with its replacement, never mapped."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    key = "harness.verify_settled_stop_after"
    args = ["config", verb, key, *(["3"] if verb == "set" else [])]
    assert entry.main(args) == 2
    err = capsys.readouterr().err
    assert f"unknown config key '{key}'" in err
    assert "'harness.settled_stop_after'" in err


def test_config_get_distinguishes_a_section_from_an_unknown_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "get", "sandbox"]) == 2
    assert "'sandbox' is not a config leaf" in capsys.readouterr().err
    assert entry.main(["config", "get", "sandbox.nope"]) == 2
    assert "unknown config key 'sandbox.nope'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "table",
    [
        '[harness.metric]\ncommand = ["true"]\npattern = "(0)"\ngoal = "minimize"\n',
        '[harness]\nmetric = { command = ["true"], pattern = "(0)", goal = "minimize" }\n',
    ],
)
def test_config_refuses_and_fix_drops_the_metric_table(
    table: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    repo_cfg = paths.repo_config_path(tmp_path)
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text(
        table + '[harness.verify]\ncommand = ["check"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="--metric"):
        config_layer.load_effective(tmp_path)

    assert entry.main(["config", "fix"]) == 0
    assert "metric" not in repo_cfg.read_text(encoding="utf-8")
    assert config_layer.load_effective(tmp_path).config.harness.verify.command == ("check",)
    for key in ("harness.metric", "harness.metric.command"):
        assert entry.main(["config", "set", key, '["true"]']) == 2
        assert "--metric" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["harness", "review", "budget"])
def test_config_get_calls_a_section_with_no_resolved_leaf_a_section(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str,
) -> None:
    """Sections with no runtime-resolved leaf were called unknown config keys."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "get", key]) == 2
    err = capsys.readouterr().err
    assert f"{key!r} is not a config leaf" in err
    assert "unknown config key" not in err


def test_config_set_help_names_inline_tables(capsys: pytest.CaptureFixture[str]) -> None:
    """The set help omitted inline tables even though they preserve types and set siblings."""
    from agent6.ui.cli import entry

    with pytest.raises(SystemExit) as exc:
        entry.main(["config", "set", "--help"])
    assert exc.value.code == 0
    assert "inline tables" in " ".join(capsys.readouterr().out.split())


def test_config_set_keeps_a_valid_write_despite_a_stale_value_elsewhere(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A value a schema change left invalid never blocks an unrelated valid key; a warning names it.
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[harness.prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = entry.main(["config", "set", "budget.max_usd", "5"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid write is kept...
    assert "Set budget" in captured.out  # ...it succeeded,
    assert "harness.prompt.decompose" in captured.err  # ...and a warning names the stale value,
    assert str(gpath) in captured.err  # the exact file,
    assert "config set harness.prompt.decompose <value>" in captured.err  # and how to fix it.

    # Overwriting the offending value clears the warning; the write is clean.
    assert entry.main(["config", "set", "harness.prompt.decompose", "off"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_rejects_a_masked_invalid_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A repo overlay masks the key; an invalid value in a lower layer is still rejected.
    import subprocess

    from agent6.ui.cli import entry

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "set", "--repo", "sandbox.run_commands", "allow"]) == 0  # the mask
    capsys.readouterr()
    # Global set of an invalid value -> rejected despite the repo mask.
    rc = entry.main(["config", "set", "sandbox.run_commands", "bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "sandbox.run_commands" in err  # a friendly per-field error, not a merge dump
    # A valid global set still succeeds.
    assert entry.main(["config", "set", "sandbox.run_commands", "deny"]) == 0


def test_config_set_rejects_a_newly_invalid_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An invalid value still fails loud and reverts, whatever stale value sits elsewhere.
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    rc = entry.main(["config", "set", "harness.prompt.decompose", "bogus"])
    assert rc == 2  # the write itself is invalid -> reverted + fail loud
    assert "harness.prompt.decompose" in capsys.readouterr().err
    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    assert not gpath.is_file() or "decompose" not in gpath.read_text(encoding="utf-8")


def test_a_refused_write_still_hands_the_config_back_to_the_operator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused write still hands the config back to the operator under sudo."""
    from agent6.ui.cli import entry

    handed: list[pathlib.Path] = []
    monkeypatch.setattr(paths, "chown_to_real_user", handed.append)
    monkeypatch.setattr(paths, "mkdir_for_real_user", handed.append)  # the dir handover
    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[budget]\nmax_usd = 5.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert entry.main(["config", "set", "harness.prompt.decompose", "bogus"]) == 2  # refused
    assert gpath in handed  # the file is the operator's
    assert gpath.parent in handed  # and so is the dir the write may have created


def test_config_set_keeps_a_write_on_an_already_invalid_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # On an already-broken config an invalid value is rejected, while a valid one lands and clears.
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[harness.prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert (
        entry.main(["config", "set", "harness.prompt.decompose", "enabled"]) == 2
    )  # invalid value: rejected
    assert "harness.prompt.decompose" in capsys.readouterr().err  # friendly per-field error
    assert (
        entry.main(["config", "set", "harness.prompt.decompose", "on"]) == 0
    )  # a valid value clears it
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_unknown_leaf_gets_a_did_you_mean(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A typo under a known section gets the near-miss and the show pointer, not pydantic's text.
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    rc = entry.main(["config", "set", "sandbox.run_command", "allow"])  # missing 's'
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'sandbox.run_command'" in err
    assert "'sandbox.run_commands'" in err  # the did-you-mean
    assert "Extra inputs" not in err


def test_config_set_unknown_section_gets_the_same_friendly_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unknown top-level section errors at the section loc, a parent of the written key.
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    rc = entry.main(["config", "set", "bogus.key", "foo"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'bogus.key'" in err
    assert "merged config layers" not in err and "extra_forbidden" not in err
    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    assert not gpath.is_file() or "bogus" not in gpath.read_text(encoding="utf-8")


def test_config_set_accepts_a_profiles_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # [presets.*] is stripped before validation, so the unknown-key reroute does not reject it.
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    rc = entry.main(["config", "set", "presets.mine.review.when", "finish"])
    assert rc == 0
    text = paths.global_config_path().read_text(encoding="utf-8")
    assert "[presets.mine.review]" in text
    assert "unknown config key" not in capsys.readouterr().err


def test_config_set_bool_error_speaks_human(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    rc = entry.main(["config", "set", "git.auto_merge", "notabool"])
    assert rc == 2
    assert "git.auto_merge: expected true or false, got 'notabool'" in capsys.readouterr().err


def test_config_set_global_keeps_a_valid_write_shadowed_by_a_stale_repo_layer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # harness.prompt.decompose is stale in the repo layer; a valid global write is kept, warned.
    from agent6.ui.cli import entry

    repo_cfg = paths.repo_config_path(tmp_path)
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text("[harness.prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = entry.main(["config", "set", "harness.prompt.decompose", "auto"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid global write is KEPT, not reverted over the repo's stale value
    assert "WARNING" in captured.err  # ...but warns the repo layer still shadows it
    assert '"auto"' in paths.global_config_path().read_text(encoding="utf-8")


def test_config_set_sub_leaf_on_an_existing_provider(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A leaf's isolated dict lacks the union tag; the parent's error is not the written child's.
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        '[providers.op]\nbackend = "openai"\nbase_url = "https://x.test/v1"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "set", "providers.op.base_url", "https://y.test/v1"]) == 0


def test_config_set_submodel_inline_table_completed_by_a_lower_layer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An inline table a lower layer completes is accepted; the missing child is not this write's.
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[models.worker]\nmodel = "p/m"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "set", "--repo", "models.worker", '{ effort = "high" }']) == 0


# --- `config fix`: drop invalid entries, print what was dropped and where -------


def test_config_fix_drops_a_bad_value_and_keeps_valid_ones(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The valid budget entry beside the stale harness.prompt.decompose survives the repair.
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        "[harness.prompt]\ndecompose = true\n[budget]\nmax_usd = 5.0\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    rc = entry.main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "harness.prompt.decompose" in out and "global" in out  # named the entry + its layer
    text = gpath.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid entry is gone
    assert "max_usd" in text  # the valid one stays
    assert entry.main(["config", "show"]) == 0  # config is valid now


def test_config_fix_drops_a_masked_invalid_global_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repo override must not hide a stale global value from config fix."""
    from agent6.config import layer
    from agent6.ui.cli import entry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text('[sandbox]\nrun_commands = "bogus"\n', encoding="utf-8")
    repo_path = paths.repo_config_path(repo)
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text('[sandbox]\nrun_commands = "allow"\n', encoding="utf-8")

    assert entry.main(["config", "fix"]) == 0
    assert "sandbox.run_commands" in capsys.readouterr().out
    assert "run_commands" not in global_path.read_text(encoding="utf-8")
    assert 'run_commands = "allow"' in repo_path.read_text(encoding="utf-8")
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    assert layer.load_effective(other_repo).config.sandbox.run_commands == "ask"


def test_config_fix_keeps_a_table_whose_validator_the_repo_layer_satisfies(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config fix` keeps a table whose validator only the repo layer satisfies."""
    from agent6.ui.cli import entry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True)
    global_text = '[web]\nhost = "0.0.0.0"\nport = 9123\n'
    global_path.write_text(global_text, encoding="utf-8")
    repo_path = paths.repo_config_path(repo)
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text("[web]\nallow_non_loopback = true\n", encoding="utf-8")

    assert entry.main(["config", "fix"]) == 0
    assert "nothing to fix" in capsys.readouterr().out.lower()
    assert global_path.read_text(encoding="utf-8") == global_text


def test_config_fix_accepts_a_table_completed_by_the_repo_layer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text('[models.worker]\nmodel = "p/m"\n', encoding="utf-8")
    repo_path = paths.repo_config_path(repo)
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text(
        '[providers.p]\nbackend = "openai"\nbase_url = "https://p.example/v1"\n'
        '[models.worker]\neffort = "high"\n',
        encoding="utf-8",
    )

    assert entry.main(["config", "fix"]) == 0
    assert "nothing to fix" in capsys.readouterr().out.lower()
    assert 'model = "p/m"' in global_path.read_text(encoding="utf-8")


def test_config_fix_drops_an_invalid_unselected_preset_leaf(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    path = paths.global_config_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        '[presets.good.review]\nwhen = "finish"\n[presets.bad.sandbox]\nnetwork = "banana"\n',
        encoding="utf-8",
    )

    assert entry.main(["config", "fix"]) == 0
    out = capsys.readouterr().out
    assert "presets.bad.sandbox.network" in out
    text = path.read_text(encoding="utf-8")
    assert "banana" not in text
    assert '[presets.good.review]\nwhen = "finish"' in text


def test_config_fix_drops_an_invalid_preset_leaf_masked_by_the_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text('[presets.demo.sandbox]\nnetwork = "banana"\n', encoding="utf-8")
    repo_path = paths.repo_config_path(repo)
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text('[presets.demo.sandbox]\nnetwork = "host"\n', encoding="utf-8")

    assert entry.main(["config", "fix"]) == 0
    out = capsys.readouterr().out
    assert "presets.demo.sandbox.network" in out
    assert str(global_path) in out
    assert "network" not in global_path.read_text(encoding="utf-8")
    assert 'network = "host"' in repo_path.read_text(encoding="utf-8")


def test_config_fix_drops_an_unknown_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nprotct_git = true\n", encoding="utf-8")  # typo of protect_git
    monkeypatch.chdir(tmp_path)

    rc = entry.main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sandbox.protct_git" in out
    assert "protct_git" not in gpath.read_text(encoding="utf-8")


@pytest.mark.parametrize("table", ["providers.p", "presets.demo.providers.p"])
def test_config_fix_drops_a_refused_provider_leaf(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    table: str,
) -> None:
    """Fix named `providers.p.openai.idle_compact_s`, a key no file holds, and dropped nothing."""
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        f'[models.default]\nmodel = "p/m"\n[{table}]\nbackend = "openai"\n'
        'base_url = "http://127.0.0.1:1/v1"\napi_key_env = "K"\nidle_compact_s = 0\nbogus = 1\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert entry.main(["config", "fix"]) == 0
    out = capsys.readouterr().out
    assert f"Removed {table}.idle_compact_s = 0" in out and f"{table}.bogus" in out
    text = gpath.read_text(encoding="utf-8")
    assert "idle_compact_s" not in text and "bogus" not in text and 'backend = "openai"' in text
    assert entry.main(["config", "show"]) == 0  # the config loads again


def test_a_load_error_names_the_provider_key_the_file_holds(tmp_path: pathlib.Path) -> None:
    """The error named `providers.p.openai.idle_compact_s`, a key no file holds."""
    path = tmp_path / "config.toml"
    path.write_text(
        '[providers.p]\nbackend = "openai"\nidle_compact_s = "soon"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError) as refused:
        config_model.load_config(path)
    assert "- providers.p.idle_compact_s: " in str(refused.value)


def test_config_set_names_the_one_write_for_a_provider_table(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The missing fields were looked up under `providers.p.openai.oauth`, so no hint matched."""
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[providers.p]\nbackend = "openai"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert entry.main(["config", "set", "providers.p.oauth.issuer", "https://issuer.example"]) != 0
    said = capsys.readouterr()
    text = said.out + said.err
    assert "- providers.p.oauth.client_id: Field required" in text
    assert "in one write: `agent6 config set providers.p.oauth '{ issuer = ..., " in text
    assert "oauth" not in gpath.read_text(encoding="utf-8")


def test_config_fix_labels_a_repo_layer_entry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    rpath = paths.repo_config_path(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text("[harness.prompt]\ndecompose = true\n", encoding="utf-8")

    rc = entry.main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "harness.prompt.decompose" in out and "repo" in out
    assert "decompose" not in rpath.read_text(encoding="utf-8")


def test_config_fix_on_valid_config_reports_nothing_to_fix(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    before = "[budget]\nmax_usd = 5.0\n"
    gpath.write_text(before, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = entry.main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to fix" in out.lower()
    assert gpath.read_text(encoding="utf-8") == before  # untouched


def test_config_fix_repairs_both_layers_and_labels_each(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[harness.prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rpath = paths.repo_config_path(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text('[sandbox]\nrun_commands = "bogus"\n', encoding="utf-8")

    rc = entry.main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "harness.prompt.decompose" in out and "global" in out
    assert "sandbox.run_commands" in out and "repo" in out
    assert "decompose" not in gpath.read_text(encoding="utf-8")
    assert "bogus" not in rpath.read_text(encoding="utf-8")


def test_config_fix_machine_overlay_leaves_the_spec_untouched(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    mfile = tmp_path / "m.asm.toml"
    mfile.write_text(_GOOD + "[config.harness.prompt]\ndecompose = true\n", encoding="utf-8")

    rc = entry.main(["config", "fix", "--machine-file", str(mfile)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "harness.prompt.decompose" in out
    text = mfile.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid overlay entry is gone
    assert 'machine = "m"' in text  # the machine spec itself is untouched


def test_config_set_unknown_provider_key_speaks_human(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The standalone minimal dict cannot resolve a union member; the member models answer directly.
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "set", "providers.p.backend", "anthropic"]) == 0
    capsys.readouterr()
    rc = entry.main(["config", "set", "providers.p.bogus_key", "x"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown provider key 'providers.p.bogus_key'" in err
    assert "Extra inputs" not in err and "anthropic.bogus" not in err
    rc = entry.main(["config", "set", "providers.p.api_key_enw", "MY_KEY"])  # typo
    assert rc == 2
    assert "'api_key_env'" in capsys.readouterr().err  # the did-you-mean


def test_config_set_invalid_provider_value_names_the_field(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "set", "providers.p.backend", "anthropic"]) == 0
    capsys.readouterr()
    rc = entry.main(["config", "set", "providers.p.http_timeout_s", "not-a-number"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "providers.p.http_timeout_s" in err
    assert "merged config layers" not in err
    # A Field-constraint violation gets the same member answer, no discriminator tag in the loc.
    rc = entry.main(["config", "set", "providers.p.http_timeout_s", "-5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "greater than 0" in err
    assert "merged config layers" not in err and ".anthropic." not in err
    # A partial-entry write some member accepts still lands (never reverted).
    assert entry.main(["config", "set", "providers.p.base_url", "https://x.example/v1"]) == 0


def test_config_fix_skips_an_entry_another_writer_already_fixed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fix` skips an entry another writer replaced with a valid value in between.

    find_invalid_entries reads unlocked and removal deletes by key name.
    """
    from agent6.config import layer

    cfg = tmp_path / "config.toml"
    cfg.write_text('[sandbox]\nrun_commands = "ask"\n', encoding="utf-8")

    # Diagnosis saw the OLD, invalid value; the file already holds the fixed one.
    stale = layer.InvalidEntry(
        leaf="sandbox.run_commands",
        value="maybe",  # what the (unlocked) diagnosis read
        layer="global",
        path=cfg,
        file_key="sandbox.run_commands",
    )
    calls = {"n": 0}

    def _diag(*_a: object, **_k: object) -> layer.ConfigDiagnosis:
        calls["n"] += 1
        return layer.ConfigDiagnosis(removable=(stale,) if calls["n"] == 1 else (), blocked=None)

    monkeypatch.setattr(config_layer, "find_invalid_entries", _diag)
    cc._cmd_config_fix(machine=None)

    assert 'run_commands = "ask"' in cfg.read_text(encoding="utf-8"), (
        "the concurrent writer's value was deleted by a stale diagnosis"
    )


def test_config_fix_removes_a_nan_entry_instead_of_claiming_valid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config fix` removes a `nan` entry instead of claiming the config valid.

    TOML `nan` never compares equal to itself, so an identity re-check reads it as replaced.
    """
    from agent6.ui.cli import entry

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nbogus_entry = nan\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = entry.main(["config", "fix"])
    out = capsys.readouterr().out
    assert "nothing to fix" not in out  # the config was NOT valid
    assert rc == 0
    assert "bogus_entry" in out  # removed and named
    assert "bogus_entry" not in gpath.read_text(encoding="utf-8")
    assert entry.main(["config", "show"]) == 0


def test_equal_tolerating_nan_matches_nan_at_every_depth() -> None:
    import math

    assert cc._equal_tolerating_nan(math.nan, math.nan)
    assert cc._equal_tolerating_nan({"x": math.nan}, {"x": math.nan})
    assert cc._equal_tolerating_nan([1.0, math.nan], [1.0, math.nan])
    assert not cc._equal_tolerating_nan({"x": math.nan}, {"x": 1.0})
    assert not cc._equal_tolerating_nan([math.nan], [math.nan, math.nan])


def test_config_set_names_the_flag_file_that_shadows_the_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 --config F config set` notes that F keeps overriding the key it just wrote."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    flag = tmp_path / "f.toml"
    flag.write_text('[sandbox]\nrun_commands = "allow"\n', encoding="utf-8")
    rc = entry.main(["--config", str(flag), "config", "set", "sandbox.run_commands", "deny"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"--config {flag} overrides sandbox.run_commands" in out
    # A key the flag file does not set carries no note.
    rc = entry.main(["--config", str(flag), "config", "set", "sandbox.protect_git", "on"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "overrides sandbox.protect_git" not in out


def test_config_set_notes_a_repo_layer_that_keeps_a_global_write_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A global write under a repo override said nothing; a run kept reading the repo value."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    assert entry.main(["config", "set", "--repo", "harness.max_turns", "7"]) == 0
    capsys.readouterr()
    assert entry.main(["config", "set", "harness.max_turns", "9"]) == 0
    out = capsys.readouterr().out
    assert "note: the repo config overrides harness.max_turns; a run here reads its value." in out
    assert entry.main(["config", "set", "--repo", "harness.max_turns", "8"]) == 0
    assert "overrides" not in capsys.readouterr().out


def test_config_set_shadow_note_names_the_last_file_setting_the_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under two `--config` files the note names the later file that sets the key."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text('[sandbox]\nrun_commands = "allow"\nprotect_git = "on"\n', encoding="utf-8")
    second.write_text('[sandbox]\nrun_commands = "ask"\n', encoding="utf-8")
    both = ["--config", str(first), "--config", str(second)]
    assert entry.main([*both, "config", "set", "sandbox.run_commands", "deny"]) == 0
    out = capsys.readouterr().out
    assert f"--config {second} overrides sandbox.run_commands" in out
    assert str(first) not in out
    # A key only the first file sets is shadowed by the first file.
    assert entry.main([*both, "config", "set", "sandbox.protect_git", "off"]) == 0
    assert f"--config {first} overrides sandbox.protect_git" in capsys.readouterr().out


def test_config_show_legend_names_the_flag_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The legend reads a "flag" source back to the one path the operator typed."""
    from agent6.ui.cli import entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    flag = tmp_path / "f.toml"
    flag.write_text('[sandbox]\nrun_commands = "allow"\n', encoding="utf-8")
    rc = entry.main(["--config", str(flag), "config", "show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"flag = {flag}" in out


def test_a_machine_overlay_refusal_reads_like_every_other_writer() -> None:
    """`config set --machine-file` refuses with the leaf lines every other writer prints."""
    raw = (
        "Config validation failed: (merged config layers + machine overlay)\n"
        "  - harness.verify.retries: Input should be greater than or equal to 0"
        " (type=greater_than_equal)"
    )
    assert cc._leaf_problems(raw) == (
        "harness.verify.retries: Input should be greater than or equal to 0"
    )
    assert cc._leaf_problems("machine file unreadable") == "machine file unreadable"


def test_config_unset_on_an_mcp_server_names_the_verb_that_removes_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config unset` on a `[mcp.servers.<name>]` table points at the verb that removes it."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[agent6]\nconfig_version = 1\n\n[mcp.servers.calc]\ncommand = ["true"]\n',
        encoding="utf-8",
    )

    rc = cc._cmd_config_unset("mcp.servers.calc", repo=False, machine=None, config_paths=(cfg,))

    assert rc == 2
    assert "agent6 mcp remove calc" in capsys.readouterr().err


def test_config_unset_on_an_unknown_key_matches_show_and_get(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[agent6]\nconfig_version = 1\n", encoding="utf-8")

    rc = cc._cmd_config_unset("mcp.servers.nope", repo=False, machine=None, config_paths=(cfg,))

    assert rc == 2
    assert capsys.readouterr().err == (
        "ERROR: unknown config key 'mcp.servers.nope'. Did you mean 'mcp.servers'?"
        " (see `agent6 config show`)\n"
    )


def test_config_unset_repairs_a_config_that_no_longer_loads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config unset` repairs a config that no longer loads.

    Its precheck does not load the merged config.
    """
    monkeypatch.chdir(tmp_path)
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    # Valid at 0.0.29; a later invariant refuses it (summarize > keep_recent).
    (cfg_home / "agent6" / "config.toml").write_text(
        "[agent6]\nconfig_version = 1\n\n[compaction]\ndrop_at_chars = 30000\n"
        "summarize_at_chars = 60000\n",
        encoding="utf-8",
    )

    rc = cc._cmd_config_unset("compaction.summarize_at_chars", repo=False, machine=None)

    assert rc == 0
    assert "summarize_at_chars" not in (cfg_home / "agent6" / "config.toml").read_text(
        encoding="utf-8"
    )
