# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`config set/add/remove --machine` re-validates the whole machine spec."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.config import layer as config_layer
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
    out = cli_completers._complete_config_values("", args)  # pyright: ignore[reportPrivateUsage]
    assert '{ provider = { sort = "throughput" } }' in out
    # a non-extra_body key is unaffected
    enum_args = argparse.Namespace(key="sandbox.isolation")
    assert cli_completers._complete_config_values("", enum_args) == [  # pyright: ignore[reportPrivateUsage]
        "auto",
        "strict",
        "hardened",
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
    args = argparse.Namespace(key="preset")
    out = cli_completers._complete_config_values("", args)  # pyright: ignore[reportPrivateUsage]
    assert "ultra" in out and "myteam" in out
    assert cli_completers._complete_config_values("ul", args) == ["ultra"]  # pyright: ignore[reportPrivateUsage]


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
    out = cli_completers._complete_config_keys("presets.")  # pyright: ignore[reportPrivateUsage]
    assert any(k.startswith("presets.myteam.review.") for k in out)
    assert not any(k.startswith("presets.ultra") for k in out)
    # the top-level `preset` leaf itself is offered alongside presets.*
    assert "preset" in cli_completers._complete_config_keys("preset")  # pyright: ignore[reportPrivateUsage]
    # a bare TAB (empty prefix) is not flooded with the generated paths
    assert not any(
        k.startswith("presets.")
        for k in cli_completers._complete_config_keys("")  # pyright: ignore[reportPrivateUsage]
    )


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
    assert completers._complete_parallel_models("w/gpt") == [  # pyright: ignore[reportPrivateUsage]
        "w/gpt-5",
        "w/gpt-5-mini",
    ]
    assert completers._complete_parallel_models("w/opus,w/gpt") == [  # pyright: ignore[reportPrivateUsage]
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
    from agent6.ui.cli import cli_main

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    cfg = gdir / "agent6" / "config.toml"
    cfg.write_text(
        "[providers.openrouter]\n"
        'api_format = "openai"\n'
        'base_url = "https://o.example/v1"\n'
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    before = cfg.read_text(encoding="utf-8")
    # The whole layer stack must read this same file, not just the write path.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.chdir(tmp_path)

    rc = cli_main(["config", "set", "providers.openrouter.extra_body.provider.sort", "price"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "providers.openrouter.extra_body" in err
    assert "Cannot declare" not in err, "the raw TOML error is not an explanation"
    assert cfg.read_text(encoding="utf-8") == before


def test_config_set_refuses_a_target_that_does_not_parse(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config set` refuses a target that does not parse instead of appending to it."""
    from agent6.ui.cli import cli_main

    cfg = tmp_path / "config.toml"
    cfg.write_text("[sandbox\nprotect_git = true\n", encoding="utf-8")  # missing ]

    def _global_path(*_a: object, **_k: object) -> pathlib.Path:
        return cfg

    monkeypatch.setattr(paths, "global_config_path", _global_path)
    monkeypatch.setattr(paths, "global_config_path", _global_path)

    rc = cli_main(["config", "set", "sandbox.run_commands", "yes"])
    out = capsys.readouterr()
    assert rc == 2, "a target that does not parse must not report success"
    assert "Set sandbox.run_commands" not in out.out
    # And the file is untouched: no surgery appended into a file we cannot read.
    assert cfg.read_text(encoding="utf-8") == "[sandbox\nprotect_git = true\n"


def test_reject_machine_protected_covers_every_spec_forbidden_key(tmp_path: pathlib.Path) -> None:
    """The machine-file write guard refuses every key the MachineSpec validator forbids.

    The compensating load re-check is skipped while the file is a `machine create` draft.
    """
    m = tmp_path / "m.asm.toml"
    for key in (
        "providers.openai.base_url",
        "sandbox.run_commands",
        "presets.ultra.sandbox.run_commands",
        "machine.notify.on_event",
        "mcp.servers",
        "notify.on_complete",
        "git.run_repo_hooks",
    ):
        assert cc._reject_machine_protected(key, m) is not None, key  # pyright: ignore[reportPrivateUsage]
    # Benign overlay keys stay writable (the forbid is surgical).
    assert cc._reject_machine_protected("git.commit.name", m) is None  # pyright: ignore[reportPrivateUsage]
    assert cc._reject_machine_protected("review.panel_size", m) is None  # pyright: ignore[reportPrivateUsage]


def test_revalidate_machine_rejects_invalid_spec_and_rolls_back(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cwd-dependent [config]-overlay validation is stubbed out.
    monkeypatch.setattr(config_layer, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_BAD, encoding="utf-8")

    err = cc._revalidate_machine(target, _GOOD)  # pyright: ignore[reportPrivateUsage]

    assert err is not None  # the invalid machine was caught (not silently left)
    assert target.read_text(encoding="utf-8") == _GOOD  # and the file was rolled back


def test_revalidate_machine_accepts_valid_spec(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_layer, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_GOOD, encoding="utf-8")

    # prior_text=_GOOD makes _machine_is_valid true, so load_machine(target) runs.
    assert cc._revalidate_machine(target, _GOOD) is None  # pyright: ignore[reportPrivateUsage]
    assert target.read_text(encoding="utf-8") == _GOOD  # untouched


def test_config_show_and_get_share_the_unknown_key_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    expected = "ERROR: unknown config key 'nope.nope' (see `agent6 config show`)\n"
    for verb in ("show", "get"):
        assert main(["config", verb, "nope.nope"]) == 2
        assert capsys.readouterr().err == expected


def test_config_get_suggests_a_key_from_the_layers_it_was_given(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config get`'s did-you-mean pool comes from the layers it was given, `--config` included."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    overlay = tmp_path / "overlay.toml"
    overlay.write_text(
        '[providers.zz]\napi_format = "openai"\nbase_url = "https://zz.test/v1"\n',
        encoding="utf-8",
    )
    assert main(["--config", str(overlay), "config", "get", "providers.zz.base_ur"]) == 2
    assert "Did you mean 'providers.zz.base_url'" in capsys.readouterr().err


def test_config_get_distinguishes_a_section_from_an_unknown_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "get", "sandbox"]) == 2
    assert "'sandbox' is not a config leaf" in capsys.readouterr().err
    assert main(["config", "get", "sandbox.nope"]) == 2
    assert "unknown config key 'sandbox.nope'" in capsys.readouterr().err


def test_config_set_keeps_a_valid_write_despite_a_stale_value_elsewhere(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A value a schema change left invalid never blocks an unrelated valid key; a warning names it.
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "set", "budget.max_usd", "5"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid write is kept...
    assert "Set budget" in captured.out  # ...it succeeded,
    assert "prompt.decompose" in captured.err  # ...and a warning names the stale value,
    assert str(gpath) in captured.err  # the exact file,
    assert "config set prompt.decompose <value>" in captured.err  # and how to fix it.

    # Overwriting the offending value clears the warning; the write is clean.
    assert main(["config", "set", "prompt.decompose", "off"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_rejects_a_masked_invalid_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A repo overlay masks the key; an invalid value in a lower layer is still rejected.
    import subprocess

    from agent6.ui.cli import main

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "--repo", "sandbox.run_commands", "yes"]) == 0  # the mask
    capsys.readouterr()
    # Global set of an invalid value -> rejected despite the repo mask.
    rc = main(["config", "set", "sandbox.run_commands", "bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "sandbox.run_commands" in err  # a friendly per-field error, not a merge dump
    # A valid global set still succeeds.
    assert main(["config", "set", "sandbox.run_commands", "no"]) == 0


def test_config_set_rejects_a_newly_invalid_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An invalid value still fails loud and reverts, whatever stale value sits elsewhere.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "prompt.decompose", "bogus"])
    assert rc == 2  # the write itself is invalid -> reverted + fail loud
    assert "prompt.decompose" in capsys.readouterr().err
    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    assert not gpath.is_file() or "decompose" not in gpath.read_text(encoding="utf-8")


def test_a_refused_write_still_hands_the_config_back_to_the_operator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused write still hands the config back to the operator under sudo.

    Every write publishes a fresh root-owned inode, the rollback of a refused value included.
    """
    from agent6.ui.cli import main

    handed: list[pathlib.Path] = []
    monkeypatch.setattr(paths, "chown_to_real_user", handed.append)
    monkeypatch.setattr(paths, "mkdir_for_real_user", handed.append)  # the dir handover
    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[budget]\nmax_usd = 5.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "set", "prompt.decompose", "bogus"]) == 2  # rolled back
    assert gpath in handed  # the rolled-back file is the operator's again
    assert gpath.parent in handed  # and so is the dir the write may have created


def test_config_set_keeps_a_write_on_an_already_invalid_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # On an already-broken config an invalid value is rejected, while a valid one lands and clears.
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "set", "prompt.decompose", "enabled"]) == 2  # invalid value: rejected
    assert "prompt.decompose" in capsys.readouterr().err  # friendly per-field error
    assert main(["config", "set", "prompt.decompose", "on"]) == 0  # a valid value clears it
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_unknown_leaf_gets_a_did_you_mean(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A typo under a known section gets the near-miss and the show pointer, not pydantic's text.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "sandbox.run_command", "yes"])  # missing 's'
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'sandbox.run_command'" in err
    assert "'sandbox.run_commands'" in err  # the did-you-mean
    assert "Extra inputs" not in err


def test_config_set_unknown_section_gets_the_same_friendly_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unknown top-level section errors at the section loc, a parent of the written key.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "bogus.key", "foo"])
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
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "presets.mine.review.trigger", "before_finish"])
    assert rc == 0
    text = paths.global_config_path().read_text(encoding="utf-8")
    assert "[presets.mine.review]" in text
    assert "unknown config key" not in capsys.readouterr().err


def test_config_set_bool_error_speaks_human(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "sandbox.protect_git", "notabool"])
    assert rc == 2
    assert "sandbox.protect_git: expected true or false, got 'notabool'" in capsys.readouterr().err


def test_config_set_global_keeps_a_valid_write_shadowed_by_a_stale_repo_layer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # prompt.decompose is stale in the repo layer; a valid global write is kept with a warning.
    from agent6.ui.cli import main

    repo_cfg = paths.repo_config_path(tmp_path)
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "set", "prompt.decompose", "auto"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid global write is KEPT, not reverted over the repo's stale value
    assert "WARNING" in captured.err  # ...but warns the repo layer still shadows it
    assert '"auto"' in paths.global_config_path().read_text(encoding="utf-8")


def test_config_set_sub_leaf_on_an_existing_provider(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A leaf's isolated dict lacks the union tag; the parent's error is not the written child's.
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        '[providers.op]\napi_format = "openai"\nbase_url = "https://x.test/v1"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.op.base_url", "https://y.test/v1"]) == 0


def test_config_set_submodel_inline_table_completed_by_a_lower_layer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An inline table a lower layer completes is accepted; the missing child is not this write's.
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[models.worker]\nmodel = "m"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "--repo", "models.worker", '{ provider = "p" }']) == 0


# --- `config fix`: drop invalid entries, print what was dropped and where -------


def test_config_fix_drops_a_bad_value_and_keeps_valid_ones(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The valid budget entry beside the stale prompt.decompose survives the repair.
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n[budget]\nmax_usd = 5.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "global" in out  # named the entry + its layer
    text = gpath.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid entry is gone
    assert "max_usd" in text  # the valid one stays
    assert main(["config", "show"]) == 0  # config is valid now


def test_config_fix_drops_a_masked_invalid_global_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repo override must not hide a stale global value from config fix."""
    from agent6.config import layer
    from agent6.ui.cli import main

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
    repo_path.write_text('[sandbox]\nrun_commands = "yes"\n', encoding="utf-8")

    assert main(["config", "fix"]) == 0
    assert "sandbox.run_commands" in capsys.readouterr().out
    assert "run_commands" not in global_path.read_text(encoding="utf-8")
    assert 'run_commands = "yes"' in repo_path.read_text(encoding="utf-8")
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    assert layer.load_effective(other_repo).config.sandbox.run_commands == "ask"


def test_config_fix_keeps_a_table_whose_validator_the_repo_layer_satisfies(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config fix` keeps a table whose validator only the repo layer satisfies."""
    from agent6.ui.cli import main

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

    assert main(["config", "fix"]) == 0
    assert "nothing to fix" in capsys.readouterr().out.lower()
    assert global_path.read_text(encoding="utf-8") == global_text


def test_config_fix_accepts_a_table_completed_by_the_repo_layer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text('[models.worker]\nmodel = "m"\n', encoding="utf-8")
    repo_path = paths.repo_config_path(repo)
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text(
        '[providers.p]\napi_format = "openai"\nbase_url = "https://p.example/v1"\n'
        '[models.worker]\nprovider = "p"\n',
        encoding="utf-8",
    )

    assert main(["config", "fix"]) == 0
    assert "nothing to fix" in capsys.readouterr().out.lower()
    assert 'model = "m"' in global_path.read_text(encoding="utf-8")


def test_config_fix_drops_an_invalid_unselected_preset_leaf(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    path = paths.global_config_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        '[presets.good.review]\ntrigger = "before_finish"\n'
        '[presets.bad.sandbox]\nnetwork = "banana"\n',
        encoding="utf-8",
    )

    assert main(["config", "fix"]) == 0
    out = capsys.readouterr().out
    assert "presets.bad.sandbox.network" in out
    text = path.read_text(encoding="utf-8")
    assert "banana" not in text
    assert '[presets.good.review]\ntrigger = "before_finish"' in text


def test_config_fix_drops_an_invalid_preset_leaf_masked_by_the_repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

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

    assert main(["config", "fix"]) == 0
    out = capsys.readouterr().out
    assert "presets.demo.sandbox.network" in out
    assert str(global_path) in out
    assert "network" not in global_path.read_text(encoding="utf-8")
    assert 'network = "host"' in repo_path.read_text(encoding="utf-8")


def test_config_fix_drops_an_unknown_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nprotct_git = true\n", encoding="utf-8")  # typo of protect_git
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sandbox.protct_git" in out
    assert "protct_git" not in gpath.read_text(encoding="utf-8")


def test_config_fix_labels_a_repo_layer_entry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rpath = paths.repo_config_path(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "repo" in out
    assert "decompose" not in rpath.read_text(encoding="utf-8")


def test_config_fix_on_valid_config_reports_nothing_to_fix(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    before = "[budget]\nmax_usd = 5.0\n"
    gpath.write_text(before, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to fix" in out.lower()
    assert gpath.read_text(encoding="utf-8") == before  # untouched


def test_config_fix_repairs_both_layers_and_labels_each(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rpath = paths.repo_config_path(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text('[sandbox]\nrun_commands = "bogus"\n', encoding="utf-8")

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "global" in out
    assert "sandbox.run_commands" in out and "repo" in out
    assert "decompose" not in gpath.read_text(encoding="utf-8")
    assert "bogus" not in rpath.read_text(encoding="utf-8")


def test_config_fix_machine_overlay_leaves_the_spec_untouched(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    mfile = tmp_path / "m.asm.toml"
    mfile.write_text(_GOOD + "[config.prompt]\ndecompose = true\n", encoding="utf-8")

    rc = main(["config", "fix", "--machine-file", str(mfile)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out
    text = mfile.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid overlay entry is gone
    assert 'machine = "m"' in text  # the machine spec itself is untouched


def test_config_set_unknown_provider_key_speaks_human(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The standalone minimal dict cannot resolve a union member; the member models answer directly.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.p.api_format", "anthropic"]) == 0
    capsys.readouterr()
    rc = main(["config", "set", "providers.p.bogus_key", "x"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown provider key 'providers.p.bogus_key'" in err
    assert "Extra inputs" not in err and "anthropic.bogus" not in err
    rc = main(["config", "set", "providers.p.api_key_enw", "MY_KEY"])  # typo
    assert rc == 2
    assert "'api_key_env'" in capsys.readouterr().err  # the did-you-mean


def test_config_set_invalid_provider_value_names_the_field(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.p.api_format", "anthropic"]) == 0
    capsys.readouterr()
    rc = main(["config", "set", "providers.p.http_timeout_s", "not-a-number"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "providers.p.http_timeout_s" in err
    assert "merged config layers" not in err
    # A Field-constraint violation gets the same member answer, no discriminator tag in the loc.
    rc = main(["config", "set", "providers.p.http_timeout_s", "-5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "greater than 0" in err
    assert "merged config layers" not in err and ".anthropic." not in err
    # A partial-entry write some member accepts still lands (never reverted).
    assert main(["config", "set", "providers.p.base_url", "https://x.example/v1"]) == 0


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
    cc._cmd_config_fix(machine=None)  # pyright: ignore[reportPrivateUsage]

    assert 'run_commands = "ask"' in cfg.read_text(encoding="utf-8"), (
        "the concurrent writer's value was deleted by a stale diagnosis"
    )


def test_config_fix_removes_a_nan_entry_instead_of_claiming_valid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config fix` removes a `nan` entry instead of claiming the config valid.

    TOML `nan` never compares equal to itself, so an identity re-check reads it as replaced.
    """
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nbogus_entry = nan\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert "nothing to fix" not in out  # the config was NOT valid
    assert rc == 0
    assert "bogus_entry" in out  # removed and named
    assert "bogus_entry" not in gpath.read_text(encoding="utf-8")
    assert main(["config", "show"]) == 0


def test_equal_tolerating_nan_matches_nan_at_every_depth() -> None:
    import math

    assert cc._equal_tolerating_nan(math.nan, math.nan)  # pyright: ignore[reportPrivateUsage]
    assert cc._equal_tolerating_nan({"x": math.nan}, {"x": math.nan})  # pyright: ignore[reportPrivateUsage]
    assert cc._equal_tolerating_nan([1.0, math.nan], [1.0, math.nan])  # pyright: ignore[reportPrivateUsage]
    assert not cc._equal_tolerating_nan({"x": math.nan}, {"x": 1.0})  # pyright: ignore[reportPrivateUsage]
    assert not cc._equal_tolerating_nan([math.nan], [math.nan, math.nan])  # pyright: ignore[reportPrivateUsage]


def test_revalidate_machine_no_lock_keeps_the_write_and_says_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the config lock the machine path keeps an invalid write and says so.

    A whole-file restore could clobber a concurrent writer's update.
    """
    monkeypatch.setattr(config_layer, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_BAD, encoding="utf-8")

    err = cc._revalidate_machine(target, _GOOD, held=False)  # pyright: ignore[reportPrivateUsage]

    assert err is not None and "kept as written" in err
    assert target.read_text(encoding="utf-8") == _BAD  # NOT rolled back


def test_config_set_names_the_flag_file_that_shadows_the_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 --config F config set` notes that F keeps overriding the key it just wrote."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    flag = tmp_path / "f.toml"
    flag.write_text('[sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    rc = main(["--config", str(flag), "config", "set", "sandbox.run_commands", "no"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"--config {flag} overrides sandbox.run_commands" in out
    # A key the flag file does not set carries no note.
    rc = main(["--config", str(flag), "config", "set", "sandbox.protect_git", "true"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "overrides sandbox.protect_git" not in out


def test_config_show_legend_names_the_flag_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The legend reads a "flag" source back to the one path the operator typed."""
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    flag = tmp_path / "f.toml"
    flag.write_text('[sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    rc = main(["--config", str(flag), "config", "show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"flag = {flag}" in out


def test_a_machine_overlay_refusal_reads_like_every_other_writer() -> None:
    """`config set --machine-file` refuses with the leaf lines every other writer prints."""
    raw = (
        "Config validation failed: (merged config layers + machine overlay)\n"
        "  - harness.verify_retries: Input should be greater than or equal to 0"
        " (type=greater_than_equal)"
    )
    assert cc._leaf_problems(raw) == (
        "harness.verify_retries: Input should be greater than or equal to 0"
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

    rc = cc._cmd_config_unset(  # pyright: ignore[reportPrivateUsage]
        "mcp.servers.calc", repo=False, machine=None, config_path=cfg
    )

    assert rc == 2
    assert "agent6 mcp remove calc" in capsys.readouterr().err


def test_config_unset_on_an_unknown_key_matches_show_and_get(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("[agent6]\nconfig_version = 1\n", encoding="utf-8")

    rc = cc._cmd_config_unset(  # pyright: ignore[reportPrivateUsage]
        "mcp.servers.nope", repo=False, machine=None, config_path=cfg
    )

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
    # Valid at 0.0.29; a later invariant refuses it (summarise > keep_recent).
    (cfg_home / "agent6" / "config.toml").write_text(
        "[agent6]\nconfig_version = 1\n\n[context]\ndrop_at_chars = 30000\n"
        "summarise_at_chars = 60000\n",
        encoding="utf-8",
    )

    rc = cc._cmd_config_unset(  # pyright: ignore[reportPrivateUsage]
        "context.summarise_at_chars", repo=False, machine=None
    )

    assert rc == 0
    assert "summarise_at_chars" not in (cfg_home / "agent6" / "config.toml").read_text(
        encoding="utf-8"
    )
