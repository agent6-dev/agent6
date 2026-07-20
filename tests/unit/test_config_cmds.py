# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`config set/add/remove --machine` re-validates the whole machine spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli import config_cmds as cc


def _noop_overlay(*_a: object, **_k: object) -> None:
    # Stub for load_effective_with_overlay so the test isolates machine-spec
    # validation from the cwd-dependent [config]-overlay validation.
    return None


def test_extra_body_value_completer_offers_routing_presets() -> None:
    # TAB after `config set providers.<name>.extra_body` suggests the routing
    # presets for any provider name (matched by suffix).
    import argparse

    from agent6.ui.cli.completers import (
        _complete_config_values,  # pyright: ignore[reportPrivateUsage]
    )

    args = argparse.Namespace(key="providers.openrouter.extra_body")
    out = _complete_config_values("", args)  # pyright: ignore[reportPrivateUsage]
    assert '{ provider = { sort = "throughput" } }' in out
    # a non-extra_body key is unaffected
    enum_args = argparse.Namespace(key="sandbox.profile")
    assert _complete_config_values("", enum_args) == [  # pyright: ignore[reportPrivateUsage]
        "auto",
        "strict",
        "hardened",
    ]


def test_parallel_models_completer_completes_after_last_comma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TAB on `run --parallel` completes model ids for the WORKER provider only
    # (lanes inherit it; only the model is overridden per lane), and completes
    # only the token AFTER the last comma so a `m1,m2,...` list grows member by
    # member (the head is preserved on each completion).
    from agent6.config import Config
    from agent6.ui.cli import completers

    cfg = Config.model_validate(
        {
            "providers": {
                "w": {"api_format": "openai", "base_url": "https://w.example/v1"},
                "s": {"api_format": "openai", "base_url": "https://s.example/v1"},
            },
            "models": {"worker": {"provider": "w", "model": "gpt-5"}},
        }
    )

    class _Eff:
        config = cfg

    def _eff(*_a: object, **_k: object) -> _Eff:
        return _Eff()

    def _models(_cp: object, provider: object) -> list[str]:
        # Per-provider catalogs: the sibling's must never be offered.
        return {"w": ["gpt-5", "gpt-5-mini", "opus"], "s": ["gpt-sibling-only"]}[str(provider)]

    monkeypatch.setattr(completers, "load_effective", _eff)
    monkeypatch.setattr(completers, "_models_for", _models)
    assert completers._complete_parallel_models("gpt") == [  # pyright: ignore[reportPrivateUsage]
        "gpt-5",
        "gpt-5-mini",
    ]
    assert completers._complete_parallel_models("opus,gpt") == [  # pyright: ignore[reportPrivateUsage]
        "opus,gpt-5",
        "opus,gpt-5-mini",
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


def test_revalidate_machine_rejects_invalid_spec_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate the machine-spec validation from the cwd-dependent [config]-overlay
    # validation by stubbing the latter.
    monkeypatch.setattr(cc, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_BAD, encoding="utf-8")

    err = cc._revalidate_config(target, _GOOD, machine=target)  # pyright: ignore[reportPrivateUsage]

    assert err is not None  # the invalid machine was caught (not silently left)
    assert target.read_text(encoding="utf-8") == _GOOD  # and the file was rolled back


def test_revalidate_machine_accepts_valid_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cc, "load_effective_with_overlay", _noop_overlay)
    target = tmp_path / "m.asm.toml"
    target.write_text(_GOOD, encoding="utf-8")

    assert cc._revalidate_config(target, None, machine=target) is None  # pyright: ignore[reportPrivateUsage]
    assert target.read_text(encoding="utf-8") == _GOOD  # untouched


def test_config_show_single_key_prints_untruncated_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "show", "sandbox.run_commands"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.run_commands" in out and "value:" in out and "source:" in out
    # A section prefix shows all its leaves.
    assert main(["config", "show", "sandbox"]) == 0
    assert "sandbox.agent_network" in capsys.readouterr().out


def test_config_show_unknown_key_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "show", "nope.nope"]) == 2
    assert "no config key matches" in capsys.readouterr().err


def test_config_add_on_scalar_key_says_not_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The key is unset in the target file, so the old file-only guard let a
    # scalar through to revalidation, which printed a self-contradictory
    # "'local' is not valid ... Input should be 'providers', 'local' or 'open'".
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "add", "sandbox.agent_network", "local"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a list field" in err
    assert "Input should be" not in err


def test_config_add_on_unset_list_key_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "add", "sandbox.allow_urls", "https://example.com"])
    assert rc == 0
    assert "Added" in capsys.readouterr().out


def test_config_add_on_unset_optional_list_key_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A `list[str] | None` field (providers.*.token_command, default None) whose
    # value is unset must not be misread as a scalar: the effective value is
    # None, which is not proof of scalar-ness. Regression for a guard that
    # refused `config add` on it with a "not a list field" error.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    global_config_path().write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    rc = main(["config", "add", "providers.anthropic.token_command", "aws-vault"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Added" in out


def test_config_set_keeps_a_valid_write_despite_a_stale_value_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A value left invalid by a schema change (prompt.decompose = true, once a bool,
    # now Literal["auto","on","off"]) must NOT block setting an unrelated valid key:
    # the write is kept (not reverted) and a WARNING names the exact file + command
    # to fix the stale value, so it is self-service. The old strict behaviour made a
    # broken config impossible to fix through `config set`.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "set", "budget.best_effort_usd_limit", "5"])
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A repo overlay masks the key in the merged view; setting an INVALID value in
    # a lower layer must still be rejected, or it lands unvalidated and only
    # explodes later where the mask is absent.
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Setting an invalid VALUE still fails loud and reverts, so a typo cannot land in
    # the config even though a stale value elsewhere no longer blocks a valid write.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "prompt.decompose", "bogus"])
    assert rc == 2  # the write itself is invalid -> reverted + fail loud
    assert "prompt.decompose" in capsys.readouterr().err
    gpath = global_config_path()
    assert not gpath.is_file() or "decompose" not in gpath.read_text(encoding="utf-8")


def test_config_set_reverts_a_write_that_trips_a_non_pydantic_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A write that trips a STANDALONE ConfigError (not a pydantic per-leaf error) --
    # e.g. a non-absolute agent6.state_dir -- must still revert. Such errors carry no
    # "  - <leaf>:" line, so the before/after comparison must count full error content
    # (else the invalid write is silently kept and bricks the config).
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "agent6.state_dir", "not-absolute"])
    assert rc == 2
    assert "absolute" in capsys.readouterr().err.lower()


def test_config_set_keeps_a_write_on_an_already_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # config set never lands a known-INVALID written value (it validates the value
    # standalone), but a config left invalid by a stale value elsewhere STAYS
    # fixable: a VALID write still succeeds. So on an already-broken config,
    # writing another invalid value is rejected, while a valid value lands + clears.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "set", "prompt.decompose", "enabled"]) == 2  # invalid value: rejected
    assert "prompt.decompose" in capsys.readouterr().err  # friendly per-field error
    assert main(["config", "set", "prompt.decompose", "on"]) == 0  # a valid value clears it
    assert "WARNING" not in capsys.readouterr().err


def test_config_set_unknown_leaf_gets_a_did_you_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unknown key under a known section spoke pydantic ("Extra inputs are
    # not permitted"); a typo deserves the near-miss and the show pointer.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "sandbox.run_command", "yes"])  # missing 's'
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'sandbox.run_command'" in err
    assert "'sandbox.run_commands'" in err  # the did-you-mean
    assert "Extra inputs" not in err


def test_config_set_unknown_section_gets_the_same_friendly_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unknown TOP-LEVEL section errors at the section loc (a parent of the
    # written key); it used to fall through to the merged-layer dump with
    # "(merged config layers)" and a raw type=extra_forbidden.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "bogus.key", "foo"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown config key 'bogus.key'" in err
    assert "merged config layers" not in err and "extra_forbidden" not in err
    gpath = global_config_path()
    assert not gpath.is_file() or "bogus" not in gpath.read_text(encoding="utf-8")


def test_config_set_accepts_a_profiles_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # [profiles.*] is meta-config stripped before validation (_apply_profile),
    # so the schema forbids it by design; the unknown-key reroute must not
    # reject a legitimate, documented profile write (it did: rc 2 + revert).
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "profiles.mine.review.trigger", "before_finish"])
    assert rc == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[profiles.mine.review]" in text
    assert "unknown config key" not in capsys.readouterr().err


def test_config_set_bool_error_speaks_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rc = main(["config", "set", "sandbox.protect_git", "notabool"])
    assert rc == 2
    assert "sandbox.protect_git: expected true or false, got 'notabool'" in capsys.readouterr().err


def test_config_set_global_keeps_a_valid_write_shadowed_by_a_stale_repo_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The exact motivating case: prompt.decompose is stale in the REPO layer; setting a
    # VALID value GLOBALLY (which the repo still shadows) must be KEPT + warn, NEVER
    # reverted -- the leaf appears in the merged error but this write is not its cause.
    from agent6.config.layer import repo_config_path_for
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    repo_cfg = repo_config_path_for(tmp_path)
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    repo_cfg.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "set", "prompt.decompose", "auto"])
    captured = capsys.readouterr()
    assert rc == 0  # the valid global write is KEPT, not reverted over the repo's stale value
    assert "WARNING" in captured.err  # ...but warns the repo layer still shadows it
    assert '"auto"' in global_config_path().read_text(encoding="utf-8")


def test_config_set_allows_a_cross_field_write_valid_given_a_set_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A value whose validity depends on a SIBLING (a cross-field @model_validator) must
    # be accepted once that sibling is set. The up-front pre-check validates the leaf
    # in isolation (siblings at defaults), so it must NOT attribute the resulting
    # parent-table error to the written child, or it would wrongly reject e.g.
    # sandbox.tool_network='allow' after sandbox.agent_network='open' is already set.
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "sandbox.agent_network", "open"]) == 0
    assert main(["config", "set", "sandbox.tool_network", "allow"]) == 0  # not over-rejected


def test_config_set_sub_leaf_on_an_existing_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # providers.<name> is a discriminated union on api_format, so a leaf's isolated dict
    # lacks the union tag and errors on the parent providers.<name>; the pre-check must
    # not attribute that to the written child, or every providers.<name>.* set on an
    # already-complete provider would be rejected.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        '[providers.op]\napi_format = "openai"\nbase_url = "https://x.test/v1"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "providers.op.base_url", "https://y.test/v1"]) == 0


def test_config_set_submodel_inline_table_completed_by_a_lower_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An inline-table value for a submodel key that a LOWER layer completes must be
    # accepted: the isolated pre-check sees only the partial table (missing a required
    # child), so it must not attribute that descendant error to the written key.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[models.worker]\nmodel = "m"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["config", "set", "--repo", "models.worker", '{ provider = "p" }']) == 0


# --- `config fix`: drop invalid entries, print what was dropped and where -------


def test_config_fix_drops_a_bad_value_and_keeps_valid_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # prompt.decompose = true is invalid (now Literal["auto","on","off"]); the valid
    # budget entry beside it must survive the repair.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        "[prompt]\ndecompose = true\n[budget]\nbest_effort_usd_limit = 5.0\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "global" in out  # named the entry + its layer
    text = gpath.read_text(encoding="utf-8")
    assert "decompose" not in text  # the invalid entry is gone
    assert "best_effort_usd_limit" in text  # the valid one stays
    assert main(["config", "show"]) == 0  # config is valid now


def test_config_fix_drops_an_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[sandbox]\nprotct_git = true\n", encoding="utf-8")  # typo of protect_git
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sandbox.protct_git" in out
    assert "protct_git" not in gpath.read_text(encoding="utf-8")


def test_config_fix_labels_a_repo_layer_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config.layer import repo_config_path_for
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    rpath = repo_config_path_for(tmp_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "prompt.decompose" in out and "repo" in out
    assert "decompose" not in rpath.read_text(encoding="utf-8")


def test_config_fix_on_valid_config_reports_nothing_to_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    before = "[budget]\nbest_effort_usd_limit = 5.0\n"
    gpath.write_text(before, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nothing to fix" in out.lower()
    assert gpath.read_text(encoding="utf-8") == before  # untouched


def test_config_fix_repairs_both_layers_and_labels_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config.layer import repo_config_path_for
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("[prompt]\ndecompose = true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rpath = repo_config_path_for(tmp_path)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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


def test_config_fix_reports_an_entry_it_cannot_auto_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A non-absolute agent6.state_dir is rejected before the model loads (it locates
    # the per-repo config dir), so fix cannot drop it as a plain leaf. It must SAY so
    # and exit non-zero, never silently report a still-broken config as fixed.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('[agent6]\nstate_dir = "not-absolute"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "state_dir" in err


def test_config_fix_drops_an_unknown_top_level_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A leftover [cli] table (from a removed feature) is an unknown TOP-LEVEL table:
    # pydantic reports extra_forbidden at "cli" (not "cli.input"), and the WHOLE table
    # must be dropped -- removing just the leaf would leave an empty [cli] that is
    # still invalid. Regression for `config fix` reporting it unfixable.
    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(
        '[cli]\ninput = "bar"\n[budget]\nbest_effort_usd_limit = 5.0\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    rc = main(["config", "fix"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cli" in out  # named the removed table
    text = gpath.read_text(encoding="utf-8")
    assert "[cli]" not in text  # the whole table is gone, not just a leaf line
    assert "best_effort_usd_limit" in text  # the valid section stays
    assert main(["config", "show"]) == 0  # config is valid now


def test_config_set_unknown_provider_key_speaks_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A provider entry is a discriminated union; the standalone minimal dict
    # cannot resolve a member, so every provider-key error used to fall through
    # to the merged-layer pydantic dump (with the member tag leaking into the
    # displayed loc). The member models answer directly now.
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    # A Field-constraint violation (gt=0) gets the same member answer, not the
    # merged dump with the discriminator tag leaking into the loc.
    rc = main(["config", "set", "providers.p.http_timeout_s", "-5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "greater than 0" in err
    assert "merged config layers" not in err and ".anthropic." not in err
    # A partial-entry write some member accepts still lands (never reverted).
    assert main(["config", "set", "providers.p.base_url", "https://x.example/v1"]) == 0


def test_config_show_key_with_json_filters_to_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config show <key> --json` dumped the ENTIRE config, silently ignoring
    the key; it must filter like the text path and error on an unknown key."""
    import json as jsonlib

    from agent6.paths import global_config_path
    from agent6.ui.cli import main

    gpath = global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "show", "sandbox.agent_network", "--json"]) == 0
    payload = jsonlib.loads(capsys.readouterr().out)
    assert set(payload) == {"sandbox.agent_network"}

    assert main(["config", "show", "sandbox", "--json"]) == 0
    payload = jsonlib.loads(capsys.readouterr().out)
    assert payload and all(k.startswith("sandbox.") for k in payload)

    assert main(["config", "show", "nope.nope", "--json"]) == 2
    assert "no config key matches" in capsys.readouterr().err
