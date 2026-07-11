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
