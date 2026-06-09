# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`config set/add/remove --machine` re-validates the whole machine spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.cli import config_cmds as cc


def _noop_overlay(*_a: object, **_k: object) -> None:
    # Stub for load_effective_with_overlay so the test isolates machine-spec
    # validation from the cwd-dependent [config]-overlay validation.
    return None


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
