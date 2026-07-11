# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`resolve_strict_egress_viability`: strict selected but this process can't
create a userns for the egress broker (surgical AppArmor profile case)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from agent6.config import Config
from agent6.ui.cli import egress


def _cfg(profile: str, agent_network: str, tool_network: str = "block") -> Config:
    return cast(
        Config,
        SimpleNamespace(
            sandbox=SimpleNamespace(
                profile=profile, agent_network=agent_network, tool_network=tool_network
            )
        ),
    )


def test_hardened_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    assert egress.resolve_strict_egress_viability(_cfg("auto", "providers"), "hardened") == (
        "hardened",
        None,
    )


def test_open_needs_no_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    assert egress.resolve_strict_egress_viability(_cfg("auto", "open"), "strict") == (
        "strict",
        None,
    )


def test_broker_viable_when_process_can_userns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: True)
    assert egress.resolve_strict_egress_viability(_cfg("auto", "providers"), "strict") == (
        "strict",
        None,
    )


def test_auto_downgrades_to_hardened(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    profile, err = egress.resolve_strict_egress_viability(_cfg("auto", "providers"), "strict")
    assert profile == "hardened" and err is None
    assert "Falling back to the hardened profile" in capsys.readouterr().err


def test_explicit_strict_refuses_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    profile, err = egress.resolve_strict_egress_viability(_cfg("strict", "providers"), "strict")
    assert profile == "strict"  # not silently downgraded for an explicit request
    assert err is not None and "REFUSING" in err and "apparmor_restrict" in err


def test_local_refuses_rather_than_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    # agent_network='local' needs the broker and has no hardened fallback, so it
    # must refuse (even for auto) -- NOT silently downgrade to hardened, which
    # would bypass _check_network_profile's local-on-hardened refusal.
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    profile, err = egress.resolve_strict_egress_viability(_cfg("auto", "local"), "strict")
    assert profile == "strict" and err is not None
    assert "REFUSING" in err and "local" in err


def test_only_explicit_states_refuses_rather_than_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tool_network='only_explicit_states' also requires strict (no hardened
    # fallback), so the downgrade must refuse it, not silently under-confine.
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    cfg = _cfg("auto", "providers", tool_network="only_explicit_states")
    profile, err = egress.resolve_strict_egress_viability(cfg, "strict")
    assert profile == "strict" and err is not None
    assert "REFUSING" in err and "only_explicit_states" in err
