# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`resolve_strict_egress_viability`: strict selected but this process can't
create a userns for the egress broker (surgical AppArmor isolation case)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent6.app import egress
from agent6.config import Config


def _cfg(isolation: str, agent_network: str, tool_network: str = "auto") -> Config:
    return cast(
        Config,
        SimpleNamespace(
            sandbox=SimpleNamespace(
                isolation=isolation, agent_network=agent_network, tool_network=tool_network
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
    isolation, err = egress.resolve_strict_egress_viability(_cfg("auto", "providers"), "strict")
    assert isolation == "hardened" and err is None
    assert "Falling back to the hardened isolation" in capsys.readouterr().err


def test_explicit_strict_refuses_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    isolation, err = egress.resolve_strict_egress_viability(_cfg("strict", "providers"), "strict")
    assert isolation == "strict"  # not silently downgraded for an explicit request
    assert err is not None and "REFUSING" in err and "apparmor_restrict" in err


def test_local_refuses_rather_than_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    # agent_network='local' needs the broker and has no hardened fallback, so it
    # must refuse (even for auto) -- NOT silently downgrade to hardened, which
    # would bypass check_network_profile's local-on-hardened refusal.
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    isolation, err = egress.resolve_strict_egress_viability(_cfg("auto", "local"), "strict")
    assert isolation == "strict" and err is not None
    assert "REFUSING" in err and "local" in err


def test_explicit_tool_network_block_blocks_the_hardened_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An EXPLICIT tool_network='block' can't be honored on hardened (no netns),
    # so auto must NOT silently downgrade to hardened -- it refuses, naming the
    # gap. The secure default 'auto' would degrade there with a warning instead.
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    isolation, err = egress.resolve_strict_egress_viability(
        _cfg("auto", "providers", tool_network="block"), "strict"
    )
    assert isolation == "strict" and err is not None
    assert "REFUSING" in err and "block" in err


def test_only_explicit_states_refuses_rather_than_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tool_network='only_explicit_states' also requires strict (no hardened
    # fallback), so the downgrade must refuse it, not silently under-confine.
    monkeypatch.setattr(egress, "probe_userns_supported", lambda: False)
    cfg = _cfg("auto", "providers", tool_network="only_explicit_states")
    isolation, err = egress.resolve_strict_egress_viability(cfg, "strict")
    assert isolation == "strict" and err is not None
    assert "REFUSING" in err and "only_explicit_states" in err


def test_broker_sockets_ignore_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The broker's provider sockets must never land where a jail can name them.

    AF_UNIX PATHNAME sockets are not namespaced by the network namespace, so the
    jail's MOUNT namespace is the only barrier -- and a read-only bind still
    permits connect(). tempfile's default base honours $TMPDIR, so a $TMPDIR
    inside the workspace put live provider tunnels in the one directory every
    jail bind-mounts read-write: an LLM-chosen argv could connect() straight to
    the operator's provider."""
    from agent6.app.egress import _egress_socket_dir  # pyright: ignore[reportPrivateUsage]

    workspace = tmp_path / "repo"
    (workspace / ".tmp").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(workspace / ".tmp"))
    sock_dir = _egress_socket_dir(Config(), workspace)
    try:
        assert workspace not in sock_dir.parents
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_broker_socket_dir_refuses_an_exposing_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Placement alone is not enough: an extra_read_paths grant (or a workspace)
    containing the chosen directory bind-mounts it into every jail, so the run
    is refused rather than started with reachable provider tunnels."""
    from agent6.app.egress import _egress_socket_dir  # pyright: ignore[reportPrivateUsage]
    from agent6.sandbox.broker import EgressBrokerError

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    cfg = Config.model_validate({"sandbox": {"extra_read_paths": [str(runtime)]}})
    with pytest.raises(EgressBrokerError, match="provider tunnel"):
        _egress_socket_dir(cfg, tmp_path / "repo")
