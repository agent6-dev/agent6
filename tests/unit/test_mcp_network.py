# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[mcp.servers.<n>.sandbox].network`: the per-server switch.

Same three values and same meaning as `[sandbox].tool_network`, because it is
the same axis at a different scope -- one vocabulary, one set of rules. The
outcome (a confined server cannot reach a live listener) is asserted in
`tests/security/test_mcp_network_confinement.py`; this file is the config
contract and the refuse/degrade split.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.confine import check_mcp_network_support
from agent6.config import Config


def _server(body: dict[str, object]) -> Config:
    return Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"s": {"command": ["x"], "sandbox": body}}}}
    )


def test_network_defaults_to_auto() -> None:
    """The secure-and-degrading default, matching every other `auto` in the
    config: no network where the host can provide a namespace, a warning where
    it cannot. The old default was permissive because a server had no other
    confinement to lose; it is confined like a command now."""
    cfg = _server({})
    sandbox = cfg.mcp.servers["s"].sandbox
    assert sandbox is not None
    assert sandbox.network == "auto"


def test_the_three_values_are_tool_networks_three_values() -> None:
    for value in ("auto", "allow", "block"):
        cfg = _server({"network": value})
        sandbox = cfg.mcp.servers["s"].sandbox
        assert sandbox is not None and sandbox.network == value
    with pytest.raises(ValueError, match="network"):
        _server({"network": "none"})  # the old word; one axis, one vocabulary


def test_explicit_block_refuses_where_there_is_no_namespace() -> None:
    """An operator who wrote `block` asked for enforcement, so a host that
    cannot give a network namespace refuses rather than running the server
    connected."""
    cfg = _server({"network": "block"})
    err = check_mcp_network_support(cfg, "hardened")
    assert err is not None and "'s'" in err and "strict" in err
    assert check_mcp_network_support(cfg, "strict") is None


def test_auto_degrades_with_a_warning_naming_the_server(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The default cannot be enforced everywhere, so where it cannot it says
    so once, per server, at MCP setup -- where the operator is already being
    told about their servers, rather than inside the isolation warner, which
    is about the level's own gaps."""
    from agent6.app._setup import start_mcp_manager_if_enabled

    cfg = _server({})
    assert check_mcp_network_support(cfg, "hardened") is None  # never a refusal
    start_mcp_manager_if_enabled(cfg, tmp_path, "hardened")
    err = capsys.readouterr().err
    assert "MCP server 's'" in err and "network" in err


def test_allow_is_silent(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """An operator who granted the network asked for it; nothing degraded."""
    from agent6.app._setup import start_mcp_manager_if_enabled

    cfg = _server({"network": "allow"})
    assert check_mcp_network_support(cfg, "hardened") is None
    start_mcp_manager_if_enabled(cfg, tmp_path, "hardened")
    assert "MCP server" not in capsys.readouterr().err
