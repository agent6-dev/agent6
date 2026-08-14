# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[mcp.servers.<n>.sandbox].network`: the per-server switch.

One axis at a different scope, so the shared words mean the same thing and the
refuse/degrade rules are identical. Each side has exactly one value the other
lacks, and each has a reason: a server can be `none` (alone) because it is one
process, and only `tool_network` has `only_explicit_states`, which is about
machine tool states. The
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


def test_the_words_mean_the_same_on_both_sides() -> None:
    """One axis, one vocabulary. A server takes every word `tool_network` takes
    except `only_explicit_states` (a machine-tool concept), plus `none`, which
    a single process can be and a group of commands cannot."""
    from agent6.config import SandboxConfig

    for value in ("auto", "none", "session", "host"):
        cfg = _server({"network": value})
        sandbox = cfg.mcp.servers["s"].sandbox
        assert sandbox is not None and sandbox.network == value
    with pytest.raises(ValueError, match="network"):
        _server({"network": "only_explicit_states"})  # a machine-tool concept
    with pytest.raises(ValueError, match="network"):
        SandboxConfig(network="none")  # type: ignore[arg-type]


def test_explicit_block_refuses_where_there_is_no_namespace() -> None:
    """An operator who wrote `block` asked for enforcement, so a host that
    cannot give a network namespace refuses rather than running the server
    connected."""
    cfg = _server({"network": "none"})
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


def test_a_table_less_server_degrades_with_the_same_warning(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A server with no `[sandbox]` table has effective network `auto`, exactly
    like one that wrote `network = "auto"`: the spawn policy already gives it a
    network of its own (degrading to the host's on hardened). The degrade
    warning has to agree, or a table-less server keeps the host's network on
    hardened with nothing said -- the policy and the warning read the same
    effective value now."""
    from agent6.app._setup import start_mcp_manager_if_enabled

    cfg = Config.model_validate({"mcp": {"enabled": True, "servers": {"s": {"command": ["x"]}}}})
    assert cfg.mcp.servers["s"].sandbox is None
    assert cfg.mcp.servers["s"].effective_network == "auto"
    assert check_mcp_network_support(cfg, "hardened") is None  # auto never refuses
    start_mcp_manager_if_enabled(cfg, tmp_path, "hardened")
    err = capsys.readouterr().err
    assert "MCP server 's'" in err and "network" in err


def test_host_is_silent(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """An operator who granted the network asked for it; nothing degraded."""
    from agent6.app._setup import start_mcp_manager_if_enabled

    cfg = _server({"network": "host"})
    assert check_mcp_network_support(cfg, "hardened") is None
    start_mcp_manager_if_enabled(cfg, tmp_path, "hardened")
    assert "MCP server" not in capsys.readouterr().err


@pytest.mark.parametrize("value", ["auto", "session", "host"])  # the shared words
@pytest.mark.parametrize("isolation", ["strict", "hardened", "none"])
def test_the_per_server_knob_answers_exactly_like_tool_network(value: str, isolation: str) -> None:
    """One axis, one vocabulary, one set of rules. They drifted: the
    per-server guard read `if isolation == "strict"` where the sibling reads
    `if isolation != "hardened"`, so the enforce value refused under `none` -- where
    nothing is confined anyway and the blanket unsandboxed warning covers it.
    A table of the two side by side is what caught it, so here is the table."""
    from agent6.app.confine import check_network_support
    from agent6.config import SandboxConfig

    per_server = check_mcp_network_support(_server({"network": value}), isolation)  # type: ignore[arg-type]
    global_knob = check_network_support(
        Config(sandbox=SandboxConfig(network=value)),  # type: ignore[arg-type]
        isolation,  # type: ignore[arg-type]
    )
    assert (per_server is None) == (global_knob is None), (
        f"{value!r} on {isolation!r}: per-server says"
        f" {'refuse' if per_server else 'run'}, tool_network says"
        f" {'refuse' if global_knob else 'run'}"
    )
