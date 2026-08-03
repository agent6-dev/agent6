# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Network confinement for a SPAWNED MCP server.

`network = "none"` puts the server in its own network namespace: it keeps a
loopback of its own and reaches nothing else. Opt-in, because most servers are
there to talk to something.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.mcp_client import (
    MCPConfinement,
    _confined_argv,  # pyright: ignore[reportPrivateUsage]
)

_SHIM = [sys.executable, "-m", "agent6.sandbox.exec_confined"]


def _server(body: dict[str, object]) -> Config:
    return Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"s": {"command": ["x"], "sandbox": body}}}}
    )


def test_network_defaults_to_host() -> None:
    """The permissive default: an MCP server usually exists to reach something,
    and a default that broke every one of them would just be turned off."""
    cfg = _server({"read_paths": ["/usr"]})
    assert cfg.mcp.servers["s"].sandbox is not None
    assert cfg.mcp.servers["s"].sandbox.network == "host"


def test_network_none_is_accepted() -> None:
    cfg = _server({"read_paths": ["/usr"], "network": "none"})
    assert cfg.mcp.servers["s"].sandbox is not None
    assert cfg.mcp.servers["s"].sandbox.network == "none"


def test_network_auto_is_refused() -> None:
    """`auto` means "the most secure available, degrade with a warning"
    everywhere else in agent6. One word cannot also mean permissive."""
    with pytest.raises(ValueError, match="network"):
        _server({"read_paths": ["/usr"], "network": "auto"})


def test_host_network_adds_no_flag() -> None:
    argv = _confined_argv(("npx", "s"), MCPConfinement(read_paths=("/usr",)))
    assert "--no-network" not in argv


def test_none_network_passes_the_flag() -> None:
    argv = _confined_argv(("npx", "s"), MCPConfinement(read_paths=("/usr",), network="none"))
    assert "--no-network" in argv


@pytest.mark.needs_namespaces
def test_the_shim_really_moves_the_server_into_its_own_netns() -> None:
    """The whole point: what the server can reach, not what it says it will.
    Compared by namespace inode, so it needs no reachable network to prove."""
    ours = Path("/proc/self/ns/net").readlink()
    got = subprocess.run(
        [
            *_SHIM,
            "--read",
            "/",
            "--no-network",
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; print(Path('/proc/self/ns/net').readlink())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert got.stdout.strip() != str(ours)


@pytest.mark.needs_namespaces
def test_the_confined_server_keeps_its_own_loopback_up() -> None:
    """A fresh netns brings `lo` up DOWN. Leaving it there breaks any server
    that binds localhost -- and confines nothing extra, since the loopback of a
    private netns reaches only that namespace."""
    got = subprocess.run(
        [
            *_SHIM,
            "--read",
            "/",
            "--no-network",
            "--",
            sys.executable,
            "-c",
            "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[0])",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == "127.0.0.1"


@pytest.mark.needs_namespaces
def test_the_confined_server_keeps_the_operators_uid() -> None:
    """A user namespace remaps everyone to nobody unless uid_map is written,
    and a server that suddenly cannot read its own files is a confusing way to
    learn that."""
    got = subprocess.run(
        [
            *_SHIM,
            "--read",
            "/",
            "--no-network",
            "--",
            sys.executable,
            "-c",
            "import os; print(os.getuid(), os.getgid())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert got.stdout.split() == [str(os.getuid()), str(os.getgid())]


# The shim confines the process it runs IN, irrevocably. Every test of it runs
# it as a subprocess: an in-process call that reached the Landlock line would
# apply the domain to pytest itself and take the rest of the suite with it.
_REFUSE_UNSHARE = (
    "import os, sys;"
    " os.unshare = lambda *_a: (_ for _ in ()).throw(OSError(1, 'Operation not permitted'));"
    " import agent6.sandbox.exec_confined as m;"
    " sys.exit(m.main(sys.argv[1:]))"
)


def test_a_host_that_cannot_unshare_refuses_rather_than_running_connected(tmp_path: Path) -> None:
    """`none` is an explicit enforce value, so it refuses when the environment
    cannot honor it -- never a warning over a server that still has the network.
    """
    got = subprocess.run(
        [
            sys.executable,
            "-c",
            _REFUSE_UNSHARE,
            "--read",
            str(tmp_path),
            "--no-network",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert got.returncode == 2, got.stderr
    assert "network" in got.stderr and "unprivileged user namespaces" in got.stderr


def test_network_only_confinement_is_expressible() -> None:
    """The two axes are independent. Requiring read_paths to reach the network
    knob made the likeliest case -- "offline, but I am not enumerating your
    filesystem" -- unsayable."""
    cfg = _server({"network": "none"})
    sandbox = cfg.mcp.servers["s"].sandbox
    assert sandbox is not None
    assert sandbox.network == "none" and sandbox.read_paths == ()


def test_a_block_that_confines_nothing_is_refused() -> None:
    """`network = "host"` with no paths is an inert block: the operator asked
    for confinement and would have got none."""
    with pytest.raises(ValueError, match="read_paths"):
        _server({"network": "host"})


def test_require_without_read_paths_is_refused() -> None:
    """`require` is about a Landlock domain; with no paths there is none to
    demand, so accepting it would promise confinement that never happens."""
    with pytest.raises(ValueError, match="read_paths"):
        _server({"network": "none", "require": True})


def test_network_only_does_not_landlock_the_server() -> None:
    """No paths named means no filesystem domain: applying an empty one grants
    NOTHING and the server dies on its own interpreter."""
    argv = _confined_argv(("npx", "s"), MCPConfinement(network="none"))
    assert "--read" not in argv and "--write" not in argv and "--require" not in argv
    assert "--no-network" in argv


@pytest.mark.needs_namespaces
def test_a_network_only_server_still_runs_and_still_has_no_network() -> None:
    ours = Path("/proc/self/ns/net").readlink()
    got = subprocess.run(
        [
            *_SHIM,
            "--no-network",
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; print(Path('/proc/self/ns/net').readlink())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert got.stdout.strip() != str(ours)


def test_the_shim_refuses_to_confine_nothing() -> None:
    """Asking for neither mechanism would apply an EMPTY Landlock domain --
    granting nothing, to the shim and to whatever it becomes."""
    got = subprocess.run(
        [*_SHIM, "--", sys.executable, "-c", "pass"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert got.returncode == 2, got.stdout
    assert "nothing to confine" in got.stderr
