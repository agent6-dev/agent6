# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Network confinement for a SPAWNED MCP server, at the security level.

`network = "none"` (the default) must leave the server with no way out, and
must not pay for that by handing it anything else: the launcher holds a full
capability set between `unshare` and `execve`, and a server that inherited it
would have MORE power in exchange for losing its network.

These assert the OUTCOME (no capabilities, a netns it cannot leave, no
reachable host), never the mechanism, so they survived MCP moving from its own
shim onto the shared jail launcher.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sandbox.jail import spawn_in_jail
from agent6.tools.policy import jail_policy
from agent6.types import NetworkMode

pytestmark = pytest.mark.needs_namespaces


def _probe(script: str, cwd: Path, *, network: NetworkMode = "none") -> str:
    """Run one probe as a SERVER would run: spawned through the jail with a
    server policy, stdio inherited, output collected off its stdout pipe."""
    argv = ("/usr/bin/python3", "-c", script)
    policy = jail_policy(cwd, Config(), "strict", argv, network=network)
    proc = spawn_in_jail(
        policy,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _err = proc.communicate(timeout=30)
    return out.decode(errors="replace")


def test_a_confined_server_gains_no_capabilities(tmp_path: Path) -> None:
    """Confinement must never be a privilege trade: the launcher holds a full
    capability set between `unshare` and `execve`, and any of it reaching the
    server would be handing third-party code MORE power in exchange for taking
    its network away.

    The uid INSIDE is namespace-local root -- that is how the jail mounts its
    own root -- so the identity question is answered outside: a file the
    server creates belongs to the operator, and its capability sets, bounding
    set included, are empty.
    """
    script = (
        "import os, re\n"
        "s = open('/proc/self/status').read()\n"
        "f = lambda k: re.search(rf'^{k}:\\s*(.*)$', s, re.M).group(1).split()[0]\n"
        "open('made-by-server.txt', 'w').write('x')\n"
        "print(f('CapPrm'), f('CapEff'), f('CapBnd'))\n"
    )
    caps = _probe(script, tmp_path).split()
    assert len(caps) == 3, caps
    assert all(int(c, 16) == 0 for c in caps), f"the server holds capabilities: {caps}"
    made = tmp_path / "made-by-server.txt"
    assert made.is_file(), "the probe did not run"
    assert made.stat().st_uid == os.getuid(), "the server acted as someone other than the operator"


def test_the_server_lands_in_a_namespace_it_cannot_leave(tmp_path: Path) -> None:
    """Rejoining the host network needs a handle on its namespace and
    CAP_SYS_ADMIN there. The server gets neither: a process in the parent user
    namespace fails the ptrace check, so `/proc/<ppid>/ns/net` will not even
    open -- and setns would refuse a capability-less process anyway."""
    script = (
        "import os\n"
        "print('NETNS', os.readlink('/proc/self/ns/net'))\n"
        "try:\n"
        "    fd = os.open(f'/proc/{os.getppid()}/ns/net', os.O_RDONLY)\n"
        "    print('ESCAPE-OPENED')\n"
        "except OSError as exc:\n"
        "    print('ESCAPE-REFUSED', type(exc).__name__)\n"
    )
    out = _probe(script, tmp_path)
    assert "ESCAPE-REFUSED" in out, out
    assert str(Path("/proc/self/ns/net").readlink()) not in out


def test_a_confined_server_cannot_reach_a_live_listener(tmp_path: Path) -> None:
    """The positive control matters: a DNS probe fails inside any jail and on
    any offline host either way, proving nothing. Connect to a REAL listener
    on this machine -- denied without the network, allowed with it."""
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        script = (
            "import socket, sys\n"
            "s = socket.socket()\n"
            "s.settimeout(3)\n"
            f"print('CONNECT', s.connect_ex(('127.0.0.1', {port})))\n"
        )
        blocked = _probe(script, tmp_path, network="none")
        allowed = _probe(script, tmp_path, network="host")
    assert "CONNECT 0" not in blocked, f"a confined server reached the host: {blocked}"
    assert "CONNECT 0" in allowed, f"network = host did not reach the listener: {allowed}"


def test_the_jail_binary_is_what_confines_a_server(tmp_path: Path) -> None:
    """One implementation, asserted: a server is confined by the same launcher
    a jailed command uses, so there is no second code path to keep in step.
    (The Python Landlock shim MCP used to carry is gone; if it comes back,
    this fails.)"""
    assert not (Path(__file__).parents[2] / "src/agent6/sandbox/exec_confined.py").exists()
    # A confined server is PID 2 in its OWN pid namespace (the launcher is PID
    # 1). An unconfined spawn keeps a host pid, so this fails if confinement is
    # ever bypassed -- unlike "python3 is in the cmdline", true of any spawn.
    script = "import os; print('PID', os.getpid())\n"
    out = _probe(script, tmp_path)
    assert "PID 2" in out, out
