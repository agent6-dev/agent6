# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`network = "none"` confines a server without handing it anything new.

Entering a user namespace is what gives an unprivileged process CAP_NET_ADMIN
over its own netns -- enough to bring `lo` up. Those capabilities are real
inside the shim, so what matters is that NONE of them survive into the server
the shim becomes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_SHIM = [sys.executable, "-m", "agent6.sandbox.exec_confined"]
_PROBE = (
    "import re\n"
    "s = open('/proc/self/status').read()\n"
    "f = lambda k: re.search(rf'^{k}:\\s*(.*)$', s, re.M).group(1).split()[0]\n"
    "from pathlib import Path\n"
    "print(f('Uid'), f('Gid'), f('CapPrm'), f('CapEff'),"
    " Path('/proc/self/ns/net').readlink())"
)


def _probe(*flags: str) -> list[str]:
    got = subprocess.run(
        [*_SHIM, *flags, "--", sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    return got.stdout.split()


def _ours() -> tuple[str, str]:
    status = Path("/proc/self/status").read_text(encoding="utf-8")

    def field(key: str) -> str:
        found = re.search(rf"^{key}:\s*(.*)$", status, re.M)
        assert found is not None
        return found.group(1).split()[0]

    return field("CapPrm"), field("CapEff")


@pytest.mark.needs_namespaces
def test_a_network_confined_server_gains_no_capabilities() -> None:
    """The shim holds a full capability set between `unshare` and `execve`.
    If any of it reached the server, `network = "none"` would be handing third
    party code MORE power in exchange for taking its network away."""
    uid, gid, cap_prm, cap_eff, _netns = _probe("--no-network")
    assert (uid, gid) == (str(os.getuid()), str(os.getgid())), "identity was remapped"
    assert (cap_prm, cap_eff) == _ours(), "the server gained capabilities agent6 does not have"
    assert int(cap_prm, 16) == 0


@pytest.mark.needs_namespaces
def test_the_server_lands_in_a_namespace_it_cannot_leave() -> None:
    """Rejoining the host network needs a handle on its namespace and
    CAP_SYS_ADMIN there. The server gets neither: a process in the parent user
    namespace fails the ptrace check, so `/proc/<ppid>/ns/net` will not even
    open -- and setns would refuse a capability-less process anyway."""
    *_caps, netns = _probe("--no-network")
    assert netns != str(Path("/proc/self/ns/net").readlink())

    escape = subprocess.run(
        [
            *_SHIM,
            "--no-network",
            "--",
            sys.executable,
            "-c",
            # The PARENT's netns: same uid, so only the namespace boundary can
            # refuse it. Either failure mode is a refusal; "rejoined" is not.
            "import ctypes, os\n"
            "try:\n"
            "    fd = os.open(f'/proc/{os.getppid()}/ns/net', os.O_RDONLY)\n"
            "except OSError as exc:\n"
            "    print('refused: no handle', exc.errno)\n"
            "else:\n"
            "    rc = ctypes.CDLL(None, use_errno=True).setns(fd, 0)\n"
            "    print('rejoined' if rc == 0 else 'refused: setns')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "rejoined" not in escape.stdout, escape.stdout + escape.stderr
    assert escape.stdout.startswith("refused"), escape.stdout + escape.stderr
