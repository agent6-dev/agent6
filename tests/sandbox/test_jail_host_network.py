# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a jail on the host network reads to resolve names and verify TLS."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import jail
from agent6.tools import policy

pytestmark = pytest.mark.needs_namespaces

_PROBE = (
    "for f in /etc/hosts /etc/resolv.conf /etc/nsswitch.conf; do"
    ' if [ -e "$f" ]; then cat "$f" >/dev/null && echo "READ $f"; else echo "ABSENT $f"; fi;'
    " done"
)


def _probe(root: pathlib.Path, network: kinds.NetworkMode) -> str:
    built = policy.jail_policy(root, Config(), "full", ("sh", "-c", _PROBE), network=network)
    res = jail.run_in_jail(built)
    return res.stdout + res.stderr


def test_full_on_the_host_network_reads_the_resolver_files(tmp_path: pathlib.Path) -> None:
    """Under `full` the host network came with an empty /etc, so every lookup by name failed."""
    out = _probe(tmp_path, "host")
    for name in ("/etc/hosts", "/etc/resolv.conf", "/etc/nsswitch.conf"):
        if pathlib.Path(name).exists():
            assert f"READ {name}" in out, out


def test_full_off_the_host_network_keeps_etc_empty(tmp_path: pathlib.Path) -> None:
    """The resolver files reach the jail with the host network only."""
    out = _probe(tmp_path, "none")
    assert "READ" not in out, out
