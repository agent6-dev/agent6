# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the policy builder grants a jailed command beyond the workspace."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.tools import policy


def _host_cfg() -> Config:
    return Config.model_validate({"sandbox": {"network": "host"}})


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_the_host_network_grants_the_resolver_and_trust_store_files(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A jail on the host network had no resolver file and no CA store, so no name resolved."""
    reads = policy.host_network_reads()
    for name in (
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/ssl",
        "/etc/pki",
        "/etc/ca-certificates",
    ):
        assert (pathlib.Path(name) in reads) == pathlib.Path(name).exists(), name
    resolv = pathlib.Path("/etc/resolv.conf")
    if resolv.exists():
        # The launcher binds the name the resolver opens; Landlock needs the file it points to.
        assert resolv in reads and resolv.resolve() in reads
    assert reads, "no host in the test matrix lacks every one of them"
    granted = policy.jail_policy(tmp_path, _host_cfg(), isolation, ("true",)).extra_ro_paths
    assert set(reads) <= set(granted)


def test_a_session_or_absent_network_grants_no_host_network_file(tmp_path: pathlib.Path) -> None:
    """The resolver and trust-store grants follow the host network only."""
    reads = set(policy.host_network_reads())
    for network in ("session", "none"):
        built = policy.jail_policy(tmp_path, Config(), "strict", ("true",), network=network)  # pyright: ignore[reportArgumentType]
        assert built.network == network
        assert not reads & set(built.extra_ro_paths)


def test_a_host_network_file_is_listed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolver file that is no symlink is its own target, granted once."""
    real = pathlib.Path("/etc/hosts")
    monkeypatch.setattr(policy, "_HOST_NETWORK_FILES", (str(real),))
    assert policy.host_network_reads() == ((real,) if real.exists() else ())
