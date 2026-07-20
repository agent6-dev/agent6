# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sandbox probe: report Landlock ABI, never fail just because the kernel is old."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.sandbox import landlock as ll
from agent6.sandbox import landlock_abi

_BIND_TCP = 1 << 0  # _LANDLOCK_ACCESS_NET_BIND_TCP
_CONNECT_TCP = 1 << 1  # _LANDLOCK_ACCESS_NET_CONNECT_TCP


def test_landlock_abi_is_nonnegative() -> None:
    # On a kernel without Landlock the call returns 0; we never want a hard fail.
    abi = landlock_abi()
    assert isinstance(abi, int)
    assert abi >= 0


def test_empty_connect_ports_does_not_handle_connect_tcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: with an EMPTY tcp_connect_ports allow-list (agent_network='open'
    on a hardened host), CONNECT_TCP must NOT be in the handled set. Handling it
    with zero allow rules would deny EVERY outbound connect() by Landlock's
    deny-unless-allowed semantics, so the agent could not reach its provider --
    the opposite of the documented "open imposes no TCP restriction". BIND_TCP
    stays handled (no agent-owned listen surface); a NON-empty port list re-arms
    CONNECT_TCP so it confines to those ports.

    Stubs the syscall layer so it runs on any kernel (ABI forced to 8)."""
    captured: dict[str, int] = {}

    def fake_create(handled_fs: int, handled_net: int, abi: int) -> int:
        captured["net"] = handled_net
        return os.open(os.devnull, os.O_RDONLY)  # a real, closeable fd

    def noop_tcp(ruleset_fd: int, port: int, allowed_net: int) -> None:
        pass

    def noop_restrict(ruleset_fd: int) -> None:
        pass

    monkeypatch.setattr(ll, "landlock_abi", lambda: 8)
    monkeypatch.setattr(ll, "_set_no_new_privs", lambda: None)
    monkeypatch.setattr(ll, "_create_ruleset", fake_create)
    monkeypatch.setattr(ll, "_add_tcp_rule", noop_tcp)
    monkeypatch.setattr(ll, "_restrict_self", noop_restrict)

    ll.apply_agent_landlock(read_paths=(), write_paths=(), tcp_connect_ports=())
    assert captured["net"] & _BIND_TCP  # bind/listen still denied
    assert not (captured["net"] & _CONNECT_TCP)  # connects NOT restricted ("open")

    ll.apply_agent_landlock(read_paths=(), write_paths=(), tcp_connect_ports=(443,))
    assert captured["net"] & _CONNECT_TCP  # a real allow-list re-arms connect confinement


_TRUNCATE = 1 << 14  # _LANDLOCK_ACCESS_FS_TRUNCATE (ABI v3)


@pytest.mark.parametrize(("abi", "expect_truncate"), [(1, False), (2, False), (3, True)])
def test_handled_fs_masks_truncate_below_abi3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, abi: int, expect_truncate: bool
) -> None:
    """Regression: the kernel EINVALs any handled_access_fs bit above its ABI,
    so passing the ABI-3 TRUNCATE bit on a 5.13-5.18 kernel (ABI 1/2) refused
    every hardened run. handled_fs must be down-masked to the probed ABI, like
    handled_net already is; pre-ABI-3 truncation is governed by WRITE_FILE, so
    nothing is lost. Stubs the syscall layer so it runs on any kernel."""
    import os as _os

    captured: dict[str, int] = {}
    rule_bits: list[int] = []

    def fake_create(handled_fs: int, handled_net: int, _abi: int) -> int:
        captured["fs"] = handled_fs
        return _os.open(_os.devnull, _os.O_RDONLY)

    def fake_add_path(ruleset_fd: int, fd: int, allowed_fs: int) -> None:
        rule_bits.append(allowed_fs)

    def noop_restrict(ruleset_fd: int) -> None:
        pass

    monkeypatch.setattr(ll, "landlock_abi", lambda: abi)
    monkeypatch.setattr(ll, "_set_no_new_privs", lambda: None)
    monkeypatch.setattr(ll, "_create_ruleset", fake_create)
    monkeypatch.setattr(ll, "_add_path_rule", fake_add_path)
    monkeypatch.setattr(ll, "_restrict_self", noop_restrict)

    ll.apply_agent_landlock(read_paths=(), write_paths=(tmp_path,), tcp_connect_ports=())
    assert bool(captured["fs"] & _TRUNCATE) is expect_truncate
    # The per-path rule bits intersect with handled_fs, so the mask propagates.
    assert all(bool(bits & _TRUNCATE) is expect_truncate for bits in rule_bits)
