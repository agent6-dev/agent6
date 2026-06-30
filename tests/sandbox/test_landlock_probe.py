# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sandbox probe: report Landlock ABI, never fail just because the kernel is old."""

from __future__ import annotations

import os

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
