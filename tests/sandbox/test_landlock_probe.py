# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sandbox probe: report Landlock ABI, never fail just because the kernel is old."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.sandbox import landlock as ll
from agent6.sandbox import landlock_abi


def test_landlock_abi_reports_the_kernel_version() -> None:
    """0 means "no Landlock" and makes warn_sandbox_gaps drop confinement
    everywhere, so `>= 0` passed for exactly the regression that matters. On a
    Landlock kernel the probe must report the real ABI (>= 1)."""
    abi = landlock_abi()
    assert isinstance(abi, int)
    lsm = Path("/sys/kernel/security/lsm")
    if not lsm.is_file():
        pytest.skip("cannot confirm kernel Landlock support")
    if "landlock" not in lsm.read_text(encoding="utf-8"):
        assert abi == 0  # honest zero on a kernel without it
    else:
        assert abi >= 1, "kernel reports Landlock but the probe returned 0"


def test_no_network_access_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent ruleset must handle NO network access.

    Landlock's net rules are port-based. CONNECT could only be denied as "any
    host on port N", which stops no exfiltration (one HTTPS endpoint suffices)
    while breaking tools on other ports. BIND blocks only inbound while
    outbound stays open, so it cost a model-run dev server its listening
    socket and bought no boundary; nothing outlives its command anyway.

    Stubs the syscall layer so it runs on any kernel (ABI forced to 8)."""
    captured: dict[str, int] = {}

    def fake_create(handled_fs: int, abi: int) -> int:
        captured["fs"] = handled_fs
        return os.open(os.devnull, os.O_RDONLY)  # a real, closeable fd

    def noop_restrict(ruleset_fd: int) -> None:
        pass

    monkeypatch.setattr(ll, "landlock_abi", lambda: 8)
    monkeypatch.setattr(ll, "_set_no_new_privs", lambda: None)
    monkeypatch.setattr(ll, "_create_ruleset", fake_create)
    monkeypatch.setattr(ll, "_restrict_self", noop_restrict)

    ll.apply_agent_landlock(read_paths=(), write_paths=())
    assert captured["fs"]  # filesystem access IS handled


_TRUNCATE = 1 << 14  # _LANDLOCK_ACCESS_FS_TRUNCATE (ABI v3)


@pytest.mark.parametrize(("abi", "expect_truncate"), [(1, False), (2, False), (3, True)])
def test_handled_fs_masks_truncate_below_abi3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, abi: int, expect_truncate: bool
) -> None:
    """Regression: the kernel EINVALs any handled_access_fs bit above its ABI,
    so passing the ABI-3 TRUNCATE bit on a 5.13-5.18 kernel (ABI 1/2) refused
    every hardened run. handled_fs must be down-masked to the probed ABI;
    pre-ABI-3 truncation is governed by WRITE_FILE, so nothing is lost. Stubs
    the syscall layer so it runs on any kernel."""
    import os as _os

    captured: dict[str, int] = {}
    rule_bits: list[int] = []

    def fake_create(handled_fs: int, _abi: int) -> int:
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

    ll.apply_agent_landlock(read_paths=(), write_paths=(tmp_path,))
    assert bool(captured["fs"] & _TRUNCATE) is expect_truncate
    # The per-path rule bits intersect with handled_fs, so the mask propagates.
    assert all(bool(bits & _TRUNCATE) is expect_truncate for bits in rule_bits)
