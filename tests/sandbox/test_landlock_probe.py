# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sandbox probe: report Landlock ABI, never fail just because the kernel is old."""

from __future__ import annotations

from pathlib import Path

import pytest

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
