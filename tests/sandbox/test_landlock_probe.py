# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sandbox probe: report Landlock ABI, never fail just because the kernel is old."""

from __future__ import annotations

from agent6.sandbox import landlock_abi


def test_landlock_abi_is_nonnegative() -> None:
    # On a kernel without Landlock the call returns 0; we never want a hard fail.
    abi = landlock_abi()
    assert isinstance(abi, int)
    assert abi >= 0
