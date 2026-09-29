# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Probe the Linux Landlock ABI version through ctypes.

`landlock_abi` feeds isolation resolution in `sandbox.detect`. Landlock rules are
applied by the jail launcher (`jail/src/main.rs`), never to the agent process
(`app/confine.py`). References: `Documentation/userspace-api/landlock.rst`,
`man 7 landlock`, `include/uapi/linux/landlock.h`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os

# The syscall numbers are the same on x86_64 and aarch64.
_SYS_landlock_create_ruleset = 444
_SYS_landlock_add_rule = 445
_SYS_landlock_restrict_self = 446

_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0


class LandlockError(Exception):
    """Landlock setup failed in an unexpected way."""


def _libc() -> ctypes.CDLL:
    """Return libc loaded with errno tracking."""
    libc_path = ctypes.util.find_library("c") or "libc.so.6"
    return ctypes.CDLL(libc_path, use_errno=True)


def _syscall(nr: int, *args: int) -> int:
    """Invoke `syscall(nr, args...)` with every argument as a 64-bit value.

    ctypes passes a bare int as a 32-bit `int`, which truncates pointers and large
    flag values on 64-bit kernels (EFAULT or EINVAL).

    Args:
        nr: The syscall number.
        *args: Unsigned 64-bit values or buffer addresses.

    Returns:
        The syscall's result.

    Raises:
        OSError: The syscall failed; errno is set.
    """
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    typed: list[object] = [ctypes.c_long(nr)]
    for arg in args:
        typed.append(ctypes.c_ulong(arg))
    result = libc.syscall(*typed)
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(result)


def landlock_abi() -> int:
    """Return the Landlock ABI version the running kernel supports, or 0.

    Returns:
        The ABI version; 0 when the kernel lacks Landlock.

    Raises:
        LandlockError: The probe failed for a reason other than a missing Landlock.
    """
    try:
        return _syscall(
            _SYS_landlock_create_ruleset,
            0,
            0,
            _LANDLOCK_CREATE_RULESET_VERSION,
        )
    except OSError as exc:
        if exc.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
            return 0
        raise LandlockError(f"landlock_create_ruleset version probe failed: {exc}") from exc
