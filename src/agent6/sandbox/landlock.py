# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Minimal ctypes wrapper for the Linux Landlock LSM.

Applied to the agent process at startup. Once applied, the restrictions are
irrevocable, even root cannot remove them. This is intentional: a compromised
Python interpreter can't undo it.

References:
- Documentation/userspace-api/landlock.rst in the Linux kernel tree
- man 7 landlock
- include/uapi/linux/landlock.h
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os

# syscall numbers (x86_64 / aarch64, Linux added these uniformly)
_SYS_landlock_create_ruleset = 444
_SYS_landlock_add_rule = 445
_SYS_landlock_restrict_self = 446

# struct landlock_ruleset_attr {
#     __u64 handled_access_fs;
#     __u64 handled_access_net;   // ABI v4+
# };
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0


class LandlockError(Exception):
    """Landlock setup failed in an unexpected way."""


def _libc() -> ctypes.CDLL:
    libc_path = ctypes.util.find_library("c") or "libc.so.6"
    return ctypes.CDLL(libc_path, use_errno=True)


def _syscall(nr: int, *args: int) -> int:
    """Invoke `syscall(nr, args...)` treating each arg as a 64-bit value.

    ctypes defaults to passing int args as 32-bit `int`, which silently
    truncates pointers and large flag values on 64-bit kernels (manifests
    as EFAULT or EINVAL). We force every variadic slot through c_ulong /
    c_void_p instead. Callers may pass either Python ints (treated as
    unsigned 64-bit) or address-of buffers.
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
    """Return the Landlock ABI version supported by the running kernel, or 0."""
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
