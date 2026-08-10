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

# fs access bits (subset we use)
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13  # ABI v2
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14  # ABI v3
_LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15  # ABI v5

# net access bits (ABI v4+)

_FS_READ_BITS = (
    _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_READ_DIR | _LANDLOCK_ACCESS_FS_EXECUTE
)
_FS_WRITE_BITS = (
    _LANDLOCK_ACCESS_FS_WRITE_FILE
    | _LANDLOCK_ACCESS_FS_REMOVE_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_FILE
    | _LANDLOCK_ACCESS_FS_MAKE_CHAR
    | _LANDLOCK_ACCESS_FS_MAKE_DIR
    | _LANDLOCK_ACCESS_FS_MAKE_REG
    | _LANDLOCK_ACCESS_FS_MAKE_SOCK
    | _LANDLOCK_ACCESS_FS_MAKE_FIFO
    | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | _LANDLOCK_ACCESS_FS_MAKE_SYM
    | _LANDLOCK_ACCESS_FS_TRUNCATE
)
_FS_ALL_BITS = _FS_READ_BITS | _FS_WRITE_BITS

# Bits that only make sense for directories (creating/removing entries).
# Passing these on a regular-file rule yields EINVAL. We mask them out
# when the rule target isn't a directory.
_DIR_ONLY_BITS = (
    _LANDLOCK_ACCESS_FS_READ_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_FILE
    | _LANDLOCK_ACCESS_FS_MAKE_CHAR
    | _LANDLOCK_ACCESS_FS_MAKE_DIR
    | _LANDLOCK_ACCESS_FS_MAKE_REG
    | _LANDLOCK_ACCESS_FS_MAKE_SOCK
    | _LANDLOCK_ACCESS_FS_MAKE_FIFO
    | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | _LANDLOCK_ACCESS_FS_MAKE_SYM
)

# prctl
_PR_SET_NO_NEW_PRIVS = 38


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
