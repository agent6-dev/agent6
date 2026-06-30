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
import struct
from dataclasses import dataclass
from pathlib import Path

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
_LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
_LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1

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


class LandlockNotSupportedError(LandlockError):
    """The running kernel does not support Landlock (ABI 0)."""


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


def _set_no_new_privs() -> None:
    libc = _libc()
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise LandlockError(f"prctl(PR_SET_NO_NEW_PRIVS) failed: {os.strerror(err)}")


def _create_ruleset(handled_fs: int, handled_net: int, abi: int) -> int:
    # struct layout depends on ABI: v1-v3 = 1x u64, v4+ = 2x u64
    if abi >= 4:
        attr = struct.pack("=QQ", handled_fs, handled_net)
    else:
        attr = struct.pack("=Q", handled_fs)
    buf = ctypes.create_string_buffer(attr, len(attr))
    return _syscall(
        _SYS_landlock_create_ruleset,
        ctypes.addressof(buf),
        len(attr),
        0,
    )


def _add_path_rule(ruleset_fd: int, fd: int, allowed_fs: int) -> None:
    # struct landlock_path_beneath_attr { __u64 allowed_access; __s32 parent_fd; }
    attr = struct.pack("=Qi", allowed_fs, fd)
    buf = ctypes.create_string_buffer(attr, len(attr))
    _syscall(
        _SYS_landlock_add_rule,
        ruleset_fd,
        1,  # LANDLOCK_RULE_PATH_BENEATH
        ctypes.addressof(buf),
        0,
    )


def _add_tcp_rule(ruleset_fd: int, port: int, allowed_net: int) -> None:
    # struct landlock_net_port_attr { __u64 allowed_access; __u64 port; }
    attr = struct.pack("=QQ", allowed_net, port)
    buf = ctypes.create_string_buffer(attr, len(attr))
    _syscall(
        _SYS_landlock_add_rule,
        ruleset_fd,
        2,  # LANDLOCK_RULE_NET_PORT
        ctypes.addressof(buf),
        0,
    )


def _restrict_self(ruleset_fd: int) -> None:
    _syscall(_SYS_landlock_restrict_self, ruleset_fd, 0)


@dataclass(frozen=True, slots=True)
class LandlockReport:
    abi: int
    fs_read: tuple[Path, ...]
    fs_write: tuple[Path, ...]
    tcp_connect_ports: tuple[int, ...]
    tcp_supported: bool


def apply_agent_landlock(
    *,
    read_paths: tuple[Path, ...],
    write_paths: tuple[Path, ...],
    tcp_connect_ports: tuple[int, ...],
) -> LandlockReport:
    """Apply Landlock to the *current process*. Irrevocable.

    Silently degrades:
    - If ABI < 4, TCP rules are not applied (no kernel support).
    - If ABI == 0, raises LandlockNotSupportedError.
    """
    abi = landlock_abi()
    if abi <= 0:
        raise LandlockNotSupportedError("Landlock is not available on this kernel (ABI 0).")

    handled_fs = _FS_ALL_BITS
    # NET_BIND_TCP is declared "handled" but NEVER granted below (only CONNECT
    # rules are added). That is deliberate, not an omission: marking it handled
    # makes Landlock DENY all TCP bind()/listen() for the agent process, which
    # is exactly "no agent-owned network surface" (SECURITY.md §7). Removing it
    # from the handled set would instead leave bind UNrestricted.
    #
    # CONNECT_TCP, by contrast, is handled ONLY when there is a connect
    # allow-list to enforce. An empty tcp_connect_ports means "no connect
    # restriction" (the documented `agent_network = "open"` fallback on hardened
    # hosts): handling CONNECT_TCP with zero allow rules would deny EVERY
    # outbound connect() by the same deny-unless-allowed mechanism, so the agent
    # could not reach its provider at all. Leave it unhandled there.
    handled_net = 0
    if abi >= 4:
        handled_net = _LANDLOCK_ACCESS_NET_BIND_TCP
        if tcp_connect_ports:
            handled_net |= _LANDLOCK_ACCESS_NET_CONNECT_TCP

    _set_no_new_privs()
    ruleset_fd = _create_ruleset(handled_fs, handled_net, abi)

    def _mask_for(path: Path, bits: int) -> int:
        try:
            is_dir = path.is_dir()
        except OSError:
            is_dir = False
        return bits if is_dir else (bits & ~_DIR_ONLY_BITS)

    try:
        for path in read_paths:
            fd = os.open(str(path), os.O_PATH | os.O_CLOEXEC)
            try:
                _add_path_rule(ruleset_fd, fd, _mask_for(path, _FS_READ_BITS & handled_fs))
            finally:
                os.close(fd)
        for path in write_paths:
            fd = os.open(str(path), os.O_PATH | os.O_CLOEXEC)
            try:
                _add_path_rule(ruleset_fd, fd, _mask_for(path, _FS_ALL_BITS & handled_fs))
            finally:
                os.close(fd)
        if abi >= 4:
            for port in tcp_connect_ports:
                _add_tcp_rule(ruleset_fd, port, _LANDLOCK_ACCESS_NET_CONNECT_TCP)
        _restrict_self(ruleset_fd)
    finally:
        os.close(ruleset_fd)

    return LandlockReport(
        abi=abi,
        fs_read=tuple(read_paths),
        fs_write=tuple(write_paths),
        tcp_connect_ports=tuple(tcp_connect_ports) if abi >= 4 else (),
        tcp_supported=abi >= 4,
    )
