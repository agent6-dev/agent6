"""Validate launcher bytes and prove the shared workspace without following symlinks."""

from __future__ import annotations

import contextlib
import os
import pathlib
import platform
import secrets
import stat
import struct
import sys

from agent6.sandbox import jail, jail_session

# Only byte count and architecture are positional arguments; no command or workspace is shell text.
BOOTSTRAP = """\
set -eu
# Subshell assignments cannot overwrite inherited exported variables in the launcher.
(
    for utility in uname mktemp head chmod; do
        command -v "$utility" >/dev/null 2>&1 || {
            printf 'agent6 bootstrap requires %s\\n' "$utility" >&2
            exit 1
        }
    done
    architecture=$(uname -m)
    [ "$architecture" = "$2" ] || {
        printf 'agent6 launcher architecture %s does not match target %s\\n' \\
            "$2" "$architecture" >&2
        exit 1
    }
)
# Positional state is byte count, saved umask, then installation path.
set -- "$1" "$(umask)"
umask 077
set -- "$@" "$(mktemp -d /tmp/a6.XXXXXXXX)"
[ -n "$3" ] || exit 1
cd -P "$3"
set -C
head -c "$1" > launcher
chmod 500 launcher
umask "$2"
printf '{"installed":true}\\n'
exec ./launcher --installed
"""


def launcher() -> tuple[bytes, str]:
    """Validate the selected static ELF before any target-side installation.

    Returns:
        The immutable bytes to transfer and their native Linux architecture.

    Raises:
        jail.JailUnavailableError: The binary is absent, dynamic, malformed or incompatible.
    """
    architecture = platform.machine()
    machines = {"x86_64": 62, "aarch64": 183}
    if sys.platform != "linux" or architecture not in machines:
        raise jail.JailUnavailableError("container launchers require Linux x86-64 or aarch64 hosts")
    try:
        with jail.require_jail_binary().open("rb") as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if not 64 <= len(data) <= 32 * 1024 * 1024 or data[:7] != b"\x7fELF\x02\x01\x01":
            raise jail.JailUnavailableError("invalid 64-bit little-endian ELF header or size")
        kind, machine = struct.unpack_from("<HH", data, 16)
        if machine != machines[architecture]:
            raise jail.JailUnavailableError(
                "ELF architecture differs from the host; cross-architecture transfer is unsupported"
            )
        if kind not in (2, 3):
            raise jail.JailUnavailableError("ELF is not an executable")
        offset = struct.unpack_from("<Q", data, 32)[0]
        size, count = struct.unpack_from("<HH", data, 54)
        if size != 56 or not count or offset < 64 or offset + size * count > len(data):
            raise jail.JailUnavailableError("invalid ELF program headers")
        for index in range(count):
            entry = offset + index * size
            segment = struct.unpack_from("<I", data, entry)[0]
            start = struct.unpack_from("<Q", data, entry + 8)[0]
            length = struct.unpack_from("<Q", data, entry + 32)[0]
            if start + length > len(data):
                raise jail.JailUnavailableError("truncated ELF segment")
            if segment == 3:
                raise jail.JailUnavailableError(
                    "launcher has an ELF interpreter; a static musl build is required"
                )
            if segment == 2:
                if length % 16:
                    raise jail.JailUnavailableError("invalid ELF dynamic segment")
                if any(
                    struct.unpack_from("<Q", data, p)[0] == 1
                    for p in range(start, start + length, 16)
                ):
                    raise jail.JailUnavailableError(
                        "launcher needs shared libraries; a static musl build is required"
                    )
    except (OSError, struct.error) as exc:
        raise jail.JailUnavailableError(f"container launcher refused: {exc}") from exc
    return data, architecture


def _directory_fd(path: pathlib.Path) -> int:
    """Open an absolute directory without following symlinks in any component.

    Args:
        path: The host workspace path.

    Returns:
        An owned directory descriptor.

    Raises:
        jail.JailUnavailableError: The path is not absolute or contains parent traversal.
        OSError: A directory component cannot be opened without following symlinks.
    """
    if not path.is_absolute() or ".." in path.parts:
        raise jail.JailUnavailableError("workspace mapping needs an absolute directory")
    fd = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd
            )
            os.close(fd)
            fd = child
        return fd
    except OSError:
        os.close(fd)
        raise


def _remove_empty_directory(parent_fd: int, name: str, directory_fd: int) -> None:
    """Remove a still-matching name without following or emptying a substituted directory.

    Args:
        parent_fd: The retained parent directory.
        name: The published directory entry.
        directory_fd: The retained child directory.
    """
    with contextlib.suppress(FileNotFoundError):
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if os.path.samestat(entry, os.fstat(directory_fd)):
            os.rmdir(name, dir_fd=parent_fd)


def prove_mapping(session: jail_session.JailSession, root: pathlib.Path) -> None:
    """Exchange two independent nonces through retained no-follow directory descriptors.

    Args:
        session: The prepared target channel.
        root: The host workspace.

    Raises:
        jail.JailUnavailableError: The target cannot exchange nonces through the shared mapping.
        OSError: A host-side probe operation failed.
    """
    name = ".agent6-probe-" + secrets.token_hex(16)
    staging = ".agent6-stage-" + secrets.token_hex(16)
    nonce = secrets.token_hex(32)
    with contextlib.ExitStack() as stack:
        root_fd = _directory_fd(root)
        stack.callback(os.close, root_fd)
        os.mkdir(staging, mode=0o700, dir_fd=root_fd)
        staging_fd = os.open(
            staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd
        )
        stack.callback(os.close, staging_fd)
        info = os.fstat(staging_fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o777 != 0o700:
            raise jail.JailUnavailableError("workspace probe staging directory is not private")
        stack.callback(_remove_empty_directory, root_fd, staging, staging_fd)
        # A substituted private parent is safe: never chmod it or remove its existing entries.
        os.mkdir(name, mode=0o700, dir_fd=staging_fd)
        probe_fd = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=staging_fd
        )
        stack.callback(os.close, probe_fd)
        stack.callback(_remove_empty_directory, staging_fd, name, probe_fd)
        stack.callback(_remove_empty_directory, root_fd, name, probe_fd)
        os.fchmod(probe_fd, 0o777)
        os.rename(name, name, src_dir_fd=staging_fd, dst_dir_fd=root_fd)
        try:
            fd = os.open(
                "host", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=probe_fd
            )
            with os.fdopen(fd, "w") as stream:
                os.fchmod(fd, 0o644)
                stream.write(nonce)
            expected = session.prove_mapping(name, nonce)
            # O_PATH avoids device open handlers; /proc/self/fd reopens the pinned inode.
            nonce_fd = os.open("target", os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=probe_fd)
            stack.callback(os.close, nonce_fd)
            if not stat.S_ISREG(os.fstat(nonce_fd).st_mode):
                raise jail.JailUnavailableError("workspace mapping nonce is not a regular file")
            fd = os.open(f"/proc/self/fd/{nonce_fd}", os.O_RDONLY | os.O_CLOEXEC)
            with os.fdopen(fd, "rb") as stream:
                actual = stream.read(65)
            if actual != expected.encode():
                raise jail.JailUnavailableError(
                    "workspace mapping returned a different target nonce"
                )
        finally:
            for entry in ("host", "target"):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(entry, dir_fd=probe_fd)
