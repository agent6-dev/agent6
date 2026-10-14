# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What `strict` does with an operator grant: the failure it names."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds
from agent6.sandbox import jail

pytestmark = pytest.mark.needs_namespaces


def _granted(cwd: pathlib.Path, grant: pathlib.Path, *, write: bool) -> kinds.JailPolicy:
    return kinds.JailPolicy(
        cwd=cwd,
        argv=("true",),
        isolation="strict",
        extra_ro_paths=() if write else (grant,),
        extra_rw_paths=(grant,) if write else (),
        timeout_s=10.0,
    )


def _launch_errors(policy: kinds.JailPolicy) -> tuple[str, str]:
    """Return the error a one-shot launch and a session open each raise for the policy."""
    with pytest.raises(jail.JailUnavailableError) as one_shot:
        jail.run_in_jail(policy)
    with pytest.raises(jail.JailUnavailableError) as served:
        jail.JailSession.open(policy)
    return str(one_shot.value), str(served.value)


@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
@pytest.mark.parametrize(
    ("grant", "why"),
    [("/", "the root directory cannot be granted"), ("/dev", "/dev cannot be granted")],
)
def test_a_grant_on_the_root_or_on_dev_fails_the_launch_by_name(
    tmp_path: pathlib.Path, grant: str, why: str, write: bool
) -> None:
    """A grant on `/` or `/dev` fails the launch with a sentence that names the grant.

    The root failed a later mount with a bare errno ("Read-only file system", "Operation not
    permitted"), and `/dev` bound, after which every command exited 127.
    """
    kind = "write" if write else "read"
    for error in _launch_errors(_granted(tmp_path, pathlib.Path(grant), write=write)):
        assert f"{kind} grant {grant}: {why}" in error


def _read_only_submount() -> tuple[pathlib.Path, pathlib.Path] | None:
    """Return a directory this host can grant and a read-only mount nested in it, or None."""
    unbindable = ("/", "/dev", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives")
    for line in pathlib.Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        mount_point, options = pathlib.Path(fields[4]), fields[5].split(",")
        parent = mount_point.parent
        if "ro" not in options or "\\" in fields[4]:
            continue
        if any(parent == pathlib.Path(u) or parent.is_relative_to(u) for u in unbindable[2:]):
            continue
        if str(parent) not in unbindable[:2] and parent.is_dir():
            return parent, mount_point
    return None


def test_a_write_grant_holding_a_read_only_mount_names_the_grant_and_the_mount(
    tmp_path: pathlib.Path,
) -> None:
    """A write grant over a read-only mount fails naming the grant, the step and the mount.

    The kernel refuses to make the nested mount writable, and the launch failed with a bare
    "Operation not permitted" that named neither.
    """
    found = _read_only_submount()
    if found is None:
        pytest.skip("this host has no read-only mount nested in a grantable directory")
    grant, nested = found
    for error in _launch_errors(_granted(tmp_path, grant, write=True)):
        assert f"write grant {grant}: remount of submount {nested.name}: EPERM" in error
