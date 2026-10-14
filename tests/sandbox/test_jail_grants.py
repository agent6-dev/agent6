# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the jail does with an operator grant: the failure it names, the tree it binds."""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterator

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import jail
from agent6.tools import dispatch
from agent6.tools import policy as tools_policy

pytestmark = pytest.mark.needs_namespaces


@pytest.fixture
def scratch() -> Iterator[pathlib.Path]:
    """A directory no isolation level grants: hardened grants the host's /tmp wholesale."""
    base = pathlib.Path("/var/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        pytest.skip("/var/tmp is not writable here")
    made = pathlib.Path(tempfile.mkdtemp(prefix="agent6-jail-grants-", dir=base)).resolve()
    try:
        yield made
    finally:
        shutil.rmtree(made, ignore_errors=True)


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


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_a_grant_repointed_between_two_launches_keeps_the_tree_the_run_started_with(
    scratch: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A grant through a link in the workspace is the same tree for every launch of a run.

    The policy re-read the grant on each launch and the launcher's bind followed the link, so
    a command that re-pointed the link gave the next command a directory outside every grant.
    """
    repo, first, secret = scratch / "repo", scratch / "first", scratch / "secret"
    for made, said in ((first, "first"), (secret, "secret")):
        made.mkdir()
        (made / "name").write_text(f"{said}\n", encoding="utf-8")
    repo.mkdir()
    (repo / "hop").symlink_to(first)
    grant = scratch / "grant"
    grant.symlink_to(repo / "hop")
    cfg = Config.model_validate(
        {
            "sandbox": {
                "isolation": isolation,
                "run_commands": "yes",
                "extra_read_paths": [str(grant)],
            }
        }
    )
    run = dispatch.ToolDispatcher(root=repo, config=cfg, isolation=isolation)
    try:
        swap = ["sh", "-c", f"cat {grant}/name && ln -sfn {secret} hop"]
        before = run.dispatch("run_command", {"argv": swap}).to_wire()
        assert before["returncode"] == 0 and before["stdout"] == "first\n", before
        assert (repo / "hop").readlink() == secret
        after = run.dispatch("run_command", {"argv": ["cat", f"{grant}/name"]}).to_wire()
        direct = run.dispatch("run_command", {"argv": ["cat", f"{secret}/name"]}).to_wire()
    finally:
        run.close()
    assert "secret" not in str(after["stdout"]), after
    assert direct["returncode"] != 0 and "secret" not in str(direct["stdout"]), direct
    if isolation == "strict":
        # The mount at the grant's name still holds the first tree.
        assert after["returncode"] == 0 and after["stdout"] == "first\n", after
    else:
        # No mount holds the name: it follows the new link, into a tree no rule grants.
        assert after["returncode"] != 0, after


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_a_symlinked_grant_is_readable_at_its_spelled_name(
    scratch: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A grant on a symlinked toolchain directory works through the name the config gives.

    Bound from its resolved path, the grant must stay at the spelled one: a toolchain is
    configured, and finds itself, by that name.
    """
    real, link, repo = scratch / "tool-1.2", scratch / "tool", scratch / "repo"
    real.mkdir()
    (real / "VERSION").write_text("1.2\n", encoding="utf-8")
    link.symlink_to(real, target_is_directory=True)
    repo.mkdir()
    cfg = Config.model_validate({"sandbox": {"extra_read_paths": [str(link)]}})
    built = tools_policy.jail_policy(
        repo, cfg, isolation, ("cat", f"{link}/VERSION"), network="none", timeout_s=10.0
    )
    assert built.extra_ro_paths[0] == link and built.grant_sources == ((link, real),)
    res = jail.run_in_jail(built)
    assert res.returncode == 0 and res.stdout == "1.2\n", res
