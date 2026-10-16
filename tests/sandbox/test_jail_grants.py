# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the jail does with an operator grant: the failure it names, the tree it binds."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterator

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import jail, jail_session
from agent6.tools import dispatch
from agent6.tools import policy as tools_policy

pytestmark = pytest.mark.needs_namespaces


@pytest.fixture
def scratch() -> Iterator[pathlib.Path]:
    """A directory no isolation level grants: partial grants the host's /tmp wholesale."""
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
        isolation="full",
        extra_ro_paths=() if write else (grant,),
        extra_rw_paths=(grant,) if write else (),
        timeout_s=10.0,
    )


def _launch_errors(policy: kinds.JailPolicy) -> tuple[str, str]:
    """Return the error a one-shot launch and a session open each raise for the policy."""
    with pytest.raises(jail.JailUnavailableError) as one_shot:
        jail.run_in_jail(policy)
    with pytest.raises(jail.JailUnavailableError) as served:
        jail_session.JailSession.open(policy)
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


@pytest.mark.parametrize("isolation", ["full", "partial"])
def test_a_grant_repointed_between_two_launches_keeps_the_tree_the_run_started_with(
    scratch: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A symlinked grant is the same tree for every launch of a run, whoever re-points it.

    The policy re-read the grant on each launch and the launcher's bind followed the link, so
    a link re-pointed mid-run gave the next command a directory outside every grant. A grant
    that links through the workspace or a write grant is dropped, so the link is re-pointed
    here from outside the jail.
    """
    repo, first, secret = scratch / "repo", scratch / "first", scratch / "secret"
    for made, said in ((first, "first"), (secret, "secret")):
        made.mkdir()
        (made / "name").write_text(f"{said}\n", encoding="utf-8")
    repo.mkdir()
    grant = scratch / "grant"
    grant.symlink_to(first)
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
        before = run.dispatch("run_command", {"argv": ["cat", f"{grant}/name"]}).to_wire()
        assert before["returncode"] == 0 and before["stdout"] == "first\n", before
        grant.unlink()
        grant.symlink_to(secret)
        after = run.dispatch("run_command", {"argv": ["cat", f"{grant}/name"]}).to_wire()
        direct = run.dispatch("run_command", {"argv": ["cat", f"{secret}/name"]}).to_wire()
    finally:
        run.close()
    assert "secret" not in str(after["stdout"]), after
    assert direct["returncode"] != 0 and "secret" not in str(direct["stdout"]), direct
    if isolation == "full":
        # The mount at the grant's name still holds the first tree.
        assert after["returncode"] == 0 and after["stdout"] == "first\n", after
    else:
        # No mount holds the name: it follows the new link, into a tree no rule grants.
        assert after["returncode"] != 0, after


@pytest.mark.parametrize("isolation", ["full", "partial"])
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


@pytest.mark.parametrize("shape", ["duplicate", "nested"])
def test_a_file_grant_binds_over_a_mount_point_an_earlier_grant_made(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A file granted twice, or inside a granted directory, binds over the earlier mount.

    The launcher created every file grant's mount point, and creating one that an earlier
    read-only bind already held failed the launch with "Read-only file system".
    """
    held = tmp_path / "held"
    held.mkdir()
    granted = held / "ca.crt"
    granted.write_text("cert\n", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    probe = f"cat {granted}; {{ echo x >> {granted}; }} 2>/dev/null && echo WRITTEN || echo refused"
    res = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=ws,
            argv=("sh", "-c", probe),
            isolation="full",
            extra_ro_paths=(granted if shape == "duplicate" else held, granted),
            timeout_s=10.0,
        )
    )
    assert res.returncode == 0 and res.stdout == "cert\nrefused\n", res
    assert granted.read_text(encoding="utf-8") == "cert\n"


def test_a_grant_is_never_mounted_through_a_link_an_earlier_grant_shows(
    tmp_path: pathlib.Path,
) -> None:
    """A symlink already at a grant's mount point is left as the earlier grant shows it.

    The launcher created the mount point through the link, which emptied the host file the
    link points to. A read grant there adds nothing to the link; a write grant fails by name.
    """
    held, target, ws = tmp_path / "held", tmp_path / "target.txt", tmp_path / "ws"
    held.mkdir()
    ws.mkdir()
    target.write_text("host content\n", encoding="utf-8")
    link = held / "link.txt"
    link.symlink_to(target)
    base = kinds.JailPolicy(cwd=ws, argv=("readlink", str(link)), isolation="full")
    res = jail.run_in_jail(dataclasses.replace(base, extra_ro_paths=(held, link)))
    assert res.returncode == 0 and res.stdout.strip() == str(target), res
    assert target.read_text(encoding="utf-8") == "host content\n"
    with pytest.raises(jail.JailUnavailableError, match=f"write grant {link}: a symbolic link"):
        jail.run_in_jail(dataclasses.replace(base, extra_ro_paths=(held,), extra_rw_paths=(link,)))
    assert target.read_text(encoding="utf-8") == "host content\n"


def test_a_hidden_path_is_masked_under_a_grants_spelled_name_too(scratch: pathlib.Path) -> None:
    """A hidden path inside a symlinked grant stayed readable through the grant's name.

    The grant is bound from its resolved tree at its spelled name, and the mask sat only where
    the hidden path itself points, a place the jail does not show.
    """
    real, link, repo = scratch / "tool-1.2", scratch / "tool", scratch / "repo"
    (real / "secret").mkdir(parents=True)
    (real / "secret" / "key").write_text("HIDDEN\n", encoding="utf-8")
    (real / "VERSION").write_text("1.2\n", encoding="utf-8")
    link.symlink_to(real, target_is_directory=True)
    repo.mkdir()
    cfg = Config.model_validate(
        {
            "sandbox": {
                "extra_read_paths": [str(link)],
                "hide_paths": [str(real / "secret")],
            }
        }
    )
    script = f"cat {link}/VERSION; cat {link}/secret/key; ls {link}/secret | wc -l"
    built = tools_policy.jail_policy(
        repo, cfg, "full", ("sh", "-c", script), network="none", timeout_s=10.0
    )
    res = jail.run_in_jail(built)
    assert "HIDDEN" not in res.stdout + res.stderr, res
    assert res.stdout.splitlines() == ["1.2", "0"], res
