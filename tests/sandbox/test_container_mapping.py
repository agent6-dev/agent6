from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import stat
import subprocess

import pytest

from agent6 import kinds
from agent6.sandbox import container, jail_session
from tests.sandbox import conftest


def test_mapping_accepts_inherited_setgid_on_private_staging(target: conftest.Target) -> None:
    """Setgid inheritance does not give group access to an owner-only staging directory."""
    root = target.discovery.target.host_root
    root.chmod(0o2775)
    inherited = root / "inherited-mode"
    inherited.mkdir(mode=0o700)
    assert stat.S_IMODE(inherited.stat().st_mode) == 0o2700
    inherited.rmdir()
    with conftest.opened(target) as session:
        result = session.run(("/bin/pwd",))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
        assert result.stdout.strip() == str(root)
    assert stat.S_IMODE(root.stat().st_mode) == 0o2775
    assert not list(root.iterdir())


@pytest.mark.needs_namespaces
@pytest.mark.parametrize("symlink", [False, True])
def test_host_mapping_needs_search_not_read_permission(
    target: conftest.Target, symlink: bool
) -> None:
    """A host search-only ancestor permits mapping, but a symlink ancestor still refuses."""
    assert os.geteuid() != 0
    parent = target.discovery.target.host_root
    root = parent / "repo"
    root.mkdir()
    alias = target.home / "mapped"
    alias.mkdir()
    link = target.home / "linked-parent"
    link.symlink_to(parent, target_is_directory=True)
    prefix = (
        "unshare",
        "-Ur",
        "-m",
        "/bin/sh",
        "-ec",
        'mount --bind "$1" "$2"; shift 2; exec "$@"',
        "search-only-host",
        str(root),
        str(alias),
        *target.discovery.exec,
    )
    discovery = container.discover(
        link / "repo" if symlink else root,
        container.ContainerOptions(exec=prefix, workspace=alias),
    )
    parent.chmod(0o111)
    try:
        assert os.access(parent, os.X_OK) and not os.access(parent, os.R_OK)
        assert root.is_dir()
        if symlink:
            with pytest.raises(container.ContainerError, match="Not a directory"):
                session = container.prepare(discovery)
                session.close()
        else:
            session = container.prepare(discovery)
            try:
                result = session.run(("/bin/pwd",))
                assert isinstance(result, kinds.CommandResult) and result.returncode == 0
                assert result.stdout.strip() == str(alias)
            finally:
                assert not session.close()
        assert not list(root.iterdir())
    finally:
        parent.chmod(0o755)


@pytest.mark.needs_namespaces
@pytest.mark.parametrize("symlink", [False, True])
def test_selected_user_mapping_needs_search_not_read_permission(
    host_tmp_path: pathlib.Path, static_launcher: pathlib.Path, symlink: bool
) -> None:
    """A non-root target can traverse a root-owned 0711 ancestor without following links."""
    uid, gid = 1000, 1001
    selected = (
        "unshare",
        "-U",
        "--map-root-user",
        "--map-auto",
        "--setuid",
        str(uid),
        "--setgid",
        str(gid),
    )
    available = subprocess.run(
        (*selected, "/bin/true"), check=False, capture_output=True, text=True
    )
    if available.returncode != 0:
        pytest.skip("subordinate UID/GID mappings unavailable: " + available.stderr.strip())
    host_tmp_path.chmod(0o755)
    parent = host_tmp_path / "search-only"
    parent.mkdir(mode=0o711)
    root = parent / "repo"
    root.mkdir()
    link = host_tmp_path / "linked-parent"
    link.symlink_to(parent, target_is_directory=True)
    prefix = (*selected, "/usr/bin/env", "-i", "HOME=/tmp", "PATH=/usr/bin:/bin")
    usable = subprocess.check_output(
        (
            *prefix,
            "/bin/sh",
            "-ec",
            'test ! -r "$1"; test -x "$1"; stat -c "%u:%a" "$1"; cd "$2"; pwd -P; id -u',
            "search-only-target",
            str(parent),
            str(root),
        ),
        text=True,
    )
    assert usable.splitlines() == ["0:711", str(root), str(uid)]
    discovery = container.discover(
        root,
        container.ContainerOptions(exec=prefix, workspace=link / "repo" if symlink else root),
    )
    if symlink:
        with pytest.raises(container.ContainerError, match="Not a directory"):
            session = container.prepare(discovery)
            session.close()
    else:
        session = container.prepare(discovery)
        try:
            conftest.assert_selected_user(session, uid, gid)
            result = session.run(("/bin/pwd",))
            assert isinstance(result, kinds.CommandResult) and result.returncode == 0
            assert result.stdout.strip() == str(root)
        finally:
            assert not session.close()
    assert not list(root.iterdir())


@pytest.mark.parametrize("direction", ["host", "target"])
def test_mapping_refuses_corrupted_regular_file_nonce(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """Each direction must compare nonce contents even when the regular-file inode matches."""
    root = target.discovery.target.host_root
    exchange = jail_session.JailSession.prove_mapping
    corrupted: list[pathlib.Path] = []

    def corrupt(session: jail_session.JailSession, name: str, nonce: str) -> str:
        expected = nonce if direction == "host" else exchange(session, name, nonce)
        entry = root / name / direction
        before = entry.lstat()
        assert stat.S_ISREG(before.st_mode) and entry.read_text() == expected
        replacement = ("0" if expected[0] != "0" else "1") + expected[1:]
        entry.write_text(replacement)
        assert os.path.samestat(before, entry.lstat())
        assert entry.read_text() == replacement != expected
        corrupted.append(entry)
        return exchange(session, name, nonce) if direction == "host" else expected

    monkeypatch.setattr(jail_session.JailSession, "prove_mapping", corrupt)
    reason = (
        "host mapping nonce differs"
        if direction == "host"
        else "workspace mapping returned a different target nonce"
    )
    with pytest.raises(container.ContainerError, match=reason):
        session = container.prepare(target.discovery)
        session.close()
    assert len(corrupted) == 1
    assert not list(root.iterdir())


@pytest.mark.parametrize("special_bits", [0, stat.S_ISGID])
@pytest.mark.parametrize("substitute", ["probe", "staging"])
def test_unopened_public_directory_cannot_grant_host_permissions(
    target: conftest.Target,
    monkeypatch: pytest.MonkeyPatch,
    substitute: str,
    special_bits: int,
) -> None:
    """Renaming a host-private directory into a public name must not expose or empty it."""
    root = target.discovery.target.host_root
    root.chmod(0o777 | special_bits)
    victim = root / "private"
    victim.mkdir(mode=0o700)
    for entry in ("host", "target"):
        (victim / entry).write_text("private-canary")
    original_open = os.open
    substitutions: list[pathlib.Path] = []

    def substitute_before_open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        name = os.fspath(path)
        prefix = ".agent6-probe-" if substitute == "probe" else ".agent6-stage-"
        if (
            not substitutions
            and name.startswith(prefix)
            and dir_fd is not None
            and os.path.samestat(os.fstat(dir_fd), root.stat())
        ):
            # Both renames need only write/search permission on the mode-0777 parent.
            (root / name).rename(root / "displaced")
            victim.rename(root / name)
            substitutions.append(root / name)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", substitute_before_open)
    with contextlib.suppress(container.ContainerError):
        session = container.prepare(target.discovery)
        session.close()
    remaining = substitutions[0] if substitutions else victim
    assert stat.S_IMODE(remaining.stat().st_mode) == 0o700 | special_bits
    assert {path.name: path.read_text() for path in remaining.iterdir()} == {
        "host": "private-canary",
        "target": "private-canary",
    }


def test_host_nonce_reader_refuses_symlinks_during_preparation(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching nonce behind a symlink must not satisfy the host reader."""
    root = target.discovery.target.host_root
    victim = target.home / "private-nonce"
    exchange = jail_session.JailSession.prove_mapping

    def substitute(session: jail_session.JailSession, name: str, nonce: str) -> str:
        expected = exchange(session, name, nonce)
        victim.write_text(expected)
        (root / name / "target").unlink()
        (root / name / "target").symlink_to(victim)
        return expected

    monkeypatch.setattr(jail_session.JailSession, "prove_mapping", substitute)
    with pytest.raises(container.ContainerError, match="nonce is not a regular file"):
        session = container.prepare(target.discovery)
        session.close()
    assert len(victim.read_text()) == 64
    assert not list(root.iterdir())


def test_host_nonce_refuses_fifo_without_opening_a_reader(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusal must precede special-file open side effects, not just reads."""
    root = target.discovery.target.host_root
    exchange = jail_session.JailSession.prove_mapping
    original_stat = os.fstat
    published: list[tuple[pathlib.Path, os.stat_result]] = []
    readers: list[bool] = []

    def substitute(session: jail_session.JailSession, name: str, nonce: str) -> str:
        expected = exchange(session, name, nonce)
        entry = root / name / "target"
        entry.unlink()
        os.mkfifo(entry, mode=0o644)
        published.append((entry, entry.lstat()))
        return expected

    def observe_reader(fd: int) -> os.stat_result:
        info = original_stat(fd)
        if published and os.path.samestat(info, published[0][1]):
            # A nonblocking writer succeeds only if an I/O reader already holds the FIFO open.
            try:
                writer = os.open(published[0][0], os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                readers.append(False)
            else:
                os.close(writer)
                readers.append(True)
        return info

    monkeypatch.setattr(jail_session.JailSession, "prove_mapping", substitute)
    monkeypatch.setattr(os, "fstat", observe_reader)
    with pytest.raises(container.ContainerError, match="nonce is not a regular file"):
        session = container.prepare(target.discovery)
        session.close()
    assert readers == [False], "host opened the special file for I/O before refusing it"
    assert not list(root.iterdir())


def test_host_nonce_reads_the_validated_inode_after_path_replacement(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking an inode must not permit a later open through its substituted name."""
    root = target.discovery.target.host_root
    exchange = jail_session.JailSession.prove_mapping
    original_stat = os.fstat
    published: list[tuple[pathlib.Path, os.stat_result]] = []
    replacements: list[pathlib.Path] = []

    def locate(session: jail_session.JailSession, name: str, nonce: str) -> str:
        expected = exchange(session, name, nonce)
        entry = root / name / "target"
        published.append((entry, entry.lstat()))
        return expected

    def replace_after_stat(fd: int) -> os.stat_result:
        info = original_stat(fd)
        if published and os.path.samestat(info, published[0][1]):
            entry = published[0][0]
            entry.unlink()
            entry.write_text("replacement-canary")
            replacements.append(entry)
        return info

    monkeypatch.setattr(jail_session.JailSession, "prove_mapping", locate)
    monkeypatch.setattr(os, "fstat", replace_after_stat)
    with conftest.opened(target) as session:
        result = session.run(("/bin/echo", "original-nonce"))
        assert isinstance(result, kinds.CommandResult)
        assert result.returncode == 0 and result.stdout == "original-nonce\n", result
    assert len(replacements) == 1
    assert not list(root.iterdir())


@pytest.mark.parametrize("symlink", [False, True])
def test_host_cleanup_uses_the_retained_probe_directory(
    target: conftest.Target, monkeypatch: pytest.MonkeyPatch, symlink: bool
) -> None:
    """Replacing the published name must not redirect nonce cleanup to another directory."""
    root = target.discovery.target.host_root
    victim = root / "private"
    victim.mkdir(mode=0o700)
    for entry in ("host", "target"):
        (victim / entry).write_text("private-canary")
    exchange = jail_session.JailSession.prove_mapping
    published: list[pathlib.Path] = []

    def substitute(session: jail_session.JailSession, name: str, nonce: str) -> str:
        expected = exchange(session, name, nonce)
        path = root / name
        path.rename(root / "displaced")
        if symlink:
            path.symlink_to(victim, target_is_directory=True)
        else:
            victim.rename(path)
        published.append(path)
        return expected

    monkeypatch.setattr(jail_session.JailSession, "prove_mapping", substitute)
    with conftest.opened(target):
        remaining = victim if symlink else published[0]
        assert stat.S_IMODE(remaining.stat().st_mode) == 0o700
        assert {path.name: path.read_text() for path in remaining.iterdir()} == {
            "host": "private-canary",
            "target": "private-canary",
        }
        assert not list((root / "displaced").iterdir())
