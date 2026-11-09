from __future__ import annotations

import contextlib
import errno
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

import pytest

from agent6 import kinds
from agent6.sandbox import container
from tests.sandbox import conftest


def test_close_preserves_command_created_tmpdir_contents(target: conftest.Target) -> None:
    """Installed-session shutdown must not recursively delete command-created temp files."""
    with conftest.opened(target) as session:
        directory = pathlib.Path(session.probe().environment["TMPDIR"])
        canary = directory / "command-created"
        try:
            result = session.run(("/bin/sh", "-ec", 'printf preserved > "$TMPDIR/command-created"'))
            assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
            assert canary.read_bytes() == b"preserved"
            before = canary.stat()
            assert not session.close()
            assert canary.is_file(), "shutdown deleted the command-created temp file"
            assert canary.read_bytes() == b"preserved"
            assert os.path.samestat(before, canary.stat())
        finally:
            canary.unlink(missing_ok=True)
            with contextlib.suppress(FileNotFoundError):
                directory.rmdir()


@pytest.mark.parametrize("needed", [False, True])
def test_elf_dependencies_without_interpreter_refuse_before_bootstrap(
    target: conftest.Target,
    static_launcher: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    needed: bool,
) -> None:
    """DT_NEEDED must refuse before spawning even when PT_INTERP is absent."""
    data = bytearray(static_launcher.read_bytes())
    offset = struct.unpack_from("<Q", data, 32)[0]
    size, count = struct.unpack_from("<HH", data, 54)
    assert size == 56
    headers = {
        entry: struct.unpack_from("<I", data, entry)[0]
        for entry in range(offset, offset + size * count, size)
    }
    assert 3 not in headers.values(), "PT_INTERP must not mask dependency validation"
    entry = next(entry for entry, kind in headers.items() if kind != 1)
    dynamic = struct.pack("<QQQQ", 1 if needed else 0, 0, 0, 0)
    struct.pack_into("<IIQQQQQQ", data, entry, 2, 4, len(data), 0, 0, 32, 32, 8)
    data.extend(dynamic)
    binary = target.home / "dynamic-without-interpreter"
    binary.write_bytes(data)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(binary))
    marker = target.home / "bootstrap-started"
    discovery = container.discover(
        target.discovery.target.host_root,
        container.ContainerOptions(
            exec=(
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            )
        ),
    )
    reason = "needs shared libraries.*static musl" if needed else "jail session died during setup"
    with pytest.raises(container.ContainerError, match=reason):
        session = container.prepare(discovery)
        session.close()
    assert marker.exists() is not needed


@pytest.mark.parametrize("symlink", [False, True])
def test_installed_launcher_refuses_a_substituted_entry(
    tmp_path: pathlib.Path, static_launcher: pathlib.Path, symlink: bool
) -> None:
    """Installed mode must not accept or delete a different inode or a symlink to itself."""
    contents = static_launcher.read_bytes()
    with tempfile.TemporaryDirectory(prefix="a6.", dir="/tmp") as temporary:
        directory = pathlib.Path(temporary)
        assert directory.stat().st_uid == os.geteuid()
        assert directory.stat().st_mode & 0o777 == 0o700
        running = directory / "running"
        shutil.copyfile(static_launcher, running)
        running.chmod(0o500)
        entry = directory / "launcher"
        if symlink:
            entry.symlink_to(running)
            reason = f"(os error {errno.ELOOP})"
        else:
            shutil.copyfile(static_launcher, entry)
            entry.chmod(0o500)
            assert entry.stat().st_dev == running.stat().st_dev
            assert entry.stat().st_ino != running.stat().st_ino
            reason = "installed launcher identity changed"
        before = entry.lstat()
        result = subprocess.run(
            (str(running), "--installed"),
            cwd=directory,
            input=json.dumps(
                {
                    "argv": ["/bin/true"],
                    "cwd": str(tmp_path),
                    "isolation": "none",
                    "network": "host",
                }
            )
            + "\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 2, result
        assert "installation: " in result.stderr and reason in result.stderr
        assert not result.stdout
        assert os.path.samestat(before, entry.lstat())
        assert entry.is_symlink() is symlink
        assert entry.read_bytes() == contents
