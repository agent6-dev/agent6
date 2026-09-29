# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 system apparmor` install/remove/status."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

import pytest

from agent6.ui.cli import main
from agent6.ui.cli import system_cmds as sc


@pytest.fixture
def priv_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the privileged argv instead of running sudo."""
    recorded: list[list[str]] = []

    def _fake_run_priv(argv: list[str], *, what: str, required: bool = True) -> bool:
        recorded.append(argv)
        return True

    monkeypatch.setattr(sc, "_run_priv", _fake_run_priv)
    return recorded


def test_status_reports_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))
    assert main(["system", "apparmor", "status"]) == 0
    assert "AppArmor profile: installed" in capsys.readouterr().out
    # and a missing profile reads not-installed
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(tmp_path / "absent"))
    assert main(["system", "apparmor", "status"]) == 0
    assert "AppArmor profile: not installed" in capsys.readouterr().out


def test_status_recognizes_apparmor_when_parser_is_outside_the_user_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _not_found(name: str) -> None:
        return None

    monkeypatch.setattr(sc, "_host_lsm", lambda: "lockdown,yama,apparmor,bpf")
    monkeypatch.setattr(shutil, "which", _not_found)
    assert sc._cmd_system_apparmor("status") == 0  # pyright: ignore[reportPrivateUsage]
    assert "does not use AppArmor" not in capsys.readouterr().out


def test_install_refused_on_non_apparmor_host(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: False)
    rc = sc._cmd_system_apparmor("install")  # pyright: ignore[reportPrivateUsage]
    assert rc == 1
    assert "does not use AppArmor" in capsys.readouterr().err


def test_install_writes_profile_and_reloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, priv_calls: list[list[str]]
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    dest = tmp_path / "agent6-jail"
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(dest))
    rc = sc._cmd_system_apparmor("install")  # pyright: ignore[reportPrivateUsage]
    assert rc == 0
    # cp <tmp> <dest>, then apparmor_parser -r <dest>
    assert priv_calls[0][:2] == ["apparmor_parser", "-r"]  # loaded from the temp file first
    assert priv_calls[1][0] == "cp" and priv_calls[1][2] == str(dest)
    # The bundled profile pins the launcher binary.
    assert "profile agent6-jail /**/agent6/sandbox/_bin/agent6-jail" in sc._APPARMOR_PROFILE  # pyright: ignore[reportPrivateUsage]


def test_bundled_profile_parses_as_apparmor() -> None:
    """apparmor_parser accepts the bundled profile.

    A text assertion would let a rename of AppArmor's own keyword ship, leaving strict silently
    unavailable on the hosts that need it most.
    """
    parser = shutil.which("apparmor_parser")
    if parser is None:
        pytest.skip("apparmor_parser not installed")
    profile = tempfile.NamedTemporaryFile("w", suffix=".aa", delete=False)  # noqa: SIM115
    with profile as fh:
        fh.write(sc._APPARMOR_PROFILE)  # pyright: ignore[reportPrivateUsage]
    # -d parses and dumps; it never loads into the kernel, so no privileges.
    done = subprocess.run([parser, "-d", profile.name], capture_output=True, text=True, check=False)
    pathlib.Path(profile.name).unlink(missing_ok=True)
    assert done.returncode == 0, done.stderr


def test_remove_absent_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(tmp_path / "nope"))
    rc = sc._cmd_system_apparmor("remove")  # pyright: ignore[reportPrivateUsage]
    assert rc == 0


def test_remove_deletes_profile_after_apparmor_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: False)
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))
    calls: list[list[str]] = []

    def _run_priv(argv: list[str], *, what: str, required: bool = True) -> bool:
        calls.append(argv)
        profile.unlink()
        return True

    monkeypatch.setattr(sc, "_run_priv", _run_priv)
    assert sc._cmd_system_apparmor("remove") == 0  # pyright: ignore[reportPrivateUsage]
    assert calls == [["rm", "-f", str(profile)]]


def test_remove_unloads_then_deletes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, priv_calls: list[list[str]]
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))

    # The mock `rm` deletes the file, so the post-removal check sees success.
    def _run_priv_rm(argv: list[str], *, what: str, required: bool = True) -> bool:
        priv_calls.append(argv)
        if argv and argv[0] == "rm":
            profile.unlink(missing_ok=True)
        return True

    monkeypatch.setattr(sc, "_run_priv", _run_priv_rm)
    rc = sc._cmd_system_apparmor("remove")  # pyright: ignore[reportPrivateUsage]
    assert rc == 0
    assert priv_calls[0][:2] == ["apparmor_parser", "-R"]  # unload first
    assert priv_calls[1][0] == "rm"  # then delete


def test_remove_reports_failure_if_file_remains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, priv_calls: list[list[str]]
) -> None:
    # A failed privileged rm is exit 1; a failed -R alone (present but not loaded) is not.
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))
    rc = sc._cmd_system_apparmor("remove")  # pyright: ignore[reportPrivateUsage]
    assert rc == 1  # priv_calls mock left the file in place


def test_remove_does_not_report_an_unloaded_profile_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))
    monkeypatch.setattr(sc.os, "geteuid", lambda: 0)

    def _run(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        if argv[0] == "rm":
            profile.unlink()
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(sc.subprocess, "run", _run)
    assert sc._cmd_system_apparmor("remove") == 0  # pyright: ignore[reportPrivateUsage]
    assert "ERROR:" not in capsys.readouterr().err


def test_a_profile_the_parser_refuses_never_reaches_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A profile the parser refuses never reaches the path.

    A failed reinstall keeps the working one.

    The kernel loads the profile from the temp file before it is copied into place.
    """
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    dest = tmp_path / "agent6-jail"
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(dest))
    calls: list[list[str]] = []

    def _run_priv(argv: list[str], *, what: str, required: bool = True) -> bool:
        calls.append(argv)
        if argv[0] == "cp":
            shutil.copyfile(argv[1], argv[2])
            return True
        return argv[0] != "apparmor_parser"

    monkeypatch.setattr(sc, "_run_priv", _run_priv)
    assert sc._cmd_system_apparmor("install") == 1  # pyright: ignore[reportPrivateUsage]
    assert not dest.exists()
    assert [c[0] for c in calls] == ["apparmor_parser"]
    dest.write_text("previous", encoding="utf-8")
    calls.clear()
    assert sc._cmd_system_apparmor("install") == 1  # pyright: ignore[reportPrivateUsage]
    assert dest.read_text(encoding="utf-8") == "previous"
    assert [c[0] for c in calls] == ["apparmor_parser"]


def test_a_failed_first_copy_removes_its_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A failed first copy removes its partial file, so nothing reads as installed."""
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    dest = tmp_path / "agent6-jail"
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(dest))

    def _run_priv(argv: list[str], *, what: str, required: bool = True) -> bool:
        if argv[0] == "cp":
            dest.write_text("partial", encoding="utf-8")
            return False
        if argv[0] == "rm":
            dest.unlink()
            return True
        return argv[0] == "apparmor_parser"

    monkeypatch.setattr(sc, "_run_priv", _run_priv)
    assert sc._cmd_system_apparmor("install") == 1  # pyright: ignore[reportPrivateUsage]
    assert not dest.exists()
