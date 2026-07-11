# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 system apparmor` install/remove/status."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli import main
from agent6.ui.cli import system_cmds as sc


@pytest.fixture
def priv_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the privileged argv instead of running sudo."""
    recorded: list[list[str]] = []

    def _fake_run_priv(argv: list[str], *, what: str) -> bool:
        recorded.append(argv)
        return True

    monkeypatch.setattr(sc, "_run_priv", _fake_run_priv)
    return recorded


def test_status_reports_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))
    rc = main(["system", "apparmor", "status"])
    assert rc == 0


def test_install_refused_on_non_apparmor_host(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: False)
    rc = sc._cmd_system_apparmor("install")  # pyright: ignore[reportPrivateUsage]
    assert rc == 1
    assert "does not use AppArmor" in capsys.readouterr().err


def test_install_writes_profile_and_reloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, priv_calls: list[list[str]]
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    dest = tmp_path / "agent6-jail"
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(dest))
    rc = sc._cmd_system_apparmor("install")  # pyright: ignore[reportPrivateUsage]
    assert rc == 0
    # cp <tmp> <dest>, then apparmor_parser -r <dest>
    assert priv_calls[0][0] == "cp" and priv_calls[0][2] == str(dest)
    assert priv_calls[1][:2] == ["apparmor_parser", "-r"]
    # The bundled profile content is well-formed and pins the launcher binary.
    assert sc._APPARMOR_PROFILE.startswith("# AppArmor profile")  # pyright: ignore[reportPrivateUsage]
    assert "profile agent6-jail /**/agent6/sandbox/_bin/agent6-jail" in sc._APPARMOR_PROFILE  # pyright: ignore[reportPrivateUsage]


def test_remove_absent_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(tmp_path / "nope"))
    rc = sc._cmd_system_apparmor("remove")  # pyright: ignore[reportPrivateUsage]
    assert rc == 0


def test_remove_unloads_then_deletes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, priv_calls: list[list[str]]
) -> None:
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))

    # The recorded mock leaves the file; have `rm` actually delete it so the
    # post-removal file check (success = file gone) sees success.
    def _run_priv_rm(argv: list[str], *, what: str) -> bool:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, priv_calls: list[list[str]]
) -> None:
    # If the privileged rm couldn't delete the file, removal failed (exit 1) --
    # but a failed -R (profile present-but-not-loaded) alone must NOT fail it.
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    profile = tmp_path / "agent6-jail"
    profile.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(profile))
    rc = sc._cmd_system_apparmor("remove")  # pyright: ignore[reportPrivateUsage]
    assert rc == 1  # priv_calls mock left the file in place


def test_install_removes_profile_when_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # cp succeeded but apparmor_parser -r refused: leaving the file installed
    # would make `system apparmor status` claim a profile the kernel never
    # loaded. The install removes it again and exits 1.
    monkeypatch.setattr(sc, "_apparmor_present", lambda: True)
    dest = tmp_path / "agent6-jail"
    monkeypatch.setattr(sc, "_APPARMOR_PROFILE_PATH", str(dest))
    calls: list[list[str]] = []

    def _run_priv(argv: list[str], *, what: str) -> bool:
        calls.append(argv)
        if argv[0] == "cp":
            dest.write_text("x", encoding="utf-8")
            return True
        if argv[0] == "apparmor_parser":
            return False
        if argv[0] == "rm":
            dest.unlink(missing_ok=True)
            return True
        return True

    monkeypatch.setattr(sc, "_run_priv", _run_priv)
    rc = sc._cmd_system_apparmor("install")  # pyright: ignore[reportPrivateUsage]
    assert rc == 1
    assert ["rm", "-f", str(dest)] in calls
    assert not dest.is_file()
    assert "Removed" in capsys.readouterr().err
