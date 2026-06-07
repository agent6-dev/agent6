# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.paths (XDG resolution, sudo/root handling)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6 import paths


def test_global_config_dir_honors_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    assert paths.global_config_dir() == tmp_path / "g"
    assert paths.global_config_path() == tmp_path / "g" / "config.toml"
    assert paths.secrets_path() == tmp_path / "g" / "secrets.toml"


def test_global_config_dir_uses_xdg_when_not_sudo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AGENT6_CONFIG_HOME", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # Not running as root -> via_sudo is False -> XDG is honored.
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert paths.global_config_dir() == tmp_path / "xdg" / "agent6"


def test_repo_config_path() -> None:
    assert paths.repo_config_path(Path("/repo")) == Path("/repo/.agent6/config.toml")


def test_effective_user_resolves_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    real_uid = os.getuid()
    real_gid = os.getgid()
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", str(real_uid))
    monkeypatch.setenv("SUDO_GID", str(real_gid))
    monkeypatch.setenv("SUDO_USER", "alice")
    user = paths.effective_user()
    assert user.via_sudo is True
    assert user.uid == real_uid
    assert user.gid == real_gid
    assert user.name == "alice"


def test_effective_user_non_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.delenv("SUDO_UID", raising=False)
    user = paths.effective_user()
    assert user.via_sudo is False


def test_root_optin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT6_ALLOW_ROOT", raising=False)
    assert paths.root_optin_enabled(False) is False
    assert paths.root_optin_enabled(True) is True
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "1")
    assert paths.root_optin_enabled(False) is True
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "0")
    assert paths.root_optin_enabled(False) is False


def test_chown_to_real_user_is_noop_when_not_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    f = tmp_path / "x"
    f.write_text("hi", encoding="utf-8")
    # Must not raise and must not attempt a chown.
    called: list[object] = []

    def _fake_lchown(*a: object) -> None:
        called.append(a)

    monkeypatch.setattr(os, "lchown", _fake_lchown)
    paths.chown_to_real_user(f)
    assert called == []
