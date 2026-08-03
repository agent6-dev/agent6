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


def test_state_dir_and_repo_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "state"
    monkeypatch.setenv("AGENT6_STATE_HOME", str(base))
    repo = tmp_path / "myrepo"
    repo.mkdir()
    rid = paths.repo_id(repo)
    # The id names the workspace's whole path, so a state dir says where it came
    # from; the trailing hash is what keeps two workspaces apart.
    assert "myrepo" in rid
    assert rid.endswith("-" + rid.rsplit("-", 1)[1]) and len(rid.rsplit("-", 1)[1]) == 6
    assert "/" not in rid and not rid.startswith(".")
    assert paths.repo_id(repo) == rid  # deterministic
    assert paths.state_dir(repo) == base / rid
    assert paths.repo_config_path(repo) == base / rid / "config.toml"
    # An explicit base override appends the same repo id.
    assert paths.state_dir(repo, base_override="/custom") == Path("/custom") / rid


def test_repo_id_distinguishes_paths(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert paths.repo_id(a) != paths.repo_id(b)


def test_repo_id_separates_paths_that_flatten_alike(tmp_path: Path) -> None:
    """`/a/b/c` and `/a/b-c` both flatten to `a-b-c`. Sharing one state dir
    between two real workspaces is worse than an unreadable name, so the hash
    has to separate them."""
    nested = tmp_path / "b" / "c"
    nested.mkdir(parents=True)
    dashed = tmp_path / "b-c"
    dashed.mkdir()
    assert paths.repo_id(nested) != paths.repo_id(dashed)


def test_repo_id_stays_a_usable_directory_name(tmp_path: Path) -> None:
    """A deep workspace must not produce a name the filesystem refuses (255
    bytes per component) or one `ls` hides."""
    deep = tmp_path.joinpath(*[f"segment{i}" for i in range(30)])
    deep.mkdir(parents=True)
    rid = paths.repo_id(deep)
    assert len(rid.encode()) < 255
    (tmp_path / rid).mkdir()  # the real filesystem accepts it


def test_state_base_uses_xdg_when_not_sudo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AGENT6_STATE_HOME", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert paths.state_base() == tmp_path / "xdg" / "agent6"


def test_data_dir_honors_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT6_DATA_HOME", str(tmp_path / "d"))
    assert paths.data_dir() == tmp_path / "d"


def test_data_dir_uses_xdg_when_not_sudo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AGENT6_DATA_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert paths.data_dir() == tmp_path / "xdg" / "agent6"


def test_data_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AGENT6_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    home = paths.effective_user().home
    assert paths.data_dir() == home / ".local" / "share" / "agent6"


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


def test_mkdir_for_real_user_hands_back_created_ancestors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under sudo, every directory the call CREATES is handed back to the real
    operator: chowning only the deepest one left a root-owned state/config
    BASE that no later non-root process could create a sibling in. Directories
    that already existed are never touched."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "1234")
    chowned: list[Path] = []

    def _record(*a: object) -> None:
        chowned.append(Path(str(a[0])))

    monkeypatch.setattr(os, "lchown", _record)
    base = tmp_path / "existing"
    base.mkdir()
    target = base / "agent6" / "repo-abc"
    paths.mkdir_for_real_user(target)
    assert target.is_dir()
    assert base / "agent6" in chowned  # the created ancestor is handed back
    assert target in chowned
    assert base not in chowned  # pre-existing dirs are never rechowned
    # Nothing missing: the handover still covers the path itself (the
    # behavior of the per-site mkdir+chown pairs this primitive replaces).
    chowned.clear()
    paths.mkdir_for_real_user(target)
    assert chowned == [target]


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
