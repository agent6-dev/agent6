# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail's HOME is a directory of agent6's own.

`/tmp/agent6-home` inside strict's private tmpfs, or the persistent cache dir where there is no
private /tmp or where `[sandbox].home = "cache"` asks for it. The policy builder creates the
persistent one; the preflight refuses one that cannot be agent6's own.
"""

from __future__ import annotations

import dataclasses
import pathlib
import stat
import subprocess

import pytest

from agent6 import paths
from agent6.app import confine
from agent6.app import reporter as app_reporter
from agent6.config import Config, SandboxConfig
from agent6.sandbox import detect, jail
from agent6.tools import policy as policy_module


def _warned(isolation: str, cfg: Config, root: pathlib.Path) -> list[str]:
    env = detect.Environment(
        in_container=False,
        container_signals=(),
        kernel=detect.KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )
    lines: list[str] = []
    reporter = app_reporter.Reporter(out=lines.append, err=lines.append)
    confine.warn_sandbox_gaps(isolation, env, cfg, root=root, reporter=reporter)  # pyright: ignore[reportArgumentType]
    return lines


def test_strict_defaults_to_the_private_tmpfs_home(tmp_path: pathlib.Path) -> None:
    policy = policy_module.jail_policy(tmp_path, Config(), "strict", ("true",))
    assert dict(policy.env)["HOME"] == str(policy_module.JAIL_TMP_HOME) == "/tmp/agent6-home"
    assert paths.jail_cache_home(tmp_path) not in policy.extra_rw_paths
    assert not paths.jail_cache_home(tmp_path).exists()


@pytest.mark.parametrize(
    ("isolation", "cfg"),
    [
        ("hardened", Config()),
        ("none", Config()),
        ("strict", Config(sandbox=SandboxConfig(home="cache"))),
    ],
)
def test_the_builder_creates_and_grants_the_persistent_home(
    tmp_path: pathlib.Path, isolation: str, cfg: Config
) -> None:
    """Without a private /tmp the HOME persists; the builder creates it 0700 on the rw grant.

    The launcher skips a missing rw path silently, so `agent6 exec` and an MCP probe need it to
    exist.
    """
    home = paths.jail_cache_home(tmp_path)
    assert not home.exists()
    policy = policy_module.jail_policy(tmp_path, cfg, isolation, ("true",))  # pyright: ignore[reportArgumentType]
    assert dict(policy.env)["HOME"] == str(home)
    assert home in policy.extra_rw_paths
    assert home.is_dir() and not home.is_symlink()
    assert stat.S_IMODE(home.lstat().st_mode) == 0o700


def test_the_preflight_check_inspects_without_creating(tmp_path: pathlib.Path) -> None:
    """The refusal check itself writes nothing; creation is the builder's."""
    assert confine.check_jail_home(Config(), "hardened", tmp_path, explicitly_set=False) is None
    assert confine.config_refusal(Config(), "strict", tmp_path) is None
    assert not paths.jail_cache_home(tmp_path).exists()


def test_a_symlink_at_the_cache_home_refuses(tmp_path: pathlib.Path) -> None:
    """A symlink at the HOME path is refused: it would redirect every jailed write.

    The preflight refuses it as a message; `agent6 exec` gets the same words from the builder.
    """
    home = paths.jail_cache_home(tmp_path)
    home.parent.mkdir(parents=True, exist_ok=True)
    home.symlink_to(tmp_path)
    msg = confine.config_refusal(Config(), "hardened", tmp_path)
    assert msg is not None
    assert str(home) in msg and "symlink" in msg
    with pytest.raises(jail.JailUnavailableError, match="symlink"):
        policy_module.jail_policy(tmp_path, Config(), "hardened", ("true",))


def test_a_dir_owned_by_someone_else_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A HOME path owned by another user is never bound, and the message names both uids.

    The other user is faked by shifting `effective_user`; the ownership comparison is the real one.
    """
    home = paths.jail_cache_home(tmp_path)
    home.mkdir(parents=True)
    me = paths.effective_user()
    monkeypatch.setattr(paths, "effective_user", lambda: dataclasses.replace(me, uid=me.uid + 1))
    msg = confine.config_refusal(Config(), "hardened", tmp_path)
    assert msg is not None
    assert str(home) in msg and f"uid {me.uid}" in msg
    with pytest.raises(jail.JailUnavailableError, match="owned by"):
        policy_module.jail_policy(tmp_path, Config(), "hardened", ("true",))


def test_a_cache_home_open_to_others_refuses(tmp_path: pathlib.Path) -> None:
    """A HOME with any group or other bit is refused, naming the mode found and the fix.

    A jailed command owns HOME and may chmod it; another local user could plant a `~/.gitconfig`.
    """
    home = paths.jail_cache_home(tmp_path)
    home.mkdir(parents=True, mode=0o700)
    home.chmod(0o777)
    msg = confine.config_refusal(Config(), "hardened", tmp_path)
    assert msg is not None
    assert str(home) in msg and "0777" in msg and f"chmod 700 {home}" in msg
    with pytest.raises(jail.JailUnavailableError, match="chmod 700"):
        policy_module.jail_policy(tmp_path, Config(), "hardened", ("true",))
    assert stat.S_IMODE(home.lstat().st_mode) == 0o777  # the operator sees it, nothing rewrites it
    home.chmod(0o700)
    assert confine.config_refusal(Config(), "hardened", tmp_path) is None
    assert (
        paths.jail_cache_home(tmp_path)
        in policy_module.jail_policy(tmp_path, Config(), "hardened", ("true",)).extra_rw_paths
    )


def test_a_cache_home_inside_a_private_dir_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`XDG_CACHE_HOME` under the state base is refused: a writable grant inside the hidden tree."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "state" / "agent6" / "cache"))
    msg = confine.config_refusal(Config(), "hardened", tmp_path / "ws")
    assert msg is not None
    assert "private dir" in msg and str(tmp_path / "state") in msg
    with pytest.raises(jail.JailUnavailableError, match="private dir"):
        policy_module.jail_policy(tmp_path / "ws", Config(), "hardened", ("true",))


def test_a_symlinked_ancestor_into_a_private_dir_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The containment check compares resolved paths; a symlink into the state base is refused.

    A symlinked ancestor elsewhere is an ordinary cache location.
    """
    state = tmp_path / "state"
    (state / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    (tmp_path / "link").symlink_to(state)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "link" / "agent6" / "cache"))
    msg = confine.config_refusal(Config(), "hardened", tmp_path / "ws")
    assert msg is not None
    assert "private dir" in msg and str(state) in msg
    with pytest.raises(jail.JailUnavailableError, match="private dir"):
        policy_module.jail_policy(tmp_path / "ws", Config(), "hardened", ("true",))
    assert not (state / "agent6" / "cache").exists()
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "elsewhere").symlink_to(real)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "elsewhere" / "cache"))
    assert confine.config_refusal(Config(), "hardened", tmp_path / "ws") is None
    policy = policy_module.jail_policy(tmp_path / "ws", Config(), "hardened", ("true",))
    homes = tmp_path / "elsewhere" / "cache" / "agent6" / "homes"
    assert homes / paths.repo_id(tmp_path / "ws") in policy.extra_rw_paths
    assert (real / "cache" / "agent6" / "homes" / paths.repo_id(tmp_path / "ws")).is_dir()


@pytest.mark.parametrize("isolation", ["hardened", "none"])
def test_an_explicit_tmp_home_refuses_where_there_is_no_private_tmp(
    tmp_path: pathlib.Path, isolation: str
) -> None:
    """`home = "tmp"` needs strict's private tmpfs: written down it refuses, as default degrades."""
    msg = confine.check_jail_home(Config(), isolation, tmp_path, explicitly_set=True)  # pyright: ignore[reportArgumentType]
    assert msg is not None
    assert "requires the strict isolation" in msg
    assert f"resolved to '{isolation}'" in msg
    assert "sandbox.home = 'cache'" in msg
    explicit = frozenset({"sandbox.home"})
    assert confine.config_refusal(Config(), isolation, tmp_path, explicit_leaves=explicit) == msg  # pyright: ignore[reportArgumentType]
    assert confine.config_refusal(Config(), isolation, tmp_path) is None  # pyright: ignore[reportArgumentType]
    assert confine.config_refusal(Config(), "strict", tmp_path, explicit_leaves=explicit) is None


def test_the_default_degrades_with_a_warning_naming_the_home(tmp_path: pathlib.Path) -> None:
    """Hardened's start-of-run warning names the persistent HOME whatever `protect_git` says."""
    assert confine.check_jail_home(Config(), "hardened", tmp_path, explicitly_set=False) is None
    for cfg in (Config(), Config(sandbox=SandboxConfig(protect_git=False))):
        home = str(paths.jail_cache_home(tmp_path))
        lines = [line for line in _warned("hardened", cfg, tmp_path) if home in line]
        assert len(lines) == 1, lines
        assert "persists across runs" in lines[0] and "executable" in lines[0]


def test_strict_cache_is_an_explicit_widening_that_warns(tmp_path: pathlib.Path) -> None:
    lines = _warned("strict", Config(sandbox=SandboxConfig(home="cache")), tmp_path)
    assert any(
        "sandbox.home = 'cache'" in line
        and str(paths.jail_cache_home(tmp_path)) in line
        and "persists" in line
        for line in lines
    ), lines
    assert not any(
        str(paths.jail_cache_home(tmp_path)) in line
        for line in _warned("strict", Config(), tmp_path)
    )


def _persistent_home(
    root: pathlib.Path, worktree_git_dir: pathlib.Path | None = None
) -> pathlib.Path:
    cfg = Config(sandbox=SandboxConfig(home="cache"))
    policy = policy_module.jail_policy(
        root, cfg, "strict", ("true",), worktree_git_dir=worktree_git_dir
    )
    return pathlib.Path(dict(policy.env)["HOME"])


def test_each_repository_gets_its_own_persistent_home(tmp_path: pathlib.Path) -> None:
    """One shared home let a file one repository's run wrote reach every other repository's."""
    one, other = tmp_path / "one", tmp_path / "other"
    one.mkdir()
    other.mkdir()

    home, other_home = _persistent_home(one), _persistent_home(other)

    assert home != other_home
    assert home == paths.jail_cache_home(one) == paths.cache_dir() / "homes" / paths.repo_id(one)
    assert stat.S_IMODE(home.parent.lstat().st_mode) == 0o700
    assert not (paths.cache_dir() / "home").exists()


def test_a_worktree_gets_its_repositorys_home_from_the_recorded_git_dir(
    tmp_path: pathlib.Path,
) -> None:
    """Keyed on the worktree's `.git` pointer, a jailed rewrite would aim HOME at another repo."""
    repo, worktree = tmp_path / "repo", tmp_path / "wt"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", str(worktree)], check=True)
    git_dir = (repo / ".git").resolve()

    assert _persistent_home(worktree, worktree_git_dir=git_dir) == paths.jail_cache_home(repo)
    assert _persistent_home(repo) == paths.jail_cache_home(repo)


def test_a_lanes_clone_and_a_machine_clone_share_the_repositorys_home(
    tmp_path: pathlib.Path,
) -> None:
    """A lane's clone and a machine state's clone each got a HOME of their own.

    A tool cache warmed for the repository was absent there. Every working tree agent6
    makes for a repository sits under that repository's directory in the parallel workdir,
    and its commands get the repository's HOME; a directory elsewhere keeps its own.
    """
    repo, base = tmp_path / "repo", tmp_path / "work"
    repo.mkdir()
    rid = paths.repo_id(repo)
    lane = base / rid / "fanout-AAAAAA" / "lane-1"
    state_clone = base / rid / "machine-digest" / "state-0003"
    stranger = tmp_path / "elsewhere" / rid / "x"
    for made in (lane, state_clone, stranger):
        made.mkdir(parents=True)
    cfg = Config.model_validate({"sandbox": {"home": "cache"}, "parallel": {"workdir": str(base)}})

    def home(root: pathlib.Path) -> pathlib.Path | None:
        return policy_module.persistent_jail_home(cfg, "strict", root)

    assert home(lane) == home(state_clone) == home(repo) == paths.jail_cache_home(repo)
    assert home(stranger) == paths.jail_cache_home(stranger)
