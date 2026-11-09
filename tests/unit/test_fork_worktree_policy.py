# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail grants a fork's git dir from the manifest, never from the worktree's pointer file.

Under `partial` the pointer sits in the writable workspace, so a jailed command could rewrite it to
name any host directory.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6.app import confine  # pyright: ignore[reportPrivateUsage]
from agent6.config import Config
from agent6.sandbox import jail
from agent6.tools import policy as tools_policy
from agent6.vcs import git_ops


def _repo_and_worktree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    worktree = tmp_path / "wt"
    git_ops.add_worktree(repo, worktree, "HEAD")
    return repo, worktree


def _rewrite_pointer(worktree: pathlib.Path, target: str) -> None:
    """What a jailed command can do under partial: point `.git` at any host dir via `commondir`."""
    evil = worktree / "evil"
    evil.mkdir()
    (evil / "commondir").write_text(f"{target}\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {evil}\n", encoding="utf-8")


def test_the_recorded_git_dir_is_granted_read_only(tmp_path: pathlib.Path) -> None:
    """With the recorded dir matching the pointer, the policy grants that dir read-only.

    The exposure scan under `partial` lists it under the same name.
    """
    repo, worktree = _repo_and_worktree(tmp_path)
    git_dir = (repo / ".git").resolve()
    policy = tools_policy.jail_policy(
        worktree, Config(), "full", ("true",), worktree_git_dir=git_dir
    )
    assert git_dir in policy.extra_ro_paths and git_dir not in policy.extra_rw_paths
    regions = dict(confine._grant_regions(Config(), worktree, "partial", git_dir))
    assert regions[git_dir].startswith("the repository's .git")


@pytest.mark.parametrize("protect_git", ["auto", "on", "off"])
def test_a_write_grant_inside_the_recorded_git_dir_refuses_the_start(
    tmp_path: pathlib.Path, protect_git: str
) -> None:
    """The jail protects paths under the workspace only, so such a grant is refused, not covered."""
    repo, worktree = _repo_and_worktree(tmp_path)
    git_dir = repo / ".git"
    admin = git_dir / "worktrees" / worktree.name
    cfg = Config.model_validate(
        {
            "sandbox": {
                "isolation": "full",
                "protect_git": protect_git,
                "extra_write_paths": [str(admin)],
            }
        }
    )
    refusal = confine.check_write_grants_outside_git_dir(cfg, worktree, git_dir)
    assert refusal is not None and str(admin) in refusal and str(git_dir) in refusal
    assert confine.check_write_grants_outside_git_dir(cfg, worktree, None) is None
    policy = tools_policy.jail_policy(worktree, cfg, "full", ("true",), worktree_git_dir=git_dir)
    assert git_dir not in policy.extra_protect_paths


def test_a_rewritten_pointer_refuses_instead_of_granting(tmp_path: pathlib.Path) -> None:
    """A pointer that no longer resolves to the recorded dir refuses the policy, naming both."""
    repo, worktree = _repo_and_worktree(tmp_path)
    git_dir = (repo / ".git").resolve()
    _rewrite_pointer(worktree, "/etc")
    with pytest.raises(jail.JailUnavailableError, match="/etc") as refused:
        tools_policy.jail_policy(worktree, Config(), "partial", ("true",), worktree_git_dir=git_dir)
    assert str(git_dir) in str(refused.value)
    with pytest.raises(jail.JailUnavailableError):
        confine._grant_regions(Config(), worktree, "partial", git_dir)


def test_a_linked_worktree_agent6_did_not_record_gets_no_grant(tmp_path: pathlib.Path) -> None:
    """Without a record there is nothing to grant: a foreign worktree gets only the workspace."""
    repo, worktree = _repo_and_worktree(tmp_path)
    assert tools_policy.jail_policy(worktree, Config(), "full", ("true",)).extra_ro_paths == ()
    _rewrite_pointer(worktree, "/etc")
    # `partial` shares the host network, whose resolver and CA store reads are the only grants.
    partial = tools_policy.jail_policy(worktree, Config(), "partial", ("true",))
    assert partial.extra_ro_paths == tools_policy.host_network_reads()
    assert (repo / ".git").resolve() not in dict(confine._grant_regions(Config(), repo, "partial"))
