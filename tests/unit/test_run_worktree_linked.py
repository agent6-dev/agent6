from __future__ import annotations

import pathlib
import time
from typing import Any

import pytest

from agent6 import kinds, paths
from agent6.app import fork, session_runtime
from agent6.app import run as run_lifecycle
from agent6.config import Config, layer
from agent6.sandbox import jail
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import policy as tools_policy
from agent6.ui.cli import cli_main
from agent6.ui.cli import run as cli_run
from tests.unit import test_run_worktree as worktree_tests


def _checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, pathlib.Path]:
    repo = worktree_tests._repo(tmp_path, monkeypatch)
    checkout = tmp_path / "user-worktree"
    worktree_tests._git(repo, "worktree", "add", "--detach", str(checkout), "HEAD")
    monkeypatch.chdir(checkout)
    return repo, checkout


@pytest.mark.parametrize("linked", [False, True], ids=["plain", "linked"])
def test_start_records_only_a_linked_checkouts_git_grant_before_tools(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, linked: bool
) -> None:
    """A user-created worktree needs the same recorded grant as an agent6-created worktree."""
    repo, checkout = _checkout(tmp_path, monkeypatch)
    if not linked:
        checkout = repo
        monkeypatch.chdir(repo)
    cfg = layer.load_effective(checkout).config
    cfg = cfg.model_copy(update={"git": cfg.git.model_copy(update={"dirty_tree": "include"})})
    session_dir = worktree_tests._run_dir(repo, "user-linked")
    policies: list[kinds.JailPolicy] = []
    real_tools = session_runtime.build_session_tools

    def tools(config: Config, **kw: Any) -> session_runtime.SessionTools:
        recorded = sessions_manifest.read_manifest(session_dir).worktree_git_dir
        assert recorded == ((repo / ".git").resolve() if linked else None)
        built = real_tools(config, **kw)
        policies.append(built.dispatcher._jail_policy(("true",)))
        return built

    monkeypatch.setattr(session_runtime, "build_session_tools", tools)
    worktree_tests._script(monkeypatch)
    assert (
        run_lifecycle.run_task(
            cfg,
            "look around",
            frontend=cli_run.session_frontend(),
            started_at=time.time(),
            session_id="user-linked",
        )
        == 0
    )
    saved = sessions_manifest.read_manifest(session_dir)
    assert saved.worktree is None, "a user-owned checkout must not become eligible for pruning"
    assert saved.worktree_git_dir == ((repo / ".git").resolve() if linked else None)
    assert len(policies) == 1
    assert policies[0].extra_ro_paths == ((repo / ".git",) if linked else ())
    assert repo not in policies[0].extra_rw_paths


def test_resume_and_forks_keep_the_recorded_git_grant(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume and a same-checkout fork must not drop or rediscover the source's grant."""
    repo, checkout = _checkout(tmp_path, monkeypatch)
    worktree_tests._script(monkeypatch)
    assert cli_main(["run", "--session-id", "linked-source", "look around"]) == 0
    granted: list[pathlib.Path | None] = []
    real_tools = session_runtime.build_session_tools

    def tools(config: Config, **kw: Any) -> session_runtime.SessionTools:
        granted.append(kw["worktree_git_dir"])
        return real_tools(config, **kw)

    monkeypatch.setattr(session_runtime, "build_session_tools", tools)
    worktree_tests._script(monkeypatch)
    assert cli_main(["resume", "linked-source", "--steer", "look again"]) == 0
    child, rc = fork.create_fork(
        (), "linked-source", new_session_id="linked-child", cwd=checkout, worktree=False
    )
    assert rc == 0 and child == "linked-child"
    saved = sessions_manifest.read_manifest(worktree_tests._run_dir(repo, child))
    assert saved.worktree is None
    assert saved.worktree_git_dir == repo / ".git"
    worktree_tests._script(monkeypatch)
    assert cli_main(["resume", child, "--steer", "look again"]) == 0
    assert granted == [repo / ".git", repo / ".git"]
    assert cli_main(["fork", child, "--session-id", "linked-new", "--no-run"]) == 0
    added = sessions_manifest.read_manifest(worktree_tests._run_dir(repo, "linked-new"))
    assert added.worktree is not None and added.worktree != checkout
    assert added.worktree_git_dir == repo / ".git"


@pytest.mark.parametrize("via_commondir", [False, True], ids=["direct", "commondir"])
@pytest.mark.parametrize("surface", ["cli", "app"])
def test_a_pointer_outside_any_repository_refuses_before_start(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    via_commondir: bool,
    surface: str,
) -> None:
    """A .git file is not permission to expose an arbitrary directory to a command."""
    _repo, checkout = _checkout(tmp_path, monkeypatch)
    pointer = checkout / ".git"
    target = pathlib.Path("/etc")
    if via_commondir:
        target = checkout / "admin"
        target.mkdir()
        (target / "commondir").write_text("/etc\n", encoding="utf-8")
    pointer.write_text(f"gitdir: {target}\n", encoding="utf-8")
    provider = worktree_tests._script(monkeypatch)
    if surface == "cli":
        rc = cli_main(["run", "--session-id", "bad-pointer", "look around"])
    else:
        rc = run_lifecycle.run_task(
            layer.load_effective(checkout).config,
            "look around",
            frontend=cli_run.session_frontend(),
            started_at=time.time(),
            session_id="bad-pointer",
        )
    assert rc == 2
    refusal = capsys.readouterr().err
    assert str(pointer) in refusal and "linked worktree" in refusal
    assert len(refusal.strip().splitlines()) == 1
    assert provider.calls == 0
    assert not list(paths.state_base().glob("*/sessions/runs/bad-pointer/manifest.json"))


def test_a_rewritten_pointer_cannot_replace_the_startup_grant(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved grant, not a replacement .git pointer, controls later command policies."""
    repo, checkout = _checkout(tmp_path, monkeypatch)
    worktree_tests._script(monkeypatch)
    assert cli_main(["run", "--session-id", "linked-rewrite", "look around"]) == 0
    saved = sessions_manifest.read_manifest(worktree_tests._run_dir(repo, "linked-rewrite"))
    (checkout / ".git").write_text("gitdir: /etc\n", encoding="utf-8")
    with pytest.raises(jail.JailUnavailableError, match="a rewritten pointer grants nothing"):
        tools_policy.jail_policy(
            checkout, Config(), "partial", ("true",), worktree_git_dir=saved.worktree_git_dir
        )
