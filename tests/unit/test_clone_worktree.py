# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A session worktree of the `clone` kind: its own git data, its commits copied into the repository.

Driven through `cli_main` with a scripted provider over a real git repository, as the linked
kind is in `test_run_worktree.py`.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import paths
from agent6.app import fork_worktrees, role_providers
from agent6.providers import ProviderResponse
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import cli_main
from agent6.vcs.git import read as git_read

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "deny"

[models.default]
model = "p/m"

[models.worker]
model = "p/m"

[providers.p]
backend = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
when = "never"
"""

_FINISH: tuple[dict[str, Any], ...] = (
    {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "done"}},
)


@pytest.fixture(autouse=True)
def _no_controlling_terminal(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent6.portable.has_controlling_tty", lambda: False)


class _Scripted:
    """One canned ProviderResponse per call; the last repeats."""

    def __init__(self, script: list[tuple[dict[str, Any], ...]]) -> None:
        self.script = script
        self.calls = 0

    def call(self, **_kw: Any) -> ProviderResponse:
        tools = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw={"content": list(tools)},
        )


def _edit(path: str, content: str) -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "tool_use",
            "id": "e1",
            "name": "apply_edit",
            "input": {"path": path, "old_string": "", "new_string": content, "kind": "overwrite"},
        },
    )


def _script(monkeypatch: pytest.MonkeyPatch, *turns: tuple[dict[str, Any], ...]) -> None:
    prov = _Scripted([*turns, _FINISH])

    def factory(*_a: object, **_k: object) -> _Scripted:
        return prov

    monkeypatch.setattr(role_providers, "build_role_provider", factory)


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, extra: str = "") -> pathlib.Path:
    """A committed repo, the process cwd, holding the operator's uncommitted work."""
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(_CONFIG + extra, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "seed")
    (repo / "a.py").write_text("x = 2  # the operator's edit\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    return repo


def _manifest(repo: pathlib.Path, session_id: str) -> sessions_manifest.SessionManifest:
    return sessions_manifest.read_manifest(paths.state_dir(repo) / "sessions/runs" / session_id)


def _run(repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, session_id: str) -> pathlib.Path:
    """Run one editing session in its own worktree and return the worktree."""
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    assert cli_main(["run", "--worktree", "--session-id", session_id, "add new.txt"]) == 0
    worktree = _manifest(repo, session_id).worktree
    assert worktree is not None
    return worktree


def test_a_run_in_a_clone_copies_its_commits_into_the_repository(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default kind: a clone with its own git data, and the repository holds its commits.

    Left in the clone alone, the run's branch could not be merged or diffed from the checkout.
    """
    repo = _repo(tmp_path, monkeypatch)
    before = _git(repo, "status", "--porcelain"), _git(repo, "rev-parse", "HEAD")

    worktree = _run(repo, monkeypatch, "cl-AAAA01")

    manifest = _manifest(repo, "cl-AAAA01")
    assert fork_worktrees.is_clone(worktree) and manifest.worktree_git_dir is None
    assert (worktree / "new.txt").read_text(encoding="utf-8") == "made by the run\n"
    tip = git_read.chain_tip(repo, git_read.chain_ref_for("cl-AAAA01"))
    assert tip is not None and tip == git_read.chain_tip(
        worktree, git_read.chain_ref_for("cl-AAAA01")
    )
    assert git_read.tree_diff_paths(repo, manifest.base_sha, tip) == ["new.txt"]
    assert git_read.branch_tip_sha(repo, "agent6/cl-AAAA01") == tip
    assert (_git(repo, "status", "--porcelain"), _git(repo, "rev-parse", "HEAD")) == before
    assert len(_git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2  # itself
    assert paths.state_dir(worktree) == paths.state_dir(repo)


def test_a_command_typed_in_the_clone_finds_the_repositorys_sessions(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clone is its repository's project, as a linked worktree is: one state, one session list."""
    repo = _repo(tmp_path, monkeypatch)
    worktree = _run(repo, monkeypatch, "cl-HERE01")
    capsys.readouterr()

    monkeypatch.chdir(worktree)
    assert cli_main(["sessions", "show", "cl-HERE01"]) == 0
    assert "cl-HERE01" in capsys.readouterr().out


def test_resume_from_the_checkout_continues_in_the_clone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second execution's commit reaches the repository too."""
    repo = _repo(tmp_path, monkeypatch)
    worktree = _run(repo, monkeypatch, "cl-RESU01")
    first = git_read.chain_tip(repo, git_read.chain_ref_for("cl-RESU01"))

    _script(monkeypatch, _edit("more.txt", "from the resume\n"))
    assert cli_main(["resume", "cl-RESU01", "--steer", "add more.txt"]) == 0

    tip = git_read.chain_tip(repo, git_read.chain_ref_for("cl-RESU01"))
    assert tip is not None and tip != first
    assert (worktree / "more.txt").exists() and not (repo / "more.txt").exists()
    assert "more.txt" in git_read.tree_paths(repo, tip)


def test_merge_lands_the_work_and_prune_removes_the_clone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions merge` reads the repository's own refs; prune then deletes the clone."""
    repo = _repo(tmp_path, monkeypatch)
    _git(repo, "checkout", "-q", "--", "a.py")
    worktree = _run(repo, monkeypatch, "cl-MERG01")
    capsys.readouterr()

    assert cli_main(["sessions", "merge", "cl-MERG01", "--method", "squash"]) == 0
    assert (repo / "new.txt").read_text(encoding="utf-8") == "made by the run\n"
    assert cli_main(["sessions", "prune"]) == 0

    assert "removed cl-MERG01's worktree (merged)" in capsys.readouterr().out
    assert not worktree.exists()


def test_prune_keeps_a_clone_that_holds_work_no_commit_has(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clone is deleted whole, so a file only it holds would go with it."""
    repo = _repo(tmp_path, monkeypatch)
    _git(repo, "checkout", "-q", "--", "a.py")
    worktree = _run(repo, monkeypatch, "cl-KEPT01")
    assert cli_main(["sessions", "merge", "cl-KEPT01", "--method", "squash"]) == 0
    (worktree / "left.txt").write_text("not committed\n", encoding="utf-8")
    capsys.readouterr()

    assert cli_main(["sessions", "prune"]) == 0

    assert (
        "kept cl-KEPT01's worktree (holds work no commit has: left.txt)" in capsys.readouterr().out
    )
    assert (worktree / "left.txt").exists()


def test_remove_deletes_the_clone_with_the_record(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    worktree = _run(repo, monkeypatch, "cl-REMO01")

    assert cli_main(["sessions", "remove", "cl-REMO01", "--yes"]) == 0

    assert not worktree.exists()
    assert git_read.chain_tip(repo, git_read.chain_ref_for("cl-REMO01")) is None


def test_auto_merge_lands_on_the_repositorys_branch_not_the_clones(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone has a `main` of its own: merged there, the run read merged and nothing landed."""
    repo = _repo(tmp_path, monkeypatch, "\n[git]\nauto_merge = true\n")
    _git(repo, "checkout", "-q", "--", "a.py")
    seed = _git(repo, "rev-parse", "HEAD")

    worktree = _run(repo, monkeypatch, "cl-AUTO01")

    assert _git(repo, "rev-parse", "main") != seed
    assert "new.txt" in git_read.tree_paths(repo, "main")
    assert _git(worktree, "rev-parse", "main") == seed
    assert _manifest(repo, "cl-AUTO01").merged is not None


def test_a_fork_gets_a_clone_at_the_turn_it_forks_from(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork's base is a commit only the source's hidden ref holds; the clone has it."""
    repo = _repo(tmp_path, monkeypatch)
    _run(repo, monkeypatch, "cl-SRC001")

    _script(monkeypatch, _edit("fork.txt", "from the fork\n"))
    assert cli_main(["fork", "cl-SRC001", "--session-id", "cl-FORK01", "--steer", "add"]) == 0

    worktree = _manifest(repo, "cl-FORK01").worktree
    assert worktree is not None and fork_worktrees.is_clone(worktree)
    tip = git_read.chain_tip(repo, git_read.chain_ref_for("cl-FORK01"))
    assert tip is not None
    assert {"new.txt", "fork.txt"} <= git_read.tree_paths(repo, tip)


def test_a_model_that_commits_itself_has_its_head_copied_as_the_chain(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `git.control = "model"` the harness commits nothing, so nothing was copied."""
    repo = _repo(tmp_path, monkeypatch, '\n[git]\ncontrol = "model"\n')

    worktree = _run(repo, monkeypatch, "cl-MODL01")

    head = _git(worktree, "rev-parse", "HEAD").strip()
    assert git_read.chain_tip(repo, git_read.chain_ref_for("cl-MODL01")) == head


def test_a_marker_the_repository_does_not_confirm_names_no_repository(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory with a forged marker would have taken another repository's state and config."""
    repo = _repo(tmp_path, monkeypatch)
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    _git(stranger, "init", "-q", "-b", "main")
    (stranger / ".git" / paths.SESSION_CLONE_ORIGIN).write_text(f"{repo}\n", encoding="utf-8")

    assert paths.session_clone_origin(stranger) is None
    assert paths.project_root(stranger) == stranger.resolve()
    assert paths.state_dir(stranger) != paths.state_dir(repo)
