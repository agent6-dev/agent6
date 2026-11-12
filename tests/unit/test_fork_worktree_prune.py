from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from agent6 import paths
from agent6.harness import loop_snapshot
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import main
from agent6.vcs import git_ops, git_read

pytestmark = pytest.mark.usefixtures("linked_worktrees")


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(params=("squash", "tasks"))
def merged_fork(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[pathlib.Path, sessions_layout.SessionLayout]:
    """Fork a committed checkpoint, commit on its chain, and land it through the CLI."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-qb", "agent6/source")
    (tmp_path / "source.txt").write_text("source work\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "source work")
    source_tip = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "update-ref", git_read.chain_ref_for("source"), source_tip)
    _git(tmp_path, "checkout", "-q", "main")
    source = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "source")
    source.ensure()
    source.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "source",
                "mode": "run",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/source",
                "user_task": "source and fork work",
            }
        ),
        encoding="utf-8",
    )
    source.logs_path.write_text(
        '{"type":"session.start","mode":"run"}\n'
        '{"type":"session.stop","reason":"finish_session"}\n',
        encoding="utf-8",
    )
    source.checkpoint_path(1).write_text(
        loop_snapshot.SessionSnapshot(
            system="system",
            messages=[{"role": "user", "content": "source and fork work"}],
            tool_calls=0,
            next_turn=1,
            root_task_id=None,
            original_task="source and fork work",
            verify_command=(),
            head_sha=source_tip,
        ).model_dump_json(),
        encoding="utf-8",
    )
    assert main(["fork", "source", "--at-turn", "1", "--no-run", "--session-id", "fork"]) == 0
    child = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "fork")
    child.logs_path.write_text(source.logs_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = sessions_manifest.read_manifest(child.session_dir)
    worktree = manifest.worktree
    assert worktree is not None
    assert paths.state_dir(worktree) == paths.state_dir(tmp_path)
    (worktree / "fork.txt").write_text("fork work\n", encoding="utf-8")
    tip = git_ops.chain_commit(
        worktree,
        "fork work",
        ref=git_read.chain_ref_for("fork"),
        fallback_parent=source_tip,
        also_branch=manifest.run_branch,
        trailers={git_read.TASK_TRAILER: "child-task"},
    )
    assert tip and _git(worktree, "rev-parse", "HEAD") == source_tip
    method: str = request.param
    assert main(["sessions", "merge", "fork", "--method", method]) == 0
    stamp = sessions_manifest.read_manifest(child.session_dir).merged
    assert stamp is not None and stamp.tip == tip and stamp.method == method
    assert _git(tmp_path, "show", "main:fork.txt") == "fork work"
    # Distinct task ranges must exercise task merging, not its single-squash fallback.
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == (
        "1" if method == "squash" else "2"
    )
    assert not git_read.is_ancestor(tmp_path, source_tip, "main")
    return worktree, child


@pytest.mark.parametrize("delete_squashed", [False, True])
def test_prune_removes_a_fork_covered_by_its_merge_stamp(
    tmp_path: pathlib.Path,
    merged_fork: tuple[pathlib.Path, sessions_layout.SessionLayout],
    capsys: pytest.CaptureFixture[str],
    *,
    delete_squashed: bool,
) -> None:
    """A squash or task merge covers the fork's detached HEAD without making it reachable."""
    worktree, _ = merged_fork
    capsys.readouterr()

    assert main(["sessions", "prune", *(["--delete-squashed"] if delete_squashed else [])]) == 0

    out = capsys.readouterr().out
    assert not worktree.exists(), out
    assert str(worktree) not in _git(tmp_path, "worktree", "list", "--porcelain")
    assert "removed fork's worktree (merged)" in out
    assert "worktrees: removed 1, kept 0" in out
    assert bool(git_read.branch_tip_sha(tmp_path, "agent6/fork")) is not delete_squashed
    assert bool(git_read.chain_tip(tmp_path, git_read.chain_ref_for("fork"))) is not delete_squashed


def test_prune_keeps_and_names_work_no_commit_has_after_a_merge(
    merged_fork: tuple[pathlib.Path, sessions_layout.SessionLayout],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid merge stamp does not authorize deleting uncommitted edits or untracked files."""
    worktree, _ = merged_fork
    (worktree / "README.md").write_text("uncommitted edit\n", encoding="utf-8")
    (worktree / "notes.txt").write_text("untracked notes\n", encoding="utf-8")
    capsys.readouterr()

    for args in ([], ["--delete-squashed"]):
        assert main(["sessions", "prune", *args]) == 0

        out = capsys.readouterr().out
        assert (worktree / "README.md").read_text(encoding="utf-8") == "uncommitted edit\n"
        assert (worktree / "notes.txt").read_text(encoding="utf-8") == "untracked notes\n"
        assert "kept fork's worktree (holds work no commit has:" in out, out
        assert "README.md" in out and "notes.txt" in out
        assert "worktrees: removed 0, kept 1" in out


@pytest.mark.parametrize("commit_to", ["branch", "chain"])
def test_prune_keeps_work_committed_after_the_merge(
    merged_fork: tuple[pathlib.Path, sessions_layout.SessionLayout],
    capsys: pytest.CaptureFixture[str],
    commit_to: str,
) -> None:
    """Neither a separate HEAD commit nor a chain past its stamp is covered by the merge."""
    worktree, child = merged_fork
    (worktree / "later.txt").write_text("post-merge work\n", encoding="utf-8")
    if commit_to == "branch":
        _git(worktree, "switch", "-c", "operator-work")
        _git(worktree, "add", ".")
        _git(worktree, "commit", "-qm", "post-merge work")
    else:
        manifest = sessions_manifest.read_manifest(child.session_dir)
        assert git_ops.chain_commit(
            worktree,
            "post-merge work",
            ref=git_read.chain_ref_for("fork"),
            fallback_parent=manifest.base_sha,
            also_branch=manifest.run_branch,
        )
    capsys.readouterr()

    assert main(["sessions", "prune"]) == 0

    out = capsys.readouterr().out
    assert (worktree / "later.txt").read_text(encoding="utf-8") == "post-merge work\n"
    assert "kept fork's worktree (" in out and "unmerged" in out
    if commit_to == "branch":
        assert "2 unmerged commits on operator-work" in out
    assert "worktrees: removed 0, kept 1" in out
