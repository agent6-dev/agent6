# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.git_ops on a temporary repository."""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
from unittest import mock

import pytest

from agent6 import commit_message, git_ops


def _init_repo(path: pathlib.Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def _stash_change(path: pathlib.Path, name: str, content: str, message: str) -> None:
    """Write content into the tracked file (committed empty first when new) and stash the change."""
    target = path / name
    if not target.exists():
        target.write_text("", encoding="utf-8")
        subprocess.run(["git", "-C", str(path), "add", "--", name], check=True)
        subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", f"track {name}"], check=True)
    target.write_text(content, encoding="utf-8")
    git_ops.stash_tracked_changes(path, message)


def test_modified_and_untracked_paths_split_the_operators_work(tmp_path: pathlib.Path) -> None:
    """Tracked modifications are the run's start question; untracked files are the operator's."""
    _init_repo(tmp_path)
    assert git_ops.modified_paths(tmp_path) == []
    assert git_ops.untracked_paths(tmp_path) == frozenset()
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")  # untracked
    (tmp_path / "odd name:here.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")  # modified
    assert git_ops.modified_paths(tmp_path) == ["README.md"]
    assert git_ops.untracked_paths(tmp_path) == {"new.txt", "odd name:here.txt"}
    st = git_ops.status(tmp_path)
    assert (st.modified_count, st.untracked_count, st.is_clean) == (1, 2, False)
    # The run's untracked_at_start set makes them invisible to status.
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    scoped = git_ops.status(tmp_path, exclude={"new.txt", "odd name:here.txt"})
    assert (scoped.modified_count, scoped.untracked_count, scoped.is_clean) == (0, 0, True)
    assert git_ops.status(tmp_path, exclude={"new.txt"}).untracked_count == 1


def test_worktree_name_status_leaves_out_the_operators_untracked_files(
    tmp_path: pathlib.Path,
) -> None:
    """`worktree_name_status` excludes `untracked_at_start` like every status call in the loop.

    A bystander untracked file would otherwise skew every checkpoint's derived type and scope.
    """
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("mine\n", encoding="utf-8")  # the operator's
    mine = git_ops.untracked_paths(tmp_path)
    (tmp_path / "feature.py").write_text("x = 1\n", encoding="utf-8")  # the run's

    changes = git_ops.worktree_name_status(tmp_path, exclude=mine)
    assert ("A", "feature.py") in changes
    assert not any(path == "notes.txt" for _status, path in changes)


def test_stash_tracked_changes_leaves_untracked_files_in_place(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("mine\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    git_ops.stash_tracked_changes(tmp_path, "agent6 auto-stash")
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hi\n"
    assert git_ops.untracked_paths(tmp_path) == {"notes.txt"}


def test_commit_paths_ignores_unrelated_staged_work(tmp_path: pathlib.Path) -> None:
    # Work the user already staged is not swept into the scaffold commit.
    _init_repo(tmp_path)
    (tmp_path / "wip.txt").write_text("in progress\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "wip.txt"], check=True)
    (tmp_path / "AGENTS.md").write_text("# scaffold\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".agent6/\n", encoding="utf-8")

    git_ops.commit_paths(tmp_path, "chore: scaffold", ("AGENTS.md", ".gitignore"))

    committed = subprocess.run(
        ["git", "-C", str(tmp_path), "show", "--name-only", "--format=", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert set(committed) == {"AGENTS.md", ".gitignore"}
    assert "wip.txt" not in committed
    # wip.txt is still staged, uncommitted.
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert staged == ["wip.txt"]


def test_status_handles_unborn_head(tmp_path: pathlib.Path) -> None:
    """A repo with an unborn HEAD does not crash `status()`, which every entry point loads first."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    st = git_ops.status(tmp_path)
    assert st.branch == "main"
    assert st.head_sha == ""
    assert not st.is_clean
    assert st.untracked_count == 1


def test_git_ops_neutralizes_repo_fsmonitor(tmp_path: pathlib.Path) -> None:
    """A repo-controlled core.fsmonitor never executes on the host when agent6 runs git."""
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.fsmonitor", f"touch {marker}"],
        check=True,
    )
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")
    # Index-refreshing op through agent6's hardened _run.
    git_ops.status(tmp_path)
    git_ops.commit_all(tmp_path, "msg")
    assert not marker.exists(), "repo core.fsmonitor command executed on the host"


def _add_pre_commit_hook(repo: pathlib.Path, marker: pathlib.Path) -> None:
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    hook.chmod(0o755)


def test_git_ops_skips_repo_hooks_by_default(tmp_path: pathlib.Path) -> None:
    """agent6's own commit never fires a repo `.git/hooks/*`, on the untouched defaults too.

    Calling set_repo_hook_policy(False) first would pin only the explicit path.
    """
    from agent6.config import Config

    assert git_ops._hook_policy["honor_repo_hooks"] is False  # the module's own default
    assert Config().git.run_repo_hooks is False  # the config default that sets it

    _init_repo(tmp_path)
    marker = tmp_path / "HOOK_FIRED"
    _add_pre_commit_hook(tmp_path, marker)
    (tmp_path / "n.txt").write_text("x\n", encoding="utf-8")
    git_ops.commit_all(tmp_path, "c")  # no set_repo_hook_policy call: the default governs
    assert not marker.exists()


def test_git_ops_runs_repo_hooks_when_enabled(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "HOOK_FIRED"
    _add_pre_commit_hook(tmp_path, marker)
    (tmp_path / "n.txt").write_text("x\n", encoding="utf-8")
    try:
        git_ops.set_repo_hook_policy(True)  # git.run_repo_hooks = true
        git_ops.commit_all(tmp_path, "c")
        assert marker.exists()
    finally:
        git_ops.set_repo_hook_policy(False)  # reset process-wide policy for other tests


def test_git_ops_neutralizes_repo_diff_external(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_DIFF"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "diff.external", f"touch {marker} ;"],
        check=True,
    )
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    out = git_ops.diff_since(tmp_path, "HEAD")
    assert not marker.exists(), "repo diff.external command executed on the host"
    # git >= 2.53 runs an empty `-c diff.external=` and the patch dies; --no-ext-diff keeps it.
    assert "README.md" in out, "diff.external neutralization silently emptied the diff"


def test_diff_range_reports_a_branch_diff_and_empty_on_bad_ref(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-q", "-b", "agent6/x"], check=True)
    (tmp_path / "feature.txt").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "feat"], check=True)
    out = git_ops.diff_range(tmp_path, base, "agent6/x")
    assert "feature.txt" in out
    assert git_ops.diff_range(tmp_path, base, "no-such-branch") == ""  # unresolvable range -> ""


def test_diff_range_survives_poisoned_diff_external(tmp_path: pathlib.Path) -> None:
    """`sessions compare` diffs via `diff_range`, hardened against a poisoned repo config too."""
    _init_repo(tmp_path)
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    marker = tmp_path / "PWNED_RANGE"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "diff.external", f"touch {marker} ;"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-q", "-b", "agent6/x"], check=True)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-aq", "-m", "change"], check=True)
    out = git_ops.diff_range(tmp_path, base, "agent6/x")
    assert not marker.exists(), "repo diff.external command executed on the host"
    assert "README.md" in out, "diff.external neutralization silently emptied the diff"


def test_commit_diff_survives_poisoned_diff_external(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_SHOW"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "diff.external", f"touch {marker} ;"],
        check=True,
    )
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    sha = git_ops.commit_all(tmp_path, "change readme")
    out = git_ops.commit_diff(tmp_path, sha)
    assert not marker.exists(), "repo diff.external command executed on the host"
    assert "README.md" in out, "git show honored the external diff and lost the patch"


def _poison_textconv(repo: pathlib.Path, marker: pathlib.Path) -> None:
    # A textconv driver via .gitattributes: only `--no-textconv` disables it.
    subprocess.run(
        ["git", "-C", str(repo), "config", "diff.pwn.textconv", f"touch {marker} ; cat"],
        check=True,
    )
    (repo / ".gitattributes").write_text("* diff=pwn\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", ".gitattributes"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "attrs"], check=True)


def test_git_ops_neutralizes_repo_diff_textconv(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_TEXTCONV_DIFF"
    _poison_textconv(tmp_path, marker)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    out = git_ops.diff_since(tmp_path, "HEAD")
    assert not marker.exists(), "repo diff.*.textconv command executed on the host"
    assert "README.md" in out, "textconv neutralization silently emptied the diff"


def test_commit_diff_survives_poisoned_textconv(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_TEXTCONV_SHOW"
    _poison_textconv(tmp_path, marker)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    sha = git_ops.commit_all(tmp_path, "change readme")
    out = git_ops.commit_diff(tmp_path, sha)
    assert not marker.exists(), "git show ran the repo textconv driver on the host"
    assert "README.md" in out, "git show lost the patch under --no-textconv"


def test_git_ops_neutralizes_repo_gpg_signing(tmp_path: pathlib.Path) -> None:
    """A repo-controlled `gpg.program` with commit.gpgsign never executes on agent6's own commit."""
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_GPG"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "gpg.program", f"touch {marker} ; false"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "commit.gpgsign", "true"],
        check=True,
    )
    (tmp_path / "n.txt").write_text("x\n", encoding="utf-8")
    git_ops.commit_all(tmp_path, "c")  # must not raise or fire the signing program
    assert not marker.exists(), "repo gpg.program executed on the host during commit"


def test_is_git_repo_false_for_tmp(tmp_path: pathlib.Path) -> None:
    assert git_ops.is_git_repo(tmp_path) is False


def test_init_repo_creates_repository(tmp_path: pathlib.Path) -> None:
    assert git_ops.is_git_repo(tmp_path) is False
    git_ops.init_repo(tmp_path)
    assert git_ops.is_git_repo(tmp_path) is True


def test_unignored_filters_gitignored_paths(tmp_path: pathlib.Path) -> None:
    git_ops.init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".agent6/\n", encoding="utf-8")
    (tmp_path / ".agent6").mkdir()
    (tmp_path / ".agent6" / "config.toml").write_text("x", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("y", encoding="utf-8")
    keep = git_ops.unignored(tmp_path, (".agent6/config.toml", "AGENTS.md", ".gitignore"))
    assert ".agent6/config.toml" not in keep  # gitignored
    assert "AGENTS.md" in keep
    assert ".gitignore" in keep


def test_status_clean_repo(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    st = git_ops.status(tmp_path)
    assert st.branch == "main"
    assert st.is_clean is True
    assert st.modified_count == 0


def test_status_dirty_repo(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    st = git_ops.status(tmp_path)
    assert st.is_clean is False
    assert st.untracked_count == 1


def test_commit_all_returns_sha_and_log(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    sha = git_ops.commit_all(tmp_path, "add f", trailers={"agent6-step": "x"})
    assert len(sha) == 40
    log = git_ops.recent_log(tmp_path, n=5)
    assert "add f" in log


def test_create_branch(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    git_ops.create_branch(tmp_path, "agent6/test")
    assert git_ops.status(tmp_path).branch == "agent6/test"


def test_create_branch_at_is_additive_no_checkout(tmp_path: pathlib.Path) -> None:
    """create_branch_at points a new branch at a sha WITHOUT moving HEAD."""
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    # Advance HEAD so base != current; the fork branch must land on base.
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    git_ops.commit_all(tmp_path, "second")
    git_ops.create_branch_at(tmp_path, "agent6/fork", base)
    assert git_ops.status(tmp_path).branch == "main", "must not check out the new branch"
    forked = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "agent6/fork"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert forked == base


def test_create_branch_at_idempotent_and_refuses_move(tmp_path: pathlib.Path) -> None:
    """A no-op when the branch already points at the sha; a GitError if it points elsewhere.

    Moving a branch would be a rewrite, which git_ops refuses.
    """
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    git_ops.create_branch_at(tmp_path, "agent6/fork", base)
    git_ops.create_branch_at(tmp_path, "agent6/fork", base)  # idempotent: no raise
    (tmp_path / "c.txt").write_text("c\n", encoding="utf-8")
    other = git_ops.commit_all(tmp_path, "third")
    with pytest.raises(git_ops.GitError):
        git_ops.create_branch_at(tmp_path, "agent6/fork", other)


def test_git_ops_never_spells_a_destructive_verb() -> None:
    """The hard rule, at argv level: no function in git_ops passes a destructive verb to git.

    The one sanctioned exception is force_delete_squash_merged_branch's `branch -D`, operator-only
    and content-safe.
    """
    from agent6 import git_ops as gm

    src = pathlib.Path(gm.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"push", "--force", "-f", "--hard", "rebase", "--amend", "filter-branch", "-D"}
    offenders: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call) or gm_run_name(call) is None:
                continue
            words = {
                a.value
                for a in call.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            }
            hit = words & forbidden
            # `stash push` is git's stash verb, not a remote push.
            if "stash" in words:
                hit -= {"push"}
            if fn.name == "force_delete_squash_merged_branch":
                hit -= {"-D"}
            if hit:
                offenders.append(f"{fn.name}: {sorted(hit)}")
    assert not offenders, f"destructive git verbs in git_ops: {offenders}"


def gm_run_name(call: ast.Call) -> str | None:
    """The callee name if this is a git-invoking call (`_run(...)`)."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == "_run":
        return func.id
    return None


def test_status_on_non_repo(tmp_path: pathlib.Path) -> None:
    with pytest.raises(git_ops.GitError):
        git_ops.status(tmp_path)


def test_verify_git_identity_uses_repo_config(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)  # configures user.name=t, user.email=t@t
    name, email = git_ops.verify_git_identity(tmp_path, git_ops.CommitIdentity())
    assert name == "t"
    assert email == "t@t"


def test_verify_git_identity_override_wins(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    name, email = git_ops.verify_git_identity(
        tmp_path, git_ops.CommitIdentity(name="bot", email="bot@example.com")
    )
    assert name == "bot"
    assert email == "bot@example.com"


def test_verify_git_identity_missing_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The global git config points at an empty file, so no identity leaks in.
    empty_cfg = tmp_path / "empty.gitconfig"
    empty_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_cfg))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    with pytest.raises(git_ops.GitError, match="Git identity not configured"):
        git_ops.verify_git_identity(repo, git_ops.CommitIdentity())


def test_commit_all_with_identity_overrides_author(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    sha = git_ops.commit_all(
        tmp_path,
        "add f",
        identity=git_ops.CommitIdentity(
            name="agent6",
            email="agent6@example.com",
            trailer="Co-authored-by: Alice <alice@example.com>",
        ),
    )
    show = subprocess.run(
        ["git", "-C", str(tmp_path), "show", "--no-patch", "--format=%an|%ae|%B", sha],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "agent6|agent6@example.com|" in show
    assert "Co-authored-by: Alice <alice@example.com>" in show


def test_commit_error_surfaces_stdout_when_stderr_empty(tmp_path: pathlib.Path) -> None:
    """A failed git's error text includes stdout when stderr is empty.

    `git commit` writes "nothing to commit, working tree clean" to stdout.
    """
    _init_repo(tmp_path)
    # `git commit` exits 1 with "nothing to commit, working tree clean" on stdout.
    with pytest.raises(git_ops.GitError) as excinfo:
        git_ops.commit_all(tmp_path, "no-op commit on clean repo")
    msg = str(excinfo.value)
    # The detail (from stdout) must be present so callers can pattern-match.
    assert "nothing to commit" in msg.lower()
    # And the prefix must still identify which git invocation failed.
    assert "git commit" in msg


def test_restore_stash_clean_apply_restores_and_drops(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    _stash_change(tmp_path, "wip.txt", "work in progress\n", "agent6 auto-stash")
    assert (tmp_path / "wip.txt").read_text(encoding="utf-8") == ""  # stashed away, tree clean
    entry = git_ops.find_stash(tmp_path, "agent6 auto-stash")
    assert entry is not None
    assert git_ops.restore_stash(tmp_path, entry) is True
    assert (tmp_path / "wip.txt").read_text(encoding="utf-8") == "work in progress\n"
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert listing.strip() == ""  # dropped after a clean apply


def test_restore_stash_conflict_keeps_stash(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("stashed change\n", encoding="utf-8")
    git_ops.stash_tracked_changes(tmp_path, "agent6 auto-stash")
    # A conflicting commit on the same line means the stash cannot apply cleanly.
    (tmp_path / "README.md").write_text("committed change\n", encoding="utf-8")
    git_ops.commit_all(tmp_path, "conflicting commit")
    entry = git_ops.find_stash(tmp_path, "agent6 auto-stash")
    assert entry is not None
    assert git_ops.restore_stash(tmp_path, entry) is False
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "agent6 auto-stash" in listing  # preserved, never dropped on conflict


def test_find_stash_targets_the_run_stash_not_the_latest(tmp_path: pathlib.Path) -> None:
    """The run's stash is found by its run-id message; a stash pushed mid-run is untouched."""
    _init_repo(tmp_path)
    _stash_change(
        tmp_path, "pre.txt", "pre-run work\n", git_ops.auto_stash_message("sunny-otter-AAA111")
    )
    _stash_change(
        tmp_path, "mid.txt", "mid-run stash by someone else\n", "user work stashed mid-run"
    )
    entry = git_ops.find_stash(tmp_path, git_ops.auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None
    assert entry.ref == "stash@{1}"
    assert git_ops.restore_stash(tmp_path, entry) is True
    assert (tmp_path / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    # the mid-run stash stays a stash
    assert (tmp_path / "mid.txt").read_text(encoding="utf-8") == ""
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "user work stashed mid-run" in listing
    assert "agent6 auto-stash" not in listing


def test_restore_stash_raced_drop_puts_the_bystander_back(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stash that shifted the stack between the list and the drop is put back.

    `git stash drop` takes only a position, so the drop can take a bystander's entry.
    """
    _init_repo(tmp_path)
    _stash_change(
        tmp_path, "pre.txt", "pre-run work\n", git_ops.auto_stash_message("sunny-otter-AAA111")
    )
    entry = git_ops.find_stash(tmp_path, git_ops.auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None

    real_run = git_ops._run  # pyright: ignore[reportPrivateUsage]
    raced = False

    def racing_run(
        path: pathlib.Path, *args: str, check: bool = True, env_extra: dict[str, str] | None = None
    ) -> git_ops.CommandResult:
        nonlocal raced
        res = real_run(path, *args, check=check, env_extra=env_extra)
        if not raced and args[:2] == ("stash", "list"):
            raced = True  # ours slides to stash@{1}; the recorded ref now names theirs
            (tmp_path / "README.md").write_text("bystander work\n", encoding="utf-8")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(tmp_path),
                    "stash",
                    "push",
                    "-q",
                    "-m",
                    "bystander",
                    "--",
                    "README.md",
                ],
                check=True,
            )
        return res

    monkeypatch.setattr(git_ops, "_run", racing_run)
    assert git_ops.restore_stash(tmp_path, entry) is True
    assert (tmp_path / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "bystander" in listing  # never silently destroyed
    # Ours survives too: dropping it again would race the same way.
    assert git_ops.auto_stash_message("sunny-otter-AAA111") in listing


def test_find_stash_does_not_prefix_match_another_runs_stash(tmp_path: pathlib.Path) -> None:
    """The stash lookup matches the pushed message exactly: `-l1` is a prefix of `-l10`."""
    _init_repo(tmp_path)
    _stash_change(tmp_path, "one.txt", "lane l1 work\n", git_ops.auto_stash_message("fanout-l1"))
    # newer: listed first
    _stash_change(tmp_path, "ten.txt", "lane l10 work\n", git_ops.auto_stash_message("fanout-l10"))
    entry = git_ops.find_stash(tmp_path, git_ops.auto_stash_message("fanout-l1"))
    assert entry is not None
    assert entry.ref == "stash@{1}"  # the l1 stash, not the newer l10 one


def test_raced_drop_failed_putback_raises_with_recovery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bystander's stash lost in the raced drop is reported, never silently gone.

    When storing it back fails, the restore raises, naming the orphaned commit and the `git stash
    store` command that puts it back.
    """
    _init_repo(tmp_path)
    _stash_change(
        tmp_path, "pre.txt", "pre-run work\n", git_ops.auto_stash_message("sunny-otter-AAA111")
    )
    entry = git_ops.find_stash(tmp_path, git_ops.auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None

    real_run = git_ops._run  # pyright: ignore[reportPrivateUsage]
    raced = False

    def racing_run(
        path: pathlib.Path, *args: str, check: bool = True, env_extra: dict[str, str] | None = None
    ) -> git_ops.CommandResult:
        nonlocal raced
        if args[:2] == ("stash", "store"):
            return git_ops.CommandResult(
                argv=("git", *args), returncode=1, stdout="", stderr="ref lock held", duration_s=0.0
            )
        res = real_run(path, *args, check=check, env_extra=env_extra)
        if not raced and args[:2] == ("stash", "list"):
            raced = True  # ours slides to stash@{1}; the recorded ref now names theirs
            (tmp_path / "README.md").write_text("bystander work\n", encoding="utf-8")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(tmp_path),
                    "stash",
                    "push",
                    "-q",
                    "-m",
                    "bystander",
                    "--",
                    "README.md",
                ],
                check=True,
            )
        return res

    monkeypatch.setattr(git_ops, "_run", racing_run)
    with pytest.raises(git_ops.GitError) as excinfo:
        git_ops.restore_stash(tmp_path, entry)
    msg = str(excinfo.value)
    assert "git stash store" in msg  # the recovery command, ready to paste
    assert "bystander" in msg  # names whose work went missing


def test_git_runs_under_a_pinned_locale(tmp_path: pathlib.Path) -> None:
    """The bystander rescue works under a non-English git locale.

    It reads git's own "Dropped stash@{0} (sha)" line, which git translates.
    """
    _init_repo(tmp_path)
    seen: dict[str, str] = {}
    real = subprocess.Popen

    def capture(argv: object, **kwargs: object) -> object:
        env = kwargs.get("env")
        if isinstance(env, dict):
            seen.update({k: v for k, v in env.items() if k in ("LC_ALL", "LANG")})  # pyright: ignore[reportUnknownArgumentType]
        return real(argv, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    with mock.patch.object(git_ops.subprocess, "Popen", capture):
        git_ops._run(tmp_path, "stash", "list", check=False)  # pyright: ignore[reportPrivateUsage]
    assert seen.get("LC_ALL") == "C"


class _HungGit:
    """Popen stand-in for a git stuck past the timeout.

    The first communicate() times out (creating *lock* if given); terminate() exits it, unless
    *ignores_term*, where only kill() reaps it.
    """

    def __init__(self, *, lock: pathlib.Path | None = None, ignores_term: bool = False) -> None:
        self.lock = lock
        self.ignores_term = ignores_term
        self.calls: list[str] = []

    def communicate(
        self, input: bytes | None = None, timeout: float | None = None
    ) -> tuple[bytes, bytes]:
        if "kill" in self.calls:
            return b"", b""
        if "terminate" in self.calls:
            if self.ignores_term:
                raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)
            return b"", b""
        if self.lock is not None:
            self.lock.write_text("", encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd="git", timeout=timeout or 0)

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")


def _hang_the_op_only(fake: _HungGit) -> object:
    """A Popen replacement that hangs only the git op, not the driver-neutralization enumeration."""
    real = git_ops.subprocess.Popen

    def fake_popen(*a: object, **k: object) -> object:
        argv = a[0] if a else k.get("args", [])
        if isinstance(argv, list | tuple) and "--get-regexp" in [str(x) for x in argv]:
            return real(*a, **k)  # type: ignore[arg-type]
        return fake

    return fake_popen


def test_timeout_terminates_first_and_leaves_a_survivor_lock(tmp_path: pathlib.Path) -> None:
    """A timed-out git gets SIGTERM, whose handler removes its lockfiles; SIGKILL follows.

    After a graceful TERM exit a lock still on disk is a concurrent git's, so nothing is deleted.
    """
    _init_repo(tmp_path)
    lock = tmp_path / ".git" / "index.lock"
    fake = _HungGit(lock=lock)

    fake_popen = _hang_the_op_only(fake)

    with (
        mock.patch.object(git_ops.subprocess, "Popen", fake_popen),
        pytest.raises(git_ops.GitError, match="timed out"),
    ):
        git_ops._run(tmp_path, "commit", "-m", "x", check=False)  # pyright: ignore[reportPrivateUsage]
    assert fake.calls == ["terminate"]  # graceful exit: never escalated to KILL
    assert lock.exists()  # a survivor lock after TERM is not ours to delete


def test_timeout_keeps_a_preexisting_index_lock(tmp_path: pathlib.Path) -> None:
    """A lock that predates the timed-out git is a concurrent git's and stays, even on SIGKILL."""
    _init_repo(tmp_path)
    lock = tmp_path / ".git" / "index.lock"
    lock.write_text("held by a concurrent git\n", encoding="utf-8")
    fake = _HungGit(ignores_term=True)

    fake_popen = _hang_the_op_only(fake)

    with (
        mock.patch.object(git_ops.subprocess, "Popen", fake_popen),
        pytest.raises(git_ops.GitError, match="timed out"),
    ):
        git_ops._run(tmp_path, "status", check=False)  # pyright: ignore[reportPrivateUsage]
    assert lock.exists()  # never ours to delete


def test_timeout_clears_the_lock_its_own_child_created(tmp_path: pathlib.Path) -> None:
    """A child that ignores TERM is killed, and a lock that appeared under it is cleared."""
    _init_repo(tmp_path)
    lock = tmp_path / ".git" / "index.lock"
    fake = _HungGit(lock=lock, ignores_term=True)

    fake_popen = _hang_the_op_only(fake)

    with (
        mock.patch.object(git_ops.subprocess, "Popen", fake_popen),
        pytest.raises(git_ops.GitError, match="timed out"),
    ):
        git_ops._run(tmp_path, "commit", "-m", "x", check=False)  # pyright: ignore[reportPrivateUsage]
    assert fake.calls == ["terminate", "kill"]  # TERM tried before KILL
    assert not lock.exists()  # the dead child's own lock is cleared


def test_find_stash_missing_returns_none(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    assert git_ops.find_stash(tmp_path, git_ops.auto_stash_message("nope")) is None


def test_restore_stash_survives_index_shift_after_lookup(tmp_path: pathlib.Path) -> None:
    """The stash entry is applied and dropped by its sha, resolved fresh at drop time.

    A stash pushed after the lookup shifts every stash@{N}.
    """
    _init_repo(tmp_path)
    _stash_change(
        tmp_path, "pre.txt", "pre-run work\n", git_ops.auto_stash_message("sunny-otter-AAA111")
    )
    entry = git_ops.find_stash(tmp_path, git_ops.auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None and entry.ref == "stash@{0}"
    # A stash pushed after the lookup makes the recorded position point at someone else's work.
    _stash_change(tmp_path, "mid.txt", "mid work\n", "pushed after lookup")
    assert git_ops.restore_stash(tmp_path, entry) is True
    assert (tmp_path / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    assert (tmp_path / "mid.txt").read_text(encoding="utf-8") == ""  # the other stash stays a stash
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "pushed after lookup" in listing
    assert "agent6 auto-stash" not in listing  # ours was dropped, by identity


def _commit_file(repo: pathlib.Path, name: str, content: str, msg: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", msg], check=True)
    return git_ops.status(repo).head_sha


def test_plumb_merge_lands_without_touching_the_checkout_medium(tmp_path: pathlib.Path) -> None:
    """A no-ff merge is pure ref plumbing: a two-parent commit and the index brought forward.

    A worktree already holding the run's files is no obstacle, and `git status` shows no phantom
    dirt.
    """
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    run_tip = _lane_commit(tmp_path, base, "feat.txt", "x\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/agent6/r1", run_tip], check=True
    )
    (tmp_path / "feat.txt").write_text("x\n", encoding="utf-8")  # worktree carries the work

    res = git_ops.plumb_merge(
        tmp_path,
        "main",
        "refs/agent6/r1",
        strategy="merge",
        message="Merge refs/agent6/r1",
        identity=git_ops.CommitIdentity(trailer="Assisted-by: agent6:m1"),
    )
    assert not res.conflicted
    assert _rev(tmp_path, "main") == res.merged_sha
    parents = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%P", res.merged_sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert parents == [base, run_tip]
    head_msg = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%B", res.merged_sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert head_msg.count("Assisted-by: agent6:m1") == 1
    assert git_ops.status(tmp_path).is_clean  # main is checked out: index brought forward


def test_a_merge_stamp_stops_holding_once_the_run_commits_past_it(tmp_path: pathlib.Path) -> None:
    """The run's record is its chain, so a branchless run's later commits are not called merged."""
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    tip = _lane_commit(tmp_path, base, "a.txt", "one\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", git_ops.chain_ref_for("branchless1"), tip],
        check=True,
    )

    assert git_ops.merge_stamp_holds(tmp_path, "branchless1", "", tip)

    later = _lane_commit(tmp_path, tip, "b.txt", "two\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", git_ops.chain_ref_for("branchless1"), later],
        check=True,
    )

    assert not git_ops.merge_stamp_holds(tmp_path, "branchless1", "", tip)
    # No chain and no branch (auto_prune took both): the claim stands.
    assert git_ops.merge_stamp_holds(tmp_path, "gone-run111", "", tip)


def test_a_merge_stamp_on_the_chain_holds_through_an_operator_commit_on_the_branch(
    tmp_path: pathlib.Path,
) -> None:
    """The stamp names the chain tip a merge took, so the listing accepts either tip as merged."""
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    chain = _lane_commit(tmp_path, base, "a.txt", "one\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", git_ops.chain_ref_for("stamped1"), chain],
        check=True,
    )
    own = _lane_commit(tmp_path, chain, "b.txt", "two\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/heads/agent6/stamped1", own], check=True
    )

    assert git_ops.merge_stamp_holds(tmp_path, "stamped1", "agent6/stamped1", chain)
    assert git_ops.merge_stamp_holds(tmp_path, "stamped1", "agent6/stamped1", own)
    assert not git_ops.merge_stamp_holds(tmp_path, "stamped1", "agent6/stamped1", base)


def test_plumb_merge_names_the_files_the_checkout_kept_its_own_version_of(
    tmp_path: pathlib.Path,
) -> None:
    """The merge brings a checked-out file forward only where it still matches the branch.

    An operator's edit survives; the run's own checkout, already at the merged content, is named by
    nothing.
    """
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    run_tip = _lane_commit(tmp_path, base, "feat.txt", "the run's line\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/agent6/r1", run_tip], check=True
    )
    (tmp_path / "feat.txt").write_text("a third version, the operator's\n", encoding="utf-8")

    res = git_ops.plumb_merge(tmp_path, "main", "refs/agent6/r1", strategy="squash", message="m")

    assert res.left_behind == ("feat.txt",)
    assert (tmp_path / "feat.txt").read_text(
        encoding="utf-8"
    ) == "a third version, the operator's\n"

    # The same merge onto a checkout that holds the run's own work: nothing kept.
    second_tip = _lane_commit(tmp_path, res.merged_sha, "next.txt", "y\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/agent6/r2", second_tip], check=True
    )
    (tmp_path / "next.txt").write_text("y\n", encoding="utf-8")
    assert (
        git_ops.plumb_merge(
            tmp_path, "main", "refs/agent6/r2", strategy="squash", message="m"
        ).left_behind
        == ()
    )


def test_add_worktree_is_detached_shares_refs_and_is_removed_alone(tmp_path: pathlib.Path) -> None:
    """`add_worktree`, `git_common_dir` and `remove_worktree` manage a detached linked worktree.

    The worktree's refs are the repository's own; `remove_worktree` deletes it with its record only,
    so the record of another worktree whose directory is missing survives, and refuses a directory
    that is not a linked worktree.
    """
    import shutil

    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _rev(repo, "HEAD")
    (repo / "README.md").write_text("later\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "second"], check=True)
    wt = tmp_path / "wt"

    git_ops.add_worktree(repo, wt, base)
    assert (wt / ".git").is_file()
    assert _rev(wt, "HEAD") == base
    assert (wt / "README.md").read_text(encoding="utf-8") == "hi\n"
    assert git_ops.git_common_dir(wt) == (repo / ".git").resolve()
    (wt / "new.txt").write_text("n\n", encoding="utf-8")
    sha = git_ops.chain_commit(wt, "step", ref="refs/agent6/w/head", fallback_parent=base)
    assert sha is not None and _rev(repo, "refs/agent6/w/head") == sha
    assert (
        _rev(repo, "HEAD") != base and git_ops.status(repo).is_clean
    )  # the main checkout: untouched

    other = tmp_path / "other"
    git_ops.add_worktree(repo, other, base)
    shutil.rmtree(other)  # missing, its record kept: not agent6's to prune
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / ".git").write_text("gitdir: /nowhere/.git/worktrees/x\n", encoding="utf-8")

    assert git_ops.remove_worktree(repo, plain) is False
    assert (plain / ".git").is_file()
    assert git_ops.remove_worktree(repo, wt) is True
    assert not wt.exists()
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert str(wt) not in listed
    assert str(other) in listed


def test_plumb_merge_from_a_linked_worktree_brings_the_main_checkout_forward(
    tmp_path: pathlib.Path,
) -> None:
    """A merge run from a linked worktree brings the main checkout forward, not the worktree.

    The main checkout has the target checked out; skipping it leaves a phantom staged reversal
    there.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _rev(repo, "HEAD")
    run_tip = _lane_commit(repo, base, "feat.txt", "x\n")
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/agent6/r1", run_tip], check=True)
    wt = tmp_path / "wt"
    git_ops.add_worktree(repo, wt, base)

    res = git_ops.plumb_merge(wt, "main", "refs/agent6/r1", strategy="merge", message="Merge")
    assert not res.conflicted
    assert _rev(repo, "main") == res.merged_sha
    assert (repo / "feat.txt").read_text(encoding="utf-8") == "x\n"
    assert git_ops.status(repo).is_clean
    assert not (wt / "feat.txt").exists()  # the worktree is not the target's checkout


def test_plumb_merge_conflict_moves_nothing(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "README.md", "run change\n")
    _commit_file(tmp_path, "README.md", "main change\n", "main edit")  # same line
    main_tip = git_ops.status(tmp_path).head_sha
    res = git_ops.plumb_merge(tmp_path, "main", theirs, strategy="merge", message=None)
    assert res.conflicted
    # The conflicted paths only; merge-tree's informational lines follow a blank line and stay out.
    assert res.conflicts == ("README.md",)
    assert _rev(tmp_path, "main") == main_tip  # nothing moved
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "main change\n"


def test_plumb_merge_ff_moves_the_ref_and_refuses_divergence(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "feat.txt", "x\n")
    (tmp_path / "feat.txt").write_text("x\n", encoding="utf-8")
    res = git_ops.plumb_merge(tmp_path, "main", theirs, strategy="ff")
    assert res.merged_sha == theirs == _rev(tmp_path, "main")
    assert git_ops.status(tmp_path).is_clean

    diverged = _lane_commit(tmp_path, base, "other.txt", "y\n")
    with pytest.raises(git_ops.GitError):
        git_ops.plumb_merge(tmp_path, "main", diverged, strategy="ff")
    assert _rev(tmp_path, "main") == theirs  # a refused ff moves nothing


def test_plumb_merge_squash_is_one_commit_and_noop_when_contained(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    step1 = _lane_commit(tmp_path, base, "a.txt", "a\n")
    step2 = _lane_commit(tmp_path, step1, "b.txt", "b\n")
    for name, content in (("a.txt", "a\n"), ("b.txt", "b\n")):
        (tmp_path / name).write_text(content, encoding="utf-8")
    res = git_ops.plumb_merge(
        tmp_path,
        "main",
        step2,
        strategy="squash",
        message="the task",
        identity=git_ops.CommitIdentity(trailer="Assisted-by: agent6:m1"),
    )
    assert not res.conflicted
    parents = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%P", res.merged_sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert parents == [base]  # ONE commit, single parent
    assert git_ops.status(tmp_path).is_clean
    # Landing the same rev again is a clean no-op returning the unchanged tip.
    again = git_ops.plumb_merge(tmp_path, "main", step2, strategy="squash", message="again")
    assert again.merged_sha == res.merged_sha and not again.conflicted


def test_plumb_merge_preserves_the_operators_own_staging(tmp_path: pathlib.Path) -> None:
    """Index entries the operator staged survive the bring-forward; only old-tip entries move."""
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "feat.txt", "x\n")
    (tmp_path / "feat.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("operator wip\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    res = git_ops.plumb_merge(tmp_path, "main", theirs, strategy="squash", message="land")
    assert not res.conflicted
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert staged == ["README.md"]  # their staging intent, nothing else


def test_clone_repo_local_clone(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    dest = tmp_path / "clone"
    git_ops.clone_repo(origin, dest)
    assert (dest / "README.md").read_text(encoding="utf-8") == "hi\n"
    assert git_ops.status(dest).head_sha == git_ops.status(origin).head_sha


def test_clone_repo_raises_on_existing_dest(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    dest = tmp_path / "clone"
    dest.mkdir()
    (dest / "occupied.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(git_ops.GitError):
        git_ops.clone_repo(origin, dest)


def test_fetch_branch_lands_branch_from_path(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    clone = tmp_path / "clone"
    git_ops.clone_repo(origin, clone)
    git_ops.create_branch(clone, "agent6/r1")
    _commit_file(clone, "feat.txt", "x\n", "add feat")
    git_ops.fetch_branch(origin, clone, "agent6/r1:agent6/r1")
    sha = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "refs/heads/agent6/r1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert sha == git_ops.status(clone).head_sha


def test_fetch_branch_refuses_non_ff_update(tmp_path: pathlib.Path) -> None:
    # A plain (unforced) refspec must not clobber an existing diverged branch.
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    git_ops.create_branch(origin, "agent6/r1")
    _commit_file(origin, "ours.txt", "o\n", "origin r1")
    origin_sha = git_ops.status(origin).head_sha
    git_ops.create_branch(origin, "main")
    clone = tmp_path / "clone"
    git_ops.clone_repo(origin, clone)
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "-q", "-b", "agent6/r1", "main"], check=True
    )
    _commit_file(clone, "theirs.txt", "t\n", "clone r1")  # diverged from origin's r1
    with pytest.raises(git_ops.GitError):
        git_ops.fetch_branch(origin, clone, "agent6/r1:agent6/r1")
    kept = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "refs/heads/agent6/r1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert kept == origin_sha  # origin's branch untouched


def test_list_run_commits_oldest_first(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    git_ops.create_branch(tmp_path, "agent6/r1")
    _commit_file(tmp_path, "a.txt", "a\n", "agent6 iter 1: add a")
    _commit_file(tmp_path, "b.txt", "b\n", "agent6 iter 2: add b")
    rows = git_ops.list_run_commits(tmp_path, base, "agent6/r1")
    assert [r.subject for r in rows] == ["agent6 iter 1: add a", "agent6 iter 2: add b"]


def test_condense_strips_prefix_and_bullets(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    git_ops.create_branch(tmp_path, "agent6/r1")
    _commit_file(tmp_path, "a.txt", "a\n", "agent6 iter 1: add a")
    _commit_file(tmp_path, "b.txt", "b\n", "agent6 iter 2: add b")
    rows = git_ops.list_run_commits(tmp_path, base, "agent6/r1")
    message = commit_message.condense_commit_message(rows, subject="implement parse_url")
    assert message.splitlines()[0] == "implement parse_url"  # headline = the task
    assert "- add a" in message and "- add b" in message  # prefix stripped, bulleted


def test_condense_subject_is_first_clause_and_the_task_stays_out(tmp_path: pathlib.Path) -> None:
    """The subject is the task's first clause, capped at 72 characters; the rest stays out."""
    task = (
        "Add a --limit flag to runs list. Then update the parser help and add a "
        "focused unit test covering the newest-N slice and the argcomplete choices"
    )
    message = commit_message.condense_commit_message((), subject=task)
    assert message == "Add a --limit flag to runs list"


def test_condense_subject_truncates_a_clauseless_run_on_with_ellipsis(
    tmp_path: pathlib.Path,
) -> None:
    task = "make the thing " * 20  # 300 chars, no sentence break
    message = commit_message.condense_commit_message((), subject=task)
    subject = message.splitlines()[0]
    assert len(subject) <= 72 and subject.endswith("…")


def test_condense_subject_drops_only_a_markdown_heading_mark() -> None:
    """A TASK.md task's `# Title` marks are dropped from the squash headline; a bare `#` stays."""
    assert (
        commit_message.condense_commit_message((), subject="# Fix the parser\n\nMore detail")
        == "Fix the parser"
    )
    assert commit_message.condense_commit_message((), subject="#include <stdio.h> fails") == (
        "#include <stdio.h> fails"
    )


def test_commit_all_appends_the_identity_trailer_once(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    line = "Assisted-by: agent6:m1 (worker)"
    git_ops.commit_all(tmp_path, "add f", identity=git_ops.CommitIdentity(trailer=line))
    head_msg = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert head_msg.count(line) == 1
    assert head_msg.splitlines()[0] == "add f"


def test_render_commit_trailer_joins_the_code_writers() -> None:
    assert commit_message.render_commit_trailer("", models=("m",)) is None
    got = commit_message.render_commit_trailer("Assisted-by: agent6:{model}", models=("m1",))
    assert got == "Assisted-by: agent6:m1"
    # Contributing models join first-seen, deduplicated, blanks dropped; the worker stays first.
    got = commit_message.render_commit_trailer(
        "Assisted-by: agent6:{model}", models=("m1", "", "m2", "m1")
    )
    assert got == "Assisted-by: agent6:m1, m2"


def test_diff_of_non_utf8_file_does_not_crash(tmp_path: pathlib.Path) -> None:
    # A latin-1 text file put non-UTF-8 bytes in the diff, and the strict decode raised mid-run.
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    (tmp_path / "latin1.txt").write_bytes(b"caf\xe9 au lait\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add latin1"], check=True)
    sha = git_ops.status(tmp_path).head_sha

    since = git_ops.diff_since(tmp_path, base)
    assert "latin1.txt" in since  # the diff was produced, not dropped to a crash
    shown = git_ops.commit_diff(tmp_path, sha)
    assert "latin1.txt" in shown


def test_list_run_commits_preserves_body_with_separator_bytes(tmp_path: pathlib.Path) -> None:
    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    git_ops.create_branch(tmp_path, "agent6/r1")
    body = "line A \x1f mid \x1e end\n\nCo-authored-by: Op <op@x>"
    _commit_file(tmp_path, "a.txt", "a\n", f"agent6 iter 1: add a\n\n{body}")
    rows = git_ops.list_run_commits(tmp_path, base, "agent6/r1")
    assert len(rows) == 1  # \x1e in the body did not split it into two records
    assert "Co-authored-by: Op <op@x>" in rows[0].message  # \x1f did not truncate the body


def test_conventional_subject_derives_type_and_scope() -> None:
    # All test files -> test; the scope is the common first dir under src/, else the top-level dir.
    assert commit_message.conventional_commit_subject(
        [("M", "tests/unit/test_a.py"), ("M", "tests/unit/test_b.py")],
        summary="cover the resolver",
    ).startswith("test(unit): cover the resolver")
    # All docs -> docs.
    assert commit_message.conventional_commit_subject(
        [("M", "docs/config.md"), ("M", "README.md")], summary="explain the lock"
    ).startswith("docs: explain the lock")
    # An added source file -> feat, scoped by the package dir.
    got = commit_message.conventional_commit_subject(
        [("A", "src/agent6/config/write.py"), ("M", "src/agent6/config/model.py")],
        summary="one write path",
    )
    assert got == "feat(config): one write path"
    # Modified-only source -> fix.
    got = commit_message.conventional_commit_subject(
        [("M", "src/agent6/git_ops.py")], summary="Trailer emitted once."
    )
    assert got == "fix(git_ops): trailer emitted once"
    # No changes at all still yields a valid subject.
    assert commit_message.conventional_commit_subject([], summary="tidy") == "chore: tidy"


def _rev(path: pathlib.Path, ref: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", ref], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_chain_commit_touches_no_head_index_or_checkout(tmp_path: pathlib.Path) -> None:
    """The detached chain records commits without moving HEAD, the shared index or the checkout.

    The run's commits land on refs/agent6/<id> while the operator's staged change survives
    byte-for-byte.
    """
    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("edited\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)

    sha = git_ops.chain_commit(
        tmp_path,
        "agent6 iter 1",
        ref="refs/agent6/t1",
        fallback_parent=head0,
        also_branch="agent6/t1",
    )
    assert sha is not None
    assert _rev(tmp_path, "refs/agent6/t1") == sha == _rev(tmp_path, "agent6/t1")
    assert _rev(tmp_path, f"{sha}^") == head0
    assert _rev(tmp_path, "HEAD") == head0  # HEAD never moves
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert staged == ["README.md"]  # the operator's staged set is untouched
    tree = subprocess.run(
        ["git", "-C", str(tmp_path), "ls-tree", "--name-only", sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "b.txt" in tree and "README.md" in tree


def test_chain_commit_brings_a_checked_out_run_branch_index_forward(tmp_path: pathlib.Path) -> None:
    """Moving a checked-out run branch advances its clean index with the ref."""
    _init_repo(tmp_path)
    base = _rev(tmp_path, "HEAD")
    git_ops.create_branch(tmp_path, "agent6/r1")
    (tmp_path / "README.md").write_text("run\n", encoding="utf-8")

    sha = git_ops.chain_commit(
        tmp_path,
        "step",
        ref="refs/agent6/r1/head",
        fallback_parent=base,
        also_branch="agent6/r1",
    )

    assert sha == _rev(tmp_path, "HEAD")
    assert git_ops.status(tmp_path).is_clean


def test_chain_merge_lands_a_lane_on_a_checked_out_run_branch(tmp_path: pathlib.Path) -> None:
    """Merging a lane onto the checked-out run branch leaves a clean status.

    The worktree sync runs before the branch's index and worktree move, or it refuses to overwrite
    the moved file.
    """
    _init_repo(tmp_path)
    base = _rev(tmp_path, "HEAD")
    git_ops.create_branch(tmp_path, "agent6/r1")
    (tmp_path / "README.md").write_text("run\n", encoding="utf-8")
    tip = git_ops.chain_commit(
        tmp_path, "step", ref="refs/agent6/r1/head", fallback_parent=base, also_branch="agent6/r1"
    )
    assert tip is not None
    lane_dir = tmp_path.parent / "lane"
    subprocess.run(
        ["git", "-C", str(tmp_path), "worktree", "add", "-q", str(lane_dir), "-b", "lane", tip],
        check=True,
    )
    (lane_dir / "lane.txt").write_text("lane\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(lane_dir), "add", "lane.txt"], check=True)
    subprocess.run(["git", "-C", str(lane_dir), "commit", "-q", "-m", "lane"], check=True)
    merge_rev = _rev(lane_dir, "HEAD")

    sha = git_ops.chain_merge(
        tmp_path, merge_rev, "merge lane", ref="refs/agent6/r1/head", also_branch="agent6/r1"
    )

    assert sha is not None
    assert sha == _rev(tmp_path, "HEAD")
    assert (tmp_path / "lane.txt").read_text(encoding="utf-8") == "lane\n"
    assert git_ops.status(tmp_path).is_clean


def test_chain_commit_skips_identical_trees_and_survives_branch_switches(
    tmp_path: pathlib.Path,
) -> None:
    """An unchanged worktree records None; the chain keeps its parentage across a branch switch."""
    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    s1 = git_ops.chain_commit(
        tmp_path, "agent6 iter 1", ref="refs/agent6/t1", fallback_parent=head0
    )
    assert s1 is not None
    assert (
        git_ops.chain_commit(tmp_path, "agent6 iter 2", ref="refs/agent6/t1", fallback_parent=None)
        is None
    )

    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-q", "-b", "model-branch"], check=True)
    (tmp_path / "c.txt").write_text("three\n", encoding="utf-8")
    s2 = git_ops.chain_commit(tmp_path, "agent6 iter 3", ref="refs/agent6/t1", fallback_parent=None)
    assert s2 is not None
    assert _rev(tmp_path, f"{s2}^") == s1
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branch == "model-branch"


def test_chain_commit_root_and_trailer(tmp_path: pathlib.Path) -> None:
    """No ref and no fallback makes a root commit, and the identity trailer lands once."""
    _init_repo(tmp_path)
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    sha = git_ops.chain_commit(
        tmp_path,
        "agent6 iter 1",
        ref="refs/agent6/t2",
        fallback_parent=None,
        identity=git_ops.CommitIdentity(name="A", email="a@a", trailer="Assisted-by: agent6:m1"),
    )
    assert sha is not None
    body = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%B%P", sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert body.count("Assisted-by: agent6:m1") == 1
    assert body.strip().endswith("m1")  # %P empty: a root commit has no parent


def _lane_commit(path: pathlib.Path, parent: str, name: str, content: str) -> str:
    """Return a commit adding a file on top of a parent, built with plumbing only."""
    env = dict(os.environ, GIT_INDEX_FILE=str(path / ".git" / "lane-index"))
    blob = subprocess.run(
        ["git", "-C", str(path), "hash-object", "-w", "--stdin"],
        input=content,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(path), "read-tree", parent], env=env, check=True)
    subprocess.run(
        ["git", "-C", str(path), "update-index", "--add", "--cacheinfo", f"100644,{blob},{name}"],
        env=env,
        check=True,
    )
    tree = subprocess.run(
        ["git", "-C", str(path), "write-tree"], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    return subprocess.run(
        ["git", "-C", str(path), "commit-tree", tree, "-p", parent, "-m", f"lane: {name}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_chain_merge_records_the_lane_and_syncs_the_worktree(tmp_path: pathlib.Path) -> None:
    """A clean lane merge lands as a two-parent chain commit with the lane's files checked out.

    HEAD and the operator's checkout stay untouched; an already-contained rev is a no-op returning
    the tip.
    """
    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    s1 = git_ops.chain_commit(
        tmp_path, "agent6 iter 1", ref="refs/agent6/t3", fallback_parent=head0
    )
    assert s1 is not None
    lane = _lane_commit(tmp_path, head0, "c.txt", "lane\n")

    merged = git_ops.chain_merge(tmp_path, lane, "merge lane", ref="refs/agent6/t3")
    assert merged is not None and merged not in (s1, lane)
    assert _rev(tmp_path, "refs/agent6/t3") == merged
    parents = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%P", merged],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert parents == [s1, lane]
    assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "lane\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two\n"
    assert _rev(tmp_path, "HEAD") == head0
    assert git_ops.chain_merge(tmp_path, head0, "again", ref="refs/agent6/t3") == merged


def test_chain_merge_syncs_a_file_the_lane_modified(tmp_path: pathlib.Path) -> None:
    """A lane that edits an existing file merges and the edit lands in the worktree.

    The temp index carries no stat data, and `read-tree -m -u` refuses an entry it cannot prove up
    to date.
    """
    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    s1 = git_ops.chain_commit(
        tmp_path, "agent6 iter 1", ref="refs/agent6/t5", fallback_parent=head0
    )
    assert s1 is not None
    lane = _lane_commit(tmp_path, head0, "README.md", "hi\nlane edit\n")

    merged = git_ops.chain_merge(tmp_path, lane, "merge lane", ref="refs/agent6/t5")
    assert merged is not None
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hi\nlane edit\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two\n"


def test_chain_merge_conflict_leaves_chain_and_worktree_alone(tmp_path: pathlib.Path) -> None:
    """A textual conflict returns None: the ref keeps its tip and no worktree file is rewritten."""
    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("ours\n", encoding="utf-8")
    s1 = git_ops.chain_commit(
        tmp_path, "agent6 iter 1", ref="refs/agent6/t4", fallback_parent=head0
    )
    lane = _lane_commit(tmp_path, head0, "b.txt", "theirs\n")

    assert git_ops.chain_merge(tmp_path, lane, "merge lane", ref="refs/agent6/t4") is None
    assert _rev(tmp_path, "refs/agent6/t4") == s1
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "ours\n"


def test_chain_commit_keeps_tracked_but_ignored_files(tmp_path: pathlib.Path) -> None:
    """A file committed before it was gitignored stays tracked in every chain commit.

    The temp index is seeded from the parent tree; to an empty index everything is untracked, so
    `add -A` applied the ignore rules and a later merge deleted such files from the operator's
    branch.
    """
    repo = _repo_with_ignored_tracked(tmp_path)
    ref = "refs/agent6/t/head"
    head = _rev(repo, "HEAD")

    # No edits: the worktree matches the chain base exactly.
    assert git_ops.chain_dirty(repo, ref, head) is False

    (repo / "code.py").write_text("print('v2')\n", encoding="utf-8")
    assert git_ops.chain_dirty_paths(repo, ref, head, 10) == ["code.py"]
    sha = git_ops.chain_commit(repo, "step", ref=ref, fallback_parent=head)
    assert sha is not None
    files = _run_git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
    assert "results.log" in files, files
    assert "scratch.log" not in files  # new ignored files still stay out
    diff = _run_git(repo, "diff", "--name-only", f"{head}..{sha}").splitlines()
    assert diff == ["code.py"], diff


def test_chain_commit_leaves_the_operators_untracked_files_out(tmp_path: pathlib.Path) -> None:
    """Files untracked when the run started are the operator's and enter no per-step commit."""
    _init_repo(tmp_path)
    head = _rev(tmp_path, "HEAD")
    ref = "refs/agent6/u/head"
    (tmp_path / "notes.txt").write_text("mine\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "we ird:name.txt").write_text("mine too\n", encoding="utf-8")
    mine = git_ops.untracked_paths(tmp_path)
    assert mine == {"notes.txt", "sub/we ird:name.txt"}

    assert git_ops.chain_dirty(tmp_path, ref, head, exclude=mine) is False
    assert git_ops.chain_dirty(tmp_path, ref, head) is True  # the same tree, unscoped
    assert (
        git_ops.chain_commit(tmp_path, "nothing", ref=ref, fallback_parent=head, exclude=mine)
        is None
    )

    (tmp_path / "README.md").write_text("edited\n", encoding="utf-8")
    (tmp_path / "sub" / "made.py").write_text("print(1)\n", encoding="utf-8")  # the model's
    assert git_ops.chain_dirty_paths(tmp_path / "sub", ref, head, 10, exclude=mine) == [
        "README.md",
        "sub/made.py",
    ]
    sha = git_ops.chain_commit(
        tmp_path / "sub", "step", ref=ref, fallback_parent=head, exclude=mine
    )
    assert sha is not None
    files = _run_git(tmp_path, "ls-tree", "-r", "--name-only", sha).splitlines()
    assert files == ["README.md", "sub/made.py"], files
    assert git_ops.chain_dirty(tmp_path, ref, head, exclude=mine) is False
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "mine\n"


def _run_git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _repo_with_ignored_tracked(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "t@t")
    _run_git(repo, "config", "user.name", "t")
    (repo / "code.py").write_text("print('v1')\n", encoding="utf-8")
    (repo / "results.log").write_text("kept\n", encoding="utf-8")
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-q", "-m", "base")
    # Ignored AFTER being tracked, the shape that bit; plus a new ignored file.
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    _run_git(repo, "add", ".gitignore")
    _run_git(repo, "commit", "-q", "-m", "ignore logs")
    (repo / "scratch.log").write_text("never tracked\n", encoding="utf-8")
    return repo


def test_diff_since_excludes_untracked_at_start(tmp_path: pathlib.Path) -> None:
    """The review diff agrees with the chain: a file untracked before the run is not its work."""
    _init_repo(tmp_path)
    base = _run_git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "operator-notes.toml").write_text("k = 1\n", encoding="utf-8")
    (tmp_path / "new-work.py").write_text("x = 1\n", encoding="utf-8")
    out = git_ops.diff_since(tmp_path, base, exclude=frozenset({"operator-notes.toml"}))
    assert "new-work.py" in out
    assert "operator-notes.toml" not in out


def test_diff_since_leaves_the_real_index_untouched(tmp_path: pathlib.Path) -> None:
    """The intent-add runs against a temp index copy, so no `-N` entry reaches the real index."""
    _init_repo(tmp_path)
    base = _run_git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "new-work.py").write_text("x = 1\n", encoding="utf-8")
    out = git_ops.diff_since(tmp_path, base)
    assert "new-work.py" in out
    porcelain = _run_git(tmp_path, "status", "--porcelain")
    assert "?? new-work.py" in porcelain
    assert "A  new-work.py" not in porcelain and " A new-work.py" not in porcelain


def test_tree_diff_paths_names_what_changed_between_two_worktree_trees(
    tmp_path: pathlib.Path,
) -> None:
    """`worktree_tree` and `tree_diff_paths` answer the flip-green notice's question from git.

    The worktree is staged into a temp index, the shared index untouched.
    """
    _init_repo(tmp_path)
    before = git_ops.worktree_tree(tmp_path, "HEAD", ())
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")
    after = git_ops.worktree_tree(tmp_path, "HEAD", ())
    assert git_ops.tree_diff_paths(tmp_path, before, after) == ["README.md", "new.txt"]
    assert git_ops.tree_diff_paths(tmp_path, before, before) == []
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--quiet"], check=False
    )
    assert staged.returncode == 0  # nothing reached the shared index


def test_plumb_merge_survives_a_recorded_but_missing_worktree(tmp_path: pathlib.Path) -> None:
    """A merge over a worktree record whose directory is gone lands and returns."""
    import shutil

    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _rev(repo, "HEAD")
    subprocess.run(["git", "-C", str(repo), "branch", "side", base], check=True)
    run_tip = _lane_commit(repo, base, "feat.txt", "x\n")
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/agent6/r1", run_tip], check=True)
    gone = tmp_path / "gone"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", str(gone), "side"], check=True)
    shutil.rmtree(gone)

    res = git_ops.plumb_merge(repo, "side", "refs/agent6/r1", strategy="merge", message="Merge")
    assert not res.conflicted
    assert _rev(repo, "side") == res.merged_sha


def test_a_chain_commit_never_rewinds_the_run_branch(tmp_path: pathlib.Path) -> None:
    """The end banner leaves the operator on the run branch, and their commit there survives.

    A compare-and-swap leaves the branch where they put it; the chain ref keeps the run's record.
    """
    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    first = git_ops.chain_commit(
        tmp_path,
        "agent6 iter 1",
        ref="refs/agent6/t2",
        fallback_parent=head0,
        also_branch="agent6/t2",
    )
    assert first is not None and _rev(tmp_path, "refs/heads/agent6/t2") == first

    # The operator commits on the run branch with plumbing; the chain's worktree is not the subject.
    theirs = subprocess.run(
        ["git", "-C", str(tmp_path), "commit-tree", f"{first}^{{tree}}", "-p", first, "-m", "mine"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/heads/agent6/t2", theirs], check=True
    )

    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    second = git_ops.chain_commit(
        tmp_path,
        "agent6 iter 2",
        ref="refs/agent6/t2",
        fallback_parent=head0,
        also_branch="agent6/t2",
    )

    assert second is not None
    assert git_ops.chain_tip(tmp_path, "refs/agent6/t2") == second  # the chain still advances
    # By full ref name: `agent6/t2` alone resolves to the CHAIN ref first.
    assert _rev(tmp_path, "refs/heads/agent6/t2") == theirs, "the operator's commit was rewound"


def test_a_merge_tree_this_git_cannot_run_names_the_floor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`merge-tree --merge-base` needs git 2.40; the refusal names the git found and the floor."""
    from agent6 import kinds

    _init_repo(tmp_path)
    base = git_ops.status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "README.md", "run change\n")
    _commit_file(tmp_path, "README.md", "main change\n", "main edit")
    real_run = git_ops._run  # pyright: ignore[reportPrivateUsage]

    def old_git(cwd: pathlib.Path, *args: str, **kw: object) -> kinds.CommandResult:
        if args[:1] == ("merge-tree",):
            return kinds.CommandResult(
                argv=("git", *args),
                returncode=129,
                stdout="",
                stderr="error: unknown option `merge-base=abc'",
                duration_s=0.0,
                exec_failed=False,
            )
        if args == ("--version",):
            return kinds.CommandResult(
                argv=("git", "--version"),
                returncode=0,
                stdout="git version 2.39.0\n",
                stderr="",
                duration_s=0.0,
                exec_failed=False,
            )
        return real_run(cwd, *args, **kw)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(git_ops, "_run", old_git)
    with pytest.raises(git_ops.GitError, match=r"git version 2\.39\.0.*git 2\.40 or newer"):
        git_ops.plumb_merge(
            tmp_path, "main", theirs, strategy="merge", message=None, merge_base=base
        )


def test_remove_worktree_reports_a_tree_it_could_not_delete(tmp_path: pathlib.Path) -> None:
    """A prune whose rmtree fails says so instead of claiming the directory removed."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    base = _rev(repo, "HEAD")
    wt = tmp_path / "wt"
    git_ops.add_worktree(repo, wt, base)
    held = wt / "held"
    held.mkdir()
    (held / "f").write_text("x\n", encoding="utf-8")
    held.chmod(0o500)
    try:
        assert git_ops.remove_worktree(repo, wt) is False
        assert held.is_dir()
    finally:
        held.chmod(0o700)


def test_a_diff_carries_a_b_prefixes_whatever_the_diff_config_says(tmp_path: pathlib.Path) -> None:
    """The operator's diff prefix settings do not change the headers agent6 reads.

    `diff.noprefix`, `diff.mnemonicPrefix` and `diff.srcPrefix` would file each file under a path no
    citation matched.
    """
    from agent6.harness import _panel

    _init_repo(tmp_path)
    base = _rev(tmp_path, "HEAD")
    for key, value in (
        ("diff.noprefix", "true"),
        ("diff.mnemonicPrefix", "true"),
        ("diff.srcPrefix", "x/"),
        ("diff.dstPrefix", "y/"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    (tmp_path / "README.md").write_text("hi\nthere\n", encoding="utf-8")
    diff = git_ops.diff_since(tmp_path, base)
    assert "diff --git a/README.md b/README.md" in diff
    assert "--- a/README.md" in diff and "+++ b/README.md" in diff
    assert _panel.is_grounded("README.md:2", _panel.diff_hunks(diff))
