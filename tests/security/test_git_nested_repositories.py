# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A nested repository's own git config runs no program under agent6's own git.

A jailed command can make a repository inside the workspace at every isolation level, and
can write the `.git` of a submodule: the nested config, attributes and hooks are workspace
content. git records such a directory as a gitlink and, on some commands, runs git inside
it under that config. Each test arms a nested repository with a filter and a textconv
program and runs the agent6 operations of one route.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from typing import Any

import pytest

from agent6 import errors, kinds, path_safety
from agent6.vcs import review_submodules
from agent6.vcs.git import command as git_command
from agent6.vcs.git import history as git_history
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import policy as git_policy
from agent6.vcs.git import read as git_read


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "x@y.z")
    _git(root, "config", "user.name", "x")
    (root / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def _program(tmp_path: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Return a program that records its run and passes its input on, with its record."""
    marker = tmp_path / "ran"
    program = tmp_path / "program"
    program.write_text(f'#!/bin/sh\necho ran >> {marker}\ncat "$@"\n', encoding="utf-8")
    program.chmod(0o755)
    return str(program), marker


def _nested(root: pathlib.Path) -> pathlib.Path:
    """Make a repository in `vendor/`, as a jailed command can; the project records nothing."""
    nested = _repo(root / "vendor")
    (nested / ".gitattributes").write_text("* filter=x diff=x\n", encoding="utf-8")
    _git(nested, "add", "-A")
    _git(nested, "commit", "-qm", "attributes")
    return nested


def _record_nested(root: pathlib.Path) -> pathlib.Path:
    """Make a repository in `vendor/` and commit it to the project as a submodule.

    Its `.gitmodules` entry asks for a full comparison (`ignore = none`), which outranks
    `diff.ignoreSubmodules`.
    """
    nested = _nested(root)
    (root / ".gitmodules").write_text(
        '[submodule "vendor"]\n\tpath = vendor\n\turl = ./vendor\n\tignore = none\n',
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "vendor")
    return nested


def _commit_nested(root: pathlib.Path, text: str) -> str:
    """Commit a change in the nested repository and record it in the project."""
    (root / "vendor" / "f.txt").write_text(text, encoding="utf-8")
    _git(root / "vendor", "commit", "-qam", text.strip())
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", f"vendor {text.strip()}")
    return _git(root, "rev-parse", "HEAD")


def _arm(nested: pathlib.Path, program: str) -> None:
    """Name the program as the nested repository's filter and textconv."""
    for key in ("filter.x.clean", "filter.x.smudge", "diff.x.textconv"):
        _git(nested, "config", key, program)


def _move(nested: pathlib.Path, text: str) -> str:
    """Commit a change in the nested repository, its filter off, and return the new commit."""
    (nested / "f.txt").write_text(text, encoding="utf-8")
    _git(nested, "-c", "filter.x.clean=", "commit", "-qam", text.strip())
    return _git(nested, "rev-parse", "HEAD")


def test_a_status_or_a_diff_runs_no_git_inside_a_nested_repository(
    tmp_path: pathlib.Path,
) -> None:
    """A worktree comparison ran `git status` in the nested repository, and its clean filter."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    assert git_read.status(root).is_clean
    assert git_read.modified_paths(root) == []
    assert not git_read.paths_dirty(root, ("vendor",))
    assert git_read.worktree_matches(root, "HEAD", ["vendor"])
    assert git_history.worktree_files(root, "HEAD") == ()
    assert git_history.worktree_patch(root, "HEAD", "vendor", max_bytes=None) == ("", False)
    assert git_command.nul_fields(root, "diff-files", "--name-only", "-z") == [""]
    assert not marker.exists(), "the nested repository's clean filter ran on the host"

    # The commit the nested repository is at still counts: git reads it without running there.
    _git(nested, "-c", "filter.x.clean=", "commit", "-qam", "edited")
    assert git_read.modified_paths(root) == ["vendor"]
    assert not git_read.worktree_matches(root, "HEAD", ["vendor"])
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_a_branch_switch_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """`checkout` and `switch` listed local changes, asking the nested repository for its own."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    git_ops.create_branch(root, "topic")
    git_ops.create_branch(root, "main")  # exists: a plain checkout
    git_ops.switch_branch(root, "topic")
    git_ops.checkout_detached(root, "main")
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_restoring_a_stash_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """`stash apply` ended with a `git status`, which asked the nested repository."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    git_ops.stash_tracked_changes(root, "kept")
    entry = git_read.find_stash(root, "kept")
    assert entry is not None
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    assert git_ops.restore_stash(root, entry)
    assert (root / "f.txt").read_text(encoding="utf-8") == "v2\n"
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_moving_a_checkout_never_recurses_into_a_submodule(tmp_path: pathlib.Path) -> None:
    """Under `submodule.recurse`, a tree move checked the submodule out with its own filters."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    older = _git(root, "rev-parse", "HEAD")
    newer = _commit_nested(root, "v2\n")
    _git(root, "config", "submodule.recurse", "true")
    _git(root, "config", "submodule.vendor.url", "./vendor")
    _arm(nested, program)

    git_ops.sync_worktree(root, newer, older)
    git_ops.checkout_detached(root, older)
    git_ops.create_branch(root, "topic", start_point=newer)
    git_ops.switch_branch(root, "main")
    assert (nested / "f.txt").read_text(encoding="utf-8") == "v2\n"  # git left it alone
    assert not marker.exists(), "the submodule's own filter ran on the host"


def test_a_diff_never_renders_inside_a_submodule(tmp_path: pathlib.Path) -> None:
    """Under `diff.submodule = diff`, a diff, show or log ran `git diff` inside the submodule."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    older = _git(root, "rev-parse", "HEAD")
    newer = _commit_nested(root, "v2\n")
    _git(root, "config", "diff.submodule", "diff")
    _arm(nested, program)

    assert "Subproject commit" in git_read.commit_diff(root, newer)
    assert "Subproject commit" in git_read.diff_range(root, older, newer)
    assert (
        "Subproject commit"
        in git_history.file_patch(root, older, newer, "vendor", max_bytes=None)[0]
    )
    assert not marker.exists(), "the submodule's own textconv ran on the host"


def test_a_status_summarizes_no_submodule(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `status.submoduleSummary`, a status ran `git log` inside each moved submodule."""
    root = _repo(tmp_path / "r")
    nested = _record_nested(root)
    (nested / "f.txt").write_text("v2\n", encoding="utf-8")
    _git(nested, "commit", "-qam", "v2")
    _git(root, "config", "status.submoduleSummary", "true")
    trace = tmp_path / "trace"
    monkeypatch.setenv("GIT_TRACE", str(trace))

    assert "modified:   vendor" in git_command.run(root, "status").stdout
    assert "cd vendor" not in trace.read_text(encoding="utf-8"), "git ran inside the submodule"


def test_a_directory_laid_out_as_a_git_dir_is_no_repository(tmp_path: pathlib.Path) -> None:
    """Below an embedded bare repository, git took that directory's config as the project's."""
    root = _repo(tmp_path / "r")
    embedded = root / "pkg"
    subprocess.run(["git", "init", "-q", "--bare", str(embedded)], check=True)
    (embedded / "src").mkdir()
    _git(embedded, "config", "core.bare", "false")
    _git(embedded, "config", "core.worktree", str(tmp_path))

    assert git_read.toplevel(embedded / "src") is None
    assert not git_read.is_git_repo(embedded / "src")
    assert git_read.toplevel(root) == root


def test_a_review_honors_no_driver_of_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """Under `git.run_repo_drivers`, review's status of a submodule ran that submodule's filter."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    with git_policy.use(git_policy.GitPolicy(honor_repo_drivers=True)):
        dirty = review_submodules.dirty_paths(
            root, ("vendor",), path_safety.host_workspace(root, ())
        )
    assert dirty == ("vendor",)
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


def test_a_review_runs_no_filter_whose_name_holds_an_equals_sign(tmp_path: pathlib.Path) -> None:
    """Review's status of a submodule ran that submodule's clean filter named `a=b`."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _git(nested, "config", "filter.a=b.clean", program)
    (nested / ".gitattributes").write_text("* filter=a=b\n", encoding="utf-8")
    # The same size as the committed file: git has to read the content to compare it.
    (nested / "f.txt").write_text("v2\n", encoding="utf-8")

    dirty = review_submodules.dirty_paths(root, ("vendor",), path_safety.host_workspace(root, ()))
    assert dirty == ("vendor",)
    assert not marker.exists(), "the nested repository's clean filter ran on the host"


_RAN = "the nested repository's clean filter ran on the host"


def test_a_checkpoint_runs_no_git_inside_the_nested_repository_it_recorded(
    tmp_path: pathlib.Path,
) -> None:
    """One checkpoint recorded a nested repository; the next one's `git add -A` ran its filter."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    base = _git(root, "rev-parse", "HEAD")
    nested = _nested(root)
    _arm(nested, program)
    ref = "refs/agent6/s/head"

    first = git_ops.chain_commit(root, "turn 1", ref=ref, fallback_parent=base)
    assert first is not None
    recorded = f"160000 commit {_git(nested, 'rev-parse', 'HEAD')}"
    assert git_read.tree_entry(root, first, "vendor") == recorded
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")
    (root / "other.txt").write_text("x\n", encoding="utf-8")
    second = git_ops.chain_commit(root, "turn 2", ref=ref, fallback_parent=base)
    assert second is not None
    assert git_read.tree_paths(root, second) == {"f.txt", "other.txt", "vendor"}
    assert git_read.tree_entry(root, second, "vendor") == recorded
    assert not git_read.chain_dirty(root, ref, base)
    assert git_read.chain_dirty_paths(root, ref, base, 5) == []
    assert not marker.exists(), _RAN


@pytest.mark.parametrize("state", ["edited", "moved", "new", "deleted"])
def test_a_checkpoint_records_the_commit_a_nested_repository_is_at(
    tmp_path: pathlib.Path, state: str
) -> None:
    """`update-index` records each gitlink as `git add -A` would, and runs nothing inside it."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _nested(root) if state == "new" else _record_nested(root)
    head = _git(root, "rev-parse", "HEAD")
    at = _git(nested, "rev-parse", "HEAD")
    _arm(nested, program)
    if state == "moved":
        at = _move(nested, "v2\n")
    elif state == "deleted":
        shutil.rmtree(nested)
    (nested.parent / "f.txt" if state == "deleted" else nested / "f.txt").write_text(
        "edited\n", encoding="utf-8"
    )

    tree = git_read.worktree_tree(root, head, ())
    entry = "" if state == "deleted" else f"160000 commit {at}"
    assert git_read.tree_entry(root, tree, "vendor") == entry
    assert (tree == git_read.tree_sha(root, head)) == (state == "edited")
    assert not marker.exists(), _RAN


def test_a_checkpoint_from_a_subdirectory_records_a_nested_repository(
    tmp_path: pathlib.Path,
) -> None:
    """`update-index` reads its paths from the directory it runs in: the checkout's root."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    (root / "pkg").mkdir()
    (root / "pkg" / "g.txt").write_text("g\n", encoding="utf-8")
    head = _git(root, "rev-parse", "HEAD")
    _arm(nested, program)
    at = _move(nested, "v2\n")
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    tree = git_read.worktree_tree(root / "pkg", head, ())
    assert git_read.tree_entry(root, tree, "vendor") == f"160000 commit {at}"
    assert git_read.tree_entry(root, tree, "pkg/g.txt").startswith("100644 blob ")
    assert not marker.exists(), _RAN


def test_an_excluded_nested_repository_keeps_the_commit_the_seed_records(
    tmp_path: pathlib.Path,
) -> None:
    """A hidden path stays out of every tree: its gitlink is neither added nor recorded anew."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    head = _git(root, "rev-parse", "HEAD")
    seeded = git_read.tree_entry(root, head, "vendor")
    _arm(nested, program)
    _move(nested, "v2\n")
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    tree = git_read.worktree_tree(root, head, {"vendor"})
    assert git_read.tree_entry(root, tree, "vendor") == seeded
    assert not marker.exists(), _RAN


def _record_ignored(root: pathlib.Path, where: str, ignore: str) -> pathlib.Path:
    """Record a repository at a path of the project, then commit ignore rules that cover it."""
    nested = _repo(root / where)
    (nested / ".gitattributes").write_text("* filter=x diff=x\n", encoding="utf-8")
    _git(nested, "add", "-A")
    _git(nested, "commit", "-qm", "attributes")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "nested")
    (root / ".gitignore").write_text(ignore, encoding="utf-8")
    _git(root, "add", "-f", ".gitignore")
    _git(root, "commit", "-qm", "ignore")
    return nested


def _plain_tree(root: pathlib.Path, head: str, index: pathlib.Path) -> str:
    """Return the tree plain `git add -A` makes of the worktree, from the commit's index."""
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    for args in (("read-tree", head), ("add", "-A")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)
    return subprocess.run(
        ["git", "-C", str(root), "write-tree"], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


@pytest.mark.parametrize(
    ("where", "ignore"),
    [
        ("deps/lib", "deps/\n"),
        ("deps/lib", "deps\n"),
        ("deps/lib", "*\n!.gitignore\n!f.txt\n"),
        ("deps/lib", "deps/*\n"),
        ("deps/lib", "deps/lib\n"),
        ("lib", "lib\n"),
    ],
)
def test_a_checkpoint_records_a_nested_repository_under_an_ignored_path(
    tmp_path: pathlib.Path, where: str, ignore: str
) -> None:
    """`git add` exits 1 naming an ignored directory above a path left out: the commit failed."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_ignored(root, where, ignore)
    head = _git(root, "rev-parse", "HEAD")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    (root / "new.txt").write_text("new\n", encoding="utf-8")
    at = _move(nested, "v2\n")
    plain = _plain_tree(root, head, tmp_path / "plain-index")
    _arm(nested, program)
    # The same size as the committed file: a status inside would read the content to compare it.
    (nested / "f.txt").write_text("v3\n", encoding="utf-8")

    tree = git_read.worktree_tree(root, head, ())
    assert tree == plain
    assert git_read.tree_entry(root, tree, where) == f"160000 commit {at}"
    assert not marker.exists(), _RAN


def test_a_checkpoint_records_a_repository_made_beside_one_under_an_ignored_directory(
    tmp_path: pathlib.Path,
) -> None:
    """The warning for a repository git adds followed its notice, and the notice did not pass."""
    root = _repo(tmp_path / "r")
    _record_ignored(root, "deps/lib", "deps/\n")
    head = _git(root, "rev-parse", "HEAD")
    made = _repo(root / "made")

    tree = git_read.worktree_tree(root, head, ())
    assert (
        git_read.tree_entry(root, tree, "made")
        == f"160000 commit {_git(made, 'rev-parse', 'HEAD')}"
    )


@pytest.mark.parametrize("paths", [["."], ["deps/lib"]])
def test_staging_records_a_nested_repository_under_an_ignored_directory(
    tmp_path: pathlib.Path, paths: list[str]
) -> None:
    """The Git page's stage of a path covering the nested repository failed on git's notice."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_ignored(root, "deps/lib", "deps/\n")
    at = _move(nested, "v2\n")
    _arm(nested, program)
    (nested / "f.txt").write_text("v3\n", encoding="utf-8")

    git_ops.stage_paths(root, paths)
    assert _git(root, "ls-files", "--stage", "deps/lib") == f"160000 {at} 0\tdeps/lib"
    assert not marker.exists(), _RAN


def test_staging_an_ignored_path_fails_beside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """Only a notice that names what agent6 left out passes: an ignored path asked for fails."""
    root = _repo(tmp_path / "r")
    _record_ignored(root, "deps/lib", "deps/\n")
    (root / "deps" / "new.txt").write_text("new\n", encoding="utf-8")

    with pytest.raises(errors.GitError, match="paths are ignored"):
        git_ops.stage_paths(root, ["deps/new.txt"])
    assert _git(root, "diff", "--cached", "--name-only") == ""


def test_a_run_diff_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """The run diff's `git add -N` ran the clean filter of each nested repository in the index."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    head = _git(root, "rev-parse", "HEAD")
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")
    (root / "new.txt").write_text("new\n", encoding="utf-8")

    diff = git_read.diff_since(root, head)
    assert "+++ b/new.txt" in diff
    assert "vendor" not in diff
    assert not marker.exists(), _RAN

    at = _move(nested, "v2\n")
    assert f"+Subproject commit {at}" in git_read.diff_since(root, head)
    assert not marker.exists(), _RAN


def test_staging_a_path_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """The Git page's stage ran the clean filter of a nested repository its path covered."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")

    git_ops.stage_paths(root, ["vendor"])
    git_ops.stage_paths(root, ["."])
    assert _git(root, "diff", "--cached", "--name-only") == "f.txt"
    assert not marker.exists(), _RAN

    at = _move(nested, "v2\n")
    git_ops.stage_paths(root, ["f.txt"])
    assert _git(root, "diff", "--cached", "--name-only") == "f.txt"  # a path beside it stages alone
    git_ops.stage_paths(root, ["vendor"])
    assert _git(root, "ls-files", "--stage", "vendor") == f"160000 {at} 0\tvendor"
    assert not marker.exists(), _RAN


def test_staging_no_path_stages_nothing_beside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """Left with a gitlink's exclude pathspec alone, `git add` takes the whole tree."""
    root = _repo(tmp_path / "r")
    _record_nested(root)
    (root / "f.txt").write_text("v2\n", encoding="utf-8")

    git_ops.stage_paths(root, [])
    assert _git(root, "diff", "--cached", "--name-only") == ""


def test_staging_leaves_a_skip_worktree_gitlink_in_the_index(tmp_path: pathlib.Path) -> None:
    """`update-index --remove` drops a skip-worktree entry unless told to leave it."""
    root = _repo(tmp_path / "r")
    nested = _record_nested(root)
    at = _git(nested, "rev-parse", "HEAD")
    _git(root, "update-index", "--skip-worktree", "vendor")
    _move(nested, "v2\n")

    git_ops.stage_paths(root, ["."])
    assert _git(root, "ls-files", "--stage", "vendor") == f"160000 {at} 0\tvendor"


def test_a_path_commit_runs_no_git_inside_a_nested_repository(tmp_path: pathlib.Path) -> None:
    """`commit_paths` ran the filter on its `git add`, and on the status of an empty commit."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    head = _git(root, "rev-parse", "HEAD")
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")

    for unchanged in ("f.txt", "vendor"):
        with pytest.raises(errors.GitError) as refused:
            git_ops.commit_paths(root, "empty", (unchanged,))
        assert not marker.exists(), _RAN
        assert "nothing to commit" in str(refused.value)
    assert _git(root, "rev-parse", "HEAD") == head

    at = _move(nested, "v2\n")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    sha = git_ops.commit_paths(root, "pointer", ("vendor",))
    assert git_read.tree_entry(root, sha, "vendor") == f"160000 commit {at}"
    assert git_read.tree_diff_paths(root, head, sha) == ["vendor"]
    assert not marker.exists(), _RAN


@pytest.mark.parametrize("commit", ["paths", "index"])
def test_a_change_undone_before_the_commit_gets_no_status_from_git(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, commit: str
) -> None:
    """Between agent6's check and `git commit` the change can go: git printed its status."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    head = _git(root, "rev-parse", "HEAD")
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    run = git_command.run

    def undo_after_the_check(cwd: pathlib.Path, *args: str, **kwargs: Any) -> kinds.CommandResult:
        result = run(cwd, *args, **kwargs)
        if args[:2] == ("diff", "--cached"):
            (root / "f.txt").write_text("v1\n", encoding="utf-8")
            _git(root, "add", "f.txt")
        return result

    monkeypatch.setattr(git_command, "run", undo_after_the_check)
    if commit == "paths":
        sha = git_ops.commit_paths(root, "raced", ("f.txt",))
    else:
        git_ops.stage_paths(root, ["f.txt"])
        sha = git_ops.commit_index(root, "raced")
    assert not marker.exists(), _RAN
    assert git_read.tree_sha(root, sha) == git_read.tree_sha(root, head)  # an empty commit


@pytest.mark.parametrize("paths", [(), ("f.txt",), ("f.txt", "vendor")])
def test_the_auto_stash_runs_no_git_inside_a_nested_repository(
    tmp_path: pathlib.Path, paths: tuple[str, ...]
) -> None:
    """`git stash push` ran the clean filter of each nested repository in the index."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    nested = _record_nested(root)
    _arm(nested, program)
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")

    git_ops.stash_tracked_changes(root, "kept", paths)
    entry = git_read.find_stash(root, "kept")
    assert entry is not None
    assert git_read.tree_diff_paths(root, "HEAD", entry.sha) == ["f.txt"]
    assert (root / "f.txt").read_text(encoding="utf-8") == "v1\n"
    assert not marker.exists(), _RAN

    # A moved nested repository is left out and stays where it is: no stash ever moved one.
    at = _move(nested, "v2\n")
    (root / "f.txt").write_text("v3\n", encoding="utf-8")
    git_ops.stash_tracked_changes(root, "later", paths)
    entry = git_read.find_stash(root, "later")
    assert entry is not None
    assert git_read.tree_diff_paths(root, "HEAD", entry.sha) == ["f.txt"]
    assert _git(nested, "rev-parse", "HEAD") == at
    assert not marker.exists(), _RAN
