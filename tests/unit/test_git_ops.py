# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.git_ops on a temporary repository."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from agent6 import git_ops
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    auto_stash_message,
    clone_repo,
    commit_all,
    commit_diff,
    commit_paths,
    condense_commit_message,
    create_branch,
    create_branch_at,
    diff_range,
    diff_since,
    dirty_paths,
    fetch_branch,
    find_stash,
    init_repo,
    is_git_repo,
    list_run_commits,
    make_run_branch_name,
    plumb_merge,
    recent_log,
    reset_to,
    restore_stash,
    set_repo_hook_policy,
    slugify,
    stash_all,
    status,
    unignored,
    verify_git_identity,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def test_dirty_paths_tags_and_caps(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    assert dirty_paths(tmp_path) == []  # clean tree
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")  # untracked
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")  # modified
    paths = dirty_paths(tmp_path)
    assert "? new.txt" in paths
    assert "M README.md" in paths
    # the cap bounds the listing
    for i in range(15):
        (tmp_path / f"f{i}.txt").write_text("y\n", encoding="utf-8")
    assert len(dirty_paths(tmp_path, limit=5)) == 5


def test_commit_paths_ignores_unrelated_staged_work(tmp_path: Path) -> None:
    # `agent6 init` scaffolds AGENTS.md + .gitignore and commits them. If the
    # user has other work already staged, that must NOT be swept into the
    # scaffold commit.
    _init_repo(tmp_path)
    (tmp_path / "wip.txt").write_text("in progress\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "wip.txt"], check=True)
    (tmp_path / "AGENTS.md").write_text("# scaffold\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".agent6/\n", encoding="utf-8")

    commit_paths(tmp_path, "chore: scaffold", ("AGENTS.md", ".gitignore"))

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


def test_status_handles_unborn_head(tmp_path: Path) -> None:
    """A freshly `git init`'d repo (no commits, unborn HEAD) must not crash
    status() — every agent6 entry point loads the repo summary first, so an
    unborn HEAD used to crash `machine create`/`run` in a brand-new repo."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    st = status(tmp_path)
    assert st.branch == "main"
    assert st.head_sha == ""
    assert not st.is_clean
    assert st.untracked_count == 1


def test_git_ops_neutralizes_repo_fsmonitor(tmp_path: Path) -> None:
    """A repo-controlled core.fsmonitor must NOT execute on the host when agent6
    runs git. Defense-in-depth against a cloned/poisoned `.git/config` firing on
    the harness's own status/commit."""
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.fsmonitor", f"touch {marker}"],
        check=True,
    )
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")
    # Index-refreshing op through agent6's hardened _run.
    status(tmp_path)
    commit_all(tmp_path, "msg")
    assert not marker.exists(), "repo core.fsmonitor command executed on the host"


def _add_pre_commit_hook(repo: Path, marker: Path) -> None:
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    hook.chmod(0o755)


def test_git_ops_skips_repo_hooks_by_default(tmp_path: Path) -> None:
    """Secure default: agent6's own commit must NOT fire a repo
    `.git/hooks/*`. Asserts the UNTOUCHED defaults too -- calling
    set_repo_hook_policy(False) first pinned the explicit-False path, so a
    flipped default (module state or config) stayed green."""
    from agent6.config import Config
    from agent6.git_ops import _hook_policy  # pyright: ignore[reportPrivateUsage]

    assert _hook_policy["honor_repo_hooks"] is False  # the module's own default
    assert Config().git.run_repo_hooks is False  # the config default that sets it

    _init_repo(tmp_path)
    marker = tmp_path / "HOOK_FIRED"
    _add_pre_commit_hook(tmp_path, marker)
    (tmp_path / "n.txt").write_text("x\n", encoding="utf-8")
    commit_all(tmp_path, "c")  # no set_repo_hook_policy call: the default governs
    assert not marker.exists()


def test_git_ops_runs_repo_hooks_when_enabled(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "HOOK_FIRED"
    _add_pre_commit_hook(tmp_path, marker)
    (tmp_path / "n.txt").write_text("x\n", encoding="utf-8")
    try:
        set_repo_hook_policy(True)  # git.run_repo_hooks = true
        commit_all(tmp_path, "c")
        assert marker.exists()
    finally:
        set_repo_hook_policy(False)  # reset process-wide policy for other tests


def test_git_ops_neutralizes_repo_diff_external(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_DIFF"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "diff.external", f"touch {marker} ;"],
        check=True,
    )
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    out = diff_since(tmp_path, "HEAD")
    assert not marker.exists(), "repo diff.external command executed on the host"
    # git >= 2.53 tries to run the empty `-c diff.external=` override and a full
    # patch dies (safe, but empty); --no-ext-diff keeps the patch. Assert the
    # diff still comes back so a broken-but-safe regression can't hide here.
    assert "README.md" in out, "diff.external neutralization silently emptied the diff"


def test_diff_range_reports_a_branch_diff_and_empty_on_bad_ref(tmp_path: Path) -> None:
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
    out = diff_range(tmp_path, base, "agent6/x")
    assert "feature.txt" in out
    assert diff_range(tmp_path, base, "no-such-branch") == ""  # unresolvable range -> ""


def test_diff_range_survives_poisoned_diff_external(tmp_path: Path) -> None:
    """`sessions compare` diffs candidates via `diff_range`; a poisoned repo config
    must not run its payload on the host, same guarantee as `diff_since`."""
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
    out = diff_range(tmp_path, base, "agent6/x")
    assert not marker.exists(), "repo diff.external command executed on the host"
    assert "README.md" in out, "diff.external neutralization silently emptied the diff"


def test_commit_diff_survives_poisoned_diff_external(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_SHOW"
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "diff.external", f"touch {marker} ;"],
        check=True,
    )
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    sha = commit_all(tmp_path, "change readme")
    out = commit_diff(tmp_path, sha)
    assert not marker.exists(), "repo diff.external command executed on the host"
    assert "README.md" in out, "git show honored the external diff and lost the patch"


def _poison_textconv(repo: Path, marker: Path) -> None:
    # A per-file textconv driver bound via .gitattributes. `-c diff.external=`
    # and `--no-ext-diff` do NOT disable textconv; only `--no-textconv` does.
    subprocess.run(
        ["git", "-C", str(repo), "config", "diff.pwn.textconv", f"touch {marker} ; cat"],
        check=True,
    )
    (repo / ".gitattributes").write_text("* diff=pwn\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", ".gitattributes"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "attrs"], check=True)


def test_git_ops_neutralizes_repo_diff_textconv(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_TEXTCONV_DIFF"
    _poison_textconv(tmp_path, marker)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    out = diff_since(tmp_path, "HEAD")
    assert not marker.exists(), "repo diff.*.textconv command executed on the host"
    assert "README.md" in out, "textconv neutralization silently emptied the diff"


def test_commit_diff_survives_poisoned_textconv(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    marker = tmp_path / "PWNED_TEXTCONV_SHOW"
    _poison_textconv(tmp_path, marker)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    sha = commit_all(tmp_path, "change readme")
    out = commit_diff(tmp_path, sha)
    assert not marker.exists(), "git show ran the repo textconv driver on the host"
    assert "README.md" in out, "git show lost the patch under --no-textconv"


def test_git_ops_neutralizes_repo_gpg_signing(tmp_path: Path) -> None:
    """A repo-controlled gpg.program + commit.gpgsign=true must NOT execute the
    configured (arbitrary host) program on agent6's own commit."""
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
    commit_all(tmp_path, "c")  # must not raise or fire the signing program
    assert not marker.exists(), "repo gpg.program executed on the host during commit"


def test_slugify_basic() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("") == "run"
    assert slugify("a" * 100).startswith("a" * 40)
    assert len(slugify("a" * 100)) == 40


def test_make_run_branch_name_format() -> None:
    name = make_run_branch_name()
    assert name.startswith("agent6/")


def test_is_git_repo_false_for_tmp(tmp_path: Path) -> None:
    assert is_git_repo(tmp_path) is False


def test_init_repo_creates_repository(tmp_path: Path) -> None:
    assert is_git_repo(tmp_path) is False
    init_repo(tmp_path)
    assert is_git_repo(tmp_path) is True


def test_unignored_filters_gitignored_paths(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".agent6/\n", encoding="utf-8")
    (tmp_path / ".agent6").mkdir()
    (tmp_path / ".agent6" / "config.toml").write_text("x", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("y", encoding="utf-8")
    keep = unignored(tmp_path, (".agent6/config.toml", "AGENTS.md", ".gitignore"))
    assert ".agent6/config.toml" not in keep  # gitignored
    assert "AGENTS.md" in keep
    assert ".gitignore" in keep


def test_status_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    st = status(tmp_path)
    assert st.branch == "main"
    assert st.is_clean is True
    assert st.modified_count == 0


def test_status_dirty_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")
    st = status(tmp_path)
    assert st.is_clean is False
    assert st.untracked_count == 1


def test_commit_all_returns_sha_and_log(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    sha = commit_all(tmp_path, "add f", trailers={"agent6-step": "x"})
    assert len(sha) == 40
    log = recent_log(tmp_path, n=5)
    assert "add f" in log


def test_create_branch(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    create_branch(tmp_path, "agent6/test")
    assert status(tmp_path).branch == "agent6/test"


def test_create_branch_at_is_additive_no_checkout(tmp_path: Path) -> None:
    """create_branch_at points a new branch at a sha WITHOUT moving HEAD."""
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    # Advance HEAD so base != current; the fork branch must land on base.
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    commit_all(tmp_path, "second")
    create_branch_at(tmp_path, "agent6/fork", base)
    assert status(tmp_path).branch == "main", "must not check out the new branch"
    forked = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "agent6/fork"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert forked == base


def test_create_branch_at_idempotent_and_refuses_move(tmp_path: Path) -> None:
    """A no-op when the branch already points at the sha; a GitError if it points
    elsewhere (moving a branch would be a rewrite, which we refuse)."""
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    create_branch_at(tmp_path, "agent6/fork", base)
    create_branch_at(tmp_path, "agent6/fork", base)  # idempotent: no raise
    (tmp_path / "c.txt").write_text("c\n", encoding="utf-8")
    other = commit_all(tmp_path, "third")
    with pytest.raises(GitError):
        create_branch_at(tmp_path, "agent6/fork", other)


def test_git_ops_never_spells_a_destructive_verb() -> None:
    """The hard rule, at argv level: no function in git_ops passes push,
    --force, reset --hard, rebase, amend, filter-branch, or branch -D to git.
    The one sanctioned exception is force_delete_squash_merged_branch's
    `branch -D` (operator-only, content-safe, never LLM-reachable).

    This replaces three refuse_* helpers that no production code ever called:
    they raised on demand in a test while git_ops could have grown a real
    `push` beside them, green."""
    from agent6 import git_ops as gm

    src = Path(gm.__file__).read_text(encoding="utf-8")
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


def test_status_on_non_repo(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        status(tmp_path)


def test_verify_git_identity_uses_repo_config(tmp_path: Path) -> None:
    _init_repo(tmp_path)  # configures user.name=t, user.email=t@t
    name, email = verify_git_identity(tmp_path, CommitIdentity())
    assert name == "t"
    assert email == "t@t"


def test_verify_git_identity_override_wins(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    name, email = verify_git_identity(tmp_path, CommitIdentity(name="bot", email="bot@example.com"))
    assert name == "bot"
    assert email == "bot@example.com"


def test_verify_git_identity_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate from any global/system git identity by pointing the global
    # config at an empty file.
    empty_cfg = tmp_path / "empty.gitconfig"
    empty_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_cfg))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_cfg))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    with pytest.raises(GitError, match="Git identity not configured"):
        verify_git_identity(repo, CommitIdentity())


def test_commit_all_with_identity_overrides_author(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("y", encoding="utf-8")
    sha = commit_all(
        tmp_path,
        "add f",
        identity=CommitIdentity(
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


def test_reset_to_soft_keeps_index_and_worktree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    start = status(tmp_path).head_sha
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    sha1 = commit_all(tmp_path, "add a")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    sha2 = commit_all(tmp_path, "add b")
    assert sha2 != start
    reset_to(tmp_path, start, mode="soft")
    assert status(tmp_path).head_sha == start
    # Worktree files survive.
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b\n"
    # Soft reset leaves changes STAGED.
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert set(staged) == {"a.txt", "b.txt"}
    # Orphaned commit object is still alive (reflog keeps it from gc).
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "cat-file", "-t", sha1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == "commit"
    )


def test_reset_to_mixed_unstages(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    start = status(tmp_path).head_sha
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    commit_all(tmp_path, "add a")
    reset_to(tmp_path, start, mode="mixed")
    assert status(tmp_path).head_sha == start
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\n"
    # Mixed reset leaves changes UNSTAGED (file shows as untracked).
    assert status(tmp_path).untracked_count == 1


def test_reset_to_rejects_hard(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    with pytest.raises(GitError, match="mode must be 'soft' or 'mixed'"):
        reset_to(tmp_path, status(tmp_path).head_sha, mode="hard")


def test_commit_error_surfaces_stdout_when_stderr_empty(tmp_path: Path) -> None:
    """`git commit` writes "nothing to commit, working tree
    clean" to STDOUT, not stderr. `_run` only captured
    stderr, producing error strings like "git commit -m X failed: "
    with no useful detail. The new behaviour must include stdout when
    stderr is empty so the operator gets actionable signal."""
    _init_repo(tmp_path)
    # `commit_all` will stage a no-op and call `git commit`, which exits
    # 1 with "nothing to commit, working tree clean" on STDOUT.
    with pytest.raises(GitError) as excinfo:
        commit_all(tmp_path, "no-op commit on clean repo")
    msg = str(excinfo.value)
    # The detail (from stdout) must be present so callers can pattern-match.
    assert "nothing to commit" in msg.lower()
    # And the prefix must still identify which git invocation failed.
    assert "git commit" in msg


def test_restore_stash_clean_apply_restores_and_drops(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "wip.txt").write_text("work in progress\n", encoding="utf-8")
    stash_all(tmp_path, "agent6 auto-stash")
    assert not (tmp_path / "wip.txt").exists()  # stashed away, tree clean
    entry = find_stash(tmp_path, "agent6 auto-stash")
    assert entry is not None
    assert restore_stash(tmp_path, entry) is True
    assert (tmp_path / "wip.txt").read_text(encoding="utf-8") == "work in progress\n"
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert listing.strip() == ""  # dropped after a clean apply


def test_restore_stash_conflict_keeps_stash(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("stashed change\n", encoding="utf-8")
    stash_all(tmp_path, "agent6 auto-stash")
    # A conflicting commit on the same line means the stash cannot apply cleanly.
    (tmp_path / "README.md").write_text("committed change\n", encoding="utf-8")
    commit_all(tmp_path, "conflicting commit")
    entry = find_stash(tmp_path, "agent6 auto-stash")
    assert entry is not None
    assert restore_stash(tmp_path, entry) is False
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "agent6 auto-stash" in listing  # preserved, never dropped on conflict


def test_find_stash_targets_the_run_stash_not_the_latest(tmp_path: Path) -> None:
    """A stash pushed DURING the run sits at stash@{0}; the old positional
    restore popped it (the wrong work) and left the pre-run work hidden. The
    run's stash is found by its run-id message and restored; the other stash
    is untouched."""
    _init_repo(tmp_path)
    (tmp_path / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    (tmp_path / "mid.txt").write_text("mid-run stash by someone else\n", encoding="utf-8")
    stash_all(tmp_path, "user work stashed mid-run")
    entry = find_stash(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None
    assert entry.ref == "stash@{1}"
    assert restore_stash(tmp_path, entry) is True
    assert (tmp_path / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    assert not (tmp_path / "mid.txt").exists()  # the mid-run stash stays a stash
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "user work stashed mid-run" in listing
    assert "agent6 auto-stash" not in listing


def test_restore_stash_raced_drop_puts_the_bystander_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git stash drop` takes only a POSITION (git refuses a sha outright), so a
    stash pushed between the list that resolves ours and the drop shifts the
    stack and the drop takes a bystander's entry. Its commit must come back."""
    _init_repo(tmp_path)
    (tmp_path / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    entry = find_stash(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None

    real_run = git_ops._run  # pyright: ignore[reportPrivateUsage]
    raced = False

    def racing_run(
        path: Path, *args: str, check: bool = True, env_extra: dict[str, str] | None = None
    ) -> git_ops.CommandResult:
        nonlocal raced
        res = real_run(path, *args, check=check, env_extra=env_extra)
        if not raced and args[:2] == ("stash", "list"):
            raced = True  # ours slides to stash@{1}; the recorded ref now names theirs
            (tmp_path / "README.md").write_text("bystander work\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(tmp_path), "stash", "push", "-q", "-m", "bystander", "--"],
                check=True,
            )
        return res

    monkeypatch.setattr(git_ops, "_run", racing_run)
    assert restore_stash(tmp_path, entry) is True
    assert (tmp_path / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "bystander" in listing  # never silently destroyed
    # Ours survives too: re-resolving to drop it again would race the same way,
    # so a raced restore leaks its own stash rather than risk a second bystander.
    assert auto_stash_message("sunny-otter-AAA111") in listing


def test_find_stash_does_not_prefix_match_another_runs_stash(tmp_path: Path) -> None:
    """Lane run ids are ordinal (`…-l1`, `…-l10`), so one run's auto-stash
    message is a PREFIX of another's stash subject; the substring lookup
    returned the newer `-l10` stash when asked for `-l1`, and finalization
    then applied and dropped the wrong work. The lookup must match the
    pushed message exactly."""
    _init_repo(tmp_path)
    (tmp_path / "one.txt").write_text("lane l1 work\n", encoding="utf-8")
    stash_all(tmp_path, auto_stash_message("fanout-l1"))
    (tmp_path / "ten.txt").write_text("lane l10 work\n", encoding="utf-8")
    stash_all(tmp_path, auto_stash_message("fanout-l10"))  # newer: listed first
    entry = find_stash(tmp_path, auto_stash_message("fanout-l1"))
    assert entry is not None
    assert entry.ref == "stash@{1}"  # the l1 stash, not the newer l10 one


def test_raced_drop_failed_putback_raises_with_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the raced drop took a bystander's stash AND storing it back fails
    (e.g. the stash ref lock is held by the very process that raced us), the
    bystander's entry is gone from the list. That loss must not pass silently:
    the restore raises, naming the orphaned commit and the exact
    `git stash store` command that puts it back."""
    _init_repo(tmp_path)
    (tmp_path / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    entry = find_stash(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None

    real_run = git_ops._run  # pyright: ignore[reportPrivateUsage]
    raced = False

    def racing_run(
        path: Path, *args: str, check: bool = True, env_extra: dict[str, str] | None = None
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
                ["git", "-C", str(tmp_path), "stash", "push", "-q", "-m", "bystander", "--"],
                check=True,
            )
        return res

    monkeypatch.setattr(git_ops, "_run", racing_run)
    with pytest.raises(GitError) as excinfo:
        restore_stash(tmp_path, entry)
    msg = str(excinfo.value)
    assert "git stash store" in msg  # the recovery command, ready to paste
    assert "bystander" in msg  # names whose work went missing


def test_git_runs_under_a_pinned_locale(tmp_path: Path) -> None:
    """The bystander rescue reads git's own sentence ("Dropped stash@{0} (sha)")
    to learn what it just dropped, and git translates that. On a host with git
    l10n and a non-English LANG the match failed AFTER the drop had happened,
    so the bystander's stash was destroyed with no record of it."""
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
    """Popen stand-in for a git stuck past the timeout: the first
    communicate() times out (creating *lock* if given -- a lock appearing
    mid-window); terminate() exits it, unless *ignores_term* (wedged
    uninterruptible), where only kill() reaps it."""

    def __init__(self, *, lock: Path | None = None, ignores_term: bool = False) -> None:
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


def test_timeout_terminates_first_and_leaves_a_survivor_lock(tmp_path: Path) -> None:
    """A timed-out git gets SIGTERM, and git's TERM handler removes its own
    lockfiles; SIGKILL only follows an ignored TERM. After a graceful TERM
    exit a lock still on disk is a concurrent git's (git cleaned its own), so
    nothing is deleted."""
    _init_repo(tmp_path)
    lock = tmp_path / ".git" / "index.lock"
    fake = _HungGit(lock=lock)

    def fake_popen(*_a: object, **_k: object) -> _HungGit:
        return fake

    with (
        mock.patch.object(git_ops.subprocess, "Popen", fake_popen),
        pytest.raises(GitError, match="timed out"),
    ):
        git_ops._run(tmp_path, "commit", "-m", "x", check=False)  # pyright: ignore[reportPrivateUsage]
    assert fake.calls == ["terminate"]  # graceful exit: never escalated to KILL
    assert lock.exists()  # a survivor lock after TERM is not ours to delete


def test_timeout_keeps_a_preexisting_index_lock(tmp_path: Path) -> None:
    """A lock that already existed when the timed-out git was spawned belongs
    to a CONCURRENT git process (operator shell, another lane); deleting it
    would break git's index mutual exclusion, even on the SIGKILL path."""
    _init_repo(tmp_path)
    lock = tmp_path / ".git" / "index.lock"
    lock.write_text("held by a concurrent git\n", encoding="utf-8")
    fake = _HungGit(ignores_term=True)

    def fake_popen(*_a: object, **_k: object) -> _HungGit:
        return fake

    with (
        mock.patch.object(git_ops.subprocess, "Popen", fake_popen),
        pytest.raises(GitError, match="timed out"),
    ):
        git_ops._run(tmp_path, "status", check=False)  # pyright: ignore[reportPrivateUsage]
    assert lock.exists()  # never ours to delete


def test_timeout_clears_the_lock_its_own_child_created(tmp_path: Path) -> None:
    """A child that ignores TERM (a wedged filesystem) is SIGKILLed, skipping
    git's own lockfile cleanup; a lock that appeared under this child is
    cleared so the run's remaining git ops recover."""
    _init_repo(tmp_path)
    lock = tmp_path / ".git" / "index.lock"
    fake = _HungGit(lock=lock, ignores_term=True)

    def fake_popen(*_a: object, **_k: object) -> _HungGit:
        return fake

    with (
        mock.patch.object(git_ops.subprocess, "Popen", fake_popen),
        pytest.raises(GitError, match="timed out"),
    ):
        git_ops._run(tmp_path, "commit", "-m", "x", check=False)  # pyright: ignore[reportPrivateUsage]
    assert fake.calls == ["terminate", "kill"]  # TERM tried before KILL
    assert not lock.exists()  # the dead child's own lock is cleared


def test_find_stash_missing_returns_none(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    assert find_stash(tmp_path, auto_stash_message("nope")) is None


def test_restore_stash_survives_index_shift_after_lookup(tmp_path: Path) -> None:
    """A stash pushed AFTER the lookup shifts every stash@{N}, so restoring by
    the recorded position applied the wrong stash (and dropped it). The entry
    is applied and dropped by its sha, resolved fresh at drop time."""
    _init_repo(tmp_path)
    (tmp_path / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    entry = find_stash(tmp_path, auto_stash_message("sunny-otter-AAA111"))
    assert entry is not None and entry.ref == "stash@{0}"
    # The shift: a stash pushed after the lookup makes the recorded position
    # point at someone else's work.
    (tmp_path / "mid.txt").write_text("mid work\n", encoding="utf-8")
    stash_all(tmp_path, "pushed after lookup")
    assert restore_stash(tmp_path, entry) is True
    assert (tmp_path / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    assert not (tmp_path / "mid.txt").exists()  # the other stash stays a stash
    listing = subprocess.run(
        ["git", "-C", str(tmp_path), "stash", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert "pushed after lookup" in listing
    assert "agent6 auto-stash" not in listing  # ours was dropped, by identity


def _commit_file(repo: Path, name: str, content: str, msg: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", msg], check=True)
    return status(repo).head_sha


def test_plumb_merge_lands_without_touching_the_checkout_medium(tmp_path: Path) -> None:
    """A no-ff merge is pure ref plumbing: the target branch gains a two-parent
    commit carrying the trailer once, and a worktree already holding the run's
    files (as after every run) is no obstacle -- afterwards `git status` shows
    no phantom dirt because the index was brought forward."""
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    run_tip = _lane_commit(tmp_path, base, "feat.txt", "x\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/agent6/r1", run_tip], check=True
    )
    (tmp_path / "feat.txt").write_text("x\n", encoding="utf-8")  # worktree carries the work

    res = plumb_merge(
        tmp_path,
        "main",
        "refs/agent6/r1",
        strategy="merge",
        message="Merge refs/agent6/r1",
        identity=CommitIdentity(trailer="Assisted-by: agent6:m1"),
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
    assert status(tmp_path).is_clean  # main is checked out: index brought forward


def test_plumb_merge_conflict_moves_nothing(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "README.md", "run change\n")
    _commit_file(tmp_path, "README.md", "main change\n", "main edit")  # same line
    main_tip = status(tmp_path).head_sha
    res = plumb_merge(tmp_path, "main", theirs, strategy="merge", message=None)
    assert res.conflicted
    assert "README.md" in res.conflicts
    assert _rev(tmp_path, "main") == main_tip  # nothing moved
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "main change\n"


def test_plumb_merge_ff_moves_the_ref_and_refuses_divergence(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "feat.txt", "x\n")
    (tmp_path / "feat.txt").write_text("x\n", encoding="utf-8")
    res = plumb_merge(tmp_path, "main", theirs, strategy="ff")
    assert res.merged_sha == theirs == _rev(tmp_path, "main")
    assert status(tmp_path).is_clean

    diverged = _lane_commit(tmp_path, base, "other.txt", "y\n")
    with pytest.raises(GitError):
        plumb_merge(tmp_path, "main", diverged, strategy="ff")
    assert _rev(tmp_path, "main") == theirs  # a refused ff moves nothing


def test_plumb_merge_squash_is_one_commit_and_noop_when_contained(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    step1 = _lane_commit(tmp_path, base, "a.txt", "a\n")
    step2 = _lane_commit(tmp_path, step1, "b.txt", "b\n")
    for name, content in (("a.txt", "a\n"), ("b.txt", "b\n")):
        (tmp_path / name).write_text(content, encoding="utf-8")
    res = plumb_merge(
        tmp_path,
        "main",
        step2,
        strategy="squash",
        message="the task",
        identity=CommitIdentity(trailer="Assisted-by: agent6:m1"),
    )
    assert not res.conflicted
    parents = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%P", res.merged_sha],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert parents == [base]  # ONE commit, single parent
    assert status(tmp_path).is_clean
    # Landing the same rev again is a clean no-op returning the unchanged tip.
    again = plumb_merge(tmp_path, "main", step2, strategy="squash", message="again")
    assert again.merged_sha == res.merged_sha and not again.conflicted


def test_plumb_merge_preserves_the_operators_own_staging(tmp_path: Path) -> None:
    """Index entries the operator staged themselves survive the bring-forward
    exactly as staged; only entries still matching the old tip move."""
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    theirs = _lane_commit(tmp_path, base, "feat.txt", "x\n")
    (tmp_path / "feat.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("operator wip\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    res = plumb_merge(tmp_path, "main", theirs, strategy="squash", message="land")
    assert not res.conflicted
    staged = subprocess.run(
        ["git", "-C", str(tmp_path), "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert staged == ["README.md"]  # their staging intent, nothing else


def test_clone_repo_local_clone(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    dest = tmp_path / "clone"
    clone_repo(origin, dest)
    assert (dest / "README.md").read_text(encoding="utf-8") == "hi\n"
    assert status(dest).head_sha == status(origin).head_sha


def test_clone_repo_raises_on_existing_dest(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    dest = tmp_path / "clone"
    dest.mkdir()
    (dest / "occupied.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(GitError):
        clone_repo(origin, dest)


def test_fetch_branch_lands_branch_from_path(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    clone = tmp_path / "clone"
    clone_repo(origin, clone)
    create_branch(clone, "agent6/r1")
    _commit_file(clone, "feat.txt", "x\n", "add feat")
    fetch_branch(origin, clone, "agent6/r1:agent6/r1")
    sha = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "refs/heads/agent6/r1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert sha == status(clone).head_sha


def test_fetch_branch_refuses_non_ff_update(tmp_path: Path) -> None:
    # A plain (unforced) refspec must not clobber an existing diverged branch.
    origin = tmp_path / "origin"
    origin.mkdir()
    _init_repo(origin)
    create_branch(origin, "agent6/r1")
    _commit_file(origin, "ours.txt", "o\n", "origin r1")
    origin_sha = status(origin).head_sha
    create_branch(origin, "main")
    clone = tmp_path / "clone"
    clone_repo(origin, clone)
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "-q", "-b", "agent6/r1", "main"], check=True
    )
    _commit_file(clone, "theirs.txt", "t\n", "clone r1")  # diverged from origin's r1
    with pytest.raises(GitError):
        fetch_branch(origin, clone, "agent6/r1:agent6/r1")
    kept = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "refs/heads/agent6/r1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert kept == origin_sha  # origin's branch untouched


def test_list_run_commits_oldest_first(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    create_branch(tmp_path, "agent6/r1")
    _commit_file(tmp_path, "a.txt", "a\n", "agent6 iter 1: add a")
    _commit_file(tmp_path, "b.txt", "b\n", "agent6 iter 2: add b")
    rows = list_run_commits(tmp_path, base, "agent6/r1")
    assert [r.subject for r in rows] == ["agent6 iter 1: add a", "agent6 iter 2: add b"]


def test_condense_strips_prefix_and_bullets(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    create_branch(tmp_path, "agent6/r1")
    _commit_file(tmp_path, "a.txt", "a\n", "agent6 iter 1: add a")
    _commit_file(tmp_path, "b.txt", "b\n", "agent6 iter 2: add b")
    rows = list_run_commits(tmp_path, base, "agent6/r1")
    message = condense_commit_message(rows, subject="implement parse_url")
    assert message.splitlines()[0] == "implement parse_url"  # headline = the task
    assert "- add a" in message and "- add b" in message  # prefix stripped, bulleted


def test_condense_subject_is_first_clause_and_wraps_the_rest(tmp_path: Path) -> None:
    # A long multi-clause task must not become one 180-char subject line: the
    # subject is the first clause (<= 72 chars); the whole task wraps into the body.
    task = (
        "Add a --limit flag to runs list. Then update the parser help and add a "
        "focused unit test covering the newest-N slice and the argcomplete choices"
    )
    message = condense_commit_message((), subject=task)
    lines = message.splitlines()
    assert lines[0] == "Add a --limit flag to runs list"  # first clause only
    assert all(len(ln) <= 72 for ln in lines)  # nothing over the subject/body cap
    assert "focused unit test" in message  # the rest is preserved, wrapped


def test_condense_subject_truncates_a_clauseless_run_on_with_ellipsis(tmp_path: Path) -> None:
    task = "make the thing " * 20  # 300 chars, no sentence break
    message = condense_commit_message((), subject=task)
    subject = message.splitlines()[0]
    assert len(subject) <= 72 and subject.endswith("…")


def test_commit_all_appends_the_identity_trailer_once(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    line = "Assisted-by: agent6:m1 (worker)"
    commit_all(tmp_path, "add f", identity=CommitIdentity(trailer=line))
    head_msg = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert head_msg.count(line) == 1
    assert head_msg.splitlines()[0] == "add f"


def test_render_commit_trailer_joins_the_code_writers() -> None:
    from agent6.git_ops import render_commit_trailer

    assert render_commit_trailer("", models=("m",)) is None
    got = render_commit_trailer("Assisted-by: agent6:{model}", models=("m1",))
    assert got == "Assisted-by: agent6:m1"
    # Several contributing models join first-seen order, deduplicated, blanks
    # dropped: the primary worker stays first.
    got = render_commit_trailer("Assisted-by: agent6:{model}", models=("m1", "", "m2", "m1"))
    assert got == "Assisted-by: agent6:m1, m2"


def test_diff_of_non_utf8_file_does_not_crash(tmp_path: Path) -> None:
    # git diff/show emit raw file bytes; a latin-1 text file (no NULs, so git
    # does not treat it as binary) put non-UTF-8 bytes in the output, and the
    # strict text=True decode raised UnicodeDecodeError mid-run with no session.end.
    # Both diff surfaces must return a (lossily-decoded) string instead.
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    (tmp_path / "latin1.txt").write_bytes(b"caf\xe9 au lait\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "add latin1"], check=True)
    sha = status(tmp_path).head_sha

    since = diff_since(tmp_path, base)
    assert "latin1.txt" in since  # the diff was produced, not dropped to a crash
    shown = commit_diff(tmp_path, sha)
    assert "latin1.txt" in shown


def test_list_run_commits_preserves_body_with_separator_bytes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    base = status(tmp_path).head_sha
    create_branch(tmp_path, "agent6/r1")
    body = "line A \x1f mid \x1e end\n\nCo-authored-by: Op <op@x>"
    _commit_file(tmp_path, "a.txt", "a\n", f"agent6 iter 1: add a\n\n{body}")
    rows = list_run_commits(tmp_path, base, "agent6/r1")
    assert len(rows) == 1  # \x1e in the body did not split it into two records
    assert "Co-authored-by: Op <op@x>" in rows[0].message  # \x1f did not truncate the body


def test_conventional_subject_derives_type_and_scope() -> None:
    from agent6.git_ops import conventional_commit_subject

    # All test files -> test; scope from the common first dir under src/ when
    # source is touched, else the common top-level dir.
    assert conventional_commit_subject(
        [("M", "tests/unit/test_a.py"), ("M", "tests/unit/test_b.py")],
        summary="cover the resolver",
    ).startswith("test(unit): cover the resolver")
    # All docs -> docs.
    assert conventional_commit_subject(
        [("M", "docs/config.md"), ("M", "README.md")], summary="explain the lock"
    ).startswith("docs: explain the lock")
    # An added source file -> feat, scoped by the package dir.
    got = conventional_commit_subject(
        [("A", "src/agent6/config/write.py"), ("M", "src/agent6/config/model.py")],
        summary="one write path",
    )
    assert got == "feat(config): one write path"
    # Modified-only source -> fix.
    got = conventional_commit_subject(
        [("M", "src/agent6/git_ops.py")], summary="Trailer emitted once."
    )
    assert got == "fix(git_ops): trailer emitted once"
    # No changes at all still yields a valid subject.
    assert conventional_commit_subject([], summary="tidy") == "chore: tidy"


def _rev(path: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", ref], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_chain_commit_touches_no_head_index_or_checkout(tmp_path: Path) -> None:
    """The detached chain records the worktree without moving HEAD, without
    reading or writing the shared index, and without a checkout: the run's
    commits land on refs/agent6/<id> (and the visible branch ref when asked)
    while the operator's staged change survives byte-for-byte."""
    from agent6.git_ops import chain_commit

    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("edited\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)

    sha = chain_commit(
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


def test_chain_commit_skips_identical_trees_and_survives_branch_switches(
    tmp_path: Path,
) -> None:
    """An unchanged worktree records nothing (None), and the chain keeps its
    own parentage when the model or the operator switches branches mid-run."""
    from agent6.git_ops import chain_commit

    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    s1 = chain_commit(tmp_path, "agent6 iter 1", ref="refs/agent6/t1", fallback_parent=head0)
    assert s1 is not None
    assert (
        chain_commit(tmp_path, "agent6 iter 2", ref="refs/agent6/t1", fallback_parent=None) is None
    )

    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-q", "-b", "model-branch"], check=True)
    (tmp_path / "c.txt").write_text("three\n", encoding="utf-8")
    s2 = chain_commit(tmp_path, "agent6 iter 3", ref="refs/agent6/t1", fallback_parent=None)
    assert s2 is not None
    assert _rev(tmp_path, f"{s2}^") == s1
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branch == "model-branch"


def test_chain_commit_root_and_trailer(tmp_path: Path) -> None:
    """No ref and no fallback makes a root commit, and the identity trailer lands once."""
    from agent6.git_ops import CommitIdentity, chain_commit

    _init_repo(tmp_path)
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    sha = chain_commit(
        tmp_path,
        "agent6 iter 1",
        ref="refs/agent6/t2",
        fallback_parent=None,
        identity=CommitIdentity(name="A", email="a@a", trailer="Assisted-by: agent6:m1"),
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


def _lane_commit(path: Path, parent: str, name: str, content: str) -> str:
    """A commit adding *name* on top of *parent* built with plumbing only, like
    an imported lane tip: it exists in the odb without any checkout."""
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


def test_chain_merge_records_the_lane_and_syncs_the_worktree(tmp_path: Path) -> None:
    """A clean lane merge lands as a two-parent chain commit and the lane's
    files appear in the worktree, while HEAD and the operator's checkout stay
    untouched; an already-contained rev is a no-op returning the tip."""
    from agent6.git_ops import chain_commit, chain_merge

    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    s1 = chain_commit(tmp_path, "agent6 iter 1", ref="refs/agent6/t3", fallback_parent=head0)
    assert s1 is not None
    lane = _lane_commit(tmp_path, head0, "c.txt", "lane\n")

    merged = chain_merge(tmp_path, lane, "merge lane", ref="refs/agent6/t3")
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
    assert chain_merge(tmp_path, head0, "again", ref="refs/agent6/t3") == merged


def test_chain_merge_conflict_leaves_chain_and_worktree_alone(tmp_path: Path) -> None:
    """A textual conflict returns None: the ref keeps its tip and no file in
    the worktree is rewritten (the coordinator reports, the model resolves)."""
    from agent6.git_ops import chain_commit, chain_merge

    _init_repo(tmp_path)
    head0 = _rev(tmp_path, "HEAD")
    (tmp_path / "b.txt").write_text("ours\n", encoding="utf-8")
    s1 = chain_commit(tmp_path, "agent6 iter 1", ref="refs/agent6/t4", fallback_parent=head0)
    lane = _lane_commit(tmp_path, head0, "b.txt", "theirs\n")

    assert chain_merge(tmp_path, lane, "merge lane", ref="refs/agent6/t4") is None
    assert _rev(tmp_path, "refs/agent6/t4") == s1
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "ours\n"
