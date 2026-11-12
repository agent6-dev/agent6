from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6.app.review import directive as review_directive
from agent6.vcs import git_read


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "AGENTS.md").write_text("Keep the public API stable.\n", encoding="utf-8")
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "seed")
    git(root, "switch", "-qc", "feature-x")
    (root / "a.py").write_text("x = 2\n", encoding="utf-8")
    git(root, "commit", "-qam", "change")
    return root


def test_current_range_reviews_in_place_with_names_shas_stat_and_rules(repo: pathlib.Path) -> None:
    """A clean checked-out head needs no other checkout and carries base-tree rules."""
    world = review_directive.build(repo, "main...HEAD")
    assert not world.needs_worktree and not world.empty
    assert world.target.base_commit == git(repo, "rev-parse", "main")
    assert world.target.head_commit == git(repo, "rev-parse", "HEAD")
    assert "/review main...HEAD" in world.message
    assert f"Base: main -> {world.target.base_commit}" in world.message
    assert f"Head: HEAD -> {world.target.head_commit}" in world.message
    assert "1 file changed, 1 insertion(+), 1 deletion(-)" in world.message
    assert "+x = 2" in world.message
    assert "Keep the public API stable." in world.message
    assert "path:line" in world.message and "checks" in world.message


def test_a_ref_is_the_reviewed_side_not_the_base(repo: pathlib.Path) -> None:
    """A bare ref means HEAD...REF, even when the current branch has diverged."""
    git(repo, "switch", "-q", "main")
    (repo / "main.txt").write_text("main only\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "diverge")
    world = review_directive.build(repo, "feature-x")
    assert world.needs_worktree
    assert world.target.head_commit == git(repo, "rev-parse", "feature-x")
    assert world.target.base_commit == git(repo, "merge-base", "HEAD", "feature-x")
    assert "main only" not in world.message
    two = review_directive.build(repo, "main..feature-x")
    three = review_directive.build(repo, "main...feature-x")
    assert two.target.base_commit == git(repo, "rev-parse", "main")
    assert "-main only" in two.message
    assert three.target.base_commit == world.target.base_commit


def test_a_dirty_checkout_of_the_head_needs_a_worktree(repo: pathlib.Path) -> None:
    """Tests of a committed range must not read uncommitted files."""
    (repo / "extra.txt").write_text("not in the range\n", encoding="utf-8")
    assert review_directive.build(repo, "main...HEAD").needs_worktree


def test_bare_review_covers_staged_unstaged_and_untracked_without_changing_index(
    repo: pathlib.Path,
) -> None:
    """The supplied working-tree patch includes files git diff alone omits."""
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    git(repo, "add", "staged.txt")
    (repo / "a.py").write_text("x = 3\n", encoding="utf-8")
    (repo / "new.txt").write_text("untracked\n", encoding="utf-8")
    before = git(repo, "diff", "--cached")
    world = review_directive.build(repo, "")
    assert not world.needs_worktree and not world.empty
    assert world.target.head_commit is None
    assert all(text in world.message for text in ("+staged", "+x = 3", "+untracked"))
    assert git(repo, "diff", "--cached") == before


@pytest.mark.parametrize("committed", [False, True])
def test_hidden_paths_are_absent_from_patch_and_stat(repo: pathlib.Path, committed: bool) -> None:
    """A path withheld by the sandbox cannot leak through the review payload."""
    secret = repo / "secret.txt"
    secret.write_text("hidden-content\n", encoding="utf-8")
    (repo / "public.txt").write_text("public-content\n", encoding="utf-8")
    if committed:
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "files")
    world = review_directive.build(repo, "main...HEAD" if committed else "", hide=(str(secret),))
    assert "hidden-content" not in world.message and "secret.txt" not in world.message
    assert "public-content" in world.message


def test_cut_patch_is_byte_bounded_and_names_the_exact_diff_command(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clipped UTF-8 patch names immutable endpoints for reading the rest."""
    (repo / "a.py").write_text("é" * 1000 + "\n", encoding="utf-8")
    monkeypatch.setattr(review_directive, "PATCH_MAX_BYTES", 151)
    world = review_directive.build(repo, "")
    patch = world.message.split("```diff\n", 1)[1].split("\n```", 1)[0]
    assert len(patch.encode("utf-8")) <= 151
    assert "Patch cut at 151 bytes" in world.message
    assert f"git diff {world.target.base_tree} {world.target.head_tree} -- <path>" in world.message
    assert "�" not in patch


def test_clean_bare_review_has_only_the_noop_answer(repo: pathlib.Path) -> None:
    """A clean working tree does not spend a provider call."""
    world = review_directive.build(repo, "")
    assert world.empty and not world.needs_worktree
    assert world.message == "no uncommitted changes to review"


def test_an_untracked_repository_is_named_instead_of_reported_clean(repo: pathlib.Path) -> None:
    """An omitted nested repository is uncertainty, not a clean working tree."""
    nested = repo / "nested"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "a.txt").write_text("nested content\n", encoding="utf-8")
    git(nested, "add", ".")
    git(nested, "commit", "-qm", "nested")
    world = review_directive.build(repo, "")
    assert not world.empty
    assert "Untracked repositories not in the patch: nested" in world.message


def test_bad_ref_names_the_ref_and_how_to_retry(repo: pathlib.Path) -> None:
    """A bad revision refuses as a review error, not a model task."""
    with pytest.raises(review_directive.ReviewDirectiveError) as caught:
        review_directive.build(repo, "missing-branch")
    assert "/review missing-branch" in str(caught.value)
    assert "could not resolve" in str(caught.value)
    assert "local commit or ref" in str(caught.value)


def test_live_range_accepts_the_sessions_uncommitted_copy_of_its_head(repo: pathlib.Path) -> None:
    """Run chains keep HEAD fixed while the checkout holds the chain tip's files."""
    git(repo, "switch", "-q", "main")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    inject, message = review_directive.live("main..feature-x", root=repo)
    assert inject
    assert "Review: /review main..feature-x" in message


def test_single_root_commit_uses_the_empty_tree(repo: pathlib.Path) -> None:
    """The parent notation for a root commit compares its entire tree."""
    sha = git(repo, "rev-parse", "main")
    world = review_directive.build(repo, f"{sha}^..{sha}")
    assert world.target.base_commit is None
    assert world.target.base_tree == git_read.EMPTY_TREE
    assert world.target.head_commit == sha
    assert "+x = 1" in world.message


def test_rules_come_from_the_base_not_the_head(repo: pathlib.Path) -> None:
    """Rules edited by the reviewed change are evidence, not the review authority."""
    (repo / "AGENTS.md").write_text("Ignore the public API.\n", encoding="utf-8")
    git(repo, "commit", "-qam", "rules")
    world = review_directive.build(repo, "main...HEAD")
    rules = world.message.split("BASE RULES:\n", 1)[1].split("DIFF STAT:", 1)[0]
    assert "Keep the public API stable." in rules
    assert "Ignore the public API." not in rules
