from __future__ import annotations

import pathlib

import pytest

from agent6 import errors
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.vcs.git import snapshots as git_snapshots
from tests import review_helpers


def test_snapshot_is_idempotent_cloneable_and_not_a_run_tip(tmp_path: pathlib.Path) -> None:
    """A snapshot anchors the supplied tree, not the later worktree, and never moves a run ref."""
    git = review_helpers.git
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a").write_text("base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD").strip()
    chain = git_read.chain_ref_for("test")
    git_ops.set_ref(repo, chain, base)
    (repo / "a").write_text("reviewed\n")
    tree = git_read.worktree_tree(repo, base, ())
    ref = git_read.review_snapshot_ref_for("test", git_read.tree_sha(repo, base), tree)
    before = git_read.run_ref_tips(repo)
    index = (repo / ".git/index").read_bytes()
    (repo / "a").write_text("later\n")
    sha = git_snapshots.snapshot_commit(repo, tree, parents=(base,), ref=ref)
    assert git_snapshots.snapshot_commit(repo, tree, parents=(base,), ref=ref) == sha
    assert git_read.run_ref_tips(repo) == before
    assert git_read.list_chain_refs(repo) == (("test", base),)
    assert git_read.session_refs(repo, "test") == (chain, ref)
    assert (repo / ".git/index").read_bytes() == index
    assert git(repo, "rev-parse", "HEAD").strip() == base
    assert (repo / "a").read_text() == "later\n"
    clone = tmp_path / "clone"
    git_ops.clone_repo(repo, clone)
    git_ops.fetch_branch(clone, repo, ref)
    git_ops.checkout_detached(clone, sha)
    assert (clone / "a").read_text() == "reviewed\n"
    with pytest.raises(errors.GitError, match="snapshot"):
        git_snapshots.snapshot_commit(repo, git_read.tree_sha(repo, base), parents=(base,), ref=ref)
    assert git_read.chain_tip(repo, ref) == sha


@pytest.mark.parametrize("side", ["base", "head"])
@pytest.mark.parametrize("fault", ["missing", "corrupt"])
@pytest.mark.parametrize("reuse", [False, True], ids=["create", "reuse"])
def test_snapshot_refuses_unreadable_descendants(
    tmp_path: pathlib.Path, side: str, fault: str, reuse: bool
) -> None:
    """Resolving a tree alone cannot prove that its blobs can be fetched and checked out."""
    git = review_helpers.git
    git(tmp_path, "init", "-q")
    (tmp_path / "file").write_text("base\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD").strip()
    base_tree = git_read.tree_sha(tmp_path, base)
    (tmp_path / "file").write_text("head\n")
    tree = git_read.worktree_tree(tmp_path, base, ())
    ref = git_read.review_snapshot_ref_for("test", base_tree, tree)
    before = (
        git_snapshots.snapshot_commit(tmp_path, tree, parents=(base,), ref=ref) if reuse else None
    )
    oid = git(tmp_path, "rev-parse", (base_tree if side == "base" else tree) + ":file").strip()
    obj = tmp_path / ".git/objects" / oid[:2] / oid[2:]
    if fault == "missing":
        obj.unlink()
    else:
        raw = obj.read_bytes()
        obj.chmod(0o600)
        obj.write_bytes(raw[:-4])
    assert git_read.tree_sha(tmp_path, tree) == tree
    with pytest.raises(errors.GitError):
        git_snapshots.snapshot_commit(tmp_path, tree, parents=(base,), ref=ref)
    assert git_read.chain_tip(tmp_path, ref) == before
