# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the read-only history helpers in agent6.vcs.git_ops, on a repository built here."""

from __future__ import annotations

import datetime
import os
import pathlib
import subprocess

import pytest

from agent6.vcs import git_files, git_history, git_read

_CLOCK = {"n": 0}
# Two valid PNGs, so an image's "before" and "after" differ in size and dimensions.
PNG_RED_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\x0cIDATx\xdac\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xf7\x03AC"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
PNG_BLUE_2X2 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00"
    b"\xfd\xd4\x9as\x00\x00\x00\x0fIDATx\xdac``\xf8\x0fF`\n\x00\x17\xf6\x03\xfd\xc7\x90\x8b\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


_EPOCH = datetime.datetime(2026, 1, 2, 10, tzinfo=datetime.UTC)


def _git(path: pathlib.Path, *args: str) -> str:
    # Each call commits a minute after the last, so no two commits share a date, however
    # many repositories one process builds.
    _CLOCK["n"] += 1
    stamp = (_EPOCH + datetime.timedelta(minutes=_CLOCK["n"])).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def _commit(path: pathlib.Path, message: str) -> str:
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def build_history(path: pathlib.Path) -> dict[str, str]:
    """Build a repository with a merge, a rename, a binary file, a PNG, two tags and foreign refs.

    Returns:
        The sha of each named commit: root, feature, main (before the merge), merge, and
        chained (reachable only from a `refs/agent6/*` chain ref).
    """
    _git(path, "init", "-q", "-b", "main")
    (path / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    (path / "d.txt").write_text("doomed\n", encoding="utf-8")
    (path / "img.bin").write_bytes(b"\x00\x01\x02binary")
    root = _commit(path, "root commit")
    _git(path, "tag", "-a", "v1", "-m", "first tag")
    _git(path, "checkout", "-q", "-b", "feature")
    _git(path, "mv", "a.txt", "b.txt")
    (path / "b.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (path / "img.bin").write_bytes(b"\x00\x01\x03binary two")
    feature = _commit(path, "feature: rename a.txt and grow it\n\nA second paragraph.")
    _git(path, "checkout", "-q", "main")
    (path / "d.txt").unlink()
    (path / "dir").mkdir()
    (path / "dir" / "f.txt").write_text("nested\n", encoding="utf-8")
    (path / "logo.png").write_bytes(PNG_RED_1X1)
    main = _commit(path, "main: drop d.txt, add dir/f.txt")
    _git(path, "merge", "-q", "--no-ff", "-m", "merge feature", "feature")
    merge = _git(path, "rev-parse", "HEAD")
    _git(path, "tag", "v2")
    _git(path, "update-ref", "refs/remotes/origin/main", merge)
    _git(path, "update-ref", "refs/agent6/run-1/head", feature)
    # A commit only a chain ref reaches: the default graph leaves it out.
    tree = _git(path, "rev-parse", f"{feature}^{{tree}}")
    chained = _git(path, "commit-tree", tree, "-p", feature, "-m", "chain-only commit")
    _git(path, "update-ref", "refs/agent6/run-2/head", chained)
    return {"root": root, "feature": feature, "main": main, "merge": merge, "chained": chained}


def make_dirty(path: pathlib.Path) -> None:
    """Leave the built repository's working tree with five uncommitted files and one ignored.

    A modified text file, a staged rename, an untracked text file, an untracked binary file,
    the PNG replaced by the 2x2 one, and `junk.log` excluded through `.git/info/exclude`.

    Args:
        path: The repository `build_history` built.
    """
    (path / "b.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    _git(path, "mv", "dir/f.txt", "dir/g.txt")
    (path / "new.txt").write_text("hello\nworld\n", encoding="utf-8")
    (path / "blob.bin").write_bytes(b"\x00\x01new binary")
    (path / "logo.png").write_bytes(PNG_BLUE_2X2)
    (path / ".git" / "info").mkdir(exist_ok=True)
    (path / ".git" / "info" / "exclude").write_text("junk.log\n", encoding="utf-8")
    (path / "junk.log").write_text("ignored\n", encoding="utf-8")


def test_only_gitlinks_are_submodules(tmp_path: pathlib.Path) -> None:
    """An ordinary directory replacing a tracked file is not a submodule."""
    build_history(tmp_path)
    _, tip = add_submodule_pointer(tmp_path)
    (tmp_path / "sub").mkdir(exist_ok=True)
    (tmp_path / "b.txt").unlink()
    (tmp_path / "b.txt").mkdir()
    committed = {f.path: f for f in git_files.tree_files(tmp_path, tip)}
    working = {f.path: f for f in git_files.worktree_tree_files(tmp_path)}
    assert committed["sub"].kind == "submodule"
    assert working["sub"].kind == "submodule"
    assert "b.txt" not in working
    assert committed["b.txt"].kind == "file"


def test_list_refs_names_kinds_and_peels_tags(tmp_path: pathlib.Path) -> None:
    """Every ref kind is listed under its short name, an annotated tag at the commit it peels to."""
    shas = build_history(tmp_path)
    rows = {r.name: r for r in git_history.list_refs(tmp_path)}
    assert {n: r.kind for n, r in rows.items()} == {
        "main": "branch",
        "feature": "branch",
        "origin/main": "remote",
        "v1": "tag",
        "v2": "tag",
        "agent6/run-1/head": "agent6",
        "agent6/run-2/head": "agent6",
    }
    assert rows["v1"].sha == shas["root"]
    assert rows["v2"].sha == shas["merge"]
    assert rows["main"].full == "refs/heads/main"
    assert rows["agent6/run-1/head"].full == "refs/agent6/run-1/head"
    assert [n for n, r in rows.items() if r.head] == ["main"]


def test_head_commit_names_the_branch_or_detaches(tmp_path: pathlib.Path) -> None:
    shas = build_history(tmp_path)
    assert git_history.head_commit(tmp_path) == (shas["merge"], "main")
    _git(tmp_path, "checkout", "-q", "--detach", shas["root"])
    assert git_history.head_commit(tmp_path) == (shas["root"], "")


def test_head_commit_on_an_unborn_branch_is_empty(tmp_path: pathlib.Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    assert git_history.head_commit(tmp_path) == ("", "main")
    assert git_history.list_refs(tmp_path) == ()


def test_resolve_commit_takes_names_and_refuses_the_unknown(tmp_path: pathlib.Path) -> None:
    """A branch, a tag, a short sha and a ref name resolve; a flag or an unknown rev does not."""
    shas = build_history(tmp_path)
    assert git_history.resolve_commit(tmp_path, "main") == shas["merge"]
    assert git_history.resolve_commit(tmp_path, "v1") == shas["root"]  # peeled through the tag
    assert git_history.resolve_commit(tmp_path, "refs/tags/v1") == shas["root"]
    assert git_history.resolve_commit(tmp_path, shas["feature"][:8]) == shas["feature"]
    assert git_history.resolve_commit(tmp_path, "HEAD~1") == shas["main"]
    assert git_history.resolve_commit(tmp_path, "-x") == ""
    assert git_history.resolve_commit(tmp_path, "--all") == ""
    assert git_history.resolve_commit(tmp_path, "no-such-rev") == ""
    assert git_history.resolve_commit(tmp_path, git_read.EMPTY_TREE) == ""  # a tree, not a commit


def test_log_commits_walk_topologically_and_page(tmp_path: pathlib.Path) -> None:
    """The merge comes first with both parents; `skip` pages the same walk."""
    shas = build_history(tmp_path)
    rows = git_history.log_commits(tmp_path, ["refs/heads/main"], limit=10)
    assert rows[0].sha == shas["merge"]
    assert rows[0].parents == (shas["main"], shas["feature"])
    assert {r.sha for r in rows} == set(shas.values()) - {shas["chained"]}
    assert rows[-1].sha == shas["root"]
    assert rows[-1].parents == ()
    assert rows[-1].subject == "root commit"
    assert rows[-1].author == "t"
    assert rows[-1].epoch == int(datetime.datetime.fromisoformat(rows[-1].date).timestamp())
    page = git_history.log_commits(tmp_path, ["refs/heads/main"], limit=2, skip=2)
    assert [r.sha for r in page] == [r.sha for r in rows[2:4]]
    assert (
        git_history.log_commits(tmp_path, ["refs/heads/feature"], limit=10)[0].sha
        == shas["feature"]
    )


def test_log_commits_on_an_unborn_branch_is_empty(tmp_path: pathlib.Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    assert git_history.log_commits(tmp_path, ["HEAD"], limit=5) == ()


def test_commit_info_carries_both_identities_and_the_body(tmp_path: pathlib.Path) -> None:
    shas = build_history(tmp_path)
    info = git_history.commit_info(tmp_path, shas["feature"])
    assert info.sha == shas["feature"]
    assert info.parents == (shas["root"],)
    assert (info.author, info.author_email) == ("t", "t@t")
    assert (info.committer, info.committer_email) == ("t", "t@t")
    assert info.subject == "feature: rename a.txt and grow it"
    assert info.body == "A second paragraph."
    assert git_history.commit_info(tmp_path, shas["root"]).body == ""


def test_changed_files_detect_renames_binaries_and_deletions(tmp_path: pathlib.Path) -> None:
    """`R` carries the old path, a binary file has no counts, the empty tree lists a root commit."""
    shas = build_history(tmp_path)
    by_path = {
        f.path: f for f in git_history.changed_files(tmp_path, shas["root"], shas["feature"])
    }
    assert set(by_path) == {"b.txt", "img.bin"}
    renamed = by_path["b.txt"]
    assert (renamed.status, renamed.old_path) == ("R", "a.txt")
    assert (renamed.additions, renamed.deletions, renamed.binary) == (1, 0, False)
    binary = by_path["img.bin"]
    assert (binary.status, binary.old_path, binary.binary) == ("M", "", True)
    assert (binary.additions, binary.deletions) == (None, None)
    dropped = {f.path: f for f in git_history.changed_files(tmp_path, shas["root"], shas["main"])}
    assert dropped["d.txt"].status == "D"
    assert dropped["dir/f.txt"].status == "A"
    root_files = git_history.changed_files(tmp_path, git_read.EMPTY_TREE, shas["root"])
    assert sorted((f.status, f.path) for f in root_files) == [
        ("A", "a.txt"),
        ("A", "d.txt"),
        ("A", "img.bin"),
    ]
    assert git_history.changed_files(tmp_path, shas["root"], shas["root"]) == ()


def test_file_patch_names_renames_and_binaries_and_cuts_at_the_cap(tmp_path: pathlib.Path) -> None:
    shas = build_history(tmp_path)
    patch, truncated = git_history.file_patch(
        tmp_path, shas["root"], shas["feature"], "b.txt", "a.txt", max_bytes=1 << 20
    )
    assert "rename from a.txt" in patch and "+three" in patch
    assert truncated is False
    binary, _ = git_history.file_patch(
        tmp_path, shas["root"], shas["feature"], "img.bin", max_bytes=1 << 20
    )
    assert "Binary files a/img.bin and b/img.bin differ" in binary
    cut, truncated = git_history.file_patch(
        tmp_path, shas["root"], shas["feature"], "b.txt", "a.txt", max_bytes=40
    )
    assert truncated is True
    assert len(cut.encode()) <= 40 and cut.endswith("\n")
    assert git_history.file_patch(tmp_path, shas["root"], shas["root"], "a.txt", max_bytes=100) == (
        "",
        False,
    )


def test_tree_has_entry_sees_files_only_in_that_tree(tmp_path: pathlib.Path) -> None:
    shas = build_history(tmp_path)
    assert git_history.tree_has_entry(tmp_path, shas["feature"], "b.txt")
    assert not git_history.tree_has_entry(tmp_path, shas["feature"], "a.txt")
    assert git_history.tree_has_entry(tmp_path, shas["root"], "a.txt")
    assert git_history.tree_has_entry(tmp_path, shas["main"], "dir/f.txt")
    assert not git_history.tree_has_entry(tmp_path, shas["main"], "dir")  # a tree, not a file
    assert not git_history.tree_has_entry(tmp_path, git_read.EMPTY_TREE, "a.txt")


def test_search_log_matches_message_text_literally(tmp_path: pathlib.Path) -> None:
    """The text is a case-insensitive literal, not a regex; empty text lists the newest commits."""
    shas = build_history(tmp_path)
    assert [r.sha for r in git_history.search_log(tmp_path, "FEATURE: rename", limit=10)] == [
        shas["feature"]
    ]
    assert [r.sha for r in git_history.search_log(tmp_path, "a.txt", limit=10)] == [shas["feature"]]
    assert git_history.search_log(tmp_path, "a.tx.", limit=10) == ()
    newest = git_history.search_log(tmp_path, "", limit=2)
    assert [r.sha for r in newest] == [shas["chained"], shas["merge"]]  # chain refs count
    assert git_history.search_log(tmp_path, "-x", limit=10) == ()


@pytest.mark.parametrize("rev", ["-x", "--output=/tmp/pwned"])
def test_history_reads_never_take_a_rev_as_an_option(tmp_path: pathlib.Path, rev: str) -> None:
    """A flag-shaped revision resolves to nothing and never reaches git as an option."""
    build_history(tmp_path)
    assert git_history.resolve_commit(tmp_path, rev) == ""
    assert git_history.log_commits(tmp_path, [rev], limit=5) == ()


def test_worktree_files_list_tracked_changes_then_untracked_files(tmp_path: pathlib.Path) -> None:
    """Tracked changes carry git's letters and counts, untracked files their line count.

    An untracked file reads `A`, a binary one `binary`; an ignored file stays out; a clean
    tree lists nothing.
    """
    shas = build_history(tmp_path)
    assert git_history.worktree_files(tmp_path, shas["merge"]) == ()
    make_dirty(tmp_path)
    rows = git_history.worktree_files(tmp_path, shas["merge"])
    assert [f.path for f in rows] == ["b.txt", "dir/g.txt", "logo.png", "blob.bin", "new.txt"]
    by_path = {f.path: f for f in rows}
    assert by_path["b.txt"] == git_history.ChangedFile("M", "b.txt", "", 1, 0, False)
    assert by_path["dir/g.txt"] == git_history.ChangedFile(
        "R", "dir/g.txt", "dir/f.txt", 0, 0, False
    )
    assert by_path["logo.png"] == git_history.ChangedFile("M", "logo.png", "", None, None, True)
    assert by_path["blob.bin"] == git_history.ChangedFile("A", "blob.bin", "", None, None, True)
    assert by_path["new.txt"] == git_history.ChangedFile("A", "new.txt", "", 2, 0, False)


def test_worktree_files_on_an_unborn_branch_list_against_the_empty_tree(
    tmp_path: pathlib.Path,
) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "s.txt").write_text("staged\n", encoding="utf-8")
    _git(tmp_path, "add", "s.txt")
    (tmp_path / "u.txt").write_text("no newline", encoding="utf-8")
    rows = git_history.worktree_files(tmp_path, git_read.EMPTY_TREE)
    assert rows == (
        git_history.ChangedFile("A", "s.txt", "", 1, 0, False),
        git_history.ChangedFile("A", "u.txt", "", 1, 0, False),
    )


def test_worktree_patch_reads_tracked_untracked_and_binary_files(tmp_path: pathlib.Path) -> None:
    """A tracked file diffs against the base, an untracked one against /dev/null.

    git's exit 1 from `--no-index` is "differs", not a failure; a binary file says so; the
    cap cuts either.
    """
    shas = build_history(tmp_path)
    make_dirty(tmp_path)
    base = shas["merge"]
    patch, cut = git_history.worktree_patch(tmp_path, base, "b.txt", max_bytes=1 << 20)
    assert "+four" in patch and cut is False
    renamed, _ = git_history.worktree_patch(
        tmp_path, base, "dir/g.txt", "dir/f.txt", max_bytes=1 << 20
    )
    assert "rename from dir/f.txt" in renamed
    new, cut = git_history.worktree_patch(
        tmp_path, base, "new.txt", untracked=True, max_bytes=1 << 20
    )
    assert "+++ b/new.txt" in new and "+hello" in new and cut is False
    binary, _ = git_history.worktree_patch(
        tmp_path, base, "blob.bin", untracked=True, max_bytes=1 << 20
    )
    assert git_history.is_binary_patch(binary)
    image, _ = git_history.worktree_patch(tmp_path, base, "logo.png", max_bytes=1 << 20)
    assert git_history.is_binary_patch(image)
    short, cut = git_history.worktree_patch(tmp_path, base, "new.txt", untracked=True, max_bytes=40)
    assert cut is True and len(short.encode()) <= 40 and short.endswith("\n")


def test_worktree_entry_names_tracked_untracked_or_nothing(tmp_path: pathlib.Path) -> None:
    """Only a file git lists classifies.

    An ignored file, a directory, a path renamed away or outside the repository is nothing,
    so a later read refuses it.
    """
    build_history(tmp_path)
    make_dirty(tmp_path)
    assert git_history.worktree_entry(tmp_path, "b.txt") == "tracked"
    assert git_history.worktree_entry(tmp_path, "dir/g.txt") == "tracked"
    assert git_history.worktree_entry(tmp_path, "new.txt") == "untracked"
    assert git_history.worktree_entry(tmp_path, "junk.log") == ""
    assert git_history.worktree_entry(tmp_path, "dir/f.txt") == ""
    assert git_history.worktree_entry(tmp_path, "dir") == ""
    assert git_history.worktree_entry(tmp_path, "nope") == ""
    assert git_history.worktree_entry(tmp_path, "../outside") == ""
    (tmp_path / "linkdir").symlink_to(tmp_path.parent)
    assert git_history.worktree_entry(tmp_path, "linkdir") == "untracked"
    assert git_history.worktree_entry(tmp_path, "linkdir/x") == ""


def test_blob_at_reads_a_regular_files_bytes_within_the_cap(tmp_path: pathlib.Path) -> None:
    """A file comes back whole with its size, a larger one as its size only.

    Anything that is not a regular file in that tree is nothing.
    """
    shas = build_history(tmp_path)
    blob = git_files.blob_at(tmp_path, shas["main"], "logo.png", max_bytes=1 << 20)
    assert blob == git_files.Blob(len(PNG_RED_1X1), PNG_RED_1X1)
    over = git_files.blob_at(tmp_path, shas["main"], "logo.png", max_bytes=10)
    assert over == git_files.Blob(len(PNG_RED_1X1), None)
    assert git_files.blob_at(tmp_path, shas["root"], "logo.png", max_bytes=1 << 20) is None
    assert git_files.blob_at(tmp_path, shas["main"], "dir", max_bytes=1 << 20) is None
    assert git_files.blob_at(tmp_path, git_read.EMPTY_TREE, "a.txt", max_bytes=1 << 20) is None


def test_worktree_blob_reads_only_a_regular_file_git_lists(tmp_path: pathlib.Path) -> None:
    """The working tree's file is read from disk when git lists it.

    An ignored file, a path renamed away or outside the repository, and a symlink are
    nothing.
    """
    build_history(tmp_path)
    make_dirty(tmp_path)
    assert git_files.worktree_blob(tmp_path, "logo.png", max_bytes=1 << 20) == git_files.Blob(
        len(PNG_BLUE_2X2), PNG_BLUE_2X2
    )
    assert git_files.worktree_blob(tmp_path, "new.txt", max_bytes=1 << 20) == git_files.Blob(
        12, b"hello\nworld\n"
    )
    assert git_files.worktree_blob(tmp_path, "new.txt", max_bytes=5) == git_files.Blob(12, None)
    assert git_files.worktree_blob(tmp_path, "junk.log", max_bytes=1 << 20) is None
    assert git_files.worktree_blob(tmp_path, "dir/f.txt", max_bytes=1 << 20) is None
    assert git_files.worktree_blob(tmp_path, "../outside", max_bytes=1 << 20) is None
    (tmp_path / "lnk").symlink_to("/etc/hostname")
    assert git_history.worktree_entry(tmp_path, "lnk") == "untracked"
    assert git_files.worktree_blob(tmp_path, "lnk", max_bytes=1 << 20) is None


def test_tree_files_list_a_commits_files_with_size_and_kind(tmp_path: pathlib.Path) -> None:
    shas = build_history(tmp_path)
    (tmp_path / "lnk").symlink_to("a.txt")
    _git(tmp_path, "add", "lnk")
    _git(tmp_path, "commit", "-q", "-m", "a symlink")
    rows = {f.path: f for f in git_files.tree_files(tmp_path, "HEAD")}
    assert sorted(rows) == ["b.txt", "dir/f.txt", "img.bin", "lnk", "logo.png"]
    assert rows["b.txt"] == git_files.TreeFile("b.txt", 14, "file")
    assert rows["logo.png"] == git_files.TreeFile("logo.png", len(PNG_RED_1X1), "file")
    assert rows["lnk"] == git_files.TreeFile("lnk", 5, "symlink")
    assert [f.path for f in git_files.tree_files(tmp_path, shas["root"])] == [
        "a.txt",
        "d.txt",
        "img.bin",
    ]
    assert git_files.tree_files(tmp_path, git_read.EMPTY_TREE) == ()


def test_worktree_tree_files_list_tracked_and_unignored_untracked_files(
    tmp_path: pathlib.Path,
) -> None:
    """The working tree's files are the index plus untracked files `.gitignore` keeps.

    A tracked file gone from disk and an ignored file are not listed; a symlink says so.
    """
    build_history(tmp_path)
    make_dirty(tmp_path)
    (tmp_path / "img.bin").unlink()
    (tmp_path / "lnk").symlink_to("b.txt")
    rows = {f.path: f for f in git_files.worktree_tree_files(tmp_path)}
    assert sorted(rows) == ["b.txt", "blob.bin", "dir/g.txt", "lnk", "logo.png", "new.txt"]
    assert rows["b.txt"] == git_files.TreeFile("b.txt", 19, "file", "modified")
    assert rows["new.txt"] == git_files.TreeFile("new.txt", 12, "file", "untracked")
    assert rows["lnk"] == git_files.TreeFile("lnk", 5, "symlink", "untracked")


def add_submodule_pointer(path: pathlib.Path) -> tuple[str, str]:
    """Commit a submodule pointer at `sub`, then move it, without a submodule checkout.

    Args:
        path: The repository `build_history` built.

    Returns:
        The two commits' shas, the pointer at its first and its second target.
    """
    shas = {line.split()[0] for line in _git(path, "rev-list", "--all").splitlines()}
    first, second = sorted(shas)[:2]
    _git(path, "update-index", "--add", "--cacheinfo", f"160000,{first},sub")
    _git(path, "commit", "-q", "-m", "a submodule pointer")
    one = _git(path, "rev-parse", "HEAD")
    _git(path, "update-index", "--add", "--cacheinfo", f"160000,{second},sub")
    _git(path, "commit", "-q", "-m", "the pointer moves")
    return one, _git(path, "rev-parse", "HEAD")


def test_a_submodule_pointer_is_a_diffable_entry(tmp_path: pathlib.Path) -> None:
    """A gitlink (`160000 commit`) is an entry a diff can name; its patch is the pointer move."""
    build_history(tmp_path)
    one, two = add_submodule_pointer(tmp_path)
    assert git_history.tree_has_entry(tmp_path, two, "sub")
    assert not git_history.tree_has_entry(tmp_path, two, "dir")
    rows = {f.path: f for f in git_history.changed_files(tmp_path, one, two)}
    assert rows["sub"].status == "M"
    patch, _ = git_history.file_patch(tmp_path, one, two, "sub", max_bytes=1 << 20)
    assert "-Subproject commit" in patch and "+Subproject commit" in patch
    assert git_files.blob_at(tmp_path, two, "sub", max_bytes=1 << 20) is None
