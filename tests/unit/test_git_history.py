# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the read-only history helpers in agent6.vcs.git_ops, on a repository built here."""

from __future__ import annotations

import datetime
import os
import pathlib
import subprocess

import pytest

from agent6.vcs import git_history, git_ops

_CLOCK = {"n": 0}


def _git(path: pathlib.Path, *args: str) -> str:
    # Each call commits a minute after the last, so no two commits share a date.
    _CLOCK["n"] += 1
    stamp = f"2026-01-02T10:{_CLOCK['n'] % 60:02d}:00Z"
    env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def _commit(path: pathlib.Path, message: str) -> str:
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def build_history(path: pathlib.Path) -> dict[str, str]:
    """Build a repository with a merge, a rename, a binary file, two tags and foreign refs.

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
    assert git_history.resolve_commit(tmp_path, git_ops.EMPTY_TREE) == ""  # a tree, not a commit


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
    root_files = git_history.changed_files(tmp_path, git_ops.EMPTY_TREE, shas["root"])
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


def test_tree_has_blob_sees_files_only_in_that_tree(tmp_path: pathlib.Path) -> None:
    shas = build_history(tmp_path)
    assert git_history.tree_has_blob(tmp_path, shas["feature"], "b.txt")
    assert not git_history.tree_has_blob(tmp_path, shas["feature"], "a.txt")
    assert git_history.tree_has_blob(tmp_path, shas["root"], "a.txt")
    assert git_history.tree_has_blob(tmp_path, shas["main"], "dir/f.txt")
    assert not git_history.tree_has_blob(tmp_path, shas["main"], "dir")  # a tree, not a file
    assert not git_history.tree_has_blob(tmp_path, git_ops.EMPTY_TREE, "a.txt")


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
