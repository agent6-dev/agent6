# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Git tab's `/api/git/*` endpoints over a repository the test builds.

Every revision and path arrives from the browser, so the pins here are the refusals as
much as the payloads: a flag-shaped rev or path, an unknown rev, a path in neither tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
import threading
from collections.abc import Iterator
from http import client
from typing import Any

import pytest

from agent6.ui.web import git_model, page
from agent6.ui.web import server as web_server
from agent6.vcs import git_files, git_read
from tests.unit.test_git_history import (
    PNG_BLUE_2X2,
    PNG_RED_1X1,
    _git,
    add_submodule_pointer,
    build_history,
    make_dirty,
)
from tests.web import signin


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, str]]:
    return tmp_path, build_history(tmp_path)


@pytest.fixture
def port(repo: tuple[pathlib.Path, dict[str, str]]) -> Iterator[int]:
    """A server reading the built repository."""
    yield from _serve(repo[0])


def _serve(cwd: pathlib.Path) -> Iterator[int]:
    srv = web_server.WebServer(("127.0.0.1", 0), cwd, "", secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


def _get(port: int, path: str) -> tuple[int, Any]:
    conn = signin.SignedIn(port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def _raw(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    """GET a path and return the status, the headers lower-cased, and the body bytes."""
    conn = signin.SignedIn(port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def _ok(port: int, path: str) -> Any:
    status, body = _get(port, path)
    assert status == 200, body
    return body


def _refused(port: int, path: str) -> str:
    status, body = _get(port, path)
    assert status == 422, body
    return str(body["error"])


def test_refs_list_every_kind_and_head(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    _, shas = repo
    body = _ok(port, "/api/git/refs")
    assert body["head"] == {"sha": shas["merge"], "branch": "main"}
    by_name = {r["name"]: r for r in body["refs"]}
    assert by_name["main"] == {
        "name": "main",
        "full": "refs/heads/main",
        "kind": "branch",
        "sha": shas["merge"],
        "head": True,
    }
    assert by_name["v1"]["sha"] == shas["root"]  # the annotated tag, peeled
    assert by_name["origin/main"]["kind"] == "remote"
    assert by_name["agent6/run-2/head"]["kind"] == "agent6"


def test_graph_walks_the_default_refs_and_decorates(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """Branches, remotes and tags are the default walk; a chain-only commit stays out of it."""
    _, shas = repo
    body = _ok(port, "/api/git/graph")
    commits = body["commits"]
    assert commits[0]["sha"] == shas["merge"]
    assert {c["sha"] for c in commits} == set(shas.values()) - {shas["chained"]}
    merge = commits[0]
    assert merge["is_merge"] is True
    assert merge["parents"] == [shas["main"], shas["feature"]]
    assert merge["short"] == shas["merge"][:7]
    assert {(r["name"], r["kind"]) for r in merge["refs"]} == {
        ("main", "branch"),
        ("origin/main", "remote"),
        ("v2", "tag"),
    }
    assert [r["head"] for r in merge["refs"] if r["name"] == "main"] == [True]
    assert merge["author"] == "t"
    assert merge["date"].startswith("2026-01-") and len(merge["when"]) == 11
    assert merge["subject"] == "merge feature"
    assert (body["has_more"], body["skip"], body["limit"]) == (False, 0, git_model.GRAPH_PAGE)


def test_graph_takes_a_ref_list_and_pages(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    _, shas = repo
    only = _ok(port, "/api/git/graph?refs=refs/agent6/run-2/head")
    assert [c["sha"] for c in only["commits"]][:2] == [shas["chained"], shas["feature"]]
    assert only["commits"][0]["refs"] == [
        {"name": "agent6/run-2/head", "kind": "agent6", "head": False}
    ]
    two = _ok(port, "/api/git/graph?refs=refs/tags/v1&refs=refs/agent6/run-2/head")
    assert {c["sha"] for c in two["commits"]} == {shas["root"], shas["feature"], shas["chained"]}
    assert _ok(port, "/api/git/graph?refs=")["commits"] == []  # present and blank: nothing
    page = _ok(port, "/api/git/graph?refs=HEAD&limit=2")
    assert len(page["commits"]) == 2 and page["has_more"] is True
    rest = _ok(port, "/api/git/graph?refs=HEAD&limit=2&skip=2")
    whole = _ok(port, "/api/git/graph?refs=HEAD")["commits"]
    assert [c["sha"] for c in page["commits"] + rest["commits"]] == [c["sha"] for c in whole]
    assert rest["commits"][-1]["sha"] == shas["root"]
    assert rest["has_more"] is False and rest["skip"] == 2
    assert _ok(port, "/api/git/graph?limit=99999")["limit"] == git_model.GRAPH_PAGE_MAX
    assert "no such ref" in _refused(port, "/api/git/graph?refs=refs/heads/nope")
    assert "no such ref" in _refused(port, "/api/git/graph?refs=main")  # short names are not refs
    assert "whole number" in _refused(port, "/api/git/graph?limit=ten")


def test_graph_pages_keep_the_original_tips_when_refs_move(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """Paging resolved tips neither repeats nor loses a row after a new commit lands."""
    path, shas = repo
    expected = _ok(port, "/api/git/graph?refs=HEAD")["commits"]
    first = _ok(port, "/api/git/graph?refs=HEAD&limit=2")
    assert first["tips"] == [shas["merge"]]
    _git(path, "commit", "--allow-empty", "-m", "a new tip")
    query = "&".join("refs=" + tip for tip in first["tips"])
    rest = _ok(port, f"/api/git/graph?{query}&skip=2")
    assert [c["sha"] for c in first["commits"] + rest["commits"]] == [c["sha"] for c in expected]
    assert rest["tips"] == first["tips"]
    refreshed = _ok(port, "/api/git/graph?refs=HEAD")
    assert refreshed["tips"] == [_git(path, "rev-parse", "HEAD")]
    assert refreshed["commits"][0]["subject"] == "a new tip"
    assert "no such ref" in _refused(port, "/api/git/graph?refs=" + "f" * 40)
    assert "no such ref" in _refused(port, "/api/git/graph?refs=--all")


def test_commit_lists_its_files_against_the_first_parent(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """A merge's files are its diff from the first parent; a root's base is the empty tree."""
    _, shas = repo
    merge = _ok(port, f"/api/git/commit/{shas['merge']}")
    assert merge["sha"] == shas["merge"] and merge["is_merge"] is True
    assert merge["base"] == shas["main"]
    assert merge["author"] == {"name": "t", "email": "t@t", "date": merge["author"]["date"]}
    assert merge["committer"]["name"] == "t"
    assert (merge["subject"], merge["body"]) == ("merge feature", "")
    assert {(f["status"], f["path"], f["old_path"]) for f in merge["files"]} == {
        ("R", "b.txt", "a.txt"),
        ("M", "img.bin", ""),
    }
    binary = next(f for f in merge["files"] if f["path"] == "img.bin")
    assert (binary["binary"], binary["additions"], binary["deletions"]) == (True, None, None)
    renamed = next(f for f in merge["files"] if f["path"] == "b.txt")
    assert (renamed["binary"], renamed["additions"], renamed["deletions"]) == (False, 1, 0)
    assert merge["truncated"] is False
    assert {r["name"] for r in merge["refs"]} == {"main", "origin/main", "v2"}

    root = _ok(port, f"/api/git/commit/{shas['root'][:8]}")
    assert root["base"] == git_read.EMPTY_TREE and root["parents"] == []
    assert sorted(f["path"] for f in root["files"]) == ["a.txt", "d.txt", "img.bin"]
    feature = _ok(port, "/api/git/commit/feature")
    assert feature["body"] == "A second paragraph."
    assert _ok(port, "/api/git/commit/v1")["sha"] == shas["root"]


def test_compare_reads_like_the_commits_own_diff(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    _, shas = repo
    body = _ok(port, f"/api/git/compare?base={shas['root']}&head=feature")
    assert body["base"] == {
        "rev": shas["root"],
        "sha": shas["root"],
        "short": shas["root"][:7],
        "subject": "root commit",
    }
    assert body["head"]["sha"] == shas["feature"] and body["head"]["rev"] == "feature"
    own = _ok(port, f"/api/git/commit/{shas['feature']}")["files"]
    assert body["files"] == own
    across = _ok(port, "/api/git/compare?base=refs/tags/v1&head=refs/tags/v2")
    assert sorted(f["path"] for f in across["files"]) == [
        "b.txt",
        "d.txt",
        "dir/f.txt",
        "img.bin",
        "logo.png",
    ]
    empty = _ok(port, f"/api/git/compare?base={git_read.EMPTY_TREE}&head=v1")
    assert empty["base"]["subject"] == "(empty tree)"
    assert len(empty["files"]) == 3
    assert _ok(port, "/api/git/compare?base=main&head=main")["files"] == []


def test_diff_returns_one_files_patch_with_its_flags(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, shas = repo
    q = f"base={shas['root']}&head={shas['feature']}"
    body = _ok(port, f"/api/git/diff?{q}&path=b.txt&old_path=a.txt")
    assert body["base"] == shas["root"] and body["head"] == shas["feature"]
    assert (body["path"], body["old_path"]) == ("b.txt", "a.txt")
    assert "rename from a.txt" in body["patch"] and "+three" in body["patch"]
    assert (body["binary"], body["truncated"]) == (False, False)
    binary = _ok(port, f"/api/git/diff?{q}&path=img.bin")
    assert binary["binary"] is True and "Binary files" in binary["patch"]
    deleted = _ok(port, f"/api/git/diff?base={shas['root']}&head={shas['main']}&path=d.txt")
    assert "-doomed" in deleted["patch"]
    monkeypatch.setattr(git_model, "DIFF_MAX_BYTES", 40)
    cut = _ok(port, f"/api/git/diff?{q}&path=b.txt&old_path=a.txt")
    assert cut["truncated"] is True and len(cut["patch"].encode()) <= 40
    assert cut["max_bytes"] == 40
    # The page is the one place that shows every byte: `whole=1` lifts the cap.
    whole = _ok(port, f"/api/git/diff?{q}&path=b.txt&old_path=a.txt&whole=1")
    assert whole["truncated"] is False and "+three" in whole["patch"]
    assert whole["max_bytes"] is None


def test_worktree_lists_the_uncommitted_files_against_head(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """A clean tree lists nothing; a dirty one lists tracked changes, then untracked files."""
    path, shas = repo
    assert _ok(port, "/api/git/worktree") == {
        "base": shas["merge"],
        "files": [],
        "truncated": False,
    }
    make_dirty(path)
    body = _ok(port, "/api/git/worktree")
    assert body["base"] == shas["merge"] and body["truncated"] is False
    assert [(f["status"], f["path"], f["old_path"]) for f in body["files"]] == [
        ("M", "b.txt", ""),
        ("R", "dir/g.txt", "dir/f.txt"),
        ("M", "logo.png", ""),
        ("A", "blob.bin", ""),
        ("A", "new.txt", ""),
    ]
    by_path = {f["path"]: f for f in body["files"]}
    assert (by_path["new.txt"]["additions"], by_path["new.txt"]["deletions"]) == (2, 0)
    assert by_path["new.txt"]["binary"] is False
    assert (by_path["blob.bin"]["additions"], by_path["blob.bin"]["binary"]) == (None, True)
    assert (by_path["b.txt"]["additions"], by_path["b.txt"]["deletions"]) == (1, 0)


def test_compare_and_diff_take_the_working_tree_as_head(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """`WORKTREE` is a head for compare and diff, never a base or a commit."""
    path, shas = repo
    make_dirty(path)
    cmp = _ok(port, "/api/git/compare?base=HEAD&head=WORKTREE")
    assert cmp["head"] == {
        "rev": "WORKTREE",
        "sha": "WORKTREE",
        "short": "WORKTREE",
        "subject": "(working tree)",
    }
    assert cmp["base"]["sha"] == shas["merge"]
    assert cmp["files"] == _ok(port, "/api/git/worktree")["files"]
    q = f"base={shas['merge']}&head=WORKTREE"
    new = _ok(port, f"/api/git/diff?{q}&path=new.txt")
    assert (new["head"], new["binary"], new["truncated"]) == ("WORKTREE", False, False)
    assert "+++ b/new.txt" in new["patch"] and "+hello" in new["patch"]
    renamed = _ok(port, f"/api/git/diff?{q}&path=dir/g.txt&old_path=dir/f.txt")
    assert "rename from dir/f.txt" in renamed["patch"]
    assert "+four" in _ok(port, f"/api/git/diff?{q}&path=b.txt")["patch"]
    assert _ok(port, f"/api/git/diff?{q}&path=blob.bin")["binary"] is True
    # A deletion staged away is in neither the index nor the working tree, only the base.
    _git(path, "rm", "-q", "img.bin")
    gone = _ok(port, f"/api/git/diff?{q}&path=img.bin")
    assert gone["binary"] is True and "/dev/null" in gone["patch"]
    listed = [(f["status"], f["path"]) for f in _ok(port, "/api/git/worktree")["files"]]
    assert ("D", "img.bin") in listed
    assert "working tree" in _refused(port, "/api/git/compare?base=WORKTREE&head=main")
    assert "working tree" in _refused(port, "/api/git/diff?base=WORKTREE&head=main&path=b.txt")
    assert "working tree" in _refused(port, "/api/git/commit/WORKTREE")
    assert "no file" in _refused(port, f"/api/git/diff?{q}&path=junk.log")  # ignored
    assert "no file" in _refused(port, f"/api/git/diff?{q}&path=dir")  # a directory
    assert "not a repository path" in _refused(port, f"/api/git/diff?{q}&path=../outside")
    assert "not a repository path" in _refused(port, f"/api/git/diff?{q}&path=/etc/passwd")


def test_blob_serves_an_image_typed_and_sandboxed_and_the_rest_as_a_download(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image's bytes come typed by extension; anything else is an untyped attachment.

    Every blob response is sandboxed and unsniffable, so a repository file opened directly
    (an SVG with a script) runs under no origin; over the cap answers 413.
    """
    path, _shas = repo
    status, headers, body = _raw(port, "/api/git/blob?rev=main&path=logo.png")
    assert (status, body, headers["content-type"]) == (200, PNG_RED_1X1, "image/png")
    csp = headers["content-security-policy"]
    assert "sandbox" in csp and "default-src 'none'" in csp
    assert headers["x-content-type-options"] == "nosniff"
    assert "content-disposition" not in headers
    make_dirty(path)
    status, headers, body = _raw(port, "/api/git/blob?rev=WORKTREE&path=logo.png")
    assert (status, body, headers["content-type"]) == (200, PNG_BLUE_2X2, "image/png")
    status, headers, body = _raw(port, "/api/git/blob?rev=WORKTREE&path=new.txt")
    assert (status, body) == (200, b"hello\nworld\n")
    assert headers["content-type"] == "application/octet-stream"
    assert headers["content-disposition"] == "attachment"
    assert headers["x-content-type-options"] == "nosniff"
    monkeypatch.setattr(git_model, "BLOB_MAX_BYTES", 10)
    status, headers, body = _raw(port, "/api/git/blob?rev=main&path=logo.png")
    assert status == 413 and "the most served is 10 bytes" in json.loads(body)["error"]


def test_blob_refuses_what_is_not_a_file_of_the_revision(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    path, _shas = repo
    make_dirty(path)
    assert "unknown revision" in _refused(port, "/api/git/blob?rev=nope&path=logo.png")
    assert "a revision is required" in _refused(port, "/api/git/blob?rev=&path=logo.png")
    assert "cannot start with '-'" in _refused(port, "/api/git/blob?rev=main&path=-x")
    assert "a path is required" in _refused(port, "/api/git/blob?rev=main&path=")
    assert "not a repository path" in _refused(port, "/api/git/blob?rev=main&path=../x")
    assert "not a repository path" in _refused(port, "/api/git/blob?rev=WORKTREE&path=/etc/passwd")
    assert "no file" in _refused(port, "/api/git/blob?rev=main&path=dir")
    assert "no file" in _refused(port, "/api/git/blob?rev=v1&path=logo.png")
    assert "no file" in _refused(port, "/api/git/blob?rev=WORKTREE&path=junk.log")
    (path / "lnk").symlink_to("/etc/hostname")
    assert "no file" in _refused(port, "/api/git/blob?rev=WORKTREE&path=lnk")


def test_text_returns_a_files_lines_with_its_language_and_spans(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text file comes back with its line count, language and highlight spans.

    A binary file says so; the 1 MiB cut is marked and `whole=1` lifts it.
    """
    path, shas = repo
    make_dirty(path)
    (path / "tool.py").write_text('def f(x):\n    return "hi"\n', encoding="utf-8")
    body = _ok(port, f"/api/git/text?rev={shas['merge']}&path=b.txt")
    assert body["rev"] == shas["merge"] and body["path"] == "b.txt"
    assert body["text"] == "one\ntwo\nthree\n" and body["lines"] == 3
    assert (body["binary"], body["truncated"], body["max_bytes"]) == (False, False, 1 << 20)
    assert (body["language"], body["spans"]) == ("", [])
    py = _ok(port, "/api/git/text?rev=WORKTREE&path=tool.py")
    assert py["rev"] == "WORKTREE" and py["lines"] == 2 and py["language"] == "python"
    assert [0, 3, "keyword"] in py["spans"][0] and len(py["spans"]) == 2
    new = _ok(port, "/api/git/text?rev=WORKTREE&path=new.txt")
    assert new["text"] == "hello\nworld\n" and new["lines"] == 2
    binary = _ok(port, "/api/git/text?rev=HEAD&path=img.bin")
    assert (binary["binary"], binary["text"], binary["lines"], binary["spans"]) == (True, "", 0, [])
    monkeypatch.setattr(git_model, "TEXT_MAX_BYTES", 8)
    cut = _ok(port, "/api/git/text?rev=HEAD&path=b.txt")
    assert (cut["text"], cut["truncated"], cut["max_bytes"], cut["lines"]) == (
        "one\ntwo\n",
        True,
        8,
        2,
    )
    whole = _ok(port, "/api/git/text?rev=HEAD&path=b.txt&whole=1")
    assert (whole["text"], whole["truncated"], whole["max_bytes"]) == (
        "one\ntwo\nthree\n",
        False,
        None,
    )
    assert "no file" in _refused(port, "/api/git/text?rev=HEAD&path=nope.txt")
    assert "no file" in _refused(port, "/api/git/text?rev=HEAD&path=dir")
    assert "no file" in _refused(port, "/api/git/text?rev=WORKTREE&path=junk.log")
    assert "cannot start with '-'" in _refused(port, "/api/git/text?rev=HEAD&path=-x")
    assert "not a repository path" in _refused(port, "/api/git/text?rev=HEAD&path=../x")
    assert "unknown revision" in _refused(port, "/api/git/text?rev=nope&path=b.txt")


@pytest.mark.parametrize("route", ["blob", "text", "blame"])
@pytest.mark.parametrize("hazard", ["ancestor", "ancestor_inside", "fifo", "device", "leaf"])
def test_worktree_reads_refuse_unsafe_files(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, route: str, hazard: str
) -> None:
    """A listed path cannot escape through an ancestor or open a special file."""
    path, _ = repo
    name = "dir/f.txt" if hazard.startswith("ancestor") else "b.txt"
    target = path / name
    target.unlink()
    if hazard.startswith("ancestor"):
        # Inside the repository, the containment check passes and only the descriptor
        # walk's no-follow refuses; outside, both would.
        inside = hazard == "ancestor_inside"
        other = path / "other" if inside else path.parent / f"{path.name}-outside"
        other.mkdir()
        (other / "f.txt").write_text("private\n", encoding="utf-8")
        (path / "dir").rmdir()
        (path / "dir").symlink_to(other, target_is_directory=True)
    elif hazard == "leaf":
        target.symlink_to("img.bin")
    elif hazard == "fifo":
        os.mkfifo(target)
    else:
        try:
            os.mknod(target, stat.S_IFCHR | 0o600, os.makedev(1, 3))
        except PermissionError:
            pytest.skip("the host does not allow device nodes")
    conn = signin.SignedIn(port, timeout=1)
    status, body = 0, b""
    try:
        conn.request("GET", f"/api/git/{route}?rev=WORKTREE&path={name}")
        response = conn.getresponse()
        status, body = response.status, response.read()
    except TimeoutError:
        pass
    finally:
        conn.close()
        if hazard == "fifo":
            # Release a reader stuck in open so a red pin leaves no blocked server thread.
            fd = os.open(target, os.O_RDWR | os.O_NONBLOCK)
            os.close(fd)
    assert status == 422, "read blocked" if not status else body
    assert "no file" in json.loads(body)["error"]


def test_blame_attributes_lines_and_marks_uncommitted_content(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blame follows renames and shares the text reader's caps and refusals."""
    path, shas = repo
    body = _ok(port, "/api/git/blame?rev=HEAD&path=b.txt")
    assert body["rev"] == shas["merge"] and body["path"] == "b.txt"
    assert [line["sha"] for line in body["lines"]] == [shas["root"], shas["root"], shas["feature"]]
    assert [line["line"] for line in body["lines"]] == [1, 2, 3]
    assert body["lines"][0]["author"] == "t"
    assert body["lines"][0]["date"].startswith("2026-01-")
    make_dirty(path)
    dirty = _ok(port, "/api/git/blame?rev=WORKTREE&path=b.txt")
    assert dirty["lines"][:3] == body["lines"]
    assert dirty["lines"][3] == {"line": 4, "sha": "", "author": "not committed", "date": ""}
    _git(path, "add", "b.txt")
    assert _ok(port, "/api/git/blame?rev=WORKTREE&path=b.txt")["lines"] == dirty["lines"]
    new = _ok(port, "/api/git/blame?rev=WORKTREE&path=new.txt")
    assert len(new["lines"]) == 2 and all(not line["sha"] for line in new["lines"])
    binary = _ok(port, "/api/git/blame?rev=HEAD&path=img.bin")
    assert binary["binary"] is True and binary["lines"] == []
    monkeypatch.setattr(git_model, "TEXT_MAX_BYTES", 8)
    cut = _ok(port, "/api/git/blame?rev=HEAD&path=b.txt")
    assert (len(cut["lines"]), cut["truncated"], cut["max_bytes"]) == (2, True, 8)
    whole = _ok(port, "/api/git/blame?rev=HEAD&path=b.txt&whole=1")
    assert (len(whole["lines"]), whole["truncated"], whole["max_bytes"]) == (3, False, None)
    for query in (
        "rev=nope&path=b.txt",
        "rev=HEAD&path=../x",
        "rev=HEAD&path=-x",
        "rev=HEAD&path=dir",
        "rev=HEAD&path=missing",
        "rev=WORKTREE&path=junk.log",
    ):
        assert _refused(port, "/api/git/blame?" + query)
    monkeypatch.setattr(git_model, "BLOB_MAX_BYTES", 1)
    assert _get(port, "/api/git/blame?rev=HEAD&path=b.txt")[0] == 413


def test_blame_counts_newlines_not_other_text_separators(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """A tab after a vertical tab is source text, not another porcelain line."""
    path, _ = repo
    (path / "b.txt").write_text("one\v\tstill one\nsecond\n", encoding="utf-8")
    _git(path, "add", "b.txt")
    _git(path, "commit", "-m", "text with separators")
    body = _ok(port, "/api/git/blame?rev=HEAD&path=b.txt")
    assert [line["line"] for line in body["lines"]] == [1, 2]


def test_tree_lists_the_files_at_a_revision_or_in_the_working_tree(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, shas = repo
    make_dirty(path)
    body = _ok(port, "/api/git/tree?rev=main")
    assert body["rev"] == shas["merge"] and body["truncated"] is False
    assert [f["path"] for f in body["files"]] == ["b.txt", "dir/f.txt", "img.bin", "logo.png"]
    assert body["files"][0] == {"path": "b.txt", "size": 14, "kind": "file", "state": ""}
    wt = _ok(port, "/api/git/tree?rev=WORKTREE")
    assert wt["rev"] == "WORKTREE"
    assert [f["path"] for f in wt["files"]] == [
        "b.txt",
        "blob.bin",
        "dir/g.txt",
        "img.bin",
        "logo.png",
        "new.txt",
    ]
    root = _ok(port, f"/api/git/tree?rev={shas['root'][:7]}")
    assert [f["path"] for f in root["files"]] == ["a.txt", "d.txt", "img.bin"]
    monkeypatch.setattr(git_model, "TREE_MAX", 2)
    cut = _ok(port, "/api/git/tree?rev=main")
    assert len(cut["files"]) == 2 and cut["truncated"] is True
    assert "unknown revision" in _refused(port, "/api/git/tree?rev=nope")
    assert "cannot start with '-'" in _refused(port, "/api/git/tree?rev=-x")
    assert "a revision is required" in _refused(port, "/api/git/tree?rev=")


def test_tree_states_distinguish_clean_tracked_and_untracked(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """Staged changes and disk changes are modified; committed trees have no state."""
    path, _ = repo
    make_dirty(path)
    states = {f["path"]: f["state"] for f in _ok(port, "/api/git/tree?rev=WORKTREE")["files"]}
    assert states == {
        "b.txt": "modified",
        "blob.bin": "untracked",
        "dir/g.txt": "modified",
        "img.bin": "tracked",
        "logo.png": "modified",
        "new.txt": "untracked",
    }
    assert all(not f["state"] for f in _ok(port, "/api/git/tree?rev=HEAD")["files"])


def test_diff_shows_a_submodule_pointer_move(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """A gitlink is a path a diff can name; the text route still has no file there."""
    path, _shas = repo
    one, two = add_submodule_pointer(path)
    body = _ok(port, f"/api/git/diff?base={one}&head={two}&path=sub")
    assert "+Subproject commit" in body["patch"] and body["binary"] is False
    assert "no file" in _refused(port, f"/api/git/text?rev={two}&path=sub")


def test_a_rev_or_path_shaped_like_a_flag_is_refused(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    _, shas = repo
    assert "cannot start with '-'" in _refused(port, "/api/git/commit/-x")
    assert "cannot start with '-'" in _refused(port, "/api/git/compare?base=--all&head=main")
    assert "cannot start with '-'" in _refused(port, "/api/git/diff?base=v1&head=v2&path=-x")
    assert "cannot start with '-'" in _refused(
        port, "/api/git/diff?base=v1&head=v2&path=b.txt&old_path=-y"
    )
    assert "unknown revision 'nope'" in _refused(port, "/api/git/commit/nope")
    assert "unknown revision" in _refused(port, "/api/git/compare?base=main&head=nope")
    assert "a revision is required" in _refused(port, "/api/git/compare?base=&head=main")
    assert "not a commit" in _refused(
        port, f"/api/git/compare?base=main&head={git_read.EMPTY_TREE}"
    )
    assert "not a revision" in _refused(port, "/api/git/commit/a%0Ab")
    assert "no file" in _refused(port, "/api/git/diff?base=v1&head=v2&path=nope.txt")
    assert "no file" in _refused(port, "/api/git/diff?base=v1&head=v2&path=dir")  # a directory
    assert "no file" in _refused(port, "/api/git/diff?base=v1&head=v2&path=b.txt&old_path=nope.txt")
    assert "a path is required" in _refused(port, "/api/git/diff?base=v1&head=v2&path=")
    # A short sha is fine: the refusal is about shape, not length.
    assert _ok(port, f"/api/git/commit/{shas['root'][:7]}")["sha"] == shas["root"]


def test_search_finds_refs_and_messages(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    _, shas = repo
    body = _ok(port, "/api/git/search?q=feat")
    assert [r["name"] for r in body["refs"]] == ["feature"]
    # Newest first; the merge's "merge feature" matches too.
    assert [c["sha"] for c in body["commits"]] == [shas["merge"], shas["feature"]]
    assert body["commits"][1]["subject"] == "feature: rename a.txt and grow it"
    assert body["commits"][1]["short"] == shas["feature"][:7]
    assert body["truncated"] is False
    tags = _ok(port, "/api/git/search?q=V")
    assert [r["name"] for r in tags["refs"]] == ["v1", "v2"]
    everything = _ok(port, "/api/git/search?q=")
    assert len(everything["refs"]) == 7 and len(everything["commits"]) == 5
    by_sha = _ok(port, f"/api/git/search?q={shas['main'][:6]}")
    assert by_sha["commits"][0]["sha"] == shas["main"]
    assert _ok(port, "/api/git/search?q=-x") == {"refs": [], "commits": [], "truncated": False}


def test_search_caps_its_lists_and_says_so(
    repo: tuple[pathlib.Path, dict[str, str]], port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(git_model, "SEARCH_MAX", 2)
    body = _ok(port, "/api/git/search?q=")
    assert (len(body["refs"]), len(body["commits"]), body["truncated"]) == (2, 2, True)


def test_outside_a_repository_every_git_route_says_so(tmp_path: pathlib.Path) -> None:
    for port in _serve(tmp_path):
        for path in (
            "/api/git/refs",
            "/api/git/graph",
            "/api/git/search?q=x",
            "/api/git/commit/x",
            "/api/git/worktree",
            "/api/git/tree?rev=HEAD",
        ):
            assert "not a git repository" in _refused(port, path)
        status, body = _get(port, "/api/git/nope")
        assert (status, body["error"]) == (404, "not found: /api/git/nope")


def test_git_routes_need_the_token(port: int) -> None:
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/api/git/refs")
        resp = conn.getresponse()
        assert resp.status == 401
        assert json.loads(resp.read())["login"] is True
    finally:
        conn.close()


def test_the_page_has_a_git_tab_in_both_navs() -> None:
    """The rail and the phone tab bar both open the Git view, and its script is on the page."""
    assert page.PAGE_HTML.count('data-tab="git"') == 2
    assert 'href="#/git"' in page.PAGE_HTML
    assert "async function renderGit(" in page.CLIENT_JS
    assert "parts[0] === 'git'" in page.CLIENT_JS


def test_text_and_blame_share_the_digest_of_the_bytes_read(
    repo: tuple[pathlib.Path, dict[str, str]], port: int
) -> None:
    """Both routes digest the same bytes, so blame rows attach only to the text shown."""
    path, _ = repo
    text = _ok(port, "/api/git/text?rev=WORKTREE&path=b.txt")
    blame = _ok(port, "/api/git/blame?rev=WORKTREE&path=b.txt")
    expected = hashlib.sha256((path / "b.txt").read_bytes()).hexdigest()
    assert text["digest"] == blame["digest"] == expected
    (path / "b.txt").write_text("changed\n", encoding="utf-8")
    assert _ok(port, "/api/git/text?rev=WORKTREE&path=b.txt")["digest"] != expected


def test_a_file_swapped_for_a_fifo_after_its_stat_is_not_waited_on(
    repo: tuple[pathlib.Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leaf opens nonblocking: a FIFO planted between the stat and the open is refused."""
    path, _ = repo
    target = path / "b.txt"
    real_stat = os.stat

    def swapping_stat(name: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        st = real_stat(name, *args, **kwargs)
        if kwargs.get("dir_fd") is not None and name == "b.txt":
            target.unlink()
            os.mkfifo(target)
        return st

    monkeypatch.setattr(os, "stat", swapping_stat)
    assert git_files.worktree_blob(path, "b.txt", max_bytes=1 << 20) is None
