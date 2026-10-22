# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Git tab's `/api/git/*` endpoints over a repository the test builds.

Every revision and path arrives from the browser, so the pins here are the refusals as
much as the payloads: a flag-shaped rev or path, an unknown rev, a path in neither tree.
"""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Iterator
from http import client
from typing import Any

import pytest

from agent6.ui.web import git_model, page
from agent6.ui.web import server as web_server
from agent6.vcs import git_ops
from tests.unit.test_git_history import build_history
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
    assert merge["date"].startswith("2026-01-02T10:") and len(merge["when"]) == 11
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
    assert root["base"] == git_ops.EMPTY_TREE and root["parents"] == []
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
    assert sorted(f["path"] for f in across["files"]) == ["b.txt", "d.txt", "dir/f.txt", "img.bin"]
    empty = _ok(port, f"/api/git/compare?base={git_ops.EMPTY_TREE}&head=v1")
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
    assert "not a commit" in _refused(port, f"/api/git/compare?base=main&head={git_ops.EMPTY_TREE}")
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
        for path in ("/api/git/refs", "/api/git/graph", "/api/git/search?q=x", "/api/git/commit/x"):
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
