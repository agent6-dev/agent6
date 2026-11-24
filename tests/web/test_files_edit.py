from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
from collections.abc import Iterator
from concurrent import futures
from http import client
from typing import Any
from urllib import parse

import pytest

from agent6.ui.web import server
from tests.unit import test_git_history
from tests.web import signin


@pytest.fixture
def served(tmp_path: pathlib.Path) -> Iterator[tuple[pathlib.Path, int]]:
    test_git_history.build_history(tmp_path)
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield tmp_path, srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()


def request(
    port: int, method: str, path: str = "plan.md", *, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    conn = signin.SignedIn(port)
    try:
        conn.request(
            method,
            "/api/files/text?" + parse.urlencode({"path": path}),
            body=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_edit_reads_complete_text_and_saves_against_its_digest(
    served: tuple[pathlib.Path, int],
) -> None:
    repo, port = served
    original = b"# Plan\r\n" + b"a\r\n" * 400_000
    (repo / "plan.md").write_bytes(original)
    status, snapshot = request(port, "GET")
    assert status == 200, snapshot
    assert snapshot == {
        "path": "plan.md",
        "text": original.decode().replace("\r\n", "\n"),
        "digest": hashlib.sha256(original).hexdigest(),
    }
    snapshot["text"] = "# Edited\n" + "a\n" * 600_000
    status, saved = request(port, "POST", body=snapshot)
    assert status == 200, saved
    assert (repo / "plan.md").read_bytes() == snapshot["text"].replace("\n", "\r\n").encode()
    assert saved["digest"] == hashlib.sha256((repo / "plan.md").read_bytes()).hexdigest()


def test_conflict_needs_the_new_digest_and_create_never_replaces(
    served: tuple[pathlib.Path, int],
) -> None:
    repo, port = served
    _, snapshot = request(port, "GET")
    snapshot["text"] = "mine\n"
    (repo / "plan.md").write_text("theirs\n", encoding="utf-8")
    status, conflict = request(port, "POST", body=snapshot)
    assert status == 409, conflict
    assert (repo / "plan.md").read_text() == "theirs\n"
    _, latest = request(port, "GET")
    snapshot["digest"] = latest["digest"]
    assert request(port, "POST", body=snapshot)[0] == 200
    copy = {"path": "copy.md", "text": "copy\n", "digest": None}
    assert request(port, "POST", body=copy)[0] == 200
    assert request(port, "POST", body=copy)[0] == 409
    assert (repo / "copy.md").read_text() == "copy\n"
    (repo / "plan.md").unlink()
    assert request(port, "POST", body=snapshot)[0] == 409
    assert not (repo / "plan.md").exists()


@pytest.mark.parametrize(
    "path",
    ["../escape", "/tmp/escape", ".git/config", "dir/.secret", ".hidden/file", "a/../b", "a//b"],
)
def test_hidden_and_escaping_paths_are_refused(served: tuple[pathlib.Path, int], path: str) -> None:
    _, port = served
    assert request(port, "POST", body={"path": path, "text": "bad", "digest": None})[0] == 422


@pytest.mark.parametrize("kind", ["leaf", "ancestor", "hardlink", "fifo", "directory"])
def test_writes_refuse_links_and_special_files(served: tuple[pathlib.Path, int], kind: str) -> None:
    repo, port = served
    protected = repo / ".git" / "protected"
    protected.write_text("keep", encoding="utf-8")
    target = repo / "target"
    path = "target"
    if kind == "leaf":
        target.symlink_to(protected)
    elif kind == "ancestor":
        target.symlink_to(repo / ".git", target_is_directory=True)
        path += "/protected"
    elif kind == "hardlink":
        target.hardlink_to(protected)
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    body = {"path": path, "text": "bad", "digest": hashlib.sha256(b"keep").hexdigest()}
    status, response = request(port, "POST", body=body)
    assert status == 422, response
    assert protected.read_text() == "keep"
    assert request(port, "GET", path)[0] == 422


@pytest.mark.parametrize(
    "content",
    [b"a\x00b", b"\xff", b"x" * (20 * 1024 * 1024 + 1)],
    ids=["binary", "invalid-utf8", "too-large"],
)
def test_binary_invalid_utf8_and_oversized_files_are_not_editable(
    served: tuple[pathlib.Path, int], content: bytes
) -> None:
    repo, port = served
    (repo / "plan.md").write_bytes(content)
    assert request(port, "GET")[0] in (413, 422)
    body = {"path": "plan.md", "text": "bad", "digest": hashlib.sha256(content).hexdigest()}
    assert request(port, "POST", body=body)[0] in (413, 422)
    assert (repo / "plan.md").read_bytes() == content


def test_only_signed_in_requests_can_read_or_write_text(served: tuple[pathlib.Path, int]) -> None:
    repo, port = served
    for method in ("GET", "POST"):
        conn = client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            method, "/api/files/text?path=plan.md", body=b"{}" if method == "POST" else None
        )
        response = conn.getresponse()
        assert response.status == 401
        response.read()
        conn.close()
    assert (repo / "plan.md").read_text() == "# Plan\n"


def test_concurrent_saves_cannot_both_use_the_same_digest(
    served: tuple[pathlib.Path, int],
) -> None:
    _, port = served
    _, snapshot = request(port, "GET")
    snapshot["text"] = "changed\n"
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(request, port, "POST", body=snapshot) for _ in range(2)]
        replies = [future.result() for future in pending]
    assert sorted(status for status, _ in replies) == [200, 409]


def test_invalid_new_text_leaves_no_file(served: tuple[pathlib.Path, int]) -> None:
    repo, port = served
    for text in ("a\x00b", "\ud800", "x" * (20 * 1024 * 1024 + 1)):
        status, _ = request(port, "POST", body={"path": "new.txt", "text": text, "digest": None})
        assert status in (413, 422)
        assert not (repo / "new.txt").exists()


def test_a_server_outside_a_git_working_tree_cannot_write(tmp_path: pathlib.Path) -> None:
    from agent6.ui.web import files_edit

    with pytest.raises(files_edit.FileEditError, match="working tree"):
        files_edit.save(tmp_path, files_edit.FileTextBody(path="new.txt", text="bad", digest=None))
    assert not (tmp_path / "new.txt").exists()


def test_the_write_body_cannot_name_a_revision_or_bypass_the_digest(
    served: tuple[pathlib.Path, int],
) -> None:
    _, port = served
    for body in (
        {"path": "plan.md", "text": "bad"},
        {"path": "plan.md", "text": "bad", "digest": "not a digest"},
        {"path": "plan.md", "text": "bad", "digest": None, "rev": "HEAD"},
        {"path": "plan.md", "text": "bad", "digest": None, "force": True},
    ):
        assert request(port, "POST", body=body)[0] == 400
