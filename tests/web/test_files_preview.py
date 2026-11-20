from __future__ import annotations

import json
import pathlib
from http import client
from typing import Any

import pytest

from agent6.ui.web import files_preview
from tests.unit import test_git_history
from tests.web import signin, test_files_edit

served = test_files_edit.served


def api(port: int, action: str = "", **body: Any) -> tuple[int, dict[str, Any]]:
    conn = signin.SignedIn(port)
    try:
        conn.request(
            "POST",
            "/api/files/preview" + action,
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def get(port: int, url: str, destination: str | None) -> tuple[int, dict[str, str], bytes]:
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(
            "GET", url, headers={"Sec-Fetch-Dest": destination} if destination is not None else {}
        )
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def test_preview_is_static_confined_and_revoked_on_close(served: tuple[pathlib.Path, int]) -> None:
    """A preview key serves only confined resources, never standalone executable documents."""
    repo, port = served
    (repo / "index.html").write_text("<h1>Preview</h1><script>window.ran = true</script>")
    (repo / "site.css").write_text("h1 { color: green }")
    status, opened = api(port, rev="WORKTREE", path="index.html")
    assert status == 200, opened
    status, headers, content = get(port, opened["url"], "iframe")
    assert status == 200
    assert b"<h1>Preview</h1>" in content
    policy = headers["Content-Security-Policy"]
    for directive in (
        "sandbox",
        "default-src 'none'",
        "script-src 'none'",
        "form-action 'none'",
        "base-uri 'none'",
        "object-src 'none'",
        "frame-ancestors 'self'",
    ):
        assert directive in policy
    assert "allow-scripts" not in policy
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-DNS-Prefetch-Control"] == "off"
    assert get(port, opened["url"].replace("index.html", "site.css"), "style")[0] == 200
    assert api(port, "/close", key=opened["key"])[0] == 200
    assert get(port, opened["url"], "iframe")[0] == 403


@pytest.mark.parametrize("destination", [None, "document", "object", "embed", "script", "worker"])
def test_preview_refuses_non_frame_documents(
    served: tuple[pathlib.Path, int], destination: str | None
) -> None:
    repo, port = served
    (repo / "index.html").write_text("<h1>Private preview</h1>")
    status, opened = api(port, rev="WORKTREE", path="index.html")
    assert status == 200, opened
    status, headers, content = get(port, opened["url"], destination)
    assert status == 403
    assert b"Previews open inside the Files page" in content
    assert b"Private preview" not in content
    assert "sandbox" in headers["Content-Security-Policy"]
    assert headers["X-DNS-Prefetch-Control"] == "off"


@pytest.mark.parametrize(
    "path", [".hidden.html", ".git/config", "alias.html", "linked/page.html", "../outside.html"]
)
def test_preview_refuses_hidden_and_symlink_paths(
    served: tuple[pathlib.Path, int], path: str
) -> None:
    repo, port = served
    (repo / ".hidden.html").write_text("secret")
    (repo / "alias.html").symlink_to(repo / ".hidden.html")
    (repo / "linked").symlink_to(repo / ".git", target_is_directory=True)
    assert api(port, rev="WORKTREE", path=path)[0] == 422


def test_preview_pins_revision_and_ignored_permission(served: tuple[pathlib.Path, int]) -> None:
    repo, port = served
    (repo / "index.html").write_text("old")
    test_git_history._git(repo, "add", "index.html")
    test_git_history._git(repo, "commit", "-qm", "html")
    status, opened = api(port, rev="HEAD", path="index.html")
    assert status == 200, opened
    (repo / "index.html").write_text("new")
    test_git_history._git(repo, "commit", "-qam", "changed html")
    assert get(port, opened["url"], "iframe")[2] == b"old"
    (repo / ".gitignore").write_text("ignored.html\n")
    (repo / "ignored.html").write_text("ignored")
    assert api(port, rev="WORKTREE", path="ignored.html")[0] == 422
    assert api(port, rev="WORKTREE", path="ignored.html", ignored=True)[0] == 200
    _, working = api(port, rev="WORKTREE", path="index.html")
    address = working["url"].replace("index.html", "ignored.html") + "?ignored=1&rev=HEAD"
    assert get(port, address, "iframe")[0] == 422


def test_preview_actions_require_the_sign_in_header(served: tuple[pathlib.Path, int]) -> None:
    repo, port = served
    (repo / "index.html").write_text("private")
    _, opened = api(port, rev="WORKTREE", path="index.html")
    for action, body in (
        ("", {"rev": "WORKTREE", "path": "index.html"}),
        ("/renew", {"key": opened["key"]}),
        ("/close", {"key": opened["key"]}),
    ):
        conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            conn.request(
                "POST",
                "/api/files/preview" + action,
                json.dumps(body),
                {"Content-Type": "application/json", "Origin": "null"},
            )
            response = conn.getresponse()
            assert response.status == 401
            response.read()
        finally:
            conn.close()
    assert get(port, opened["url"], "iframe")[0] == 200
    assert api(port, "/renew", key=opened["key"])[0] == 200


def test_preview_expiry_cannot_be_renewed_from_its_address(
    served: tuple[pathlib.Path, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, port = served
    (repo / "index.html").write_text("private")
    monkeypatch.setattr(files_preview, "LEASE_SECONDS", -1)
    _, opened = api(port, rev="WORKTREE", path="index.html")
    assert get(port, opened["url"] + "?renew=1", "iframe")[0] == 403
    assert api(port, "/renew", key=opened["key"])[0] == 403


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "PUT"])
def test_preview_method_refusals_keep_the_sandbox_policy(
    served: tuple[pathlib.Path, int], method: str
) -> None:
    _, port = served
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, "/preview/unknown/index.html")
        response = conn.getresponse()
        assert response.status == 403
        assert response.getheader("Content-Security-Policy") == files_preview.DENY_CSP
        assert response.getheader("Referrer-Policy") == "no-referrer"
        response.read()
    finally:
        conn.close()


@pytest.mark.parametrize(
    "path",
    [".hidden.html", "alias.html", "linked/config", ".git/config", "../index.html", "large.html"],
)
def test_preview_resources_follow_the_files_reader(
    served: tuple[pathlib.Path, int], path: str
) -> None:
    repo, port = served
    (repo / "index.html").write_text("private")
    (repo / ".hidden.html").write_text("hidden")
    (repo / "alias.html").symlink_to(repo / ".hidden.html")
    (repo / "linked").symlink_to(repo / ".git")
    (repo / "large.html").write_bytes(b"x" * (20 * 1024 * 1024 + 1))
    _, opened = api(port, rev="WORKTREE", path="index.html")
    url = opened["url"].replace("index.html", path)
    status, headers, _content = get(port, url, "empty")
    assert status == (413 if path == "large.html" else 422)
    assert headers["Content-Security-Policy"] == files_preview.DENY_CSP
