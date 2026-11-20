from __future__ import annotations

import hashlib
import json
import pathlib
from http import client

from tests.unit import test_git_history
from tests.web import test_files_edit, test_files_preview

served = test_files_edit.served


def submit(
    port: int,
    url: str,
    fields: str,
    *,
    origin: str = "null",
    destination: str = "iframe",
    mime: str = "application/x-www-form-urlencoded",
) -> int:
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(
            "POST",
            url,
            fields,
            {"Sec-Fetch-Dest": destination, "Origin": origin, "Content-Type": mime},
        )
        response = conn.getresponse()
        assert response.getheader("Referrer-Policy") == "no-referrer"
        assert "sandbox" in (response.getheader("Content-Security-Policy") or "")
        response.read()
        return response.status
    finally:
        conn.close()


def test_form_submits_one_proposal_and_cannot_choose_its_destination(
    served: tuple[pathlib.Path, int],
) -> None:
    """A submission cannot write a file or replace the proposal the UI already picked up."""
    repo, port = served
    folder = repo / "forms"
    folder.mkdir()
    (folder / "choose.v2.html").write_text("<form method=post></form>")
    _, page = test_files_preview.api(port, rev="WORKTREE", path="forms/choose.v2.html")
    url, key = page["url"], page["key"]
    assert submit(port, url, "x=one", destination="document") == 403
    assert submit(port, url, "x=one", origin="https://elsewhere.test") == 403
    assert submit(port, url, "x=one", mime="application/json") == 415
    assert submit(port, url.replace("choose.v2.html", "other.html"), "x=one") == 403
    assert submit(port, url, "x=first") == 200
    assert submit(port, url, "pick=red&pick=green&blank=&path=other.txt") == 200
    answers = folder / "choose.v2.answers.json"
    assert not answers.exists()
    status, result = test_files_preview.api(port, "/answers", key=key)
    assert status == 200
    proposal = result["pending"]
    assert proposal["path"] == "forms/choose.v2.answers.json"
    assert proposal["digest"] is None
    assert json.loads(proposal["text"]) == {
        "pick": ["red", "green"],
        "blank": "",
        "path": "other.txt",
    }
    assert test_files_preview.api(port, "/answers", key=key) == (200, {"pending": None})
    answers.write_text("external change")
    assert test_files_edit.request(port, "POST", body=proposal)[0] == 409
    assert answers.read_text() == "external change"
    assert submit(port, url, "pick=last") == 200
    assert test_files_preview.api(port, "/close", key=key)[0] == 200
    assert test_files_preview.api(port, "/answers", key=key)[0] == 403
    assert submit(port, url, "x=one") == 403
    assert not (folder / "other.txt").exists()


def test_form_refuses_markdown_and_history(served: tuple[pathlib.Path, int]) -> None:
    """Only the opened working-tree HTML file can submit answers."""
    repo, port = served
    (repo / "index.html").write_text("<h1>Guide</h1>")
    test_git_history._git(repo, "add", "index.html")
    test_git_history._git(repo, "commit", "-qm", "guide")
    for rev, path in (("HEAD", "index.html"), ("WORKTREE", "plan.md")):
        status, page = test_files_preview.api(
            port, rev=rev, path=path, digest=hashlib.sha256((repo / path).read_bytes()).hexdigest()
        )
        assert status == 200, page
        assert submit(port, page["url"], "x=one") == 403
        _, headers, _ = test_files_preview.get(port, page["url"], "iframe")
        assert "form-action 'none'" in headers["Content-Security-Policy"]
