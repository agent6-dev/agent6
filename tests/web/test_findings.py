from __future__ import annotations

import json
import os
import pathlib
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from agent6 import paths
from agent6.app import review_actions
from agent6.sessions import event_log, layout, lock, process
from agent6.ui import spawn
from agent6.ui.web import server
from tests import review_helpers
from tests.web import signin


@pytest.fixture
def api(tmp_path: pathlib.Path) -> Iterator[int]:
    review_helpers.reviewed_session(tmp_path)
    srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()


def _request(
    port: int, path: str = "", body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    conn = signin.SignedIn(port, timeout=10)
    try:
        conn.request(
            "GET" if body is None else "POST",
            "/api/session/review-run/findings" + path,
            None if body is None else json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_findings_routes_share_versions_and_idempotent_results(api: int) -> None:
    code, view = _request(api)
    assert code == 200
    assert len(view["findings"]) == 2 and view["version"] == 1
    body = dict(
        request_id="resolve",
        expected_version=1,
        resolution=dict(disposition="fixed", reason="Reproduced and fixed"),
    )
    code, result = _request(api, "/f1/resolve", body)
    assert code == 200 and result["status"] == "resolved"
    assert _request(api, "/f1/resolve", body) == (code, result)
    stale = {**body, "request_id": "stale"}
    assert _request(api, "/f2/resolve", stale)[0] == 409
    assert len(_request(api)[1]["findings"]) == 1
    assert len(_request(api, "?status=all")[1]["findings"]) == 2
    assert (
        _request(
            api,
            "/f1/reopen",
            dict(request_id="reopen", expected_version=2, reason="Still reproducible"),
        )[0]
        == 200
    )


def test_findings_routes_return_line_evidence(tmp_path: pathlib.Path, api: int) -> None:
    code, legacy = _request(api)
    assert code == 200
    assert legacy["findings"][0]["line_origin"]["kind"] == "unknown"
    review_helpers.reviewed_repo(tmp_path)
    code, view = _request(api)
    assert code == 200
    assert [f["line_origin"]["kind"] for f in view["findings"]] == [
        "pre_existing",
        "changed",
        "changed",
        "unknown",
    ]
    origin = view["findings"][0]["line_origin"]
    assert origin["base_location"]["path"] == "old.txt"
    assert origin["commits"] == [view["panels"][0]["target"]["base_commit"]]
    assert len(view["merge_guard"]["open_block_ids"]) == 4


@pytest.mark.parametrize("acknowledge_before_return", [False, True])
def test_confirmed_resume_returns_shared_receipt(
    tmp_path: pathlib.Path,
    api: int,
    monkeypatch: pytest.MonkeyPatch,
    acknowledge_before_return: bool,
) -> None:
    saved = layout.SessionLayout(paths.state_dir(tmp_path), "review-run")
    fd: int | None = None

    def resume(cwd: pathlib.Path, session_id: str, *, steer: str, **kwargs: object) -> str:
        nonlocal fd
        assert cwd == tmp_path and session_id == saved.session_id
        assert "Review finding f1" in steer
        fd = lock.acquire_single_writer(saved.session_dir)
        assert fd is not None
        process.write_worker_pid(saved.session_dir, os.getpid())
        if acknowledge_before_return:
            review_actions.drain(
                event_log.EventSink(saved.logs_path), execution_id="resumed", resume_steer=steer
            )
        return ""

    monkeypatch.setattr(spawn, "spawn_detached_resume", resume)
    try:
        code, result = _request(
            api, "/f1/fix", dict(request_id="web-resume", expected_version=1, resume=True)
        )
        assert code == (200 if acknowledge_before_return else 202)
        assert result == dict(
            request_id="web-resume",
            finding_id="f1",
            version=3 if acknowledge_before_return else 2,
            status="submitted" if acknowledge_before_return else "queued",
            message="run resumed to fix finding f1"
            + ("" if acknowledge_before_return else "; submission acknowledgement pending"),
        )
        view = review_actions.read_review(saved.session_dir)
        assert view.version == result["version"]
        assert view.findings[0].status == "open"
        assert len(view.findings[0].actions) == (2 if acknowledge_before_return else 1)
        assert view.findings[0].actions[0].source == "web"
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)


def test_findings_routes_distinguish_missing_invalid_and_conflicting(api: int) -> None:
    base = dict(request_id="fix", expected_version=1)
    assert _request(api, "/missing/fix", base)[0] == 404
    code, result = _request(api, "/f1/fix", base)
    assert code == 409 and "Resume to fix" in result["error"]
    assert (
        _request(api, "/f1/resolve", {**base, "resolution": dict(disposition="fixed", reason=" ")})[
            0
        ]
        == 422
    )
    assert _request(api, "/f1/reopen", {**base, "reason": "x" * 2001})[0] == 422
    assert _request(api, "?status=bogus")[0] == 422
