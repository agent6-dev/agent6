# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web composer sends images with a task, a steer and a resume; the server checks each."""

from __future__ import annotations

import base64
import json
import pathlib
import threading
from collections.abc import Iterator
from http import client
from typing import Any

import pytest

from agent6 import kinds, media, paths
from agent6.sessions import attachments, ipc
from agent6.ui import operator_images, spawn
from agent6.ui.web import page, server
from tests.unit import test_operator_images
from tests.web import session_logs, signin

png = test_operator_images.png


def _body(*shades: int, media_type: str = "image/png") -> list[dict[str, str]]:
    return [
        {"media_type": media_type, "data": base64.b64encode(png(shade)).decode("ascii")}
        for shade in shades
    ]


@pytest.fixture
def port(tmp_path: pathlib.Path) -> Iterator[int]:
    srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", secret=signin.SECRET)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


def _post(port: int, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    conn = signin.SignedIn(port)
    try:
        conn.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_a_steer_stores_its_image_and_queues_the_reference(
    port: int, tmp_path: pathlib.Path
) -> None:
    """The steer route took text alone: a pasted screenshot had no way to the run."""
    session = session_logs.write_run(tmp_path, "live-AAA111", [], live=True)

    status, data = _post(
        port, "/api/session/live-AAA111/steer", {"text": "what is this?", "images": _body(3)}
    )

    assert (status, data["ok"]) == (200, True)
    (queued,) = ipc.queued_steers(session)
    (ref,) = queued.images
    assert queued.text == f"what is this?\n\n[{ref.label()}]"
    assert attachments.load(session, ref).data == png(3)
    assert [p.name for p in (session / "attachments").iterdir()] == [ref.sha256]
    # An image with no words is a message too.
    status, data = _post(port, "/api/session/live-AAA111/steer", {"images": _body(4)})
    assert (status, data["ok"]) == (200, True)
    assert ipc.queued_steers(session)[1].text.startswith("[image image/png")


@pytest.mark.parametrize(
    ("images", "why"),
    [
        (_body(1, media_type="image/jpeg"), "image 1 was sent as 'image/jpeg' but is image/png"),
        ([{"media_type": "image/png", "data": "@@not base64@@"}], "image 1 is not valid base64"),
        (
            [{"media_type": "image/svg+xml", "data": base64.b64encode(b"<svg/>").decode()}],
            "image 1 is not a static PNG, JPEG, GIF or WebP image of at most 2 MiB",
        ),
        (
            [{"media_type": "application/pdf", "data": base64.b64encode(b"%PDF-1.7").decode()}],
            "image 1 is a PDF; name its path in the workspace and the model reads it",
        ),
        (_body(1, 2, 3, 4, 5), "a message carries at most four images"),
    ],
)
def test_a_refused_image_stores_nothing_and_says_why(
    port: int, tmp_path: pathlib.Path, images: list[dict[str, str]], why: str
) -> None:
    """Only a static PNG, JPEG, GIF or WebP whose bytes are what it claims is taken."""
    session = session_logs.write_run(tmp_path, "live-AAA111", [], live=True)

    status, data = _post(port, "/api/session/live-AAA111/steer", {"text": "look", "images": images})

    assert (status, data["error"]) == (422, f"not sent: {why}")
    assert not (session / "attachments").exists()
    assert ipc.queued_steers(session) == ()


def test_an_oversized_image_is_refused_before_it_is_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base64 length bounds an image: a decode would hold the whole payload first."""
    decoded: list[int] = []
    real = base64.b64decode

    def spy(data: Any, *args: Any, **kwargs: Any) -> bytes:
        decoded.append(len(data))
        return real(data, *args, **kwargs)

    monkeypatch.setattr(base64, "b64decode", spy)
    oversized = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(media.IMAGE_MAX_BYTES)).decode()

    with pytest.raises(operator_images.ImageRefusedError, match="image 1 is over the 2 MiB limit"):
        operator_images.decode([("image/png", oversized)])

    assert decoded == []


def test_a_body_larger_than_a_messages_images_is_refused_unread(port: int) -> None:
    """The image routes take a larger body than the megabyte every other route allows."""
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        for path, limit in (
            ("/api/session/x/steer", 1048576 + operator_images.ENCODED_TOTAL_MAX),
            ("/api/session/x/answer", 1048576),
        ):
            conn.putrequest("POST", path)
            conn.putheader("Content-Length", str(limit + 1))
            conn.endheaders()
            resp = conn.getresponse()
            assert (resp.status, json.loads(resp.read())["error"]) == (
                413,
                f"body larger than {limit} bytes",
            )
            conn.close()
    finally:
        conn.close()


def test_an_answer_and_a_btw_take_no_image_and_say_so(port: int, tmp_path: pathlib.Path) -> None:
    """An answer is text and a `/btw` is its own session: neither carries an image."""
    session = session_logs.write_run(tmp_path, "live-AAA111", [], live=True)

    status, data = _post(
        port,
        "/api/session/live-AAA111/answer",
        {"id": "q1", "answers": ["yes"], "images": _body(1)},
    )
    assert (status, data["error"]) == (
        422,
        "an answer takes no image; send the image as an instruction",
    )
    status, data = _post(
        port, "/api/session/live-AAA111/steer", {"text": "/btw what is this?", "images": _body(1)}
    )
    assert (status, data["error"]) == (
        422,
        "`/btw` takes no image; send the image with an ordinary instruction",
    )
    assert not (session / "attachments").exists()


def test_an_upload_needs_the_sign_in_every_write_needs(port: int, tmp_path: pathlib.Path) -> None:
    session = session_logs.write_run(tmp_path, "live-AAA111", [], live=True)
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps({"text": "look", "images": _body(1)}).encode()
        conn.request(
            "POST", "/api/session/live-AAA111/steer", body, {"Content-Type": "application/json"}
        )
        assert conn.getresponse().status == 401
    finally:
        conn.close()
    assert not (session / "attachments").exists()


def test_new_work_reserves_the_session_and_launches_it_with_the_images(
    port: int, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An opening's image is stored before the run exists, in the directory the run will use."""
    launched: list[list[str]] = []

    def fake_spawn(
        argv: list[str], _cwd: pathlib.Path, **_kwargs: object
    ) -> tuple[pathlib.Path, str]:
        launched.append(argv)
        session_id = argv[argv.index("--session-id") + 1]
        return paths.state_dir(tmp_path) / "sessions" / "runs" / session_id, ""

    monkeypatch.setattr(spawn, "spawn_and_locate", fake_spawn)

    status, data = _post(
        port, "/api/new", {"mode": "run", "task": "match this mockup", "images": _body(7)}
    )

    assert (status, data["ok"]) == (200, True)
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / data["session_id"]
    (blob,) = (session / "attachments").iterdir()
    assert blob.read_bytes() == png(7)
    task = launched[0][-1]
    assert launched[0][-4:-1] == ["--session-id", session.name, "--"]
    assert task.startswith("match this mockup\n\n[image image/png, 1 KiB, sha256 ")
    (ref,) = ipc.take_launch_images(session, task)
    assert ref.sha256 == blob.name
    # An ask reserves no session, and a /parallel task has several: both refuse in words.
    status, data = _post(port, "/api/new", {"mode": "ask", "task": "what?", "images": _body(1)})
    assert status == 422 and data["error"].startswith("an ask takes no image")
    status, data = _post(
        port, "/api/new", {"mode": "run", "task": "/parallel 2 -- go", "images": _body(1)}
    )
    assert status == 422 and "drop the worktree, the seed session and the images" in data["error"]


def test_a_launch_that_never_starts_leaves_no_image_behind(
    port: int, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(_argv: list[str], _cwd: pathlib.Path, **_kwargs: object) -> tuple[None, str]:
        return None, "the run refused to start"

    monkeypatch.setattr(spawn, "spawn_and_locate", refused)

    status, data = _post(port, "/api/new", {"mode": "run", "task": "t", "images": _body(1)})

    assert (status, data["error"]) == (422, "the run refused to start")
    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []


def test_a_resume_stores_the_images_and_launches_with_their_labels(
    port: int, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ended session's composer sends a resume, which carries an image as a steer does."""
    session = session_logs.write_run(tmp_path, "ended-AAA111", [{"type": "session.stop"}])
    launched: list[list[str]] = []

    def fake_confirm(argv: list[str], _cwd: pathlib.Path, **_kwargs: object) -> str:
        launched.append(argv)
        return ""

    monkeypatch.setattr(spawn, "spawn_and_confirm", fake_confirm)

    status, data = _post(
        port, "/api/session/ended-AAA111/resume", {"text": "now fix this", "images": _body(2)}
    )

    assert (status, data["ok"]) == (200, True)
    steer = next(a for a in launched[0] if a.startswith("--steer=")).removeprefix("--steer=")
    (ref,) = ipc.take_launch_images(session, steer)
    assert steer == f"now fix this\n\n[{ref.label()}]"
    assert attachments.load(session, ref) == kinds.MediaAttachment(png(2), "image/png", "")


def test_the_clients_limits_are_the_servers() -> None:
    """The page checks an image before it uploads it; its numbers are `agent6.media`'s."""
    assert f"IMAGE_MAX_BYTES = {media.IMAGE_MAX_BYTES // (1024 * 1024)} * 1024 * 1024" in (
        page.CLIENT_JS
    )
    assert f"IMAGE_MAX_COUNT = {media.IMAGE_MAX_COUNT};" in page.CLIENT_JS
    assert "['image/png', 'image/jpeg', 'image/gif', 'image/webp']" in page.CLIENT_JS
