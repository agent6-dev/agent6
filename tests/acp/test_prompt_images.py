# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An ACP prompt's image blocks reach the run it starts or resumes."""

from __future__ import annotations

import base64
import pathlib
import time
from typing import Any

import pytest

from agent6 import paths
from agent6.app import resume, run
from agent6.app.session import setup as session_setup
from agent6.sessions import attachments, ipc
from agent6.ui.acp import session as session_mod
from tests.acp import test_runner
from tests.unit import test_operator_images

png = test_operator_images.png


def _image(shade: int = 0, **extra: Any) -> dict[str, Any]:
    data = base64.b64encode(png(shade)).decode("ascii")
    return {"type": "image", "mimeType": "image/png", "data": data, **extra}


def _reply(wire: test_runner._Wire) -> dict[str, Any]:
    """Return the answer to the prompt, past the updates sent before it."""
    reply = wire.until("")
    while "result" not in reply and "error" not in reply:
        reply = wire.until("")
    return reply


def test_the_handshake_says_a_prompt_may_carry_an_image() -> None:
    wire = test_runner._Wire()
    try:
        wire.send(id=1, method="initialize", params={"protocolVersion": 1})
        capabilities = wire.recv()["result"]["agentCapabilities"]["promptCapabilities"]
        assert capabilities == {"embeddedContext": False, "image": True}
    finally:
        wire.close()


def test_a_prompts_image_is_stored_for_the_run_it_starts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image block was dropped: the run started from the prompt's text alone."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_setup, "load_session_config", test_runner._loaded)
    started: list[tuple[str, bytes]] = []

    def _run_task(_cfg: object, task: str, **kw: Any) -> int:
        session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / str(kw["session_id"])
        (ref,) = ipc.take_launch_input(session_dir, task).images
        started.append((task, attachments.load(session_dir, ref).data))
        return 0

    monkeypatch.setattr(run, "run_task", _run_task)
    wire = test_runner._Wire()
    try:
        session_id = wire.new_session(test_runner._repo(tmp_path))
        blocks = [
            {"type": "text", "text": "match this mockup"},
            # The uri is the editor's: only the inline data is read.
            _image(5, uri="file:///etc/hostname"),
        ]
        wire.send(id=3, method="session/prompt", params={"sessionId": session_id, "prompt": blocks})
        assert _reply(wire).get("result") == {"stopReason": "end_turn"}
    finally:
        wire.close()
    ((task, stored),) = started
    assert task.startswith("match this mockup\n\n[image image/png, 1 KiB, sha256 ")
    assert stored == png(5)


def test_a_follow_up_prompts_image_goes_with_the_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later prompt resumes the run with a steer, which carries an image as a task does."""
    monkeypatch.chdir(tmp_path)
    cwd = test_runner._repo(tmp_path / "repo")
    session = session_mod.Session(acp_id="one", session_id="one", cwd=cwd)
    session.images = (("image/png", base64.b64encode(png(6)).decode("ascii")),)
    session_dir = session.layout(paths.state_dir(cwd)).session_dir
    resumed: list[str] = []

    def _resume_task(*_args: object, **kw: Any) -> int:
        resumed.append(str(kw["steer"]))
        return 0

    monkeypatch.setattr(resume, "resume_task", _resume_task)

    test_runner._bridge({})._run(session, "", resuming=True, started_at=time.time())

    (steer,) = resumed
    (ref,) = ipc.take_launch_input(session_dir, steer).images
    assert steer == f"[{ref.label()}]"  # an image alone is a prompt
    assert attachments.load(session_dir, ref).data == png(6)


@pytest.mark.parametrize(
    ("blocks", "why"),
    [
        ([_image(1) | {"mimeType": "image/jpeg"}], "image 1 was sent as 'image/jpeg'"),
        (
            [{"type": "image", "mimeType": "image/svg+xml", "data": "PHN2Zy8+"}],
            "image 1 is not a static PNG, JPEG, GIF or WebP image",
        ),
        ([{"type": "image", "mimeType": "image/png"}], "block 1 data must be a string"),
        ([_image(n) for n in range(5)], "at most four images"),
        ([{"type": "text", "text": "/review main"}, _image(1)], "a directive takes no image"),
    ],
)
def test_a_refused_image_answers_the_prompt_with_the_reason(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, blocks: list[Any], why: str
) -> None:
    """The refusal reaches the editor as the request's error, and nothing is stored."""
    monkeypatch.chdir(tmp_path)
    wire = test_runner._Wire()
    try:
        session_id = wire.new_session(test_runner._repo(tmp_path))
        wire.send(id=3, method="session/prompt", params={"sessionId": session_id, "prompt": blocks})
        reply = _reply(wire)
    finally:
        wire.close()
    assert why in reply["error"]["message"]
    assert not (paths.state_dir(tmp_path) / "sessions" / "runs" / session_id).exists()
