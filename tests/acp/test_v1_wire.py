# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pin the frames a v1 editor receives for the recorded golden run.

Two routes reach the wire: a prompt whose run writes the journal, and `session/load`
replaying it. Each frame is compared with `data/v1-prompt.json` and `data/v1-load.json`,
so a change to the layer behind the wire shows as a difference here.

Regenerate only with a deliberate, reviewed change to the v1 wire:

.venv/bin/python -m tests.acp.test_v1_wire
"""

from __future__ import annotations

import functools
import json
import pathlib
import tempfile
import time
from typing import Any

import pytest

from agent6 import paths
from agent6.app import run
from agent6.app.session import setup as session_setup
from agent6.sessions import layout as sessions_layout
from tests.acp import test_runner, test_session_lifecycle

_DATA = pathlib.Path(__file__).parent / "data"
_RECORDED = pathlib.Path(__file__).parent.parent / "unit" / "data" / "golden_session_logs.jsonl"
_RUN = "pinned-run-AAAAAA"


def _normal(frames: list[dict[str, Any]], names: dict[str, str]) -> list[dict[str, Any]]:
    """Return the frames with each run-specific string replaced by its stable name."""
    text = json.dumps(frames, ensure_ascii=False)
    for value, name in names.items():
        text = text.replace(json.dumps(value)[1:-1], name)
    stable: list[dict[str, Any]] = json.loads(text)
    for frame in stable:
        # The release number changes with every release; the frame's shape is what is pinned.
        if "agentInfo" in frame.get("result", {}):
            frame["result"]["agentInfo"]["version"] = "<version>"
    return stable


def _load_frames(tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    """Return what `session/load` sends for the golden run, the reply last."""
    cwd = test_session_lifecycle._repo(tmp_path)
    saved = test_session_lifecycle._record(cwd, _RUN, "Fix the bug", mtime=time.time())
    (saved / "logs.jsonl").write_bytes(_RECORDED.read_bytes())
    frames = test_session_lifecycle._drive(
        test_session_lifecycle._msg(1, "initialize", protocolVersion=1, clientCapabilities={})
        + b"\n"
        + test_session_lifecycle._msg(2, "session/load", cwd=str(cwd), sessionId=_RUN)
        + b"\n",
        test_session_lifecycle._sessions(None),
    )
    return _normal(frames, {str(cwd): "<cwd>"})


def _prompt_frames(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Return what one prompt sends when its run writes the golden journal, the reply last."""
    monkeypatch.chdir(tmp_path)
    cwd = test_runner._repo(tmp_path)

    def _writing_run(*_a: object, **kw: Any) -> int:
        saved = sessions_layout.SessionLayout(
            state_dir=paths.state_dir(cwd), session_id=kw["session_id"]
        )
        saved.session_dir.mkdir(parents=True, exist_ok=True)
        saved.logs_path.write_bytes(_RECORDED.read_bytes())
        return 0

    monkeypatch.setattr(run, "run_task", _writing_run)
    monkeypatch.setattr(session_setup, "load_session_config", test_runner._loaded)
    wire = test_runner._Wire()
    frames: list[dict[str, Any]] = []
    try:
        wire.send(
            id=1, method="initialize", params={"protocolVersion": 1, "clientCapabilities": {}}
        )
        frames.append(wire.recv())
        wire.send(id=2, method="session/new", params={"cwd": str(cwd), "mcpServers": []})
        frames.append(wire.recv())
        session_id = str(frames[-1]["result"]["sessionId"])
        wire.prompt(session_id, "do the thing")
        while frames[-1].get("id") != 3:
            frames.append(wire.recv())
    finally:
        wire.close()
    return _normal(frames, {str(cwd): "<cwd>", session_id: "<session>"})


def test_a_v1_prompt_sends_the_pinned_frames(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change behind the wire leaves a v1 editor's live stream as it was."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    pinned = json.loads((_DATA / "v1-prompt.json").read_text(encoding="utf-8"))
    assert _prompt_frames(tmp_path, monkeypatch) == pinned


def test_a_v1_load_replays_the_pinned_frames(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change behind the wire leaves a v1 editor's replayed conversation as it was."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    pinned = json.loads((_DATA / "v1-load.json").read_text(encoding="utf-8"))
    assert _load_frames(tmp_path) == pinned


def _regenerate() -> None:
    """Rewrite both pins from the current code. Manual, reviewed only."""
    patch = pytest.MonkeyPatch()
    try:
        for name, frames_for in (
            ("v1-prompt.json", functools.partial(_prompt_frames, monkeypatch=patch)),
            ("v1-load.json", _load_frames),
        ):
            with tempfile.TemporaryDirectory() as td:
                patch.setenv("XDG_STATE_HOME", str(pathlib.Path(td) / "state"))
                root = pathlib.Path(td) / "work"
                root.mkdir()
                frames = frames_for(root)
            text = json.dumps(frames, indent=1, ensure_ascii=False) + "\n"
            (_DATA / name).write_text(text, encoding="utf-8")
            print(f"regenerated {_DATA / name}")
    finally:
        patch.undo()


if __name__ == "__main__":
    _regenerate()
