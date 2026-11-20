# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child reads its bound journal without gaining a roster of its parent's sessions."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import config, paths
from agent6.tools import dispatch, tool_errors


def test_child_reads_only_its_exact_id_outside_its_clone_state(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    clone = tmp_path / "clone"
    for root in (origin, clone):
        root.mkdir()
    origin_sessions = paths.state_dir(origin) / "sessions" / "runs"
    parent = origin_sessions / "calm-heron-parent"
    child = origin_sessions / "calm-heron-parent-p1-l1"
    for directory, body in ((parent, "parent instruction"), (child, "child note")):
        directory.mkdir(parents=True)
        entry = {"type": "model.result", "text": body}
        (directory / "logs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    reader = dispatch.ToolDispatcher(
        root=clone, state_dir=paths.state_dir(clone), session_dir=child, config=config.Config()
    )
    page = reader.dispatch("read_session", {"id": child.name, "query": "child note"}).to_wire()
    assert "child note" in page["conversation"]
    for other in (parent.name, child.name[:-1]):
        with pytest.raises(tool_errors.ToolError, match="no session"):
            reader.dispatch("read_session", {"id": other})
    assert not reader.dispatch("read_session", {}).to_wire()["sessions"]
