# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web session view says what kind of session it is showing.

The page opens for any session, but its snapshot carried no mode -- so the
details panel was headed a hard-coded "Run" and the composer said "continue the
run" over a plan or an ask. The heading is exactly where the mode belongs, and
it was stating the opposite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.sessions.layout import bucket_dir
from agent6.ui.web import model


def _session(state: Path, bucket: str, session_id: str, mode: str) -> Path:
    session = bucket_dir(state, bucket) / session_id
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": session_id, "mode": mode, "user_task": "t"}),
        encoding="utf-8",
    )
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    return session


@pytest.mark.parametrize(("bucket", "mode"), [("runs", "run"), ("plans", "plan"), ("asks", "ask")])
def test_the_snapshot_carries_the_mode(tmp_path: Path, bucket: str, mode: str) -> None:
    session = _session(tmp_path, bucket, "brave-oak-AAAAAA", mode)
    assert model.session_snapshot(session)["mode"] == mode


def test_the_page_heads_the_panel_with_the_mode_not_a_fixed_word() -> None:
    """A hard-coded 'Run' is right one time in three."""
    client = (Path(model.__file__).with_name("client.js")).read_text(encoding="utf-8")
    assert "opts.title || 'Run'" not in client
