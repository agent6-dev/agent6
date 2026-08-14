# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions` lists everything the CLI can open by id.

The TUI and the web hub give `machine create` drafts their own card, so their
session list leaves them out. The CLI has no such card -- so excluding drafts
there made a session that `attach` opens happily appear in no listing at all,
findable only by keeping the id from the create output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.layout import bucket_dir
from agent6.ui.cli import main


def _session(state: Path, bucket: str, session_id: str, mode: str) -> None:
    session = bucket_dir(state, bucket) / session_id
    session.mkdir(parents=True)
    # The task text is neutral on purpose: `f"a {mode}"` would put the mode
    # word in the TASK column and satisfy the mode-column assertions vacuously.
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "a task"}) + "\n",
        encoding="utf-8",
    )


def test_a_machine_draft_appears_in_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    state = resolved_state_dir(tmp_path)
    _session(state, "machines", "fair-trail-AAAAAA", "machine")

    assert main(["sessions", "list"]) == 0
    out = capsys.readouterr().out
    assert "fair-trail-AAAAAA" in out, out
    # The mode column is what tells them apart, so it must say which it is.
    assert "machine" in out, out


def test_every_bucket_is_listed_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    state = resolved_state_dir(tmp_path)
    for bucket, mode, sid in (
        ("runs", "run", "runny-one-AAAAAA"),
        ("plans", "plan", "planny-two-BBBBB"),
        ("asks", "ask", "asky-three-CCCCC"),
        ("machines", "machine", "drafty-four-DDDD"),
    ):
        _session(state, bucket, sid, mode)

    assert main(["sessions", "list"]) == 0
    out = capsys.readouterr().out
    for sid in ("runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC", "drafty-four-DDDD"):
        assert sid in out, f"{sid} missing from:\n{out}"
