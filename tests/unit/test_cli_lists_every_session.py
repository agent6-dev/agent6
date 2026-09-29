# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions` lists everything the CLI can open by id.

The TUI and the web hub give `machine create` drafts their own card; the CLI has no such card, so a
draft `attach` opens appears in this listing.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import paths
from agent6.sessions import layout
from agent6.ui.cli import main


def _session(state: pathlib.Path, bucket: str, session_id: str, mode: str) -> None:
    session = layout.bucket_dir(state, bucket) / session_id
    session.mkdir(parents=True)
    # The task text is neutral: `f"a {mode}"` would satisfy the mode-column assertions vacuously.
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "a task"}) + "\n",
        encoding="utf-8",
    )


def test_a_machine_draft_appears_in_the_listing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
    _session(state, "machines", "fair-trail-AAAAAA", "machine")

    assert main(["sessions", "list"]) == 0
    out = capsys.readouterr().out
    assert "fair-trail-AAAAAA" in out, out
    # The mode column is what tells them apart, so it must say which it is.
    assert "machine" in out, out


def test_an_undone_run_lists_as_undone_and_never_unmerged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An undone run lists as undone in the listing and its JSON, never as unmerged."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    monkeypatch.chdir(tmp_path)
    git("init", "-q", "-b", "main")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "run")
    git("branch", "agent6/undone-one-AAAAA", "HEAD")
    git("reset", "-q", "--hard", base)
    session = layout.bucket_dir(paths.state_dir(tmp_path), "runs") / "undone-one-AAAAA"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "user_task": "a task",
                "base_sha": base,
                "run_branch": "agent6/undone-one-AAAAA",
            }
        ),
        encoding="utf-8",
    )
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "a task"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "undone", "all_passed": False})
        + "\n",
        encoding="utf-8",
    )

    assert main(["sessions", "list", "--json"]) == 0
    [row] = json.loads(capsys.readouterr().out)
    assert (row["status"], row["reason"], row["unmerged"]) == ("undone", "", False)
    assert main(["sessions", "list"]) == 0
    out = capsys.readouterr().out
    assert "undone" in out and "stopped" not in out and "unmerged" not in out


def test_every_bucket_is_listed_together(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
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
