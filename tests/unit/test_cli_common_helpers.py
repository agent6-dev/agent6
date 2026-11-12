# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared helper resolve_or_newest_layout: a run by id, or the latest across every bucket."""

from __future__ import annotations

import os
import pathlib

import pytest

from agent6 import paths
from agent6.sessions import id
from agent6.ui.cli import common


def _session_dir(
    state: pathlib.Path, bucket: str, session_id: str, *, log_mtime: float
) -> pathlib.Path:
    """Seed a run dir with a logs.jsonl at a controlled mtime (what newest_session_dir sorts by)."""
    d = state / "sessions" / bucket / session_id
    d.mkdir(parents=True)
    log = d / "logs.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    os.utime(log, (log_mtime, log_mtime))
    return d


# --- resolve_or_newest_layout ------------------------------------------------


def test_explicit_id_resolves_across_buckets(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = paths.state_dir(repo)
    (state / "sessions" / "asks" / "ask-xyz").mkdir(parents=True)
    (state / "sessions" / "asks" / "ask-xyz" / "logs.jsonl").write_text("{}\n", encoding="utf-8")

    layout = common.resolve_or_newest_layout(repo, "ask-")
    assert layout is not None
    assert layout.subdir == "asks" and layout.session_id == "ask-xyz"


def test_empty_id_picks_the_newest_across_buckets(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = paths.state_dir(repo)
    _session_dir(state, "runs", "old-run", log_mtime=1000.0)
    _session_dir(state, "asks", "new-ask", log_mtime=2000.0)

    layout = common.resolve_or_newest_layout(repo, "")
    assert layout is not None
    assert layout.session_id == "new-ask" and layout.subdir == "asks"
    assert layout.session_dir == state / "sessions" / "asks" / "new-ask"


def test_empty_id_never_picks_a_review_seat(tmp_path: pathlib.Path) -> None:
    """During a panel a seat's record is the newest: a bare attach must mean the run."""
    repo = tmp_path / "repo"
    repo.mkdir()
    state = paths.state_dir(repo)
    _session_dir(state, "runs", "the-run", log_mtime=1000.0)
    seat = _session_dir(state, "read_only", "the-run-rpanel-l1", log_mtime=2000.0)
    (seat / "transcripts").mkdir()
    (state / "sessions" / "runs" / "the-run" / "transcripts").mkdir()

    layout = common.resolve_or_newest_layout(repo, "")
    assert layout is not None and layout.session_id == "the-run"
    held = common.newest_layout_holding(repo, "transcripts")
    assert held is not None and held.session_id == "the-run"
    # Its id still resolves it.
    named = common.resolve_or_newest_layout(repo, "the-run-rpanel-l1")
    assert named is not None and named.subdir == "read_only"


def test_empty_id_with_no_runs_returns_none(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert common.resolve_or_newest_layout(repo, "") is None


def test_bad_explicit_id_raises(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (paths.state_dir(repo) / "sessions" / "runs" / "run-abc").mkdir(parents=True)
    (paths.state_dir(repo) / "sessions" / "runs" / "run-abc" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(id.SessionIdError):
        common.resolve_or_newest_layout(repo, "nope")
