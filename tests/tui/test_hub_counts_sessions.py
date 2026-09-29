# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The hub's header counts what the hub is showing.

It lists every session (runs, plans and asks) under a fixed "N runs"
label, so a hub of one run, one plan and one ask announced "3 runs".
"""

from __future__ import annotations

import asyncio
import json
import pathlib

from agent6.sessions import layout
from agent6.ui.tui import home


def _session(state: pathlib.Path, bucket: str, session_id: str, mode: str) -> None:
    session = layout.bucket_dir(state, bucket) / session_id
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )


def _subtitle(state: pathlib.Path, repo: pathlib.Path) -> str:
    """The header the hub paints, from a real mounted app."""

    async def scenario() -> str:
        app = home.Agent6HomeApp(state, repo)
        async with app.run_test(size=(120, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            return str(app.sub_title)

    return asyncio.run(scenario())


def test_a_mixed_hub_does_not_call_them_all_runs(tmp_path: pathlib.Path) -> None:
    state, repo = tmp_path / "state", tmp_path / "repo"
    repo.mkdir()
    _session(state, "runs", "runny-one-AAAAAA", "run")
    _session(state, "plans", "planny-two-BBBBB", "plan")
    _session(state, "asks", "asky-three-CCCCC", "ask")

    subtitle = _subtitle(state, repo)
    assert "3 sessions" in subtitle, subtitle


def test_one_session_is_singular(tmp_path: pathlib.Path) -> None:
    state, repo = tmp_path / "state", tmp_path / "repo"
    repo.mkdir()
    _session(state, "runs", "runny-one-AAAAAA", "run")

    subtitle = _subtitle(state, repo)
    assert "1 session" in subtitle and "sessions" not in subtitle, subtitle


def test_an_empty_hub_says_what_to_do_next(tmp_path: pathlib.Path) -> None:
    """An empty hub names the next step, as the CLI, the web and the machines screen do."""
    state, repo = tmp_path / "state", tmp_path / "repo"
    repo.mkdir()

    subtitle = _subtitle(state, repo)

    assert "no sessions yet" in subtitle and "agent6 run" in subtitle, subtitle


def test_the_hub_names_the_repository_by_its_directory(tmp_path: pathlib.Path) -> None:
    """The header carries the repository's name, not its full path, so the count fits 80 columns."""
    state, repo = tmp_path / "state", tmp_path / "some-repo"
    repo.mkdir()
    _session(state, "runs", "runny-one-AAAAAA", "run")
    sub = _subtitle(state, repo)
    assert sub.startswith("some-repo · ") and str(tmp_path) not in sub
