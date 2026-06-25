# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""resolve_run_layout finds a run under runs/ OR asks/ (so history/graph work
for ask runs, whose state lives under asks/<id>)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.cli._common import resolve_run_layout
from agent6.config_layer import resolved_state_dir
from agent6.run_id import RunIdError


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "st"))


def test_resolves_runs_and_asks_with_correct_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = resolved_state_dir(repo)
    (state / "runs" / "run-abc").mkdir(parents=True)
    (state / "asks" / "ask-xyz").mkdir(parents=True)

    run_layout = resolve_run_layout(repo, "run-abc")
    assert run_layout.subdir == "runs" and run_layout.run_id == "run-abc"

    ask_layout = resolve_run_layout(repo, "ask-xyz")
    assert ask_layout.subdir == "asks" and ask_layout.run_id == "ask-xyz"
    # The layout points at the ask's own directory (where its graph now lives).
    assert ask_layout.run_dir == state / "asks" / "ask-xyz"

    # Unique-prefix resolution works too.
    assert resolve_run_layout(repo, "ask-").run_id == "ask-xyz"


def test_raises_when_no_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (resolved_state_dir(repo) / "runs" / "run-abc").mkdir(parents=True)
    with pytest.raises(RunIdError):
        resolve_run_layout(repo, "nope")
