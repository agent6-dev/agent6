# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""resolve_run_layout finds a run under runs/ OR asks/ (so history/graph work
for ask runs, whose state lives under asks/<id>)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.runs.id import RunIdError
from agent6.ui.cli._common import resolve_run_layout


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


def test_prefix_must_be_unique_across_runs_and_asks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = resolved_state_dir(repo)
    (state / "runs" / "same-run").mkdir(parents=True)
    (state / "asks" / "same-ask").mkdir(parents=True)

    with pytest.raises(RunIdError) as exc:
        resolve_run_layout(repo, "same-")
    assert exc.value.ambiguous
    assert "runs/same-run" in str(exc.value)
    assert "asks/same-ask" in str(exc.value)


def test_exact_match_wins_over_cross_bucket_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = resolved_state_dir(repo)
    (state / "runs" / "run").mkdir(parents=True)
    (state / "asks" / "run-question").mkdir(parents=True)

    layout = resolve_run_layout(repo, "run")
    assert layout.subdir == "runs"
    assert layout.run_id == "run"


def test_empty_query_is_invalid(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (resolved_state_dir(repo) / "runs" / "run-abc").mkdir(parents=True)

    with pytest.raises(RunIdError, match="empty run id"):
        resolve_run_layout(repo, "")


def test_raises_when_no_match(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (resolved_state_dir(repo) / "runs" / "run-abc").mkdir(parents=True)
    with pytest.raises(RunIdError):
        resolve_run_layout(repo, "nope")
