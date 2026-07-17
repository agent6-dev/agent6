# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two shared _common glue helpers: resolve_or_newest_layout (a run by id,
or the latest across every bucket) and load_config_or_exit (config or a printed
CONFIG ERROR + exit code 2)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.config.layer import EffectiveConfig, resolved_state_dir
from agent6.runs.id import RunIdError
from agent6.ui.cli._common import load_config_or_exit, resolve_or_newest_layout


def _run_dir(state: Path, bucket: str, run_id: str, *, log_mtime: float) -> Path:
    """Seed a run dir with a logs.jsonl at a controlled mtime (what
    newest_run_dir sorts by)."""
    d = state / bucket / run_id
    d.mkdir(parents=True)
    log = d / "logs.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    os.utime(log, (log_mtime, log_mtime))
    return d


# --- resolve_or_newest_layout ------------------------------------------------


def test_explicit_id_resolves_across_buckets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = resolved_state_dir(repo)
    (state / "asks" / "ask-xyz").mkdir(parents=True)

    layout = resolve_or_newest_layout(repo, "ask-")
    assert layout is not None
    assert layout.subdir == "asks" and layout.run_id == "ask-xyz"


def test_empty_id_picks_the_newest_across_buckets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state = resolved_state_dir(repo)
    _run_dir(state, "runs", "old-run", log_mtime=1000.0)
    _run_dir(state, "asks", "new-ask", log_mtime=2000.0)

    layout = resolve_or_newest_layout(repo, "")
    assert layout is not None
    assert layout.run_id == "new-ask" and layout.subdir == "asks"
    assert layout.run_dir == state / "asks" / "new-ask"


def test_empty_id_with_no_runs_returns_none(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert resolve_or_newest_layout(repo, "") is None


def test_bad_explicit_id_raises(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (resolved_state_dir(repo) / "runs" / "run-abc").mkdir(parents=True)
    with pytest.raises(RunIdError):
        resolve_or_newest_layout(repo, "nope")


# --- load_config_or_exit ------------------------------------------------------


def test_load_config_returns_the_effective_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    eff = load_config_or_exit(repo, None)
    assert isinstance(eff, EffectiveConfig)
    assert eff.config.sandbox.protect_git is True  # a real, defaulted leaf


def test_load_config_prints_config_error_and_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    bad = tmp_path / "bad.toml"
    bad.write_text("[models\n", encoding="utf-8")  # invalid TOML

    rc = load_config_or_exit(repo, bad)
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("CONFIG ERROR:\n")
    assert "bad.toml" in err
