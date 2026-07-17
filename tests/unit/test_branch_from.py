# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""git.branch_from: where a run's branch is cut from when you are not on the
base branch (stack on the current branch vs start clean from the base line)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app.preflight import resolve_base_branch
from agent6.config import Config
from agent6.runs.layout import RunLayout
from agent6.ui.cli._preflight import choose_branch_start_point


def _cfg(branch_from: str) -> Config:
    return Config.model_validate({"git": {"branch_from": branch_from}})


def _manifest(state_dir: Path, run_id: str, base_branch: str) -> None:
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps({"version": 2, "run_id": run_id, "base_branch": base_branch}) + "\n",
        encoding="utf-8",
    )


def test_resolve_base_walks_the_run_branch_chain(tmp_path: Path) -> None:
    # agent6/c cut from agent6/b cut from agent6/a cut from main -> base is main.
    _manifest(tmp_path, "a", "main")
    _manifest(tmp_path, "b", "agent6/a")
    _manifest(tmp_path, "c", "agent6/b")
    assert resolve_base_branch(tmp_path, "agent6/c") == "main"
    # a non-run branch is its own base.
    assert resolve_base_branch(tmp_path, "feature-x") == "feature-x"
    # a run branch with no manifest resolves to itself (best effort).
    assert resolve_base_branch(tmp_path, "agent6/orphan") == "agent6/orphan"


def test_resolve_base_survives_a_manifest_cycle(tmp_path: Path) -> None:
    # A corrupt chain that loops must not hang.
    _manifest(tmp_path, "x", "agent6/y")
    _manifest(tmp_path, "y", "agent6/x")
    assert resolve_base_branch(tmp_path, "agent6/x") in ("agent6/x", "agent6/y")


def test_branch_from_current_always_stacks(tmp_path: Path) -> None:
    _manifest(tmp_path, "a", "main")
    choice = choose_branch_start_point(_cfg("current"), tmp_path, "agent6/a")
    assert choice.start_point is None and choice.abort is False


def test_branch_from_base_cuts_from_the_base_line(tmp_path: Path) -> None:
    _manifest(tmp_path, "a", "main")
    choice = choose_branch_start_point(_cfg("base"), tmp_path, "agent6/a")
    assert choice.start_point == "main" and choice.abort is False


def test_branch_from_base_is_a_noop_when_already_on_the_base(tmp_path: Path) -> None:
    # On 'main' (a base branch) there is nothing to stack on: cut from HEAD.
    choice = choose_branch_start_point(_cfg("base"), tmp_path, "main")
    assert choice.start_point is None


def test_branch_from_ask_headless_falls_back_to_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No tty: ask can't prompt, so it picks the clean base (un-surprising).
    import agent6.ui.cli._preflight as pf

    monkeypatch.setattr(pf.sys.stdin, "isatty", lambda: False)
    _manifest(tmp_path, "a", "main")
    choice = choose_branch_start_point(_cfg("ask"), tmp_path, "agent6/a")
    assert choice.start_point == "main"
