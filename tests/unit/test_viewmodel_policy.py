# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One fold supplies the run's policy facts to every surface."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.viewmodel import run_policy


def _manifest(run_dir: Path, **over: object) -> None:
    data: dict[str, object] = {
        "version": 3,
        "mode": "run",
        "run_id": "r",
        "models": {"driver": {"provider": "anthropic", "model": "claude-x"}},
        "policy": {"run_commands": "ask", "isolation": "strict"},
        "workflow": {"verify_command": ["uv", "run", "pytest"], "verify_origin": "configured"},
    }
    data.update(over)
    (run_dir / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_the_line_carries_what_an_operator_needs(tmp_path: Path) -> None:
    """Model, sandbox, command setting and the gate -- the facts that used to
    need an interrupt or a config read."""
    _manifest(tmp_path)
    line = run_policy(tmp_path).line()
    assert line == "claude-x · strict · commands ask · uv run pytest (configured)"


def test_the_gate_says_whose_it_is(tmp_path: Path) -> None:
    """An inferred gate came from a file the model can edit; a configured one
    did not. A surface that hides the difference hides the only thing that
    makes "passed" mean something."""
    _manifest(tmp_path, workflow={"verify_command": ["make", "test"], "verify_origin": "inferred"})
    assert run_policy(tmp_path).gate() == "make test (inferred)"


def test_a_gateless_run_says_so(tmp_path: Path) -> None:
    _manifest(tmp_path, workflow={})
    assert run_policy(tmp_path).gate() == "no verify gate"


def test_an_unreadable_dir_reports_nothing_rather_than_guessing(tmp_path: Path) -> None:
    assert run_policy(tmp_path).line() == "no verify gate"
