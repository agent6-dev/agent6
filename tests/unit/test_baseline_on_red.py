# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A red gate is not the same fact as a broken change."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.app.baseline import Baseline, gate_on_base

pytestmark = pytest.mark.needs_namespaces


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x",
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    (repo / "gate.sh").write_text("#!/bin/sh\nexit 0\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "green base"], check=True, env=env)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    # The working tree now fails, as a run that broke the gate would leave it.
    (repo / "gate.sh").write_text("#!/bin/sh\nexit 1\n")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "break it"], check=True, env=env)
    return repo, sha


def test_a_gate_green_at_base_means_this_run_broke_it(tmp_path: Path) -> None:
    got = gate_on_base(
        *_repo(tmp_path), argv=("/bin/sh", "gate.sh"), isolation="hardened", timeout_s=30.0
    )
    assert got.ran and got.returncode == 0
    assert "this run broke it" in got.line()


def test_a_gate_already_red_at_base_is_not_this_run_s_failure(tmp_path: Path) -> None:
    """The case that makes a red run misleading: the tests were broken before
    anyone touched them, or the task WAS to change them."""
    repo, _sha = _repo(tmp_path)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    got = gate_on_base(
        repo, head, argv=("/bin/sh", "gate.sh"), isolation="hardened", timeout_s=30.0
    )
    assert got.ran and got.returncode == 1
    assert "already failed on the base commit" in got.line()


def test_the_live_checkout_is_never_touched(tmp_path: Path) -> None:
    """It answers a question ABOUT the run; disturbing the run to do so would
    be worse than not answering."""
    repo, sha = _repo(tmp_path)
    before = (repo / "gate.sh").read_text()
    gate_on_base(repo, sha, argv=("/bin/sh", "gate.sh"), isolation="hardened", timeout_s=30.0)
    assert (repo / "gate.sh").read_text() == before


def test_nothing_to_check_says_so_rather_than_guessing(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    assert gate_on_base(repo, sha, argv=(), isolation="hardened", timeout_s=5.0) == Baseline(
        ran=False, returncode=None, detail="no gate or no base commit recorded"
    )
    assert not gate_on_base(repo, "", argv=("/bin/true",), isolation="hardened", timeout_s=5.0).ran


def test_a_bad_base_sha_reports_instead_of_raising(tmp_path: Path) -> None:
    repo, _sha = _repo(tmp_path)
    got = gate_on_base(repo, "0" * 40, argv=("/bin/true",), isolation="hardened", timeout_s=5.0)
    assert not got.ran
    assert "could not check the base commit" in got.line()
