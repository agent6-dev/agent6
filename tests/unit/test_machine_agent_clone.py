# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A mode="run" machine state works a fresh clone at the machine chain's tip.

The lane mechanism, sequential where lanes are parallel: the chain ref
carries state-to-state continuation, and the visible `agent6/machine-<id>`
branch tracks the same tip for the operator (the run story: changes arrive
on a branch). The operator's checkout is never touched; work lands back per
state, on every outcome.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agent6.app import machine_agent as ma
from agent6.git_ops import chain_ref_for, chain_tip
from agent6.machine import AgentRequest

BRANCH = "agent6/machine-m1"
CHAIN = chain_ref_for("machine-m1")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _origin(tmp_path: Path) -> Path:
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


_REAL_POPEN = subprocess.Popen


def _fake_popen(argv: list[str], **kw: Any) -> Any:
    """Fake only the machine-agent spawn; git's own Popen calls stay real
    (the monkeypatch lands on the shared stdlib module object)."""
    if not any("machine_agent" in str(a) for a in argv):
        return _REAL_POPEN(argv, **kw)
    return _FakeChild(argv, **kw)


class _FakeChild:
    """Stands in for the machine-agent subprocess: commits one file in the
    request's cwd and writes a clean result.json."""

    captured_env: ClassVar[dict[str, str]] = {}
    workdirs: ClassVar[list[Path]] = []
    seen_files: ClassVar[list[set[str]]] = []

    def __init__(self, argv: list[str], **kw: Any) -> None:
        _FakeChild.captured_env = dict(kw.get("env") or {})
        req = json.loads(Path(argv[-2]).read_text(encoding="utf-8"))
        cwd = Path(req["cwd"])
        _FakeChild.workdirs.append(cwd)
        _FakeChild.seen_files.append({e.name for e in cwd.iterdir()})
        seq = req["request"]["step_seq"]
        if req["request"]["mode"] == "run":
            # Mirror the nested loop: commit the tree, advance the chain ref
            # (never a branch), leave HEAD where it was.
            (cwd / f"work{seq}.txt").write_text(f"state {seq}\n", encoding="utf-8")
            _git(cwd, "add", "-A")
            _git(cwd, "commit", "-q", "-m", f"agent6 iter 1: state {seq}")
            _git(cwd, "update-ref", CHAIN, "HEAD")
        Path(argv[-1]).write_text(
            json.dumps({"reason": "finish_session", "payload": None, "usd": 0.0}),
            encoding="utf-8",
        )
        self.pid = 4242
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _runner(origin: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    return ma.build_machine_agent_runner(
        {},
        origin,
        "none",
        tmp_path / "m1" / "agent_transcripts",
        (),
        None,
        machine_id="m1",
        clone_root=tmp_path / "clones",
    )


def _req(seq: int, mode: str = "run") -> AgentRequest:
    return AgentRequest(prompt="p", timeout_s=60.0, mode=mode, state_name=f"s{seq}", step_seq=seq)


def test_run_states_continue_the_machine_branch_and_never_touch_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    runner = _runner(origin, tmp_path)

    r1 = runner(_req(0), None)
    assert r1.reason == "finish_session"
    r2 = runner(_req(1), None)
    assert r2.reason == "finish_session"

    # Sequential continuation: state 1 built on state 0's tree, and the
    # visible branch tracks the chain's tip.
    files = _git(origin, "ls-tree", "--name-only", BRANCH)
    assert "work0.txt" in files and "work1.txt" in files
    assert chain_tip(origin, CHAIN) == chain_tip(origin, BRANCH)
    # The operator's checkout never moved and holds neither file.
    assert _git(origin, "branch", "--show-current") == "main"
    assert not (origin / "work0.txt").exists()
    # Clones are gone once landed; the subprocess ran subordinate.
    assert not any((tmp_path / "clones").glob("state-*"))
    assert _FakeChild.captured_env["AGENT6_SUBRUN"] == "1"


def test_read_only_states_see_the_machine_tree_and_land_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only judge in a machine with run states ran in the operator's
    checkout, so it could not see the work it was judging. It now runs in a
    fresh clone at the chain tip like every state of such a machine; it
    commits nothing, so the branch does not move and its clone is cleaned up
    by the landing's no-op path."""
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    _FakeChild.workdirs = []
    _FakeChild.seen_files = []
    runner = _runner(origin, tmp_path)
    assert runner(_req(0), None).reason == "finish_session"
    tip_after_run = chain_tip(origin, BRANCH)
    assert runner(_req(1, mode="agent"), None).reason == "finish_session"
    judge_dir = _FakeChild.workdirs[-1]
    assert judge_dir != origin
    assert "work0.txt" in _FakeChild.seen_files[-1]  # the run state's committed work
    assert chain_tip(origin, BRANCH) == tip_after_run  # nothing landed
    assert not any((tmp_path / "clones").glob("state-*"))  # cleaned up


def test_without_a_machine_tree_read_only_states_run_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No machine_id/clone_root (`machine create`, a machine with no run
    states): a read-only request runs in cwd, as before."""
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    _FakeChild.workdirs = []
    runner = ma.build_machine_agent_runner(
        {}, origin, "none", tmp_path / "m1" / "agent_transcripts", (), None
    )
    assert runner(_req(0, mode="agent"), None).reason == "finish_session"
    assert _FakeChild.workdirs == [origin]
    assert not (tmp_path / "clones").exists()


def test_machine_tool_runner_runs_each_call_in_the_machine_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool state's jail mounted the operator's checkout, which a run
    state's commits never reach, so an edit-then-check loop never converged
    (the shipped code-fixer burned its whole attempt budget this way). The
    runner clones the chain tip per call, remaps the bundle protect paths to
    the clone's own copy, and discards the tree after."""
    from agent6.app.machine import run as machine_run
    from agent6.types import CommandResult, JailPolicy

    origin = _origin(tmp_path)
    _git(origin, "checkout", "-q", "-b", "scratch")
    (origin / "work.txt").write_text("landed\n", encoding="utf-8")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "agent6 iter 1: work")
    _git(origin, "update-ref", CHAIN, "HEAD")
    _git(origin, "checkout", "-q", "main")
    _git(origin, "branch", "-D", "scratch")
    assert not (origin / "work.txt").exists()

    captured: dict[str, object] = {}

    def fake_jail(policy: JailPolicy) -> CommandResult:
        captured["cwd"] = policy.cwd
        captured["saw_work"] = (policy.cwd / "work.txt").exists()
        captured["protect"] = policy.extra_protect_paths
        return CommandResult(argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(machine_run, "run_in_jail", fake_jail)
    runner = machine_run.machine_tool_runner(origin, "m1", tmp_path / "clones")
    policy = JailPolicy(cwd=origin, argv=("x",), extra_protect_paths=(origin / "scripts",))
    assert runner(policy).returncode == 0
    clone_cwd = captured["cwd"]
    assert isinstance(clone_cwd, Path) and clone_cwd != origin
    assert captured["saw_work"] is True
    assert captured["protect"] == (clone_cwd / "scripts",)
    assert not clone_cwd.exists()  # scratch tree, discarded


def test_import_failure_keeps_the_clone_and_routes_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state ran and its commits are real, but they did not land: the
    outcome routes failed with no captured payload, and the clone (the only
    copy) is kept for the adopt/prune verbs."""
    from agent6.git_ops import GitError

    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)

    def _boom(*_a: object, **_k: object) -> None:
        raise GitError("refs locked")

    monkeypatch.setattr(ma, "fetch_branch", _boom)
    runner = _runner(origin, tmp_path)
    r = runner(_req(0), None)
    assert r.reason.startswith("import of")
    assert r.payload is None
    assert chain_tip(origin, CHAIN) is None  # nothing landed
    assert list((tmp_path / "clones").glob("state-*"))  # evidence kept
