# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-turn checkpoint store + `agent6 fork` (clone-to-new-session recovery).

Covers:
- the loop's `_save_resume_snapshot` ALSO writes append-only `checkpoints/<NNNN>.json`
  carrying head_sha + graph_version,
- `agent6 fork` clones state, writes lineage manifest fields, cuts the branch,
  and appends `lineage.jsonl`,
- forking a pre-checkpoint (old) run degrades gracefully.
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.cli._common import _state_dir  # pyright: ignore[reportPrivateUsage]
from agent6.cli.fork import _cmd_fork  # pyright: ignore[reportPrivateUsage]
from agent6.cli.run import _cmd_resume  # pyright: ignore[reportPrivateUsage]
from agent6.graph.storage import RunLayout, list_checkpoint_turns
from agent6.workflows._run_state import load_checkpoint
from agent6.workflows.loop import (
    Workflow,
    _LoopState,  # pyright: ignore[reportPrivateUsage]
)


def _silent(_: str) -> None:
    return None


def _git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return sp.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def _wf(**kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": Path("/tmp"),
        "config": MagicMock(prompt=MagicMock(system_prompt_file="")),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
    }
    defaults.update(kw)
    return Workflow(**defaults)


# --- checkpoint store -------------------------------------------------------


def test_save_snapshot_writes_per_turn_checkpoint(tmp_path: Path) -> None:
    """`_save_resume_snapshot` writes both loop_state.json AND a per-turn
    checkpoints/<NNNN>.json carrying head_sha + graph_version."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snap = run_dir / "loop_state.json"

    graph_client = MagicMock()
    graph_client.get_state.return_value = {"nodes": {}, "cursor": None, "graph_version": 7}
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=(), metric=SimpleNamespace(goal=None))
    )
    wf = _wf(root=repo, config=config, resume_state_path=snap, graph_client=graph_client)
    state = _LoopState(original_task="t", tool_calls=0)

    wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
        system="s", messages=[], tool_calls=0, next_iteration=3, root_task_id=None, state=state
    )

    # checkpoints live next to loop_state.json (the run dir).
    cp = run_dir / "checkpoints" / "0003.json"
    assert snap.is_file()
    assert cp.is_file(), "per-turn checkpoint must be written"

    loaded = load_checkpoint(cp)
    assert loaded.turn == 3
    assert loaded.head_sha == head
    assert loaded.graph_version == 7
    # The checkpoint payload is a superset of loop_state.json (same core fields).
    assert loaded.payload["next_iteration"] == 3
    assert json.loads(snap.read_text())["head_sha"] == head


def test_checkpoints_are_append_only(tmp_path: Path) -> None:
    """Each turn writes a distinct checkpoint; older ones are never overwritten."""
    repo = tmp_path / "repo"
    _git_repo(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snap = run_dir / "loop_state.json"
    config = SimpleNamespace(
        workflow=SimpleNamespace(verify_command=(), metric=SimpleNamespace(goal=None))
    )
    wf = _wf(root=repo, config=config, resume_state_path=snap)
    state = _LoopState(original_task="t", tool_calls=0)
    for turn in (1, 2, 3):
        wf._save_resume_snapshot(  # pyright: ignore[reportPrivateUsage]
            system="s",
            messages=[{"role": "user", "content": f"turn {turn}"}],
            tool_calls=0,
            next_iteration=turn,
            root_task_id=None,
            state=state,
        )
    cp_dir = run_dir / "checkpoints"
    assert sorted(p.name for p in cp_dir.glob("*.json")) == ["0001.json", "0002.json", "0003.json"]
    # Turn 1's payload was not clobbered by later turns.
    assert load_checkpoint(cp_dir / "0001.json").payload["messages"][0]["content"] == "turn 1"


def test_list_checkpoint_turns_empty_for_old_run(tmp_path: Path) -> None:
    """A run dir with no checkpoints/ dir lists no turns (old-run detection)."""
    layout = RunLayout(state_dir=tmp_path, run_id="old")
    (tmp_path / "runs" / "old").mkdir(parents=True)
    assert list_checkpoint_turns(layout) == []


# --- fork command -----------------------------------------------------------


def _seed_source_run(
    state_dir: Path, run_id: str, *, head_sha: str, turns: tuple[int, ...], mode: str = "run"
) -> RunLayout:
    """Lay down a source run dir with a manifest, graph DAG, and checkpoints."""
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": run_id,
                "mode": mode,
                "user_task": "do the thing",
                "base_sha": "basesha000",
                "base_branch": "main",
                "run_branch": f"agent6/{run_id}",
            }
        ),
        encoding="utf-8",
    )
    # A curator DAG artifact to be cloned verbatim.
    (layout.graph_dir / "root.md").write_text("---\nid: root\n---\nnode\n", encoding="utf-8")
    layout.journal_path.write_text('{"op": "add_subtask"}\n', encoding="utf-8")
    layout.cursor_path.write_text('{"node_id": "root"}', encoding="utf-8")
    for turn in turns:
        payload = {
            "version": 1,
            "system": "sys",
            "messages": [{"role": "user", "content": f"turn {turn}"}],
            "tool_calls": 0,
            "next_iteration": turn,
            "root_task_id": "root",
            "head_sha": head_sha,
            "graph_version": turn,
        }
        layout.checkpoint_path(turn).write_text(json.dumps(payload), encoding="utf-8")
    # loop_state.json mirrors the latest checkpoint.
    layout.run_dir.joinpath("loop_state.json").write_text(
        layout.checkpoint_path(turns[-1]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return layout


def test_fork_preserves_source_run_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Forking a plan run must resume in plan mode. Stamping mode="run" would pair
    # the frozen planning system prompt (which drives finish_planning) with
    # run-mode mutating tools and auto-commits.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    _seed_source_run(state_dir, "plan-src-AAAA11", head_sha=head, turns=(1, 2), mode="plan")

    rc = _cmd_fork(None, "plan-src", new_run_id="plan-fork-BBBB22", no_run=True)
    assert rc == 0

    dst = RunLayout(state_dir=state_dir, run_id="plan-fork-BBBB22")
    assert json.loads(dst.manifest_path.read_text(encoding="utf-8"))["mode"] == "plan"


def test_fork_cleans_up_run_dir_when_branch_cut_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the fork branch already exists at a DIFFERENT sha, create_branch_at
    # refuses (we never move a branch) -- the just-materialized run dir must be
    # cleaned up, not left orphaned.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    _seed_source_run(state_dir, "src-AAAA11", head_sha=head, turns=(1,))
    # A second commit, and pre-create the fork branch pointing at it (≠ head).
    (repo / "b.txt").write_text("y\n")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "c2"], cwd=repo, check=True)
    other = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    sp.run(["git", "branch", "agent6/child-BBBB22", other], cwd=repo, check=True)

    rc = _cmd_fork(None, "src", new_run_id="child-BBBB22", no_run=True)
    assert rc == 1
    assert not RunLayout(state_dir=state_dir, run_id="child-BBBB22").run_dir.exists()


def test_fork_clones_state_writes_lineage_and_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 fork --no-run` clones the checkpoint + DAG into a new run, writes
    the lineage manifest fields, cuts agent6/<new> at the checkpoint sha, and
    appends lineage.jsonl."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    src = _seed_source_run(state_dir, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))

    rc = _cmd_fork(None, "sunny-otter", new_run_id="brave-yak-BBBB22", no_run=True)
    assert rc == 0

    dst = RunLayout(state_dir=state_dir, run_id="brave-yak-BBBB22")
    assert dst.run_dir.is_dir()
    # loop_state.json + seed checkpoint 0000.json carry the latest (turn 3) state.
    seed = load_checkpoint(dst.checkpoint_path(0))
    assert seed.payload["messages"][0]["content"] == "turn 3"
    assert (dst.run_dir / "loop_state.json").is_file()
    # DAG cloned verbatim.
    assert (dst.graph_dir / "root.md").is_file()
    assert dst.journal_path.read_text(encoding="utf-8") == '{"op": "add_subtask"}\n'
    assert dst.cursor_path.read_text(encoding="utf-8") == '{"node_id": "root"}'

    # Lineage manifest fields.
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_run_id"] == "sunny-otter-AAAA11"
    assert manifest["forked_from_turn"] == 3
    assert manifest["forked_from_sha"] == head
    assert manifest["base_sha"] == "basesha000"
    assert manifest["base_branch"] == "main"
    assert manifest["run_branch"] == "agent6/brave-yak-BBBB22"

    # Branch cut at the checkpoint sha, WITHOUT moving HEAD off main.
    branch_sha = sp.run(
        ["git", "rev-parse", "agent6/brave-yak-BBBB22"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branch_sha == head
    current = sp.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current == "main", "fork must not move the operator's checkout"

    # lineage.jsonl appended under the state dir root.
    lineage = (state_dir / "lineage.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lineage) == 1
    ev = json.loads(lineage[0])
    assert ev["child"] == "brave-yak-BBBB22"
    assert ev["parent"] == "sunny-otter-AAAA11"
    assert ev["turn"] == 3
    assert ev["sha"] == head
    assert ev["ts"]

    # Source run is untouched: no new checkpoints, manifest unchanged.
    assert sorted(list_checkpoint_turns(src)) == [1, 2, 3]


def test_fork_at_turn_selects_that_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--at-turn N` forks from checkpoint N, not the latest."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    _seed_source_run(state_dir, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))

    rc = _cmd_fork(None, "sunny-otter", at_turn=2, new_run_id="kid-CCCC33", no_run=True)
    assert rc == 0
    dst = RunLayout(state_dir=state_dir, run_id="kid-CCCC33")
    assert load_checkpoint(dst.checkpoint_path(0)).payload["messages"][0]["content"] == "turn 2"
    assert json.loads(dst.manifest_path.read_text(encoding="utf-8"))["forked_from_turn"] == 2


def test_fork_unknown_turn_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--at-turn` with no matching checkpoint is a clean error, no fork dir."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    _seed_source_run(state_dir, "sunny-otter-AAAA11", head_sha=head, turns=(1, 2, 3))

    rc = _cmd_fork(None, "sunny-otter", at_turn=99, new_run_id="kid-DDDD44", no_run=True)
    assert rc == 2
    assert not (state_dir / "runs" / "kid-DDDD44").exists()


def test_fork_pre_checkpoint_run_degrades_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old run with only loop_state.json (no checkpoints/) still forks, from
    that latest snapshot."""
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    layout = RunLayout(state_dir=state_dir, run_id="old-run-EEEE55")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps({"version": 1, "run_id": "old-run-EEEE55", "base_sha": "x", "base_branch": "m"}),
        encoding="utf-8",
    )
    # Old run: loop_state.json exists but no checkpoints dir content. We DO carry
    # head_sha now (older snapshots without it cannot cut a branch).
    layout.run_dir.joinpath("loop_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "system": "s",
                "messages": [{"role": "user", "content": "legacy"}],
                "tool_calls": 0,
                "next_iteration": 4,
                "root_task_id": None,
                "head_sha": head,
            }
        ),
        encoding="utf-8",
    )
    # Remove the (empty) checkpoints dir created by .ensure() so it's truly "old".
    for p in layout.checkpoints_dir.glob("*"):
        p.unlink()
    layout.checkpoints_dir.rmdir()

    rc = _cmd_fork(None, "old-run", new_run_id="fresh-FFFF66", no_run=True)
    assert rc == 0
    dst = RunLayout(state_dir=state_dir, run_id="fresh-FFFF66")
    seed = load_checkpoint(dst.checkpoint_path(0))
    assert seed.payload["messages"][0]["content"] == "legacy"
    assert seed.turn == 4
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_run_id"] == "old-run-EEEE55"
    assert manifest["forked_from_turn"] == 4


# --- resume gets onto the run branch ---------------------------------------


def _current_branch(repo: Path) -> str:
    return sp.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _layout_with_run_branch(state_dir: Path, run_id: str, run_branch: str | None) -> RunLayout:
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    layout.ensure()
    body: dict[str, Any] = {"version": 2, "run_id": run_id, "mode": "run"}
    if run_branch is not None:
        body["run_branch"] = run_branch
    layout.manifest_path.write_text(json.dumps(body), encoding="utf-8")
    return layout


def test_ensure_on_run_branch_checks_out_the_fork_branch(tmp_path: Path) -> None:
    # Reproduces the fork bug: the branch exists (cut additively) but HEAD is on
    # the operator's branch, so resume must switch onto it before committing.
    from agent6.cli.run import _ensure_on_run_branch  # pyright: ignore[reportPrivateUsage]

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    sp.run(["git", "branch", "agent6/child", head], cwd=repo, check=True)
    assert _current_branch(repo) == "main"

    layout = _layout_with_run_branch(tmp_path / "state", "child", "agent6/child")
    assert _ensure_on_run_branch(repo, layout) is None
    assert _current_branch(repo) == "agent6/child"


def test_ensure_on_run_branch_refuses_dirty_tree(tmp_path: Path) -> None:
    from agent6.cli.run import _ensure_on_run_branch  # pyright: ignore[reportPrivateUsage]

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    sp.run(["git", "branch", "agent6/child", head], cwd=repo, check=True)
    (repo / "seed.txt").write_text("dirty\n")  # uncommitted change

    layout = _layout_with_run_branch(tmp_path / "state", "child", "agent6/child")
    err = _ensure_on_run_branch(repo, layout)
    assert err is not None and "uncommitted changes" in err
    assert _current_branch(repo) == "main", "must not switch with modified tracked files"


def test_ensure_on_run_branch_allows_untracked_files(tmp_path: Path) -> None:
    # Untracked files are carried across a checkout, so they must NOT block the
    # switch (only modified tracked files do).
    from agent6.cli.run import _ensure_on_run_branch  # pyright: ignore[reportPrivateUsage]

    repo = tmp_path / "repo"
    head = _git_repo(repo)
    sp.run(["git", "branch", "agent6/child", head], cwd=repo, check=True)
    (repo / "scratch.txt").write_text("untracked\n")  # untracked only

    layout = _layout_with_run_branch(tmp_path / "state", "child", "agent6/child")
    assert _ensure_on_run_branch(repo, layout) is None
    assert _current_branch(repo) == "agent6/child"
    assert (repo / "scratch.txt").exists()  # untracked file preserved


# --- resume/fork accept an omitted run id (most recent) --------------------


def test_fork_without_id_forks_most_recent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    _seed_source_run(state_dir, "only-run-AAAA11", head_sha=head, turns=(1,))
    rc = _cmd_fork(None, "", new_run_id="child-BBBB22", no_run=True)
    assert rc == 0
    dst = RunLayout(state_dir=state_dir, run_id="child-BBBB22")
    manifest = json.loads(dst.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_run_id"] == "only-run-AAAA11"  # the only/most-recent run


def test_fork_continue_resumes_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default `agent6 fork` continue path just cloned the checkpoint and cut
    # the branch at its head_sha, so the resume head guard passes by
    # construction. force stays OFF so a genuinely misaligned fork still
    # refuses instead of resuming against the wrong worktree.
    repo = tmp_path / "repo"
    head = _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    _seed_source_run(state_dir, "src-AAAA11", head_sha=head, turns=(1,))
    captured: dict[str, Any] = {}

    def _fake_resume(config_path: object, run_id: str, *, force: bool, **_kw: object) -> int:
        captured["force"] = force
        captured["run_id"] = run_id
        return 0

    monkeypatch.setattr("agent6.cli.fork._cmd_resume", _fake_resume)
    rc = _cmd_fork(None, "src", new_run_id="child-BBBB22")  # default: continue
    assert rc == 0
    assert captured["force"] is False
    assert captured["run_id"] == "child-BBBB22"


def test_fork_without_id_and_no_runs_errors_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _git_repo(repo)
    monkeypatch.chdir(repo)
    rc = _cmd_fork(None, "")  # no id, no runs -> clean error, not a crash
    assert rc == 2
    assert "nothing to fork" in capsys.readouterr().err


def test_resume_without_id_and_no_runs_errors_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _git_repo(repo)
    monkeypatch.chdir(repo)
    rc = _cmd_resume(None, "", force=False)
    assert rc == 2
    assert "nothing to resume" in capsys.readouterr().err


def test_ensure_on_run_branch_noop_without_run_branch(tmp_path: Path) -> None:
    # branch_per_run was off: no run_branch recorded, so HEAD is left alone.
    from agent6.cli.run import _ensure_on_run_branch  # pyright: ignore[reportPrivateUsage]

    repo = tmp_path / "repo"
    _git_repo(repo)
    layout = _layout_with_run_branch(tmp_path / "state", "child", None)
    assert _ensure_on_run_branch(repo, layout) is None
    assert _current_branch(repo) == "main"
