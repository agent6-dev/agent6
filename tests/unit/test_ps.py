# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps` lists every live session machine-wide, machine instances included.

Their worker.pid sits at the instance root, not under a bucket.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from agent6 import paths
from agent6.sessions import process
from agent6.ui.cli import main, ps_cmd

WAITER = """
machine = "waiter_demo"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
delay_s = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def test_ps_reads_a_live_machine_through_the_shared_status_word(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ps` reads a live machine through the shared status word.

    A live worker in a wait state is "waiting" on every other surface (`machine_word_for_dir`), not
    a hardcoded "running". The repo has run only a machine, so its state dir holds no sessions/ at
    all: ps must still list it.
    """
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER, encoding="utf-8")
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert not (paths.state_dir(tmp_path) / "sessions").exists()
    instance = paths.state_dir(tmp_path) / "machines" / "waiter_demo"
    # this process: alive by the pid rule
    process.write_worker_pid(instance, os.getpid())
    assert ps_cmd.cmd_ps() == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if "waiter_demo" in line)
    assert "machine" in row and "waiting" in row and str(os.getpid()) in row, row
    assert "running" not in row
    assert "agent6 machine status <id>" in out


def test_ps_lists_a_linked_lane_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lane's session dir is linked under the parent repo as well as its own state dir.

    One live session, one row.
    """
    base = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    lane = base / "agent6" / "lane-repo-000000" / "sessions" / "runs" / "fan-l1"
    lane.mkdir(parents=True)
    (lane / "manifest.json").write_text(json.dumps({"mode": "run"}), encoding="utf-8")
    (lane / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(lane, os.getpid())
    parent = base / "agent6" / "parent-repo-111111" / "sessions" / "runs"
    parent.mkdir(parents=True)
    (parent / "fan-l1").symlink_to(lane, target_is_directory=True)
    assert ps_cmd.cmd_ps() == 0
    assert capsys.readouterr().out.count("fan-l1") == 1


def test_ps_lists_same_named_live_sessions_from_different_repositories(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    repo_ids = ("first-repo-111111", "second-repo-222222")
    for repo_id in repo_ids:
        session = base / "agent6" / repo_id / "sessions" / "runs" / "same-run"
        session.mkdir(parents=True)
        (session / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": repo_id}) + "\n",
            encoding="utf-8",
        )
        process.write_worker_pid(session, os.getpid())

    assert ps_cmd.cmd_ps(as_json=True) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {(row["repo_id"], row["id"]) for row in rows} == {
        ("first-repo-111111", "same-run"),
        ("second-repo-222222", "same-run"),
    }


def test_ps_json_carries_the_row_facts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ps --json` carries the row facts.

    Without it the cross-repo view cannot be read by a script.
    """
    base = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    lane = base / "agent6" / "lane-repo-000000" / "sessions" / "runs" / "fan-l1"
    lane.mkdir(parents=True)
    (lane / "manifest.json").write_text(json.dumps({"mode": "run"}), encoding="utf-8")
    (lane / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(lane, os.getpid())
    assert main(["ps", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["id"] == "fan-l1" and row["mode"] == "run" and row["pid"] == os.getpid()
    assert row["attached"] is False and row["repo_id"] == "lane-repo-000000"
    assert row["directory"] is None  # a state-dir id with no checkout behind it


def test_ps_json_waiting_facts(
    tmp_path: pathlib.Path,
    short_repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON keeps the absolute directory and the separate status and detail of a waiting session."""
    home = short_repo
    repo = home / "repo"
    repo.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(time, "time", lambda: 1767225600.0)
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "waiting-run"
    _live(session_dir, {"mode": "run"})
    with (session_dir / "logs.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {"type": "approval.prompt", "id": "approval-1", "ts": "2026-01-01T00:00:00Z"}
            )
            + "\n"
        )

    assert main(["ps", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row == {
        "directory": str(repo),
        "repo_id": paths.repo_id(repo),
        "id": "waiting-run",
        "mode": "run",
        "status": "waiting",
        "status_detail": "approval <1m",
        "pid": os.getpid(),
        "attached": False,
        "parent": "",
        "lanes": [],
        "worktree": None,
    }

    assert main(["ps"]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if "waiting-run" in line)
    assert line.startswith("~/repo ")
    assert "waiting · approval <1m" in line
    assert str(home) not in line


def _live(session_dir: pathlib.Path, manifest: dict[str, object]) -> None:
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(session_dir, os.getpid())


@pytest.mark.parametrize(
    "args",
    [["sessions"], ["sessions", "show", "step-AAAAAA"], ["ps"]],
)
def test_pending_step_stop_is_visible_on_cli_listings(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
) -> None:
    """A stop after the turn must not stay plain running on sessions, show or ps."""
    monkeypatch.chdir(tmp_path)
    session_dir = paths.state_dir(tmp_path) / "sessions" / "runs" / "step-AAAAAA"
    _live(session_dir, {"mode": "run", "session_id": "step-AAAAAA", "user_task": "t"})
    assert main(["stop", "step-AAAAAA", "--after-turn"]) == 0
    assert "stops after its current turn" in capsys.readouterr().out
    assert main(args) == 0
    assert "running · stopping after this turn" in capsys.readouterr().out


def test_ps_folds_a_live_lane_under_its_live_parent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ps` folds a live lane under its live parent.

    A group in flight is one row with its live lane count; `--lanes` lists the lanes under it; a
    lane whose parent is not live stays a row of its own.
    """
    base = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    runs = base / "agent6" / "parent-repo-111111" / "sessions" / "runs"
    _live(runs / "fan", {"mode": "run"})
    _live(
        runs / "fan-l1",
        {"mode": "run", "parallel": {"group": "fan", "lane": 1, "parent": "fan"}},
    )
    _live(
        runs / "old-l1",
        {"mode": "run", "parallel": {"group": "old", "lane": 1, "parent": "old"}},
    )

    assert ps_cmd.cmd_ps() == 0
    out = capsys.readouterr().out
    assert "fan (1 lane)" in out and "fan-l1" not in out and "old-l1" in out
    assert ps_cmd.cmd_ps(lanes=True) == 0
    assert "└ fan-l1" in capsys.readouterr().out
    assert ps_cmd.cmd_ps(as_json=True) == 0
    rows = {r["id"]: r for r in json.loads(capsys.readouterr().out)}
    assert set(rows) == {"fan", "old-l1"}
    assert [ln["id"] for ln in rows["fan"]["lanes"]] == ["fan-l1"]


def test_ps_names_the_worktree_a_run_works_in(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every live run of one repository showed the checkout, whichever tree it worked in."""
    base = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv(
        "HOME", str(tmp_path)
    )  # the table contracts a path under HOME; JSON does not
    runs = base / "agent6" / "some-repo-000000" / "sessions" / "runs"
    worktree = tmp_path / "cache" / "apart"
    _live(runs / "apart", {"mode": "run", "worktree": str(worktree)})
    _live(runs / "inplace", {"mode": "run"})

    assert ps_cmd.cmd_ps() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["directory", "id", "mode", "status", "pid", "front-end", "worktree"]
    assert next(line for line in lines if " apart " in line).endswith("  ~/cache/apart")
    assert str(worktree) not in next(line for line in lines if " inplace " in line)

    assert ps_cmd.cmd_ps(as_json=True) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["apart"]["worktree"] == str(worktree) and rows["inplace"]["worktree"] is None
