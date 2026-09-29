# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions` (list): the winner marker on fan-out compare winners."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import paths
from agent6.ui.cli import (
    _common,  # pyright: ignore[reportPrivateUsage]
    sessions_cmds,
)


def _run(runs: pathlib.Path, session_id: str, *, winner: bool | None = None) -> None:
    d = runs / session_id
    d.mkdir(parents=True)
    manifest: dict[str, object] = {"mode": "run"}
    if winner is not None:
        rank = 1 if winner else 2
        manifest["compare"] = {"group": "fan", "rank": rank, "of": 2, "winner": winner}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": session_id})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )


def test_runs_list_marks_the_fan_out_winner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _common._runs_dir(repo)
    _run(runs, "fan-l1", winner=False)
    _run(runs, "fan-l2", winner=True)
    _run(runs, "solo")  # a run outside any fan-out: no marker

    assert sessions_cmds._cmd_list() == 0
    out = capsys.readouterr().out
    assert "fan-l2 ★" in out  # the winner id carries the ★
    assert "fan-l1 ★" not in out and "solo ★" not in out  # losers / non-lanes do not


def test_runs_list_json_carries_the_row_facts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions list --json` is the table's rows as data: the listing facts.

    The winner is a boolean; no styling.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert sessions_cmds._cmd_list(as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == []  # the empty listing is data too
    runs = _common._runs_dir(repo)
    _run(runs, "fan-l2", winner=True)
    assert sessions_cmds._cmd_list(as_json=True) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["session_id"] for r in rows] == ["fan-l2"]
    row = rows[0]
    assert row["winner"] is True
    assert (row["mode"], row["status"], row["task"]) == ("run", "passed", "fan-l2")
    # The one listing row shape, shared with `/api/hub` (viewmodel.summary_row).
    assert set(row) == {
        "session_id",
        "mode",
        "status",
        "reason",
        "label",
        "level",
        "unmerged",
        "verify_ok",
        "cost_usd",
        "cost",
        "id_cell",
        "usd_partial",
        "plan_consumed",
        "plan_cap",
        "model",
        "model_from_flag",
        "mtime",
        "when",
        "winner",
        "task",
        "task_line",
        "lanes",
        "lane",
        "coordinator",
    }


def test_sessions_dir_names_a_sessions_own_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions dir <id>` prints the session's directory; an unknown id is an error.

    An unknown id never resolves to the repo root.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _common._runs_dir(repo)
    _run(runs, "solo-ABC123")
    assert sessions_cmds._cmd_sessions_dir("solo") == 0
    assert capsys.readouterr().out.strip() == str(runs / "solo-ABC123")
    assert sessions_cmds._cmd_sessions_dir("nope") == 2
    assert "ERROR" in capsys.readouterr().err


def test_runs_list_uses_plan_points_for_a_plan_metered_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _common._runs_dir(repo)
    _run(runs, "plan-metered")
    log = runs / "plan-metered" / "logs.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    budget = {
        "type": "budget.update",
        "usd_total": 0.0,
        "plan_consumed": 2.5,
        "plan_cap": 6.0,
    }
    log.write_text("\n".join([lines[0], json.dumps(budget), *lines[1:]]) + "\n", encoding="utf-8")

    assert sessions_cmds._cmd_list() == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "plan-metered" in line)
    assert "2.5pt" in row
    assert "$" not in row

    assert sessions_cmds._cmd_list(as_json=True) == 0
    (data,) = json.loads(capsys.readouterr().out)
    assert (data["plan_consumed"], data["plan_cap"], data["cost"]) == (2.5, 6.0, "2.5pt")


def test_runs_list_marks_a_partial_cost(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cost the scanner knows is a lower bound renders with the '~' marker.

    `sessions show` renders it the same way.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _common._runs_dir(repo)
    d = runs / "unpriced"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"mode": "run"}), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": 0.0123, "usd_partial": True})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    assert sessions_cmds._cmd_list() == 0
    assert "~$0.01" in capsys.readouterr().out


def test_styled_status_colors_stale_red_and_parked_yellow() -> None:
    """The CLI status colors mirror the TUI and web: a lost worker is red.

    A parked submission is yellow.
    """
    stale, _ = _common.styled_status("stale", "", color=True)
    assert "\x1b[1;31m" in stale  # the error level, like failed: the run header + web pill agree
    parked, _ = _common.styled_status("parked", "resume to start", color=True)
    assert "\x1b[33m" in parked  # yellow: attention, not a neutral done


def test_runs_list_columns_stay_aligned_with_a_machine_draft(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every row's cost column starts where the header's does.

    A `machine create` draft's mode cell is wider than the fixed four-column cell.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _run(_common._runs_dir(repo), "runny-one-AAAAAA")
    draft = _common._runs_dir(repo).parent / "machines" / "drafty-two-BBBBBB"
    draft.mkdir(parents=True)
    (draft / "manifest.json").write_text(json.dumps({"mode": "machine"}), encoding="utf-8")
    (draft / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "machine", "user_task": "draft it"})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    assert sessions_cmds._cmd_list() == 0
    lines = capsys.readouterr().out.splitlines()
    id_col = lines[0].index("  id  ") + 2
    assert lines[1].index("drafty-two-BBBBBB") == id_col
    assert lines[2].index("runny-one-AAAAAA") == id_col


def test_listing_status_label_folds_mode_reason_and_unmerged() -> None:
    """One status cell folds the mode, the reason and the unmerged mark.

    The mode shows when the word does not imply it; the unmerged mark on ended runs only.
    """
    from agent6.viewmodel import format

    assert format.listing_status_label("run", "passed") == "passed"
    assert format.listing_status_label("run", "passed", unmerged=True) == "passed · unmerged"
    assert format.listing_status_label("plan", "planned") == "planned"
    assert format.listing_status_label("plan", "running") == "plan · running"
    assert format.listing_status_label("ask", "answered") == "answered"
    assert format.listing_status_label("machine", "finished") == "machine · finished"
    assert (
        format.listing_status_label("run", "failed", "provider_error", unmerged=True)
        == "failed · provider error · unmerged"
    )
    # A live run's branch is unmerged by definition: no mark.
    assert format.listing_status_label("run", "running", unmerged=True) == "running"


def test_runs_list_marks_an_unmerged_run_and_drops_the_mark_after_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finished run whose branch holds commits reads `unmerged`.

    A merge or a zero-commit branch drops it.

    Merged means stamp tip == branch tip; zero commits means tip == base.
    """
    import subprocess

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "agent6/unmerged-run-AAAAAA")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "work")
    tip = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")

    runs = _common._runs_dir(repo)
    for sid, branch, merged in (
        ("unmerged-run-AAAAAA", "agent6/unmerged-run-AAAAAA", None),
        (
            "merged-run-BBBBBB",
            "agent6/unmerged-run-AAAAAA",
            {"into": "main", "sha": tip, "tip": tip},
        ),
    ):
        d = runs / sid
        d.mkdir(parents=True)
        manifest: dict[str, object] = {
            "version": 2,
            "session_id": sid,
            "mode": "run",
            "user_task": "t",
            "base_sha": base,
            "run_branch": branch,
        }
        if merged:
            manifest["merged"] = merged
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (d / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
            + "\n"
            + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
            + "\n",
            encoding="utf-8",
        )
    assert sessions_cmds._cmd_list() == 0
    out = capsys.readouterr().out
    unmerged_row = next(line for line in out.splitlines() if "unmerged-run-AAAAAA" in line)
    merged_row = next(line for line in out.splitlines() if "merged-run-BBBBBB" in line)
    assert "passed · unmerged" in unmerged_row
    assert "unmerged" not in merged_row.replace("unmerged-run", "")
    assert "mode" not in out.splitlines()[0]  # the column folded into status


def test_runs_list_marks_a_branchless_chain_unmerged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A branch_per_run-off run's hidden chain carries the unmerged mark like a visible branch."""
    import subprocess

    from agent6 import git_ops

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    base = git("rev-parse", "HEAD")
    session_id = "branchless-run-AAAAAA"
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    git_ops.chain_commit(
        repo,
        "agent6 iter 1: work",
        ref=git_ops.chain_ref_for(session_id),
        fallback_parent=base,
    )

    _run(_common._runs_dir(repo), session_id)
    manifest = _common._runs_dir(repo) / session_id / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "mode": "run",
                "base_sha": base,
                "run_branch": None,
            }
        ),
        encoding="utf-8",
    )

    assert sessions_cmds._cmd_list() == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if session_id in line)
    assert "passed · unmerged" in row


def test_a_merge_stamped_on_a_diverged_branch_reads_merged_everywhere(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A merge stamped on a diverged branch reads merged everywhere.

    The operator's own commit on the run branch moves the branch and not the chain.
    """
    import subprocess

    from agent6 import git_ops
    from agent6.viewmodel import summarize_session_dir

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    base = git("rev-parse", "HEAD")
    session_id = "diverged-run-AAAAAA"
    branch = f"agent6/{session_id}"
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    chain = git_ops.chain_commit(
        repo,
        "agent6 iter 1: work",
        ref=git_ops.chain_ref_for(session_id),
        fallback_parent=base,
        also_branch=branch,
    )
    assert chain is not None
    own = git("commit-tree", f"{chain}^{{tree}}", "-p", chain, "-m", "operator")
    git("update-ref", f"refs/heads/{branch}", own)

    _run(_common._runs_dir(repo), session_id)
    session_dir = _common._runs_dir(repo) / session_id
    manifest: dict[str, object] = {
        "session_id": session_id,
        "mode": "run",
        "base_sha": base,
        "run_branch": branch,
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        summarize_session_dir(session_dir, branch_tips=git_ops.run_ref_tips(repo)).unmerged is True
    )
    manifest["merged"] = {"into": "main", "sha": own, "tip": own}
    (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        summarize_session_dir(session_dir, branch_tips=git_ops.run_ref_tips(repo)).unmerged is False
    )

    from agent6.ui.cli import sessions_show  # pyright: ignore[reportPrivateUsage]

    assert sessions_show._cmd_status(session_id) == 0
    out = capsys.readouterr().out
    assert f"changes:    {branch} (merged into main)" in out
    assert "merge with:" not in out


def test_model_controlled_run_refuses_the_git_surfaces() -> None:
    """A git_control = "model" manifest turns the git surfaces away with one message.

    sessions diff, merge, commits and fork: the record is the model's own commits.
    """
    from agent6.sessions import manifest as sessions_manifest

    agent6_run = sessions_manifest.SessionManifest(mode="run", session_id="x1")
    assert sessions_manifest.model_git_refusal(agent6_run, "sessions") is None
    model_run = sessions_manifest.SessionManifest(mode="run", session_id="x2", git_control="model")
    msg = sessions_manifest.model_git_refusal(model_run, "sessions diff")
    assert msg is not None and "model" in msg and "x2" in msg


def test_the_json_row_carries_the_whole_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `--json` row carries the whole task; only the table clips for width."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    from agent6.sessions import layout as sessions_layout

    task = "fix the parser bug\nsecond line of the task\nthird line"
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-TASK11")
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": task}) + "\n",
        encoding="utf-8",
    )

    from agent6.ui.cli import main

    assert main(["sessions", "list", "--json"]) == 0

    (row,) = json.loads(capsys.readouterr().out)
    assert row["task"] == task


def _fan_out(runs: pathlib.Path) -> None:
    """A `run --parallel` fan-out as it lands: the coordinator's record and two lanes naming it."""
    _run(runs, "fan")
    (runs / "fan" / "manifest.json").write_text(
        json.dumps({"mode": "run", "fanout": {"lanes": 2, "spec": "2"}}), encoding="utf-8"
    )
    for lane in (1, 2):
        _run(runs, f"fan-l{lane}", winner=lane == 2)
        (runs / f"fan-l{lane}" / "manifest.json").write_text(
            json.dumps(
                {
                    "mode": "run",
                    "models": {
                        "driver": {"provider": "openai", "model": f"lane-{lane}"},
                        "driver_from_flag": True,
                    },
                    "parallel": {"group": "fan", "lane": lane, "coordinator": "fan"},
                    "compare": {"rank": 3 - lane, "of": 2, "winner": lane == 2},
                }
            ),
            encoding="utf-8",
        )


def test_runs_list_folds_a_fan_outs_lanes_under_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fan-out is one row with its lane count; `--lanes` indents the lanes under it.

    The JSON row nests them always.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _fan_out(_common._runs_dir(repo))

    assert sessions_cmds._cmd_list() == 0
    out = capsys.readouterr().out
    assert "fan (2 lanes)" in out and "fan-l1" not in out
    assert sessions_cmds._cmd_list(lanes=True) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "fan " in lines[1] and "(2 lanes)" not in lines[1]
    assert "└ fan-l1" in lines[2] and "└ fan-l2 ★" in lines[3]
    assert sessions_cmds._cmd_list(as_json=True) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["session_id"] == "fan"
    assert [ln["session_id"] for ln in row["lanes"]] == ["fan-l1", "fan-l2"]
    assert row["lanes"][1]["winner"] is True
    assert [lane["model"] for lane in row["lanes"]] == ["openai/lane-1", "openai/lane-2"]
    assert all(lane["model_from_flag"] is True for lane in row["lanes"])


def test_a_folded_fan_out_shows_its_groups_latest_activity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A folded fan-out sorts by and shows its group's latest activity.

    The coordinator's own journal is quiet for the whole fan-out.
    """
    import os

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _common._runs_dir(repo)
    _fan_out(runs)
    _run(runs, "solo")
    old, mid, new = 1_700_000_000, 1_700_003_600, 1_700_007_200
    os.utime(runs / "fan" / "logs.jsonl", (old, old))
    os.utime(runs / "solo" / "logs.jsonl", (mid, mid))
    for lane in (1, 2):
        os.utime(runs / f"fan-l{lane}" / "logs.jsonl", (new, new))

    assert sessions_cmds._cmd_list(as_json=True) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["session_id"] for r in rows] == ["fan", "solo"]
    assert rows[0]["mtime"] == new
    assert sessions_cmds._cmd_list() == 0
    lines = capsys.readouterr().out.splitlines()
    from agent6.viewmodel import format

    assert lines[1].startswith(format.format_when(new)) and "fan (2 lanes)" in lines[1]
