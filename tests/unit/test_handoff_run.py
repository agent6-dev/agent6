"""A parent and six detached children run the fixed job in each arm."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tomllib
from typing import Any

import pytest

from tests import launch_env


def _events(log: pathlib.Path, kind: str) -> list[dict[str, Any]]:
    return [e for line in log.read_text().splitlines() if (e := json.loads(line))["type"] == kind]


def test_scripted_pair_uses_the_same_queue_and_real_child_journals(tmp_path: pathlib.Path) -> None:
    from bench.handoff import job

    out = tmp_path / "set"
    proc = subprocess.run(
        [sys.executable, "-m", "bench.handoff.run", "--scripted", "--reps", "1", "--out", out],
        env=os.environ | {"AGENT6_ALLOW_ROOT": "1"},
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rows] == ["text", "id"]
    assert len({r["seed"] for r in rows}) == 1
    for row in rows:
        assert row["issues"] == [] and row["exit_code"] == 0
        assert row["all_accepted"] is True and row["stop_reason"] == "finish_session"
        assert row["started_twice"] == row["never_started"] == row["off_arm_calls"] == 0
        assert list(row["starts_by_task"].values()) == [1] * 6
        assert row["under_two_s"] > 0
        assert row["input_tokens"] == 6 * row["turns"] and row["output_tokens"] == 2 * row["turns"]
        assert row["cache_read_tokens"] == 4 * row["turns"] and row["cache_write_tokens"] is None
        root = out / row["journal"]
        assert [q["title"] for q in _events(root, "task.queued")] == list(job.TITLES)
        assert _events(root, "pin.restored")[0]["pins"] == [job.PIN]
        parallel = [
            e["args_preview"] for e in _events(root, "tool.call") if e["name"] == "parallel"
        ]
        field = "task" if row["arm"] == "text" else "task_id"
        assert parallel == [
            {field: x} for x in (job.TITLES if field == "task" else row["starts_by_task"])
        ]
        child_logs = sorted(root.parent.parent.glob(f"{root.parent.name}-*/logs.jsonl"))
        assert len(child_logs) == 6
        for name, child_log in zip(job.FILES, child_logs, strict=True):
            (read,) = [e for e in _events(child_log, "tool.call") if e["name"] == "read_file"]
            assert read["args_preview"]["path"] == name
            (stop,) = _events(child_log, "session.stop")
            repo = (out / row["journal"]).relative_to(out).parts[0]
            assert stop["summary"] == (out / repo / "repo" / name).read_text().strip()


def test_a_deviating_parent_keeps_its_measures(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused list argument and a ranking reviewer's start each left known measures unknown."""
    from bench.handoff import job, run, scripted

    follow = scripted.parent_reply
    refused = False

    def deviate(opening: str, answers: list[dict[str, Any]]) -> launch_env.Reply:
        nonlocal refused
        reply = follow(opening, answers)
        if isinstance(reply, list) and reply[0] == ("parallel", {"task": job.TITLES[0]}):
            if refused:
                return "parallel", {"task": job.TITLES[0], "spec": "2"}
            refused = True
            return "parallel", {"task": [job.TITLES[0]]}
        return reply

    monkeypatch.setattr(scripted, "parent_reply", deviate)
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "1")
    with scripted.profile() as profile:
        row = run.sample(tmp_path, profile, "scripted/worker", "text", 1, scripted=True)
    assert row["issues"] == []
    assert (row["off_arm_calls"], row["all_accepted"]) == (1, True)
    # The scripted parent refills by children started, so the second attempt displaces beech.
    assert list(row["starts_by_task"].values()) == [2, 0, 1, 1, 1, 1]
    assert (row["started_twice"], row["never_started"]) == (1, 1)
    assert row["under_two_s"] > 0


def test_live_profile_cannot_override_the_job_or_isolation(tmp_path: pathlib.Path) -> None:
    from bench.handoff import run

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.toml").write_text('[sandbox]\nisolation = "none"\n')
    out = tmp_path / "refused"
    with pytest.raises(SystemExit):
        run.main(["--profile", str(profile), "--out", str(out)])
    assert not out.exists()
    cfg = tomllib.loads(run.config("sample/model", profile, scripted=False))
    assert cfg["sandbox"]["isolation"] == "full" and cfg["sandbox"]["run_commands"] == "deny"
    assert cfg["git"]["control"] == "model" and cfg["harness"]["parallel"]["max_children"] == 2
