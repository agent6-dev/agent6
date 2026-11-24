"""Drive the tool benchmark against a local Chat Completions endpoint."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from agent6.sandbox import jail
from tests import launch_env

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_run_set_counts_failed_edits_and_grades_the_final_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected patch and a successful edit have different outcomes in the same run."""
    changes = [
        ("if x < lo:\n        return hi", "if x < lo:\n        return lo"),
        ("sum(xs) // len(xs)", "sum(xs) / len(xs)"),
        ("    n = len(xs)", "    xs = sorted(xs)\n    n = len(xs)"),
        ("    return a\n", "    return abs(a)\n"),
        ("while i * i < n:", "while i * i <= n:"),
        ('(500, "D"),', '(900, "CM"), (500, "D"), (400, "CD"),'),
        ("    cur = 0", "    cur = xs[0] if xs else 0"),
    ]
    replies: list[tuple[str, dict[str, Any]] | str] = [
        ("apply_patch", {"patch": "not a patch"}),
        (
            "apply_edit",
            {
                "path": "shapes.py",
                "edits": [{"old_string": a, "new_string": b} for a, b in changes],
            },
        ),
        ("run_verify_command", {}),
        launch_env.FINISH,
    ]
    with launch_env.scripted_provider(replies) as (url, calls):
        rows = _run(tmp_path, monkeypatch, url, "short", "bugs")
    assert {"parallel", "ask_user", "apply_edit", "apply_patch"} <= {
        t["function"]["name"] for t in calls[0]["tools"]
    }
    (row,) = rows
    assert row["error"] is None, row
    assert row["passed"] and row["cases_passed"] == row["cases_total"] > 0
    assert (row["patch_calls"], row["patch_failures"]) == (1, 1)
    assert (row["edit_calls"], row["edit_failures"]) == (1, 0)
    assert (row["turns"], row["input_tokens"], row["output_tokens"]) == (4, 4, 4)


@pytest.mark.parametrize("arm", ["control", "no-docs"])
def test_ask_scores_the_journal_answer_not_a_tool_result(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """Each replicate starts fresh, even when the preceding answer was correct."""
    with launch_env.scripted_provider(
        [("agent6_docs", {"name": "CONFIG"}), "`--max-plan-points`", "--max-usd"],
        usage={
            "prompt_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens": 2,
        },
    ) as (url, calls):
        rows = _run(tmp_path, monkeypatch, url, arm, "plan-cap", reps=2)
    assert ("agent6_docs" in {t["function"]["name"] for t in calls[0]["tools"]}) == (
        arm == "control"
    )
    assert len(rows) == 2 and rows[0]["session_id"] != rows[1]["session_id"]
    assert [r["passed"] for r in rows] == [True, False]
    assert [r["turns"] for r in rows] == [2, 1]
    assert rows[0]["input_tokens"] == 20 and rows[0]["output_tokens"] == 4
    assert rows[0]["answer"] == "`--max-plan-points`"
    for r in rows:
        assert (r["arm"], r["split"], r["model"], r["effort"]) == (arm, "eval", "local/m", "medium")
        log = tmp_path / "out" / f"plan-cap-{r['rep']}" / r["journal"]
        events = [json.loads(line) for line in log.read_text().splitlines()]
        budget = next(event for event in events if event["type"] == "budget.update")
        assert (budget["plan_cap"], budget["usd_cap"], budget["tokens_fallback_cap"]) == (
            2,
            1,
            100000,
        )
        docs = next(
            (e for e in events if e["type"] == "tool.result" and e["name"] == "agent6_docs"), None
        )
        if r["rep"] == 1:
            assert docs and docs["ok"] == (arm == "control")


def test_offline_install_runs_the_patched_wheel(tmp_path: pathlib.Path) -> None:
    """The CLI uses the arm's install, not an agent6 already on PATH."""
    env = os.environ.copy()
    # Pytest isolates XDG; uv needs its prepared interpreters and package cache for the build.
    env.setdefault("UV_CACHE_DIR", str(pathlib.Path.home() / ".cache/uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(pathlib.Path.home() / ".local/share/uv/python"))
    probe = subprocess.run(
        ["uv", "python", "find", "--offline", "3.13"], env=env, capture_output=True, check=False
    )
    if probe.returncode:
        pytest.skip("offline wheel smoke needs uv's Python 3.13")
    with launch_env.scripted_provider() as (url, calls):
        argv = _arguments(tmp_path, url, "no-patch", "textkit", 1)
        built = subprocess.run(
            [sys.executable, "-m", "bench.tools.run", *argv],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    if (
        built.returncode
        and "Packages were unavailable because the network was disabled" in built.stderr
    ):
        pytest.skip(
            "offline wheel smoke needs cached build and runtime dependencies for Python 3.13"
        )
    assert built.returncode == 0, built.stderr
    tools = {t["function"]["name"] for t in calls[0]["tools"]}
    assert "apply_edit" in tools and "apply_patch" not in tools
    row = json.loads((tmp_path / "out/results.jsonl").read_text())
    assert row["error"] is None and not row["passed"]
    assert row["cases_passed"] == 0 < row["cases_total"]


def _run(
    where: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    arm: str,
    task: str,
    reps: int = 1,
) -> list[dict[str, Any]]:
    from bench.tools import run

    # Use the suite's installed dependencies without requiring a populated offline wheel cache.
    def install(source: pathlib.Path, venv: pathlib.Path) -> pathlib.Path:
        entry = venv / "bin/agent6"
        entry.parent.mkdir(parents=True)
        entry.write_text(
            f"#!{sys.executable} -I\nimport sys\nsys.path.insert(0, {str(source / 'src')!r})\n"
            "from agent6.ui.cli.entry import cli_main\nraise SystemExit(cli_main())\n"
        )
        entry.chmod(0o755)
        return entry

    monkeypatch.setattr(run, "install", install)
    argv = _arguments(where, url, arm, task, reps)
    out = where / "out"
    assert run.main(argv) == 0, [log.read_text() for log in out.rglob("agent.log")]
    return [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]


def _arguments(where: pathlib.Path, url: str, arm: str, task: str, reps: int) -> list[str]:
    profile = where / "profile"
    profile.mkdir()
    (profile / "config.toml").write_text(
        f'[providers.local]\nbackend = "openai"\nbase_url = "{url}"\n'
        '[sandbox]\nisolation = "partial"\n'
    )
    return [
        "--arm",
        arm,
        "--launcher",
        str(jail.require_jail_binary()),
        "--model",
        "local/m",
        "--profile",
        str(profile),
        "--tasks",
        task,
        "--reps",
        str(reps),
        "--out",
        str(where / "out"),
    ]
