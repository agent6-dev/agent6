"""Drive the tool benchmark against a local Chat Completions endpoint."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Generator
from typing import Any
from unittest import mock

import pytest

from agent6.sandbox import detect, jail
from tests import launch_env

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_runtime_resolves_commands_only_in_system_directories(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither a relative entry nor an absolute trial path enters the runtime search."""
    from bench.tools import run

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.toml").touch()
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = repo / "bench-only-tool"
    marker.touch(mode=0o700)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("PATH", f":.:bin:{repo}:/usr/bin:")
    with run.environment(tmp_path, profile) as env:
        path = env["PATH"]
        assert set(path.split(os.pathsep)) <= {"/bin", "/usr/bin"}
        assert shutil.which("git", path=path) in {"/bin/git", "/usr/bin/git"}
        assert shutil.which(marker.name, path=path) is None


def test_profile_cannot_select_partial_isolation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench.tools import run

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.toml").write_text(
        '[sandbox]\nisolation = "partial"\n[providers.chatgpt]\nbackend = "chatgpt"\n'
    )
    (profile / "secrets.toml").touch(mode=0o600)
    monkeypatch.setattr(run, "archive", mock.Mock(side_effect=AssertionError("unexpected build")))
    with pytest.raises(SystemExit) as exc:
        run.main(
            [
                "--arm",
                "control",
                "--tasks",
                "textkit",
                "--launcher",
                "unused",
                "--profile",
                str(profile),
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 2 and not (tmp_path / "out").exists()


def test_config_masks_the_source_profile_and_the_arms_build(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bench.tools import run

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.toml").write_text('[providers.chatgpt]\nbackend="chatgpt"\n')
    (profile / "secrets.toml").touch(mode=0o600)
    alias = tmp_path / "profile-link"
    alias.symlink_to(profile, target_is_directory=True)
    build = tmp_path / "build"
    monkeypatch.setattr(run, "archive", mock.Mock(return_value=build / "src"))
    monkeypatch.setattr(run, "install", mock.Mock(return_value=build / "venv/bin/agent6"))

    def inspect_config(repo: pathlib.Path, env: dict[str, str]) -> None:
        config = tomllib.loads((repo.parent / "run.toml").read_text())["sandbox"]
        assert config["isolation"] == "full"
        grant = pathlib.Path(env["XDG_CONFIG_HOME"]) / "agent6/secrets.toml"
        assert grant.samefile(profile / "secrets.toml")
        assert {str(build), str(profile)} <= set(config["hide_paths"])
        raise OSError("inspected before launch")

    monkeypatch.setattr(run, "seed", inspect_config)
    args = [
        "--arm",
        "control",
        "--tasks",
        "plan-cap",
        "--reps",
        "1",
        "--profile",
        str(alias),
        "--launcher",
        "unused",
        "--out",
        str(tmp_path / "out"),
    ]
    assert run.main(args) == 1
    row = json.loads((tmp_path / "out/results.jsonl").read_text())
    assert row["error"] == "inspected before launch"


def test_grader_policy_keeps_the_submission_and_its_metadata_read_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build the policy for the captured exec invocation without starting a jail."""
    from bench.tools import run

    from agent6.config import layer
    from agent6.ui.cli import exec_cmds

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.toml").touch()
    source = tmp_path / "build/src"
    task = pathlib.Path("bench/coreagent/tasks/eventflow")
    shutil.copytree(ROOT / task, source / task)
    trial = tmp_path / "trial"
    binary = tmp_path / "fixture"
    binary.write_text(
        f"#!{sys.executable} -I\nimport json, os, sys\n"
        "if 'exec' in sys.argv:\n"
        "    sys.stdin.buffer.read()\n"
        "    print(json.dumps([os.getcwd(), sys.argv[1:]]), file=sys.stderr)\n"
        "    print(json.dumps(dict(cases_passed=0, cases_total=25, components_passed=0, "
        "components_total=8, score=0)))\n"
        "else:\n"
        "    print(os.getcwd())\n"
    )
    binary.chmod(0o700)
    monkeypatch.setattr(run.journals, "measure", mock.Mock(return_value={}))
    args = argparse.Namespace(profile=profile, model="local/m", effort="medium", max_plan_points=2)
    run.sample(binary, source, args, "eventflow", trial, {})

    repo = pathlib.Path((trial / "agent.log").read_text().strip())
    cwd, argv = json.loads((trial / "grader.stderr").read_text())
    workspace = pathlib.Path(cwd)
    assert not repo.is_relative_to(workspace) and not workspace.is_relative_to(repo)
    assert (workspace / ".git/HEAD").is_file() and (repo / ".git/HEAD").is_file()
    command = tuple(argv[argv.index("--new") + 1 :])
    assert pathlib.Path(command[-1]) == repo
    configs = tuple(pathlib.Path(argv[i + 1]) for i, arg in enumerate(argv) if arg == "--config")
    assert all(not path.is_relative_to(workspace) for path in configs)
    cfg = layer.load_effective(workspace, configs).config
    assert cfg.sandbox.isolation == "full"
    with exec_cmds.exec_policy(workspace, cfg, "full", command) as policy:
        assert policy.extra_ro_paths == (repo,)
        assert policy.extra_rw_paths == ()
        assert workspace / ".git" in policy.extra_protect_paths
        assert {profile, source.parent} <= set(policy.hide_paths)


@pytest.fixture
def scripted_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[pathlib.Path]:
    """Use partial isolation only for fixed test replies on a host without namespaces."""
    from bench.tools import run

    if detect.environment().userns_supported:
        yield tmp_path
        return
    if ROOT.is_relative_to("/tmp"):
        pytest.skip("partial test needs its build outside the host tmp grant")
    monkeypatch.setattr(run, "ISOLATION", "partial")
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".scripted-bench-") as where:
        yield pathlib.Path(where)


def test_run_set_counts_failed_edits_and_grades_the_final_tree(
    scripted_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
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
        rows = _run(scripted_root, monkeypatch, url, "short", "bugs")
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
    scripted_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, arm: str
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
        rows = _run(scripted_root, monkeypatch, url, arm, "plan-cap", reps=2)
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
        log = scripted_root / "out" / f"plan-cap-{r['rep']}" / r["journal"]
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


def test_offline_install_runs_the_patched_wheel(
    scripted_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """The harness uses the arm's install, not an agent6 already on PATH."""
    from bench.tools import run

    # Pytest isolates XDG; uv needs its prepared interpreters and package cache for the build.
    for key, path in (
        ("UV_CACHE_DIR", ".cache/uv"),
        ("UV_PYTHON_INSTALL_DIR", ".local/share/uv/python"),
    ):
        if key not in os.environ:
            monkeypatch.setenv(key, str(pathlib.Path.home() / path))
    probe = subprocess.run(
        ["uv", "python", "find", "--offline", "3.13"], capture_output=True, check=False
    )
    if probe.returncode:
        pytest.skip("offline wheel smoke needs uv's Python 3.13")
    with launch_env.scripted_provider() as (url, calls):
        argv = _arguments(scripted_root, url, "no-patch", "textkit", 1)
        try:
            result = run.main(argv)
        except subprocess.CalledProcessError:
            if (
                "Packages were unavailable because the network was disabled"
                in capfd.readouterr().err
            ):
                pytest.skip(
                    "offline wheel smoke needs cached Python 3.13 build and runtime dependencies"
                )
            raise
    assert result == 0, capfd.readouterr().err
    tools = {t["function"]["name"] for t in calls[0]["tools"]}
    assert "apply_edit" in tools and "apply_patch" not in tools
    row = json.loads((scripted_root / "out/results.jsonl").read_text())
    assert row["error"] is None and not row["passed"]
    assert row["cases_passed"] == 0 < row["cases_total"]


def test_unscored_replicate_keeps_its_row_and_the_set_continues(
    scripted_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    message = "scripted module diagnostic"
    edit = {
        "path": "textkit.py",
        "kind": "overwrite",
        "old_string": "",
        "new_string": f"import sys\nprint('textkit')\nprint({message!r}, file=sys.stderr)\n",
    }
    with launch_env.scripted_provider([("apply_edit", edit), launch_env.FINISH]) as (url, _):
        first, second = _run(
            scripted_root, monkeypatch, url, "control", "textkit", reps=2, expected_exit=1
        )
    assert first["passed"] is None and first["error"] == "expected one JSON grader report"
    assert second["error"] is None and second["passed"] is False and second["cases_total"] == 30
    assert message not in capfd.readouterr().err
    assert message in (scripted_root / "out/textkit-1/grader.stderr").read_text()


def _run(
    where: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    arm: str,
    task: str,
    reps: int = 1,
    expected_exit: int = 0,
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
    assert run.main(argv) == expected_exit, [log.read_text() for log in out.rglob("agent.log")]
    return [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]


def _arguments(where: pathlib.Path, url: str, arm: str, task: str, reps: int) -> list[str]:
    profile = where / "profile"
    profile.mkdir()
    (profile / "config.toml").write_text(
        f'[providers.local]\nbackend = "openai"\nbase_url = "{url}"\n'
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
