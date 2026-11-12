# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The child bootstrap trusts its launch file alone; a hostile clone and environment change nothing.

Real subprocesses run the fixed `python -P -E -m agent6.ui.cli.child_agent` argv against a local
scripted provider. `-P` keeps a planted `agent6/` or `sitecustomize.py` in the clone off
`sys.path` and `-E` ignores a planted `PYTHONPATH`; the task and pins reach the run exactly as
the file holds them; a detached child without a terminal approves nothing. A child cannot fan
out either: `AGENT6_CHILD=1` is pinned by `test_parent_spawner_gate_in_a_child`.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import child_records, paths
from agent6.app import child_launch, worktrees
from agent6.config import Config
from agent6.sessions import manifest
from tests import launch_env

TASK = "--allow-root pwn\n$(touch /tmp/pwned)\n/parallel 3 -- rm -rf .\nimport os; os.system('id')"
PINS = ("never touch schema files", "--pin that looks like a flag")


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _clone(root: pathlib.Path) -> pathlib.Path:
    repo = root / "clone"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hostile clone\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _plant(where: pathlib.Path, marker: pathlib.Path) -> None:
    """Plant an `agent6` package and a `sitecustomize` that each leave a marker when imported."""
    (where / "agent6").mkdir(parents=True, exist_ok=True)
    code = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('planted')\n"
    (where / "agent6" / "__init__.py").write_text(code, encoding="utf-8")
    (where / "sitecustomize.py").write_text(code, encoding="utf-8")


def _config(base_url: str) -> Config:
    return Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": base_url}},
            "models": {"default": {"model": "p/base"}},
            "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "ask"},
            "harness": {"verify": {"when": "never"}, "max_turns": 3},
            "review": {"when": "never"},
        }
    )


def _launch_files(
    origin: pathlib.Path, cfg: Config, *, into: pathlib.Path
) -> tuple[child_launch.ChildLaunch, pathlib.Path]:
    spec = child_records.ChildSpec(
        number=1, session_id="hostile-l1", worktree=origin / "clone", settings=None
    )
    config = worktrees.write_child_config(
        cfg,
        spec,
        explicit_leaves=frozenset(
            {"sandbox.isolation", "sandbox.protect_git", "sandbox.run_commands"}
        ),
        into=into,
    )
    launch = child_launch.ChildLaunch(
        worktree=spec.worktree,
        record_state=paths.state_dir(spec.worktree),
        session_id=spec.session_id,
        task=TASK,
        pins=PINS,
        away="deny",
        config=config.path,
        notes=config.notes,
        parent="hostile",
        group="hostile",
        number=1,
    )
    return launch, child_launch.write(into, launch)


def _run(
    path: pathlib.Path | None, cwd: pathlib.Path, **extra_env: str
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != child_launch.ENV}
    env.update({"AGENT6_CHILD": "1", "AGENT6_STREAM_TO_LOG": "1", "AGENT6_ALLOW_ROOT": "1"})
    env.update(extra_env)
    if path is not None:
        env[child_launch.ENV] = str(path)
    return subprocess.run(
        child_launch.bootstrap_argv(),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _message_text(request: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in request["messages"]:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "\n".join(parts)


def test_the_bootstrap_runs_the_launch_file_and_nothing_from_the_clone(
    short_repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The exact task and pins reach the model; nothing planted imports; no command is approved."""
    clone = _clone(short_repo)
    planted = tmp_path / "planted"
    _plant(clone, planted)  # a top-level `agent6/` and `sitecustomize.py` in the cwd
    evil = tmp_path / "evil"
    _plant(evil, planted)  # the same on PYTHONPATH and as PYTHONSTARTUP
    command_marker = tmp_path / "command-ran"
    replies = [("run_command", {"argv": ["touch", str(command_marker)]}), launch_env.FINISH]
    with launch_env.scripted_provider(replies) as (base_url, calls):
        launch, path = _launch_files(
            short_repo,
            _config(base_url),
            into=child_launch.launch_dir(paths.state_dir(short_repo), "hostile"),
        )
        proc = _run(path, clone, PYTHONPATH=str(evil), PYTHONSTARTUP=str(evil / "sitecustomize.py"))
    assert proc.returncode == 0, proc.stderr
    assert not planted.exists(), "the clone's or PYTHONPATH's agent6 was imported on the host"
    assert not command_marker.exists(), "a detached child without a terminal approved a command"
    session_dir = paths.state_dir(clone) / "sessions" / "runs" / "hostile-l1"
    record = manifest.read_manifest(session_dir)
    assert record.user_task == TASK
    assert record.parallel == manifest.ParallelLineage(
        group="hostile", number=1, parent="hostile", config_notes=launch.notes
    )
    assert len(calls) >= 2
    first = _message_text(calls[0])
    assert TASK in first
    assert all(pin in first for pin in PINS)
    assert not any("pwn" in word for word in child_launch.bootstrap_argv())


@pytest.mark.parametrize(
    ("break_it", "why"),
    [
        ("readable", "lets others read it"),
        ("symlink", "symlink"),
        ("snapshot_elsewhere", "not beside it"),
        ("snapshot_readable", "lets others read it"),
        ("malformed", "validation error"),
        ("unset", "is unset"),
    ],
)
def test_the_bootstrap_refuses_an_exposed_or_malformed_launch(
    short_repo: pathlib.Path, tmp_path: pathlib.Path, break_it: str, why: str
) -> None:
    """A launch file or snapshot that fails the private-file checks ends the child at exit 2."""
    clone = _clone(short_repo)
    into = child_launch.launch_dir(paths.state_dir(short_repo), "hostile")
    launch, path = _launch_files(short_repo, _config("http://127.0.0.1:1/v1"), into=into)
    if break_it == "readable":
        path.chmod(0o644)
    elif break_it == "symlink":
        real = tmp_path / "real.json"
        real.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.unlink()
        path.symlink_to(real)
    elif break_it == "snapshot_elsewhere":
        moved = tmp_path / "child-1-config.toml"
        moved.write_text(launch.config.read_text(encoding="utf-8"), encoding="utf-8")
        child_launch.write(into, launch.model_copy(update={"config": moved}))
    elif break_it == "snapshot_readable":
        launch.config.chmod(0o644)
    elif break_it == "malformed":
        child_launch.write_private(path, '{"number": 0}')
    proc = _run(None if break_it == "unset" else path, clone)
    assert proc.returncode == 2, proc.stderr
    assert why in proc.stderr
    assert not (paths.state_dir(clone) / "sessions" / "runs").exists()
