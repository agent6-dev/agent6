"""The real detached launcher carries child snapshots and their diagnostic metadata."""

from __future__ import annotations

import pathlib
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import pytest

from agent6 import child_records, kinds, paths
from agent6.app import child_wait, children, stop
from agent6.config import Config, RoleModel
from agent6.sessions import manifest, process
from agent6.ui.cli import parallel
from agent6.viewmodel.session import log as session_log
from tests import launch_env


@pytest.fixture
def local_provider() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Answer a real child's model call with finish_session, without a paid provider."""
    with launch_env.scripted_provider() as served:
        yield served


def test_detached_child_loads_snapshot_and_records_notes(
    short_repo: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    local_provider: tuple[str, list[dict[str, Any]]],
) -> None:
    """Dropping either snapshot marker at the launcher changes the child's durable facts."""
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "1")
    subprocess.run(["git", "init", "-q", "-b", "main", str(short_repo)], check=True)
    (short_repo / "README.md").write_text("Local child smoke.\n")
    subprocess.run(["git", "-C", str(short_repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(short_repo), "commit", "-q", "-m", "init"], check=True)
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_text(
        '[models.default]\nmodel="p/base"\neffort="high"\ntemperature=0.7\n'
        '[models.worker]\neffort="max"\n'
    )
    base_url, calls = local_provider
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": base_url}},
            "models": {"default": {"model": "p/base", "effort": "high", "temperature": 0.7}},
            "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "deny"},
            "harness": {"verify": {"when": "never"}, "max_turns": 3},
            "review": {"when": "never"},
        }
    )
    cfg = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"worker": RoleModel(effort=None, temperature=None)}
            )
        }
    )
    spec = child_records.ChildSpec(
        number=1,
        session_id="detached-l1",
        worktree=tmp_path / "child-1",
        settings=kinds.ModelSettings(kinds.ModelRoute("p", "local-reasoning")),
    )
    # The child's record is in the dispatching repository's state from its first event.
    session_dir = paths.state_dir(short_repo) / "sessions" / "runs" / spec.session_id
    try:
        result = children.bridge_spawner(
            spec,
            "Finish immediately without edits.",
            cfg=cfg,
            explicit_leaves=frozenset(
                {"sandbox.isolation", "sandbox.protect_git", "sandbox.run_commands"}
            ),
            origin=short_repo,
            record_state=paths.state_dir(short_repo),
            max_usd=None,
            group_id="detached",
            parent="detached",
            runtime=parallel.child_runtime(),
        )
        assert result.ok, result.error
        deadline = time.monotonic() + 30
        assert child_wait.await_child(
            result, poll_interval_s=0.05, should_stop=lambda: time.monotonic() >= deadline
        ), "detached child did not end before the deadline"
        log = session_dir / "logs.jsonl"
        assert session_log.scan_session_log(log).stop_reason == "finish_session", log.read_text()
        record = manifest.read_manifest(session_dir)
        assert record.models.driver is not None
        assert record.models.driver.model == "local-reasoning"
        assert record.models.driver.effort == "low"
        assert record.models.driver.effort_source == "backend default"
        assert record.parallel is not None
        assert record.parallel.config_notes == (
            "models.worker.temperature: selected None; TOML key omitted; reload uses 0.0.",
        )
        assert calls and all(
            call["model"] == "local-reasoning"
            and call["temperature"] == 0.0
            and call["reasoning"] == {"effort": "low"}
            for call in calls
        )
    finally:
        if process.worker_is_alive(session_dir):
            stop.stop_session(session_dir, wait_s=1, grace_s=1)
