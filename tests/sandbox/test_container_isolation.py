from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from agent6 import paths
from agent6.app import environment_lease, session_runtime, stamps
from agent6.app import reporter as app_reporter
from agent6.config import Config, RoleModel
from agent6.sandbox import container
from agent6.sessions import event_log, layout, manifest
from agent6.tools import operator_prompts
from tests.sandbox import conftest


def test_a_prepared_container_run_settles_its_lease_through_the_dispatcher(
    target: conftest.Target, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one emitter: the dispatcher reports the lease's outcome after the session closed."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = target.discovery.target.host_root
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"isolation": "container", "run_commands": "allow"},
            "container": {"exec": list(target.discovery.exec)},
            "review": {"when": "never"},
        }
    )
    state_dir = paths.state_dir(root)
    saved = layout.SessionLayout(state_dir, "container-run")
    saved.ensure()
    stamps.write_manifest(saved.manifest_path, manifest.SessionManifest(session_id="container-run"))
    notes: list[str] = []
    reporter = app_reporter.Reporter(out=notes.append, err=notes.append)
    access = session_runtime.prepare_container_execution(
        cfg,
        root,
        session_runtime.container_options(cfg),
        state_dir=state_dir,
        session_id="container-run",
        execution_id="e1",
        reporter=reporter,
    )
    assert access.session is not None and access.session.target is not None
    key = environment_lease.lease_key(
        environment_lease.selector(root, session_runtime.container_options(cfg))
    )
    assert access.session.target.lifecycle_ref == key
    assert any("opaque exec prefix" in note for note in notes), notes
    lease = environment_lease.read_lease(state_dir, key)
    assert lease is not None and lease.owner is None
    assert [b.session_id for b in lease.borrowers] == ["container-run"]
    events = event_log.EventSink(saved.logs_path)
    tools = session_runtime.build_session_tools(
        cfg,
        cwd=root,
        state_dir=state_dir,
        layout=saved,
        isolation="none",
        mode="run",
        events=events,
        prompts=operator_prompts.OperatorPrompts(session_dir=saved.session_dir),
        loop_log=lambda _line: None,
        mcp_manager=None,
        rm_role=RoleModel(provider="test", model="test"),
        command_access=access,
    )
    try:
        result = tools.dispatcher.run_setup(("/bin/sh", "-c", "echo inside"))
        assert result.returncode == 0 and result.stdout == "inside\n"
    finally:
        tools.dispatcher.close()
    rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
    environments = [row for row in rows if row["type"] == "execution.environment"]
    assert len(environments) == 1
    assert environments[0]["outcome"] == "left_running"
    assert environments[0]["reason"] == "attached, not owned"
    targets = [row for row in rows if row["type"] == "execution.target"]
    assert targets and targets[0]["target"]["lifecycle_ref"] == key
    after = environment_lease.read_lease(state_dir, key)
    assert after is not None and after.borrowers == ()


def test_a_hidden_path_refusal_releases_the_lease(
    target: conftest.Target, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation that refuses after acquiring leaves no borrower behind."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = target.discovery.target.host_root
    cfg = Config.model_validate(
        {
            "sandbox": {"isolation": "container", "hide_paths": [str(tmp_path / "secret")]},
            "container": {"exec": list(target.discovery.exec)},
        }
    )
    state_dir = paths.state_dir(root)
    options = session_runtime.container_options(cfg)
    try:
        session_runtime.prepare_container_execution(
            cfg, root, options, state_dir=state_dir, session_id="r", execution_id="e"
        )
    except container.ContainerMappingError as exc:
        assert "cannot be inspected" in str(exc)
    else:
        raise AssertionError("an uninspectable hidden path must refuse")
    lease = environment_lease.read_lease(
        state_dir, environment_lease.lease_key(environment_lease.selector(root, options))
    )
    assert lease is not None and lease.borrowers == ()
