# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 records and shows that a model's commands could write a checkout's `.git`."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import evidence, kinds, paths
from agent6.app import reporter as app_reporter
from agent6.app.session import runtime as session_runtime
from agent6.config import model as config_model
from agent6.sandbox import container, jail_session
from agent6.tools import target_policy
from agent6.ui.cli import check_sandbox
from tests.unit import test_environment_lease as lease


def _controls(raw: dict[str, object], isolation: str) -> tuple[kinds.EffectiveControl, ...]:
    cfg = config_model.Config.model_validate(raw)
    if isolation == "container":
        return target_policy.container_controls(cfg)
    levels: dict[str, kinds.IsolationLevel] = {"full": "full", "partial": "partial", "none": "none"}
    return target_policy.effective_controls(cfg, levels[isolation])


def _target(controls: tuple[kinds.EffectiveControl, ...]) -> kinds.ExecutionTargetStamp:
    return kinds.ExecutionTargetStamp(
        transport="local",
        host_root=pathlib.Path("/w"),
        command_root=pathlib.Path("/w"),
        user=None,
        home=None,
        environment=kinds.ExecutionEnvironmentPolicy(inherit=False, variable_names=()),
        controls=controls,
    )


@pytest.mark.parametrize(
    ("raw", "isolation", "writable"),
    [
        ({}, "full", False),
        ({"sandbox": {"protect_git": "on"}}, "full", False),
        ({"git": {"control": "model"}}, "full", True),
        ({"sandbox": {"protect_git": "off"}}, "full", True),
        ({}, "partial", True),
        ({}, "none", True),
        ({"sandbox": {"isolation": "container"}}, "container", True),
    ],
)
def test_a_session_says_when_commands_could_write_git(
    raw: dict[str, object], isolation: str, writable: bool
) -> None:
    """No surface said that a run's commands could write `.git`, only which mount was absent."""
    lines = evidence.target_lines("t", _target(_controls(raw, isolation)))
    assert (evidence.GIT_WRITABLE in lines) is writable


def test_the_checkout_keeps_the_record_under_a_later_config(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A config that protects `.git` again erased every trace that commands once wrote it."""
    checkout = tmp_path / "repo"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.chdir(checkout)
    protected = _controls({}, "full")
    session_runtime.record_git_exposure(checkout, protected)
    assert not paths.git_exposure_record(checkout).exists()
    check_sandbox._git_exposure(protected)
    assert evidence.GIT_WRITABLE not in capsys.readouterr().out

    session_runtime.record_git_exposure(checkout, _controls({"git": {"control": "model"}}, "full"))
    first = paths.git_exposure_record(checkout).read_text(encoding="utf-8")
    session_runtime.record_git_exposure(checkout, protected)
    assert paths.git_exposure_record(checkout).read_text(encoding="utf-8") == first
    check_sandbox._git_exposure(protected)
    assert evidence.GIT_WRITABLE in capsys.readouterr().out


def test_a_local_execution_records_the_access_before_its_session_opens(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record's pins called `record_git_exposure` themselves.

    The call in `prepare_local_execution` could go and every test stayed green: a run's
    commands then wrote `.git` with no record that they could.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    checkout = tmp_path / "repo"
    (checkout / ".git").mkdir(parents=True)
    record = paths.git_exposure_record(checkout)
    recorded: list[bool] = []
    open_session = jail_session.JailSession.open

    def _open(policy: kinds.JailPolicy, **kwargs: Any) -> jail_session.JailSession:
        recorded.append(record.exists())
        return open_session(policy, **kwargs)

    monkeypatch.setattr(jail_session.JailSession, "open", _open)

    session_runtime.prepare_local_execution(config_model.Config(), checkout, "none").close()

    assert recorded == [True]


def _recorded_when_a_clones_container_is_prepared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, bool]:
    """Prepare a child clone's container execution; return which checkouts had a record by then."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "workspace"
    clone = workspace / ".agent6/worktrees/repo-1/run-p1/child-1"
    clone.mkdir(parents=True)
    fake = lease.Fake(tmp_path, workspace)
    fake.running("a" * 64)
    for name in ("exposures", "check_hidden_paths"):
        monkeypatch.setattr(container, name, mock.Mock(return_value=()))
    recorded: dict[str, bool] = {}

    def _prepare(_found: container.ContainerDiscovery, **_kwargs: object) -> mock.Mock:
        recorded["clone"] = paths.git_exposure_record(clone).exists()
        recorded["workspace"] = paths.git_exposure_record(workspace).exists()
        return mock.Mock()

    monkeypatch.setattr(container, "prepare", _prepare)
    session_runtime.prepare_container_execution(
        config_model.Config(),
        workspace,
        fake.options(),
        state_dir=paths.state_dir(workspace),
        session_id="child",
        execution_id="e1",
        checkout=clone,
        controls=_controls({"sandbox": {"isolation": "container"}}, "container"),
        reporter=app_reporter.Reporter(out=lambda _line: None, err=lambda _line: None),
    )
    return recorded


def test_a_container_execution_records_the_access_before_it_is_prepared(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call in `prepare_container_execution` could go and every test stayed green."""
    assert _recorded_when_a_clones_container_is_prepared(tmp_path, monkeypatch)["clone"]


def test_a_container_run_of_a_clone_records_the_checkout_the_container_mounts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child's container run recorded its clone alone.

    The container mounts the repository the clone lies in, so the child's commands could write
    that repository's `.git` too, and `agent6 check sandbox` there said nothing.
    """
    assert _recorded_when_a_clones_container_is_prepared(tmp_path, monkeypatch)["workspace"]
