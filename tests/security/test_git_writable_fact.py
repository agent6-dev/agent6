# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 records and shows that a model's commands could write a checkout's `.git`."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import evidence, kinds, paths
from agent6.app.session import runtime as session_runtime
from agent6.config import Config
from agent6.tools import target_policy
from agent6.ui.cli import check_sandbox


def _controls(raw: dict[str, object], isolation: str) -> tuple[kinds.EffectiveControl, ...]:
    cfg = Config.model_validate(raw)
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
