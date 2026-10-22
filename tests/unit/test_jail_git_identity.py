# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[git.commit]` identity rides into every jailed command's env.

A `git commit` the model runs under `git.control = "model"` sees no identity in the jail's
own HOME; the four GIT_*_NAME / GIT_*_EMAIL variables carry the operator's.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6 import config, kinds
from agent6.tools import policy

_IDENTITY = {
    "GIT_AUTHOR_NAME": "Eric Lesiuta",
    "GIT_COMMITTER_NAME": "Eric Lesiuta",
    "GIT_AUTHOR_EMAIL": "eric@example.com",
    "GIT_COMMITTER_EMAIL": "eric@example.com",
}


def _model_control(**commit: str) -> config.Config:
    return config.Config.model_validate(
        {"git": {"control": "model", "commit": commit}, "sandbox": {"protect_git": "off"}}
    )


@pytest.mark.parametrize("isolation", ["full", "partial", "none"])
def test_the_configured_identity_reaches_the_jail(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """Regression: a model-controlled run's commits were authored `agent6 <agent6@localhost>`.

    The jail's HOME is agent6's own, so git found no identity there and the model improvised
    one with `-c user.name`; the configured identity is the operator's decision for every
    commit, whoever runs git.
    """
    cfg = _model_control(name="Eric Lesiuta", email="eric@example.com")
    env = dict(policy.jail_policy(tmp_path, cfg, isolation, ("git", "commit")).env)
    assert {k: env.get(k) for k in _IDENTITY} == _IDENTITY


def test_a_server_and_a_verify_get_the_same_identity(tmp_path: pathlib.Path) -> None:
    """One owner: a caller's own env base is widened the same way as the passthrough set."""
    cfg = _model_control(name="Eric Lesiuta", email="eric@example.com")
    env = dict(policy.jail_policy(tmp_path, cfg, "full", ("srv",), env_base={"X": "1"}).env)
    assert env["X"] == "1"
    assert {k: env.get(k) for k in _IDENTITY} == _IDENTITY


def test_an_unset_identity_passes_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset leaves git its identity (a repo-local user.name serves); the shell's never leaks."""
    for key in _IDENTITY:
        monkeypatch.setenv(key, "leaked")
    env = dict(policy.jail_policy(tmp_path, _model_control(), "full", ("git", "commit")).env)
    assert not set(env) & set(_IDENTITY)


def test_each_half_rides_alone(tmp_path: pathlib.Path) -> None:
    """A name without an email sets the two name variables only, as agent6's own commits do."""
    env = dict(policy.jail_policy(tmp_path, _model_control(name="Eric"), "full", ("git",)).env)
    assert env["GIT_AUTHOR_NAME"] == env["GIT_COMMITTER_NAME"] == "Eric"
    assert "GIT_AUTHOR_EMAIL" not in env and "GIT_COMMITTER_EMAIL" not in env
