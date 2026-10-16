# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A jailed git reads the `[git.commit]` identity, live."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from agent6 import config
from agent6.sandbox import jail
from agent6.tools import policy as tools_policy

pytestmark = pytest.mark.needs_namespaces


def test_a_jailed_git_var_reads_the_configured_identity(tmp_path: pathlib.Path) -> None:
    """Regression: under `git.control = "model"` the jailed git saw no identity in its HOME."""
    if shutil.which("git") is None:
        pytest.skip("git is not on the jail PATH")
    cfg = config.Config.model_validate(
        {
            "git": {"control": "model", "commit": {"name": "Eric Lesiuta", "email": "e@x.io"}},
            "sandbox": {"protect_git": False},
        }
    )
    policy = tools_policy.jail_policy(
        tmp_path, cfg, "full", ("git", "var", "GIT_AUTHOR_IDENT"), network="none"
    )
    res = jail.run_in_jail(policy)
    assert res.returncode == 0, (res.returncode, res.stdout, res.stderr)
    assert res.stdout.startswith("Eric Lesiuta <e@x.io> "), res.stdout
