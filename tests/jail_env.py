# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared gate for tests that build real user-namespace jails.

Healthy kernels pass through. Environments that truly cannot run the jail
(non-Linux, a container, no Landlock) skip, as they always did. A host
policy blocking an otherwise capable kernel (the userns sysctl, the Ubuntu
AppArmor restriction) FAILS with the fix, so a dev machine cannot read
green while the jail suite never ran. AGENT6_TEST_SKIP_JAIL=1 turns that
failure into an explicit skip for machines the operator cannot change.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import detect
from agent6.tools import policy


def require_userns_jail() -> None:
    env = detect.environment()
    if not env.sandbox_available:
        pytest.skip(f"no kernel sandbox on {sys.platform!r}")
    if env.userns_supported:
        return
    cause = detect.degrade_reason(env) or "user namespaces unavailable"
    if env.in_container or env.landlock_abi < 1:
        pytest.skip(f"jail unavailable here: {cause}")
    if os.environ.get("AGENT6_TEST_SKIP_JAIL") == "1":
        pytest.skip(f"jail tests skipped explicitly (AGENT6_TEST_SKIP_JAIL=1): {cause}")
    pytest.fail(
        f"this machine can run the jail but a host policy blocks it: {cause}."
        " Fix the policy, or set AGENT6_TEST_SKIP_JAIL=1 to skip explicitly.",
        pytrace=False,
    )


def clone_jail(clone: pathlib.Path) -> kinds.JailPolicy:
    """Return the policy a command in a clone gets under the default config on this host."""
    return policy.workspace_jail(clone, Config(), ran_under="")
