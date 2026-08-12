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
import sys

import pytest

from agent6.sandbox.detect import degrade_reason, detect


def require_userns_jail() -> None:
    env = detect()
    if not env.sandbox_available:
        pytest.skip(f"no kernel sandbox on {sys.platform!r}")
    if env.userns_supported:
        return
    cause = degrade_reason(env) or "user namespaces unavailable"
    if env.in_container or env.landlock_abi < 1:
        pytest.skip(f"jail unavailable here: {cause}")
    if os.environ.get("AGENT6_TEST_SKIP_JAIL") == "1":
        pytest.skip(f"jail tests skipped explicitly (AGENT6_TEST_SKIP_JAIL=1): {cause}")
    pytest.fail(
        f"this machine can run the jail but a host policy blocks it: {cause}."
        " Fix the policy, or set AGENT6_TEST_SKIP_JAIL=1 to skip explicitly.",
        pytrace=False,
    )
