# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` reports the network commands get on this host, not the knob.

`sandbox.isolation` and `sandbox.network` both default to `auto`, which names no network.
"""

from __future__ import annotations

import pytest

from agent6.config import Config
from agent6.tools import policy


@pytest.mark.parametrize(
    ("configured", "isolation", "expected"),
    [
        ("auto", "full", "session"),  # the default: the run's own network
        ("session", "full", "session"),
        ("host", "full", "host"),  # asked for the host's, and gets it
        ("auto", "partial", "host"),  # no namespaces to give: degraded
        ("session", "partial", "host"),  # preflight refuses this; the policy never lies
        ("auto", "none", "host"),
    ],
)
def test_the_resolved_network_follows_the_level_and_the_knob(
    configured: str, isolation: str, expected: str
) -> None:
    cfg = Config.model_validate({"sandbox": {"network": configured}})
    assert policy.resolve_network(cfg, isolation) == expected  # type: ignore[arg-type]


def test_a_callers_own_answer_wins_but_still_cannot_outrun_the_level() -> None:
    """A caller's own answer wins but cannot outrun the level.

    An MCP server's reachability is the operator's per-server choice; on a level with no namespaces
    it is still the host's network.
    """
    cfg = Config()
    assert policy.resolve_network(cfg, "full", override="none") == "none"
    assert policy.resolve_network(cfg, "partial", override="none") == "host"
