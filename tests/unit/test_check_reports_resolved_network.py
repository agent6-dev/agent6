# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` reports the network commands get on this host, not the knob.

`auto` is the default on both sandbox knobs, so the configured value answers nothing; the level line
cannot claim "the run's own network" under `sandbox.network = "host"`.
"""

from __future__ import annotations

import pytest

from agent6.config import Config
from agent6.tools import policy
from agent6.ui.cli import check_cmds  # pyright: ignore[reportPrivateUsage]


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


def test_the_level_line_does_not_promise_a_network_the_config_can_decline() -> None:
    """The level line does not promise a network the config can decline.

    `check sandbox` runs before any config is loaded.
    """
    full = check_cmds._isolation_means("full")
    assert "sandbox.network" in full, full
    for level in ("partial", "none"):
        assert "the run's own network" not in check_cmds._isolation_means(level)  # type: ignore[arg-type]
