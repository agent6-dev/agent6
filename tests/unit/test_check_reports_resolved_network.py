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
        ("auto", "strict", "session"),  # the default: the run's own network
        ("session", "strict", "session"),
        ("host", "strict", "host"),  # asked for the host's, and gets it
        ("auto", "hardened", "host"),  # no namespaces to give: degraded
        ("session", "hardened", "host"),  # preflight refuses this; the policy never lies
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
    assert policy.resolve_network(cfg, "strict", override="none") == "none"
    assert policy.resolve_network(cfg, "hardened", override="none") == "host"


def test_the_level_line_does_not_promise_a_network_the_config_can_decline() -> None:
    """The level line does not promise a network the config can decline.

    `check sandbox` runs before any config is loaded.
    """
    strict = check_cmds._isolation_means("strict")
    assert "sandbox.network" in strict, strict
    for level in ("hardened", "none"):
        assert "the run's own network" not in check_cmds._isolation_means(level)  # type: ignore[arg-type]
