# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context compaction: its driver, the elision of old tool results, its rules and its window.

Part of `harness`: it imports `tools` and the layers below, never `app` or `ui`.
`default_strategy` picks the built-in strategy for the driving route.
Main modules: `driver`, `strategy`, `rules`, `elision`, `window`, `retention`, `cold_start`.
"""

from __future__ import annotations

from agent6.harness.compaction import driver, strategy
from agent6.providers import provider_types


def default_strategy(endpoint: provider_types.EndpointCompaction) -> strategy.Strategy:
    """Create the built-in strategy for one harness execution.

    Args:
        endpoint: The history operations of the driving route.

    Returns:
        A strategy with a fresh restart floor.
    """
    return driver.Endpoint() if endpoint.writes_summary else driver.Tiered()
