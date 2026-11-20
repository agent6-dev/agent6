# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context compaction: its driver, the elision of old tool results, its rules and its window."""

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
