# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context compaction: its driver, the elision of old tool results, its rules and its window."""

from __future__ import annotations

from agent6.harness.compaction import driver, strategy


def default_strategy() -> strategy.Strategy:
    """Create the built-in strategy for one harness execution.

    Returns:
        A strategy with a fresh restart floor.
    """
    return driver.Tiered()
