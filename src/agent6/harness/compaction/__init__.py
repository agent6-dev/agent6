# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Context compaction: its driver, the elision of old tool results, its rules and its window.

Part of `harness`: it imports `tools` and the layers below, never `app` or `ui`.
`driver.default_strategy` picks the built-in strategy for the driving route.
Main modules: `driver`, `strategy`, `rules`, `elision`, `window`, `retention`, `cold_start`.
"""

from __future__ import annotations
