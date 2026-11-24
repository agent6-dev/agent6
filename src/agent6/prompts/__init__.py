# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Static prompt text: the strings agent6 sends to models.

A shared leaf: any layer may import it, and it imports nothing from agent6.
The text is pure, with `{...}` placeholders its callers fill.
Main modules: `loop`, `review`, `review_directive`, `compaction`, `machine`.
"""

from __future__ import annotations
