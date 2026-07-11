# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-surface presentation constants shared by the CLI, TUI, and web.

The single source of truth for how run/task state reads to a human, so the same
state never renders differently across surfaces (the per-front-end glyph maps had
already drifted). The web SPA can't import Python, so it mirrors these exact
characters in page.py; keep them in sync.
"""

from __future__ import annotations

# Task-node status glyphs. Text characters (not graphics) so every terminal font
# renders them. ruff's ambiguous-glyph rule (RUF001) flags the en-dash /
# multiplication-sign, which is the intended distinct look here.
TASK_STATUS_GLYPH = {
    "passed": "✓",
    "failed": "✗",
    "in_progress": "▸",
    "pending": "·",
    "skipped": "–",  # noqa: RUF001
    "obsolete": "×",  # noqa: RUF001
}
