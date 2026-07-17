# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Public face of the review surfaces.

Re-exports the freeform review call (`code_review`, driving `agent6 review`)
and the adversarial review panel (`ReviewContext`, `render_findings`,
`run_panel`, `ReviewSeat`, `parse_seat_spec`) from their private `_review`/`_panel`
siblings, so `ui/cli` imports both from one workflow-layer module instead of
reaching into privates.
"""

from __future__ import annotations

from agent6.workflows._panel import ReviewContext, render_findings
from agent6.workflows._review import ReviewSeat, parse_seat_spec, run_panel
from agent6.workflows.code_review import CodeReviewError, code_review

__all__ = [
    "CodeReviewError",
    "ReviewContext",
    "ReviewSeat",
    "code_review",
    "parse_seat_spec",
    "render_findings",
    "run_panel",
]
