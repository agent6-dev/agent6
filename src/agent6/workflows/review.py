# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read-only `agent6 review` workflow, and the public face of the review panel.

`run_review` is a thin wrapper around `agents.code_review.code_review` so the
CLI can stay on the right side of the workflows-vs-agents module boundary
(tach forbids `cli -> agents` directly). `ReviewContext`, `render_findings`,
and `run_panel` are re-exported from the private `_panel`/`_review` siblings
so `ui/cli` (and any other cross-boundary consumer) imports the adversarial
review panel from here instead of reaching into those private modules.
"""

from __future__ import annotations

from agent6.agents.code_review import CodeReviewError, code_review
from agent6.providers import Provider
from agent6.workflows._panel import ReviewContext, render_findings
from agent6.workflows._review import Seat, parse_seat_spec, run_panel


def run_review(
    reviewer: Provider,
    *,
    diff: str,
    agents_md: str = "",
    recent_log: str = "",
    extra_context: str = "",
) -> str:
    """Return the reviewer's markdown verdict for *diff*."""
    return code_review(
        reviewer,
        diff=diff,
        agents_md=agents_md,
        recent_log=recent_log,
        extra_context=extra_context,
    )


__all__ = [
    "CodeReviewError",
    "ReviewContext",
    "Seat",
    "parse_seat_spec",
    "render_findings",
    "run_panel",
    "run_review",
]
