# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read-only `agent6 review` workflow.

Thin wrapper around `agents.code_review.code_review` so the CLI can stay on
the right side of the workflows-vs-agents module boundary (tach forbids
`cli -> agents` directly).
"""

from __future__ import annotations

from agent6.agents.code_review import CodeReviewError, code_review
from agent6.providers import Provider


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


__all__ = ["CodeReviewError", "run_review"]
