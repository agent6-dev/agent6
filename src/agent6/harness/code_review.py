# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review a diff in one provider call, markdown out.

`agent6 review` uses it: a freeform, read-only review with no structured verdict.
"""

from __future__ import annotations

from agent6 import review_records
from agent6.harness import _panel
from agent6.prompts import review
from agent6.providers import Provider, ProviderError, ProviderResponse


class CodeReviewError(Exception):
    """The code-review sub-agent failed to produce a response."""


_SYSTEM = (
    review.RULES_AUTHORITY
    + "\n"
    + """You are a senior code reviewer.

You are given a diff and (optionally) base-tree rules plus recent commit log for
context. Produce a markdown review.

Cover, in this priority order, and only if relevant:
1. Correctness bugs (race conditions, off-by-one, wrong API, missed error
   path, broken invariants). Cite file:line.
2. Security issues (injection, path traversal, weakened sandbox, leaked
   secrets, unsafe deserialization, missing authn/authz).
3. AGENTS.md / convention drift. If the diff violates a documented project
   convention, name the convention and the line.
4. Over-engineering: new abstractions used in one place; speculative error
   handling; unrelated refactors; reformat-only churn.
5. Test gaps: behavior changed but no test added/changed.
6. Minor: naming, dead code, clearer alternatives.

Format:
- Start with a one-line VERDICT: `LGTM`, `LGTM with nits`, `Needs changes`,
  or `Block`.
- Then bullet points, each prefixed with severity in brackets: [bug],
  [security], [convention], [over-engineering], [test], [nit].
- Reference files as `path:line` where possible.
- If there is nothing to say in a category, omit it. Do not pad.
"""
)


def review_diff(
    provider: Provider,
    *,
    diff: str,
    rules: review_records.ReviewRules | None = None,
    recent_log: str = "",
    extra_context: str = "",
    max_tokens: int = 2048,
) -> str:
    """Ask the reviewer model to critique the diff.

    Args:
        provider: The reviewer's provider.
        diff: The diff under review.
        rules: The complete base-tree rules and their sources.
        recent_log: The recent commit log.
        extra_context: Extra text for the reviewer.
        max_tokens: The output cap of the call.

    Returns:
        The review as markdown.

    Raises:
        CodeReviewError: When the call fails or returns nothing.
    """
    rules = rules or review_records.ReviewRules()
    if not rules.complete:
        raise CodeReviewError(rules.reason or "review rules incomplete")
    parts: list[str] = [_panel.rules_message(rules)]
    if recent_log.strip():
        parts.append(f"RECENT COMMITS:\n{recent_log.strip()[:2000]}")
    if extra_context.strip():
        parts.append(extra_context.strip()[:4000])
    parts.append(f"DIFF:\n{diff}")
    user = "\n\n".join(parts)
    try:
        resp: ProviderResponse = provider.call(
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        raise CodeReviewError(f"provider call failed: {exc}") from exc
    text = resp.text.strip()
    if not text:
        raise CodeReviewError("reviewer returned empty response")
    return text
