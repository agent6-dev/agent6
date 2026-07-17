# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Standalone freeform code review: one provider call, markdown out.

Used by `agent6 review`. Freeform (no Step / acceptance criterion), emitting
markdown text rather than a structured verdict. Read-only: given a diff plus
optional context, it returns a human-readable review.
"""

from __future__ import annotations

from agent6.providers import Provider, ProviderError, ProviderResponse


class CodeReviewError(Exception):
    """The code-review sub-agent failed to produce a response."""


_SYSTEM = """You are a senior code reviewer.

You are given a diff and (optionally) AGENTS.md plus recent commit log for
context. Produce a concise, actionable markdown review. Be terse. Skip praise.

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


def code_review(
    provider: Provider,
    *,
    diff: str,
    agents_md: str = "",
    recent_log: str = "",
    extra_context: str = "",
    max_tokens: int = 2048,
) -> str:
    """Ask the reviewer model to critique *diff*. Returns markdown text."""
    parts: list[str] = []
    if agents_md.strip():
        # AGENTS.md holds the conventions the reviewer is told to check, so the
        # old 4000-char cap silently dropped most of them. Use a generous bound
        # (the diff itself is allowed 60k) that fits any realistic AGENTS.md
        # while still guarding against a pathologically huge one.
        parts.append(f"AGENTS.md:\n{agents_md.strip()[:16000]}")
    if recent_log.strip():
        parts.append(f"RECENT COMMITS:\n{recent_log.strip()[:2000]}")
    if extra_context.strip():
        parts.append(extra_context.strip()[:4000])
    parts.append(f"DIFF:\n{diff[:60_000]}")
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
