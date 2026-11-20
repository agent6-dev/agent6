# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fresh input and output counts, independent of how a call was billed."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class TokenCounts:
    """Hold reported counts; a missing count stays unknown through a sum or a delta.

    Attributes:
        input_tokens: Fresh input tokens, or null when unreported.
        output_tokens: Output tokens, or null when unreported.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __add__(self, other: TokenCounts) -> TokenCounts:
        return TokenCounts(
            None
            if self.input_tokens is None or other.input_tokens is None
            else self.input_tokens + other.input_tokens,
            None
            if self.output_tokens is None or other.output_tokens is None
            else self.output_tokens + other.output_tokens,
        )

    def since(self, prior: TokenCounts) -> TokenCounts:
        """Return the counts beyond a continuation's settled baseline."""
        return TokenCounts(
            None
            if self.input_tokens is None or prior.input_tokens is None
            else max(0, self.input_tokens - prior.input_tokens),
            None
            if self.output_tokens is None or prior.output_tokens is None
            else max(0, self.output_tokens - prior.output_tokens),
        )

    def label(self) -> str:
        """Return input and output counts, with missing counters labelled unreported."""
        return (
            f"in={self.input_tokens if self.input_tokens is not None else 'unreported'} "
            f"out={self.output_tokens if self.output_tokens is not None else 'unreported'}"
        )


UNKNOWN = TokenCounts()
ZERO = TokenCounts(0, 0)


@dataclasses.dataclass(frozen=True, slots=True)
class TokenBreakdown:
    """Split a session's own calls from those charged through child settlement.

    Attributes:
        parent: Usage from the session's own provider calls.
        children: Usage charged through the settlement of attempts and reviewers.
    """

    parent: TokenCounts = UNKNOWN
    children: TokenCounts = UNKNOWN

    @property
    def total(self) -> TokenCounts:
        """Return counts across the parent and its settled children."""
        return self.parent + self.children

    def __add__(self, other: TokenBreakdown) -> TokenBreakdown:
        return TokenBreakdown(self.parent + other.parent, self.children + other.children)

    def lines(self) -> tuple[str, str, str]:
        """Return the same labelled counts for the end summary and the receipt."""
        return (
            f"Parent: {self.parent.label()}",
            f"Children: {self.children.label()} (settled)",
            f"TOTAL: {self.total.label()}",
        )


NO_PRIOR_USAGE = TokenBreakdown(ZERO, ZERO)
