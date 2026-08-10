# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The wordlist curation rules are executable: sorted, lowercase ASCII, and
no word is a prefix of another (a prefix pair completes ambiguously in an
id). The docstring stated a rule the lists did not keep; now the lists state
the rules and this holds them to it."""

from __future__ import annotations

from agent6._data.words import ADJECTIVES, NOUNS


def test_curation_rules_hold() -> None:
    for ws in (ADJECTIVES, NOUNS):
        assert list(ws) == sorted(ws)
        assert all(w.isascii() and w.isalpha() and w.islower() for w in ws)
        prefix_pairs = [(a, b) for a in ws for b in ws if a != b and b.startswith(a)]
        assert not prefix_pairs
