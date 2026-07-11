# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for the broadened grep ReDoS static screen.

The original screen only caught NESTED unbounded quantifiers ((a+)+, (.*)*).
These single-quantifier catastrophic shapes slipped through and could hang a
single in-process re.search under the GIL:

  * overlapping alternation under an unbounded repeat: (a|a)*, (a|ab)*
  * adjacent unbounded quantifiers over the same atom: a*a*a*...

Must NOT regress the existing safe patterns (e.g. (ab|cd)+, a.*b).
"""

from __future__ import annotations

import pytest

from agent6.tools._grep_safety import (
    _has_adjacent_unbounded_quantifiers,  # pyright: ignore[reportPrivateUsage]
    _has_overlapping_alternation_under_quantifier,  # pyright: ignore[reportPrivateUsage]
    reject_pathological_regex,
)
from agent6.tools.errors import ToolError

CATASTROPHIC = [
    "(a|a)*$",
    "(a|ab)*$",
    "(a|a|b)+",
    "(?:a|a)*",
    "a*a*a*a*b",
    "a+a+a+c",
    ".*.*x",
]

SAFE = [
    "hello",
    "foo+bar",
    "(foo)bar",
    "(foo+)bar",
    "[a-z]+",
    "a.*b",
    "(ab|cd)+",
    "x{2,4}",
    "a*b*c*",  # distinct atoms, not adjacent-same
    "(abc|def|ghi)*",  # disjoint branches
    "foo.*bar.*baz",  # distinct atoms between .*
]


@pytest.mark.parametrize("pat", CATASTROPHIC)
def test_reject_catastrophic_single_quantifier(pat: str) -> None:
    with pytest.raises(ToolError, match="backtracking"):
        reject_pathological_regex(pat)


@pytest.mark.parametrize("pat", SAFE)
def test_safe_patterns_pass(pat: str) -> None:
    # Must not raise.
    reject_pathological_regex(pat)


def test_overlapping_alternation_detector_precision() -> None:
    assert _has_overlapping_alternation_under_quantifier("(a|a)*")
    assert _has_overlapping_alternation_under_quantifier("(a|ab)+")
    assert not _has_overlapping_alternation_under_quantifier("(ab|cd)+")
    assert not _has_overlapping_alternation_under_quantifier("(ab|cd)")  # no quant
    assert not _has_overlapping_alternation_under_quantifier("(a|b){2,4}")  # bounded


def test_adjacent_quantifier_detector_precision() -> None:
    assert _has_adjacent_unbounded_quantifiers("a*a*")
    assert _has_adjacent_unbounded_quantifiers(".*.*")
    assert not _has_adjacent_unbounded_quantifiers("a*b*")
    assert not _has_adjacent_unbounded_quantifiers("a*")
    assert not _has_adjacent_unbounded_quantifiers("a{2,4}b{1,3}")
