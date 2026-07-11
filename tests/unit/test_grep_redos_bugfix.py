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
    "^(a{1,2})+$",  # bounded-but-variable inner repeat under an unbounded outer
    r"(\w{2,5})+",
    "((ab){1,3})+",
    "(?:x{0,2})*",
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
    "(a{3})+",  # fixed inner length: deterministic tiling, not catastrophic
    "(foo{2,4}bar)+",  # variable repeat has fixed boundaries around it
    "(ab){1,3}",  # bounded outer, not repeated unboundedly
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


def test_bounded_variable_repeat_detector_precision() -> None:
    from agent6.tools._grep_safety import (
        _has_bounded_variable_repeat_under_quantifier as det,  # pyright: ignore[reportPrivateUsage]
    )

    assert det("(a{1,2})+")
    assert det(r"(\w{2,5})*")
    assert det("(?:x{0,3}){1,}")  # unbounded outer via {1,}
    assert not det("(a{3})+")  # fixed inner: no variability
    assert not det("(foo{2,4}bar)+")  # not a single quantified atom
    assert not det("(a{1,2}){1,3}")  # bounded outer
