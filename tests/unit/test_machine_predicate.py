# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the restricted predicate language (allow-list + evaluator)."""

from __future__ import annotations

import pytest

from agent6.machine import predicate


def test_parses_comparison_and_collects_reference() -> None:
    pred = predicate.parse_predicate("verdict.confidence >= 0.7")
    assert pred.references == (predicate.Reference("verdict", ("confidence",)),)


def test_collects_multiple_references_in_order() -> None:
    pred = predicate.parse_predicate("len(pending) == 0 and verdict.label == 'urgent'")
    assert [r.dotted for r in pred.references] == ["pending", "verdict.label"]


def test_len_is_the_only_allowed_call() -> None:
    assert predicate.parse_predicate("len(pending) > 0")
    with pytest.raises(predicate.PredicateError, match="calls are restricted"):
        predicate.parse_predicate("open('f')")


def test_rejects_attribute_call_getattr_style() -> None:
    with pytest.raises(predicate.PredicateError, match="calls are restricted"):
        predicate.parse_predicate("os.system('rm -rf /')")


def test_rejects_comprehension() -> None:
    with pytest.raises(predicate.PredicateError, match="unsupported syntax"):
        predicate.parse_predicate("[x for x in pending]")


def test_rejects_lambda() -> None:
    with pytest.raises(predicate.PredicateError):
        predicate.parse_predicate("(lambda: 1)()")


def test_rejects_arithmetic_binop() -> None:
    """The BinOp rejection names the constraint and the fix, not the AST class."""
    with pytest.raises(predicate.PredicateError, match="arithmetic is not allowed"):
        predicate.parse_predicate("a + b == 2")


def test_rejects_keyword_argument_to_len() -> None:
    with pytest.raises(predicate.PredicateError, match="keyword"):
        predicate.parse_predicate("len(x=pending) == 0")


def test_rejects_syntax_error() -> None:
    with pytest.raises(predicate.PredicateError, match="not a valid expression"):
        predicate.parse_predicate("verdict.label ==")


def test_evaluate_equality_and_membership() -> None:
    pred = predicate.parse_predicate("label in ['urgent', 'spam']")
    assert predicate.evaluate(pred, {"label": "urgent"}) is True
    assert predicate.evaluate(pred, {"label": "normal"}) is False


def test_evaluate_record_navigation() -> None:
    pred = predicate.parse_predicate("verdict.label == 'urgent' and verdict.confidence >= 0.7")
    assert predicate.evaluate(pred, {"verdict": {"label": "urgent", "confidence": 0.9}}) is True
    assert predicate.evaluate(pred, {"verdict": {"label": "urgent", "confidence": 0.5}}) is False
    assert predicate.evaluate(pred, {"verdict": {"label": "normal", "confidence": 0.9}}) is False


def test_evaluate_len_and_not() -> None:
    pred = predicate.parse_predicate("not (len(pending) == 0)")
    assert predicate.evaluate(pred, {"pending": ["a"]}) is True
    assert predicate.evaluate(pred, {"pending": []}) is False


def test_evaluate_unknown_reference_raises() -> None:
    pred = predicate.parse_predicate("missing == 1")
    with pytest.raises(predicate.PredicateError, match="unknown reference"):
        predicate.evaluate(pred, {})


def test_evaluate_navigation_into_non_record_raises() -> None:
    pred = predicate.parse_predicate("scalar.field == 1")
    with pytest.raises(predicate.PredicateError, match="non-record"):
        predicate.evaluate(pred, {"scalar": 3})


def test_evaluate_chained_comparison() -> None:
    pred = predicate.parse_predicate("0 < n and n < 10")
    assert predicate.evaluate(pred, {"n": 5}) is True
    assert predicate.evaluate(pred, {"n": 50}) is False


def test_order_preserves_large_int_precision() -> None:
    """Ordering compares ints exactly, never through float(), which collapses values above 2^53."""
    a = 1_750_000_000_000_000_100
    b = 1_750_000_000_000_000_000
    pred = predicate.parse_predicate("a > b")
    assert predicate.evaluate(pred, {"a": a, "b": b}) is True
    assert predicate.evaluate(pred, {"a": b, "b": a}) is False
    # The fail-loud path for non-comparable operands is intact.
    with pytest.raises(predicate.PredicateError, match="cannot order"):
        predicate.evaluate(predicate.parse_predicate("x < y"), {"x": 1, "y": "s"})


def test_in_with_an_unhashable_left_operand_is_a_predicate_error() -> None:
    """`item in record` on an unhashable operand is a PredicateError, never a raw TypeError."""
    pred = predicate.parse_predicate("item in blob")
    with pytest.raises(predicate.PredicateError, match="`in`"):
        predicate.evaluate(pred, {"item": [1, 2], "blob": {"a": 1}})


def test_has_guards_an_absent_reference() -> None:
    """`has(ref)` is the presence guard an optional record field needs, and `and` short-circuits."""
    board = {"out": {"summary": "hi"}}
    assert predicate.evaluate(predicate.parse_predicate("has(out.summary)"), board) is True
    assert predicate.evaluate(predicate.parse_predicate("has(out.score)"), board) is False
    assert predicate.evaluate(predicate.parse_predicate("has(out)"), board) is True
    assert predicate.evaluate(predicate.parse_predicate("has(missing)"), board) is False
    # The guard composes: the read after `and` never evaluates on absence.
    assert (
        predicate.evaluate(predicate.parse_predicate("has(out.score) and out.score > 0"), board)
        is False
    )
    assert (
        predicate.evaluate(
            predicate.parse_predicate("has(out.score) and out.score > 0"), {"out": {"score": 3}}
        )
        is True
    )
    # Navigating INTO a non-record is a type mismatch, not absence.
    with pytest.raises(predicate.PredicateError, match="non-record"):
        predicate.evaluate(predicate.parse_predicate("has(out.summary.deeper)"), board)


def test_has_takes_only_a_reference() -> None:
    """`has(1)` or `has(len(x))` is refused at parse time."""
    with pytest.raises(predicate.PredicateError, match="reference"):
        predicate.parse_predicate("has(1)")
    with pytest.raises(predicate.PredicateError, match="reference"):
        predicate.parse_predicate("has(len(x))")
