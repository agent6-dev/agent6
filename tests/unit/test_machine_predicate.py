# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the restricted predicate language (allow-list + evaluator)."""

from __future__ import annotations

import pytest

from agent6.machine.predicate import (
    PredicateError,
    Reference,
    evaluate,
    parse_predicate,
)


def test_parses_comparison_and_collects_reference() -> None:
    pred = parse_predicate("verdict.confidence >= 0.7")
    assert pred.references == (Reference("verdict", ("confidence",)),)


def test_collects_multiple_references_in_order() -> None:
    pred = parse_predicate("len(pending) == 0 and verdict.label == 'urgent'")
    assert [r.dotted for r in pred.references] == ["pending", "verdict.label"]


def test_len_is_the_only_allowed_call() -> None:
    assert parse_predicate("len(pending) > 0")
    with pytest.raises(PredicateError, match="calls are restricted"):
        parse_predicate("open('f')")


def test_rejects_attribute_call_getattr_style() -> None:
    with pytest.raises(PredicateError, match="calls are restricted"):
        parse_predicate("os.system('rm -rf /')")


def test_rejects_comprehension() -> None:
    with pytest.raises(PredicateError, match="unsupported syntax"):
        parse_predicate("[x for x in pending]")


def test_rejects_lambda() -> None:
    with pytest.raises(PredicateError):
        parse_predicate("(lambda: 1)()")


def test_rejects_arithmetic_binop() -> None:
    with pytest.raises(PredicateError, match="unsupported syntax"):
        parse_predicate("a + b == 2")


def test_rejects_keyword_argument_to_len() -> None:
    with pytest.raises(PredicateError, match="keyword"):
        parse_predicate("len(x=pending) == 0")


def test_rejects_syntax_error() -> None:
    with pytest.raises(PredicateError, match="not a valid expression"):
        parse_predicate("verdict.label ==")


def test_evaluate_equality_and_membership() -> None:
    pred = parse_predicate("label in ['urgent', 'spam']")
    assert evaluate(pred, {"label": "urgent"}) is True
    assert evaluate(pred, {"label": "normal"}) is False


def test_evaluate_record_navigation() -> None:
    pred = parse_predicate("verdict.label == 'urgent' and verdict.confidence >= 0.7")
    assert evaluate(pred, {"verdict": {"label": "urgent", "confidence": 0.9}}) is True
    assert evaluate(pred, {"verdict": {"label": "urgent", "confidence": 0.5}}) is False
    assert evaluate(pred, {"verdict": {"label": "normal", "confidence": 0.9}}) is False


def test_evaluate_len_and_not() -> None:
    pred = parse_predicate("not (len(pending) == 0)")
    assert evaluate(pred, {"pending": ["a"]}) is True
    assert evaluate(pred, {"pending": []}) is False


def test_evaluate_unknown_reference_raises() -> None:
    pred = parse_predicate("missing == 1")
    with pytest.raises(PredicateError, match="unknown reference"):
        evaluate(pred, {})


def test_evaluate_navigation_into_non_record_raises() -> None:
    pred = parse_predicate("scalar.field == 1")
    with pytest.raises(PredicateError, match="non-record"):
        evaluate(pred, {"scalar": 3})


def test_evaluate_chained_comparison() -> None:
    pred = parse_predicate("0 < n and n < 10")
    assert evaluate(pred, {"n": 5}) is True
    assert evaluate(pred, {"n": 50}) is False
