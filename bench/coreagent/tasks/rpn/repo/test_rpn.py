"""Starter tests for rpn (basics only; runs via ./verify.sh)."""

from __future__ import annotations

import unittest

import rpn as r


class TestTokenize(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(r.tokenize("3   4   +"), ["3", "4", "+"])

    def test_empty(self) -> None:
        self.assertEqual(r.tokenize("   "), [])


class TestEvaluate(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(r.evaluate(["3", "4", "+"]), 7.0)

    def test_div_by_zero(self) -> None:
        with self.assertRaises(ValueError):
            r.evaluate(["1", "0", "/"])


class TestEvaluateExpr(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(r.evaluate_expr("10 2 /"), 5.0)

    def test_leftover(self) -> None:
        with self.assertRaises(ValueError):
            r.evaluate_expr("1 2 3 +")


class TestRPNCalculator(unittest.TestCase):
    def test_push_records_history(self) -> None:
        c = r.RPNCalculator()
        self.assertEqual(c.push("3 4 +"), 7.0)
        self.assertEqual(c.history, [("3 4 +", 7.0)])

    def test_last_empty_is_none(self) -> None:
        self.assertIsNone(r.RPNCalculator().last())


if __name__ == "__main__":
    unittest.main()
