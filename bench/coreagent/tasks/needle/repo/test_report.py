"""Starter tests for report (basics of the combined behavior; runs via ./verify.sh).

These cover a few end-to-end inputs only. The full rule-by-rule battery lives in
the hidden grader. Implement every rule from ref1.md through ref5.md.
"""

from __future__ import annotations

import unittest

import report as r


class TestRender(unittest.TestCase):
    def test_sort_format_header(self) -> None:
        rows = [{"name": "alpha", "value": 1.5}, {"name": "beta", "value": 3.0}]
        expected = "REPORT (2 items)\nbeta: 3.00\nalpha: 1.50"
        self.assertEqual(r.render(rows), expected)

    def test_skip_negative_keep_zero(self) -> None:
        rows = [
            {"name": "a", "value": -2.0},
            {"name": "b", "value": 0.0},
            {"name": "c", "value": 4.0},
        ]
        expected = "REPORT (2 items)\nc: 4.00\nb: 0.00"
        self.assertEqual(r.render(rows), expected)

    def test_empty(self) -> None:
        self.assertEqual(r.render([]), "REPORT (0 items)")


if __name__ == "__main__":
    unittest.main()
