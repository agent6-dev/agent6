import unittest
from pathlib import Path

from src.pricing import cart_total, weekend_price


class TestWeekend(unittest.TestCase):
    def test_weekend_prices(self):
        self.assertEqual(weekend_price("A-101"), 621)
        self.assertEqual(weekend_price("B-204"), 1494)
        self.assertEqual(weekend_price("D-023"), 1287)

    def test_rounding_half_up(self):
        self.assertEqual(weekend_price("F-310"), 909)

    def test_weekend_cart(self):
        self.assertEqual(cart_total(["A-101", "F-310"], weekend=True), 1530)

    def test_shelf_cart_unchanged(self):
        self.assertEqual(cart_total(["A-101", "C-077"]), 969)

    def test_catalog_has_weekend_column(self):
        header = Path("data/catalog.tsv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("weekend_cents", header.split("\t"))


if __name__ == "__main__":
    unittest.main()
