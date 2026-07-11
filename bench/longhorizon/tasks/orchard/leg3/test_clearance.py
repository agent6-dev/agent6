import unittest
from pathlib import Path

from src.pricing import cart_total, clearance_price


class TestClearance(unittest.TestCase):
    def test_clearance_prices(self):
        self.assertEqual(clearance_price("B-350"), 620)
        self.assertEqual(clearance_price("C-250"), 282)
        self.assertEqual(clearance_price("F-310"), 474)

    def test_off_clearance_falls_back_to_shelf(self):
        self.assertEqual(clearance_price("A-101"), 540)

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            clearance_price("Z-999")

    def test_clearance_cart(self):
        self.assertEqual(cart_total(["B-350", "A-101"], clearance=True), 1160)

    def test_shelf_cart_unchanged(self):
        self.assertEqual(cart_total(["A-101", "C-077"]), 969)

    def test_feed_file(self):
        lines = Path("data/clearance.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split("\t"), ["sku", "clearance_cents"])
        skus = [line.split("\t")[0] for line in lines[1:]]
        self.assertNotIn("D-550", skus)
        self.assertIn("F-310", skus)


if __name__ == "__main__":
    unittest.main()
