import unittest

from src.catalog import lookup
from src.pricing import cart_total, shelf_price


class TestCatalog(unittest.TestCase):
    def test_active_rows_present(self):
        self.assertEqual(lookup("A-101")["name"], "almond biscotti")

    def test_inactive_rows_absent(self):
        with self.assertRaises(KeyError):
            lookup("D-550")

    def test_shelf_prices(self):
        self.assertEqual(shelf_price("A-101"), 540)
        self.assertEqual(shelf_price("B-350"), 775)
        self.assertEqual(shelf_price("B-204"), 1299)

    def test_cart_total(self):
        self.assertEqual(cart_total(["A-101", "C-077"]), 969)


if __name__ == "__main__":
    unittest.main()
