import unittest
from shop import ShoppingCart, checkout, apply_discount

class T(unittest.TestCase):
    def test_subtotal(self):
        c = ShoppingCart(); c.add("a", 2.0); c.add("b", 3.0)
        self.assertEqual(c.subtotal(), 5.0)
    def test_apply_discount(self):
        c = ShoppingCart(); c.add("a", 10.0)
        self.assertAlmostEqual(apply_discount(c, 25.0), 7.5)
    def test_checkout(self):
        c = ShoppingCart(); c.add("a", 4.0); c.add("b", 6.0)
        self.assertEqual(checkout(c, 50.0), "items=2 total=5.00")
    def test_discount_out_of_range(self):
        with self.assertRaises(ValueError):
            apply_discount(ShoppingCart(), 150.0)

if __name__ == "__main__":
    unittest.main()
