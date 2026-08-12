The test file `test_shop.py` imports `ShoppingCart` from `shop`, but the code
defines the class as `Cart`. Rename the class `Cart` to `ShoppingCart`
everywhere it appears in the `shop/` package so that the existing tests pass
unchanged.

Specifics:

1. Rename `class Cart` to `class ShoppingCart` in `shop/cart.py`.
2. Update every import / type-annotation reference to `Cart` in
   `shop/__init__.py`, `shop/discount.py`, and `shop/checkout.py` to use
   `ShoppingCart`.
3. Do not modify `test_shop.py`.

The verify command `python3 -m unittest -v` must report all 4 tests passing.
