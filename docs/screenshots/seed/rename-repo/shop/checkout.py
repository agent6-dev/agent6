"""Checkout finalises a Cart and returns a receipt string."""
from __future__ import annotations
from shop.cart import Cart
from shop.discount import apply_discount

def checkout(cart: Cart, discount_percent: float = 0.0) -> str:
    total = apply_discount(cart, discount_percent)
    return f"items={len(cart.items)} total={total:.2f}"
