"""Discounts operate on a Cart subtotal."""
from __future__ import annotations
from shop.cart import Cart

def apply_discount(cart: Cart, percent: float) -> float:
    if percent < 0 or percent > 100:
        raise ValueError("percent out of range")
    return cart.subtotal() * (1.0 - percent / 100.0)
