"""Cart maintains line items."""
from __future__ import annotations

class Cart:
    def __init__(self) -> None:
        self.items: list[tuple[str, float]] = []
    def add(self, name: str, price: float) -> None:
        self.items.append((name, price))
    def subtotal(self) -> float:
        return sum(p for _, p in self.items)
