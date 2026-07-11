"""Price queries the registers call. All amounts are integer cents."""

from src.catalog import lookup


def shelf_price(sku):
    return lookup(sku)["shelf_cents"]


def cart_total(skus):
    return sum(shelf_price(sku) for sku in skus)
