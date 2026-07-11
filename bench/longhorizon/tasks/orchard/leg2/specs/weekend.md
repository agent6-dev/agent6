# WEEKEND pricing tier

Weekend prices apply Saturday and Sunday at every register. This spec adds
the tier end to end:

1. `data/catalog.tsv` gains a `weekend_cents` column (after `shelf_cents`)
   carrying the weekend price of every active product, so the registers can
   read it straight from the feed.
2. The weekend price is the shelf price plus a 15% weekend surcharge,
   computed per the house money conventions (docs/NOTES.md).
3. `src/pricing.py` gains `weekend_price(sku) -> int` returning cents
   (KeyError for unknown skus, same as shelf_price), and `cart_total` grows
   a `weekend=False` keyword: weekend carts sum weekend prices instead of
   shelf prices.
4. `test_weekend.py` is the acceptance suite for the tier; `./verify.sh`
   runs it with the rest. Done when the whole suite is green.
