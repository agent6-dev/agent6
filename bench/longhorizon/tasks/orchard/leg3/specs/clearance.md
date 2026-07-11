# CLEARANCE feed

Merchandising flags slow movers for clearance; the registers need the
marked-down prices in the feed. `tools/clearance_source.tsv` lists the
clearance products (`sku`, `clearance_pct`).

1. `data/clearance.tsv` is a new feed file: columns `sku` and
   `clearance_cents`, one row per clearance product still in the catalog
   feed (a listed sku missing from the catalog is discontinued: skip it).
2. The discount is `clearance_pct` percent of the shelf price, in cents;
   the clearance price is the shelf price minus the discount.
3. `src/pricing.py` gains `clearance_price(sku) -> int`: the clearance
   price for products on clearance, the shelf price for everything else
   (KeyError for unknown skus, same as `shelf_price`). `cart_total` grows
   a `clearance=False` keyword: clearance carts sum clearance-aware
   prices.
4. `test_clearance.py` is the acceptance suite for the feed;
   `./verify.sh` runs it with the rest. Done when the whole suite is
   green.
