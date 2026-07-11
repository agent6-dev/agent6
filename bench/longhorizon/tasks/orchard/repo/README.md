# orchard-mart pricing tools

Utilities behind the orchard-mart shelf-price feed: the product catalog and
the pricing helpers the registers call.

- `src/` - catalog loading (`catalog.py`) and price queries (`pricing.py`).
- `tools/` - build scripts and source tables.
- `data/` - the catalog feed the registers read.
- `docs/NOTES.md` - house conventions; read it before touching money math.
- `verify.sh` - runs the test suite.
