# House conventions

- Money is integer cents everywhere: prices, totals, surcharges. No floats
  in money math. When a computation demands rounding, round half-up on the
  cent (12.5 cents rounds to 13). Python's `round()` is half-to-even; do not
  use it for money.
- Data files under `data/` are built artifacts. Never hand-edit them: change
  the source tables under `tools/` and rebuild (`verify.sh` rebuilds before
  testing, so a hand-edit does not survive a verify).
- SKUs are one uppercase letter, a hyphen, three digits. Every table stays
  sorted by sku; the build scripts enforce it, keep it that way in sources
  too.
- Product names are lowercase; categories come from the fixed set bakery,
  pantry, home, wellness.
- Tests are stdlib unittest, run via `./verify.sh`. Keep it green.
