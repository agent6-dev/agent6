"""Load the catalog feed the registers read."""

import csv
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.tsv"


def load_catalog():
    rows = {}
    with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["shelf_cents"] = int(row["shelf_cents"])
            rows[row["sku"]] = row
    return rows


def lookup(sku):
    rows = load_catalog()
    if sku not in rows:
        raise KeyError(sku)
    return rows[sku]
