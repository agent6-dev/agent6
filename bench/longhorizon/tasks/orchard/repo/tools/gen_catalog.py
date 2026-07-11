"""Build data/catalog.tsv from tools/catalog_source.tsv.

Run from the repo root: python3 tools/gen_catalog.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools" / "catalog_source.tsv"
OUT = ROOT / "data" / "catalog.tsv"

FIELDS = ["sku", "name", "category", "shelf_cents"]


def build_rows():
    rows = []
    with open(SOURCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] != "1":
                continue
            base = int(row["base_cents"])
            margin = int(row["margin_pct"])
            # Money is integer cents, rounding half-up (docs/NOTES.md).
            shelf = base + (base * margin + 50) // 100
            rows.append(
                {
                    "sku": row["sku"],
                    "name": row["name"],
                    "category": row["category"],
                    "shelf_cents": str(shelf),
                }
            )
    rows.sort(key=lambda r: r["sku"])
    return rows


def main():
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
