#!/usr/bin/env python3
"""Authoritative hidden grader for orchard. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir> <leg>    # leg: fix | weekend | clearance

Grades semantics, not diffs: the `regen` component copies the worktree,
DELETES data/*.tsv, runs every tools/gen_*.py THEY left behind, and checks
the feeds the build reproduces against the reference tables (a hand-written
feed no generator can rebuild scores zero); the `api` component probes src/
in a subprocess. Leg `weekend` adds the `rounding` component (half-up vs
banker's divergence, F-310: shelf 790 -> weekend 909, float round() gives
908). Leg `clearance` re-probes both conventions with fresh values: the
discount rounds half-up on the cent before subtracting, so A-140 448 /
E-905 1858 catch round() (449/1859) and B-204 649 catches int() truncation
(650). Leg `clearance` does not grade the weekend column: a leg-2 failure
already paid in leg-2's record, and re-billing it here would conflate the
legs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_SHELF = {
    "A-101": 540,
    "A-140": 897,
    "B-204": 1299,  # post-fix; the seeded source bug yields 1250
    "B-350": 775,
    "C-077": 429,
    "C-250": 313,
    "C-410": 1846,
    "D-023": 1119,
    "E-905": 2655,
    "F-310": 790,
}
EXPECTED_WEEKEND = {
    "A-101": 621,
    "A-140": 1032,
    "B-204": 1494,
    "B-350": 891,
    "C-077": 493,
    "C-250": 360,
    "C-410": 2123,
    "D-023": 1287,
    "E-905": 3053,
    "F-310": 909,
}
# clearance = shelf - half_up(shelf * pct / 100); D-550 is listed in the
# clearance source but discontinued (absent from the catalog feed): skipped.
EXPECTED_CLEARANCE = {
    "A-140": 448,  # discount 448.5 -> 449 half-up; round() keeps 448 -> price 449
    "B-204": 649,  # discount 649.5 -> 650 both ways; int() gives 649 -> price 650
    "B-350": 620,
    "C-250": 282,
    "E-905": 1858,  # discount 796.5 -> 797 half-up; round() keeps 796 -> price 1859
    "F-310": 474,
}

FIX_PROBE = """\
import json
out = {}
def t(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name] = "raised:" + type(exc).__name__
from src.catalog import lookup
from src.pricing import cart_total, shelf_price
t("a101_name", lambda: lookup("A-101")["name"])
t("d550", lambda: lookup("D-550"))
t("zzz", lambda: lookup("Z-999"))
t("c250", lambda: shelf_price("C-250"))
t("b204", lambda: shelf_price("B-204"))
t("cart", lambda: cart_total(["A-101", "C-077"]))
print(json.dumps(out))
"""

WEEKEND_PROBE = """\
import json
out = {}
def t(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name] = "raised:" + type(exc).__name__
from src.pricing import cart_total, shelf_price
try:
    from src.pricing import weekend_price
except Exception:
    def weekend_price(sku):
        raise RuntimeError("weekend_price missing")
t("wk_a101", lambda: weekend_price("A-101"))
t("wk_b204", lambda: weekend_price("B-204"))
t("wk_c410", lambda: weekend_price("C-410"))
t("wk_e905", lambda: weekend_price("E-905"))
t("wk_zzz", lambda: weekend_price("Z-999"))
t("wk_cart", lambda: cart_total(["A-101", "F-310"], weekend=True))
t("cart_plain", lambda: cart_total(["A-101", "C-077"]))
t("wk_f310", lambda: weekend_price("F-310"))
t("wk_b350", lambda: weekend_price("B-350"))
t("wk_c077", lambda: weekend_price("C-077"))
print(json.dumps(out))
"""

CLEARANCE_PROBE = """\
import json
out = {}
def t(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name] = "raised:" + type(exc).__name__
from src.pricing import cart_total, shelf_price
try:
    from src.pricing import clearance_price
except Exception:
    def clearance_price(sku):
        raise RuntimeError("clearance_price missing")
t("clr_b350", lambda: clearance_price("B-350"))
t("clr_c250", lambda: clearance_price("C-250"))
t("clr_f310", lambda: clearance_price("F-310"))
t("fallback_a101", lambda: clearance_price("A-101"))
t("fallback_c410", lambda: clearance_price("C-410"))
t("clr_zzz", lambda: clearance_price("Z-999"))
t("clr_cart", lambda: cart_total(["B-350", "A-101"], clearance=True))
t("cart_plain", lambda: cart_total(["A-101", "C-077"]))
t("clr_a140", lambda: clearance_price("A-140"))
t("clr_b204", lambda: clearance_price("B-204"))
t("clr_e905", lambda: clearance_price("E-905"))
print(json.dumps(out))
"""


def _copy_tree(workdir: str) -> Path:
    dst = Path(tempfile.mkdtemp(prefix="orchard-grade-")) / "wt"
    shutil.copytree(
        workdir,
        dst,
        ignore=shutil.ignore_patterns(".git", ".state*", "agent-*.log", "_run_config.toml"),
    )
    return dst


def _regen(tree: Path) -> None:
    """Rebuild the data feeds from THEIR build: delete data/*.tsv, then run
    every tools/gen_*.py they left behind. Deleting first is the built-artifact
    probe: a hand-written feed that no generator reproduces simply stays gone,
    and its cases fail. Each script runs independently (a crashing one only
    loses the feeds it would have produced)."""
    for feed in (tree / "data").glob("*.tsv"):
        feed.unlink()
    for gen in sorted((tree / "tools").glob("gen_*.py")):
        try:
            subprocess.run(
                [sys.executable, str(gen.relative_to(tree))],
                cwd=tree,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception:
            continue


def _read_feed(tree: Path, name: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Parse data/<name> into (header, rows-by-sku). ([], {}) on any failure."""
    try:
        lines = (tree / "data" / name).read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], {}
    if not lines:
        return [], {}
    header = lines[0].split("\t")
    rows: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        vals = line.split("\t")
        if len(vals) == len(header):
            rows[vals[header.index("sku")] if "sku" in header else vals[0]] = dict(
                zip(header, vals, strict=True)
            )
    return header, rows


def _probe(tree: Path, script: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def _int_of(row: dict[str, str] | None, col: str) -> int | None:
    if row is None:
        return None
    try:
        return int(row.get(col, ""))
    except ValueError:
        return None


_PROBES = {"fix": FIX_PROBE, "weekend": WEEKEND_PROBE, "clearance": CLEARANCE_PROBE}


def grade(workdir: str, leg: str) -> dict[str, Any]:
    tree = _copy_tree(workdir)
    _regen(tree)
    header, rows = _read_feed(tree, "catalog.tsv")
    probe = _probe(tree, _PROBES[leg])

    components: dict[str, dict[str, bool]] = {"regen": {}, "api": {}}
    regen = components["regen"]
    api = components["api"]

    regen["d550_absent"] = bool(rows) and "D-550" not in rows
    regen["sorted_by_sku"] = bool(rows) and list(rows) == sorted(rows)
    for sku, want in EXPECTED_SHELF.items():
        regen[f"shelf_{sku}"] = _int_of(rows.get(sku), "shelf_cents") == want

    if leg == "fix":
        api["a101_name"] = probe.get("a101_name") == "almond biscotti"
        api["unknown_raises"] = probe.get("zzz") == "raised:KeyError"
        api["inactive_raises"] = probe.get("d550") == "raised:KeyError"
        api["shelf_c250"] = probe.get("c250") == 313
        api["shelf_b204"] = probe.get("b204") == 1299
        api["cart"] = probe.get("cart") == 969
    elif leg == "weekend":
        regen["weekend_column"] = "weekend_cents" in header
        for sku, want in EXPECTED_WEEKEND.items():
            regen[f"weekend_{sku}"] = _int_of(rows.get(sku), "weekend_cents") == want
        api["wk_a101"] = probe.get("wk_a101") == 621
        api["wk_b204"] = probe.get("wk_b204") == 1494
        api["wk_c410"] = probe.get("wk_c410") == 2123
        api["wk_e905"] = probe.get("wk_e905") == 3053
        api["unknown_raises"] = probe.get("wk_zzz") == "raised:KeyError"
        api["wk_cart"] = probe.get("wk_cart") == 1530
        api["cart_plain_unchanged"] = probe.get("cart_plain") == 969
        components["rounding"] = {
            "half_up_f310": probe.get("wk_f310") == 909,
            "round_down_b350": probe.get("wk_b350") == 891,
            "round_down_c077": probe.get("wk_c077") == 493,
        }
    else:
        clr_header, clr_rows = _read_feed(tree, "clearance.tsv")
        regen["clr_file"] = clr_header[:1] == ["sku"] and "clearance_cents" in clr_header
        regen["clr_sorted"] = bool(clr_rows) and list(clr_rows) == sorted(clr_rows)
        regen["clr_d550_absent"] = bool(clr_rows) and "D-550" not in clr_rows
        for sku, want in EXPECTED_CLEARANCE.items():
            regen[f"clr_{sku}"] = _int_of(clr_rows.get(sku), "clearance_cents") == want
        api["clr_b350"] = probe.get("clr_b350") == 620
        api["clr_c250"] = probe.get("clr_c250") == 282
        api["clr_f310"] = probe.get("clr_f310") == 474
        api["fallback_a101"] = probe.get("fallback_a101") == 540
        api["fallback_c410"] = probe.get("fallback_c410") == 1846
        api["unknown_raises"] = probe.get("clr_zzz") == "raised:KeyError"
        api["clr_cart"] = probe.get("clr_cart") == 1160
        api["cart_plain_unchanged"] = probe.get("cart_plain") == 969
        components["rounding"] = {
            "half_up_a140": probe.get("clr_a140") == 448,
            "half_up_e905": probe.get("clr_e905") == 1858,
            "no_trunc_b204": probe.get("clr_b204") == 649,
        }

    shutil.rmtree(tree.parent, ignore_errors=True)

    results: dict[str, dict[str, int]] = {}
    component_scores: dict[str, float] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in components.items():
        p = sum(1 for ok in cases.values() if ok)
        results[name] = {"passed": p, "total": len(cases)}
        component_scores[name] = round(p / len(cases), 4) if cases else 0.0
        cases_passed += p
        cases_total += len(cases)
        if p == len(cases):
            components_passed += 1

    return {
        "task": "orchard",
        "leg": leg,
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "score": round(cases_passed / cases_total, 4) if cases_total else 0.0,
        "components": results,
        "component_scores": component_scores,
        "components_passed": components_passed,
        "components_total": len(components),
    }


if __name__ == "__main__":
    wt = sys.argv[1] if len(sys.argv) > 1 else "."
    leg = sys.argv[2] if len(sys.argv) > 2 else "fix"
    print(json.dumps(grade(wt, leg)))
