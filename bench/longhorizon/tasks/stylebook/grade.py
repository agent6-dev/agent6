#!/usr/bin/env python3
"""Authoritative hidden grader for stylebook. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Imports <worktree-dir>/audit.py and runs a per-rule case battery. Components
r01..r10 isolate one rule each (violation detected AND near-miss left clean);
`clean` holds fully compliant reports; `multi` holds cross-rule combinations
with exact expected code lists. Prints one JSON line with partial-credit
score, per-component pass counts, and per-component fractional scores (the
per-rule retention curve).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any

BASE = [
    "# routine survey",
    "DATE: 2024-03-11",
    "SITE: north-ridge",
    "CREW: Ana Ruiz, Bo Chen",
    "TEMP: -12C",
    "SAMPLE: S0100 topsoil",
    "SAMPLE: S0142 subsoil",
    "NOTE: overcast morning",
    "  wind picked up after noon",
    "END:",
]


def doc(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def base(
    *,
    repl: dict[str, str] | None = None,
    drop: tuple[str, ...] = (),
    add_before_end: tuple[str, ...] = (),
    append: tuple[str, ...] = (),
) -> str:
    """BASE with per-line replacements, removals, inserts before END:, appends after."""
    lines: list[str] = []
    for ln in BASE:
        if ln in drop:
            continue
        if ln == "END:" and add_before_end:
            lines.extend(add_before_end)
        lines.append((repl or {}).get(ln, ln))
    lines.extend(append)
    return doc(lines)


Case = tuple[str, list[str]]

CLEAN: list[Case] = [
    (base(), []),
    (doc(["DATE: 2024-03-11", "SITE: base", "CREW: Ana, Bo", "END:"]), []),
    (
        doc(
            [
                "# nightly survey",
                "DATE: 2024-02-29",
                "",
                "SITE: east-fork",
                "CREW: Ana Ruiz, Bo Chen-Li, M. Okafor",
                "TEMP: 60C",
                "TEMP: -60C",
                "TEMP: 0C",
                "SAMPLE: S0001",
                "SAMPLE: S0930 mid slope",
                "SAMPLE: S9999 summit",
                "NOTE: alpha",
                "  beta",
                "  # keeps going",
                "NOTE: solo note",
                "ZZ: unclassified aside",
                "",
                "# wrap up",
                "END:",
                "",
                "# archived by relay",
            ]
        ),
        [],
    ),
    (
        doc(
            [
                "DATE: 2000-02-29",
                "SITE: ridge  ",
                "CREW: Ana, Bo, Cy, Di, Ed, Fay",
                "NOTE: " + "x" * 120,
                "SAMPLE: S0500",
                "END:",
            ]
        ),
        [],
    ),
    (
        doc(
            [
                "# comment at top",
                "DATE: 2024-03-11",
                "SITE: north",
                "   ",
                "CREW: Ana, Bo",
                "#immediate comment no space",
                "NOTE: base note",
                "  # continuation not comment",
                "TEMP: 7C",
                "END:",
            ]
        ),
        [],
    ),
]

R01: list[Case] = [
    (base(repl={"TEMP: -12C": "TEMP:-12C"}), ["R01"]),
    (base(repl={"TEMP: -12C": "Temp: -12C"}), ["R01"]),
    (base(add_before_end=("T: cold",)), ["R01"]),
    (base(add_before_end=("LONGTAGXX: v",)), ["R01"]),
    (base(repl={"TEMP: -12C": "TEMP:  -12C"}), ["R01"]),
    (base(add_before_end=(" SITE: annex",)), ["R01"]),
    (base(add_before_end=("GRID2: k7",)), ["R01"]),
    (base(add_before_end=("OK: fine", "WAYPOINT: w1")), []),
    (base(repl={"TEMP: -12C": "TEMP:    "}), ["R01"]),
]

R02: list[Case] = [
    (base(drop=("CREW: Ana Ruiz, Bo Chen",)), ["R02"]),
    (
        base(
            drop=("DATE: 2024-03-11", "SAMPLE: S0100 topsoil", "SAMPLE: S0142 subsoil"),
        ),
        ["R02"],
    ),
    (doc(["TEMP: 5C", "NOTE: only prose", "END:"]), ["R02"]),
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREWS: Ana, Bo"}), ["R02"]),
]

R03: list[Case] = [
    (base(repl={"DATE: 2024-03-11": "DATE: 2024-13-01"}), ["R03"]),
    (base(repl={"DATE: 2024-03-11": "DATE: 2024-02-30"}), ["R03"]),
    (base(repl={"DATE: 2024-03-11": "DATE: 2023-02-29"}), ["R03"]),
    (base(repl={"DATE: 2024-03-11": "DATE: 2024-3-11"}), ["R03"]),
    (base(repl={"DATE: 2024-03-11": "DATE: 2024-03-11 noon"}), ["R03"]),
    (base(repl={"DATE: 2024-03-11": "DATE: 1900-02-29"}), ["R03"]),
    (base(repl={"DATE: 2024-03-11": "DATE: 2000-02-29"}), []),
    (base(add_before_end=("DATE: 2024-00-10",)), ["R03"]),
]

R04: list[Case] = [
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: Ana Ruiz"}), ["R04"]),
    (
        base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: Ana, Bo, Cy, Di, Ed, Fay, Gus"}),
        ["R04"],
    ),
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: Ana Ruiz,Bo Chen"}), ["R04"]),
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: Ana,  Bo"}), ["R04"]),
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: Ana, Ana"}), ["R04"]),
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: Ana, Bo, "}), ["R04"]),
    (base(repl={"CREW: Ana Ruiz, Bo Chen": "CREW: ana, Ana"}), []),
]

R05: list[Case] = [
    (base(repl={"TEMP: -12C": "TEMP: +7C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: 7 C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: 7.5C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: 07C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: 61C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: -61C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: -0C"}), ["R05"]),
    (base(repl={"TEMP: -12C": "TEMP: 12c"}), ["R05"]),
    (base(add_before_end=("TEMP: 60C", "TEMP: -60C", "TEMP: 0C")), []),
]

R06: list[Case] = [
    (
        doc(
            [
                "SITE: north-ridge",
                "CREW: Ana, Bo",
                "SAMPLE: S0100 topsoil",
                "SAMPLE: S0142 subsoil",
                "DATE: 2024-03-11",
                "END:",
            ]
        ),
        ["R06"],
    ),
    (base(add_before_end=("DATE: 2024-03-12",)), []),
    (base(repl={"DATE: 2024-03-11": "DATE: 2024-13-01"}), ["R03"]),
    (
        doc(
            [
                "DATE:2024-03-11",
                "SITE: s",
                "CREW: Ana, Bo",
                "SAMPLE: S0100",
                "DATE: 2024-03-12",
                "END:",
            ]
        ),
        ["R01", "R06"],
    ),
]

R07: list[Case] = [
    (base(repl={"SAMPLE: S0100 topsoil": "SAMPLE: S123 topsoil"}), ["R07"]),
    (base(repl={"SAMPLE: S0142 subsoil": "SAMPLE: S12345"}), ["R07"]),
    (base(repl={"SAMPLE: S0100 topsoil": "SAMPLE: s0100 topsoil"}), ["R07"]),
    (base(repl={"SAMPLE: S0142 subsoil": "SAMPLE: S0100 again"}), ["R07"]),
    (
        base(
            repl={
                "SAMPLE: S0100 topsoil": "SAMPLE: S0142 subsoil2",
                "SAMPLE: S0142 subsoil": "SAMPLE: S0100 topsoil",
            }
        ),
        ["R07"],
    ),
    (base(add_before_end=("SAMPLE: 0100S oops", "SAMPLE: S0200 deep")), ["R07"]),
    (base(repl={"SAMPLE: S0142 subsoil": "SAMPLE: S0142 replaces S0099"}), []),
    (base(repl={"SAMPLE: S0142 subsoil": "SAMPLE: S0142"}), []),
]

R08: list[Case] = [
    (
        base(
            repl={"NOTE: overcast morning": "NOTE: " + "x" * 121},
            drop=("  wind picked up after noon",),
        ),
        ["R08"],
    ),
    (
        base(
            repl={"NOTE: overcast morning": "NOTE: " + "x" * 120},
            drop=("  wind picked up after noon",),
        ),
        [],
    ),
    (
        base(
            repl={
                "NOTE: overcast morning": "NOTE: " + "x" * 100,
                "  wind picked up after noon": "  " + "y" * 25,
            }
        ),
        ["R08"],
    ),
    (
        base(
            repl={
                "NOTE: overcast morning": "NOTE: " + "x" * 100,
                "  wind picked up after noon": "  " + "y" * 19,
            }
        ),
        [],
    ),
    (
        base(add_before_end=("NOTE: " + "a" * 90, "NOTE: " + "b" * 121)),
        ["R08"],
    ),
    (
        base(
            repl={"NOTE: overcast morning": "NOTE: short"},
            drop=("  wind picked up after noon",),
            add_before_end=("", "  " + "y" * 150),
        ),
        ["R09"],
    ),
]

R09: list[Case] = [
    (
        base(
            drop=("  wind picked up after noon",),
            add_before_end=("TEMP: 5C", "  and falling"),
        ),
        ["R09"],
    ),
    (doc(["  stray first line", "DATE: 2024-03-11", "SITE: s", "CREW: Ana, Bo", "END:"]), ["R09"]),
    (
        base(
            repl={"  wind picked up after noon": ""},
            add_before_end=("  wind rising",),
        ),
        ["R09"],
    ),
    (
        base(
            repl={"  wind picked up after noon": "# interruption"},
            add_before_end=("  wind rising",),
        ),
        ["R09"],
    ),
    (base(repl={"  wind picked up after noon": "   triple indent"}), ["R09"]),
    (
        base(
            add_before_end=("NOTE: multi", "  one", "  two", "  three"),
        ),
        [],
    ),
    (
        base(
            repl={"NOTE: overcast morning": "NOTE:missing sep"},
        ),
        ["R01", "R09"],
    ),
]

R10: list[Case] = [
    (base(drop=("END:",)), ["R10"]),
    (
        doc(
            [
                "DATE: 2024-03-11",
                "SITE: s",
                "END:",
                "CREW: Ana, Bo",
                "END:",
            ]
        ),
        ["R10"],
    ),
    (base(append=("TEMP: 5C",)), ["R10"]),
    (base(repl={"END:": "END: departed at dusk"}), ["R10"]),
    (base(add_before_end=("END: noted",)), ["R10"]),
    (base(append=("", "# post comment")), []),
    (base(repl={"END:": "END: "}), ["R01", "R10"]),
]

MULTI: list[Case] = [
    (base(drop=("DATE: 2024-03-11",)), ["R02", "R06"]),
    (
        doc(["date: 2024-03-11", "SITE: s", "CREW: Ana, Bo", "END:"]),
        ["R01", "R02"],
    ),
    ("", ["R02", "R10"]),
    (doc(["# only a comment", "", "   "]), ["R02", "R10"]),
    (
        doc(
            [
                "DATE: 2024-02-30",
                "SITE: s",
                "CREW: Ana Ruiz",
                "TEMP: +7C",
                "SAMPLE: S0100 a",
                "SAMPLE: S0100 b",
            ]
        ),
        ["R03", "R04", "R05", "R07", "R10"],
    ),
    (base(append=("  stray after end",)), ["R09", "R10"]),
]

COMPONENTS: dict[str, list[Case]] = {
    "clean": CLEAN,
    "r01": R01,
    "r02": R02,
    "r03": R03,
    "r04": R04,
    "r05": R05,
    "r06": R06,
    "r07": R07,
    "r08": R08,
    "r09": R09,
    "r10": R10,
    "multi": MULTI,
}


def _load(worktree: str) -> Any:
    spec = importlib.util.spec_from_file_location("audit", f"{worktree}/audit.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(worktree: str) -> dict[str, Any]:
    total_components = len(COMPONENTS)
    try:
        mod = _load(worktree)
    except Exception as exc:
        return {
            "task": "stylebook",
            "import_error": str(exc)[:200],
            "cases_passed": 0,
            "cases_total": sum(len(c) for c in COMPONENTS.values()),
            "score": 0.0,
            "components": {},
            "component_scores": {},
            "components_passed": 0,
            "components_total": total_components,
        }

    fn = getattr(mod, "audit", None)
    results: dict[str, dict[str, int]] = {}
    component_scores: dict[str, float] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in COMPONENTS.items():
        p = 0
        for text, expected in cases:
            cases_total += 1
            try:
                ok = callable(fn) and fn(text) == expected
            except Exception:
                ok = False
            if ok:
                p += 1
                cases_passed += 1
        results[name] = {"passed": p, "total": len(cases)}
        component_scores[name] = round(p / len(cases), 4)
        if p == len(cases):
            components_passed += 1

    return {
        "task": "stylebook",
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "score": round(cases_passed / cases_total, 4) if cases_total else 0.0,
        "components": results,
        "component_scores": component_scores,
        "components_passed": components_passed,
        "components_total": total_components,
    }


if __name__ == "__main__":
    print(json.dumps(grade(sys.argv[1] if len(sys.argv) > 1 else ".")))
