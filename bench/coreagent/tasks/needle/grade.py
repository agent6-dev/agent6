#!/usr/bin/env python3
"""Authoritative hidden grader for needle. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Imports <worktree-dir>/report.py, runs a thorough per-component battery, and
prints one JSON line: cases_passed/cases_total (fine score) and per-component
pass (coarse "did it forget a rule" signal). Each component isolates one of the
five rules buried in ref1.md through ref5.md. A component counts as passed only
if every one of its cases passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any


def _load(worktree: str) -> Any:
    spec = importlib.util.spec_from_file_location("report", f"{worktree}/report.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(fn: Callable[..., Any], args: tuple[Any, ...], expected: Any) -> bool:
    try:
        return fn(*args) == expected
    except Exception:
        return False


def grade(worktree: str) -> dict[str, Any]:
    try:
        m = _load(worktree)
    except Exception as exc:
        return {
            "task": "needle",
            "import_error": str(exc)[:200],
            "cases_passed": 0,
            "cases_total": 1,
            "score": 0.0,
            "components": {},
            "components_passed": 0,
            "components_total": 5,
        }

    # Each component isolates one rule (ref1..ref5). The expected value is the
    # full render() output string for that input. Inputs are shaped so a build
    # that breaks only this rule fails the component's cases.
    components: dict[str, list[tuple[tuple[Any, ...], Any]]] = {
        # ref1: sort by value DESC, ties broken by name ASC.
        "sort_value_desc_name_asc": [
            (
                (
                    [
                        {"name": "a", "value": 1.0},
                        {"name": "b", "value": 3.0},
                        {"name": "c", "value": 2.0},
                    ],
                ),
                "REPORT (3 items)\nb: 3.00\nc: 2.00\na: 1.00",
            ),
            (
                (
                    [
                        {"name": "zebra", "value": 5.0},
                        {"name": "apple", "value": 5.0},
                        {"name": "mango", "value": 5.0},
                    ],
                ),
                "REPORT (3 items)\napple: 5.00\nmango: 5.00\nzebra: 5.00",
            ),
            (
                (
                    [
                        {"name": "b", "value": 2.0},
                        {"name": "a", "value": 2.0},
                        {"name": "c", "value": 10.0},
                    ],
                ),
                "REPORT (3 items)\nc: 10.00\na: 2.00\nb: 2.00",
            ),
            (
                (
                    [
                        {"name": "x", "value": 1.0},
                        {"name": "y", "value": 2.0},
                        {"name": "z", "value": 3.0},
                    ],
                ),
                "REPORT (3 items)\nz: 3.00\ny: 2.00\nx: 1.00",
            ),
            (
                (
                    [
                        {"name": "d", "value": 4.0},
                        {"name": "a", "value": 4.0},
                        {"name": "c", "value": 4.0},
                        {"name": "b", "value": 8.0},
                    ],
                ),
                "REPORT (4 items)\nb: 8.00\na: 4.00\nc: 4.00\nd: 4.00",
            ),
        ],
        # ref2: each row renders as "{name}: {value}" with value to EXACTLY 2 decimals.
        "value_two_decimals": [
            (([{"name": "a", "value": 3.0}],), "REPORT (1 items)\na: 3.00"),
            (([{"name": "a", "value": 3.14159}],), "REPORT (1 items)\na: 3.14"),
            (([{"name": "a", "value": 3.1}],), "REPORT (1 items)\na: 3.10"),
            (([{"name": "a", "value": 1.999}],), "REPORT (1 items)\na: 2.00"),
            (([{"name": "a", "value": 42.0}],), "REPORT (1 items)\na: 42.00"),
            (([{"name": "a", "value": 2.5}],), "REPORT (1 items)\na: 2.50"),
        ],
        # ref4: first line is "REPORT ({n} items)" where n counts rendered rows (post ref5 filter).
        "header_item_count": [
            (([],), "REPORT (0 items)"),
            (([{"name": "a", "value": 1.0}],), "REPORT (1 items)\na: 1.00"),
            (
                (
                    [
                        {"name": "a", "value": 1.0},
                        {"name": "b", "value": 2.0},
                        {"name": "c", "value": 3.0},
                    ],
                ),
                "REPORT (3 items)\nc: 3.00\nb: 2.00\na: 1.00",
            ),
            (
                (
                    [
                        {"name": "a", "value": 1.0},
                        {"name": "b", "value": -1.0},
                        {"name": "c", "value": 2.0},
                    ],
                ),
                "REPORT (2 items)\nc: 2.00\na: 1.00",
            ),
            (
                (
                    [
                        {"name": "a", "value": 1.0},
                        {"name": "b", "value": 2.0},
                        {"name": "c", "value": 3.0},
                        {"name": "d", "value": 4.0},
                        {"name": "e", "value": 5.0},
                    ],
                ),
                "REPORT (5 items)\ne: 5.00\nd: 4.00\nc: 3.00\nb: 2.00\na: 1.00",
            ),
        ],
        # ref3: header first, rows joined by newlines, output ends WITHOUT a trailing newline.
        "join_no_trailing_newline": [
            (([{"name": "solo", "value": 5.0}],), "REPORT (1 items)\nsolo: 5.00"),
            (
                ([{"name": "a", "value": 2.0}, {"name": "b", "value": 1.0}],),
                "REPORT (2 items)\na: 2.00\nb: 1.00",
            ),
            (([],), "REPORT (0 items)"),
            (
                (
                    [
                        {"name": "a", "value": 3.0},
                        {"name": "b", "value": 2.0},
                        {"name": "c", "value": 1.0},
                    ],
                ),
                "REPORT (3 items)\na: 3.00\nb: 2.00\nc: 1.00",
            ),
        ],
        # ref5: skip any row whose value is negative; a row with value exactly 0 is kept.
        "skip_negative_keep_zero": [
            (
                ([{"name": "a", "value": -1.0}, {"name": "b", "value": 2.0}],),
                "REPORT (1 items)\nb: 2.00",
            ),
            (
                ([{"name": "a", "value": 0.0}, {"name": "b", "value": -5.0}],),
                "REPORT (1 items)\na: 0.00",
            ),
            (([{"name": "a", "value": -1.0}, {"name": "b", "value": -2.0}],), "REPORT (0 items)"),
            (
                (
                    [
                        {"name": "keep", "value": 0.0},
                        {"name": "drop", "value": -0.5},
                        {"name": "big", "value": 9.0},
                    ],
                ),
                "REPORT (2 items)\nbig: 9.00\nkeep: 0.00",
            ),
            (
                ([{"name": "a", "value": 100.0}, {"name": "b", "value": -200.0}],),
                "REPORT (1 items)\na: 100.00",
            ),
        ],
    }

    fn = getattr(m, "render", None)
    results: dict[str, dict[str, int]] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in components.items():
        p = 0
        for args, expected in cases:
            cases_total += 1
            if callable(fn) and _case(fn, args, expected):
                p += 1
                cases_passed += 1
        results[name] = {"passed": p, "total": len(cases)}
        if p == len(cases):
            components_passed += 1

    return {
        "task": "needle",
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "score": round(cases_passed / cases_total, 4) if cases_total else 0.0,
        "components": results,
        "components_passed": components_passed,
        "components_total": len(components),
    }


if __name__ == "__main__":
    wt = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(grade(wt)))
