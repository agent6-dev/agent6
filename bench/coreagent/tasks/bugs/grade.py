#!/usr/bin/env python3
"""Authoritative hidden grader for bugs. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Imports <worktree-dir>/shapes.py, runs a thorough per-component battery, and
prints one JSON line: cases_passed/cases_total (fine score) and per-component
pass (coarse "did it leave a bug" signal). A component counts as passed only if
every one of its cases passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any


def _load(worktree: str) -> Any:
    spec = importlib.util.spec_from_file_location("shapes", f"{worktree}/shapes.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load shapes.py")
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
            "task": "bugs",
            "import_error": str(exc)[:200],
            "cases_passed": 0,
            "cases_total": 1,
            "score": 0.0,
            "components": {},
            "components_passed": 0,
            "components_total": 7,
        }

    components: dict[str, list[tuple[tuple[Any, ...], Any]]] = {
        "clamp": [
            ((5, 0, 10), 5),
            ((-5, 0, 10), 0),  # below lo
            ((15, 0, 10), 10),  # above hi
            ((0, 0, 10), 0),  # on lower bound
            ((10, 0, 10), 10),  # on upper bound
            ((7, 1, 5), 5),
            ((-3, -10, -1), -3),  # in range, negative bounds
            ((-20, -10, -1), -10),  # below lo, negative bounds
        ],
        "mean": [
            (([1.0, 2.0],), 1.5),  # non-integer mean
            (([1.0, 2.0, 3.0, 4.0],), 2.5),  # non-integer mean
            (([],), 0.0),  # empty
            (([7.0],), 7.0),
            (([2.0, 4.0],), 3.0),
            (([10.0, 5.0],), 7.5),  # non-integer mean
            (([1.0, 2.0, 3.0],), 2.0),
        ],
        "median": [
            (([3.0, 1.0, 2.0],), 2.0),  # odd, unsorted
            (([4.0, 1.0, 2.0, 3.0],), 2.5),  # even, unsorted
            (([],), 0.0),  # empty
            (([5.0],), 5.0),
            (([1.0, 2.0, 3.0],), 2.0),  # odd, sorted
            (([1.0, 2.0, 3.0, 4.0],), 2.5),  # even, sorted
            (([5.0, 3.0, 1.0, 4.0, 2.0],), 3.0),  # odd, unsorted
            (([10.0, 2.0, 8.0, 4.0],), 6.0),  # even, unsorted
        ],
        "gcd": [
            ((12, 8), 4),
            ((12, -8), 4),  # negative b
            ((-12, -8), 4),  # both negative
            ((0, 5), 5),  # zero a
            ((5, 0), 5),  # zero b
            ((0, 0), 0),  # both zero
            ((17, 5), 1),  # coprime
            ((-48, -36), 12),  # both negative
        ],
        "is_prime": [
            ((2,), True),
            ((3,), True),
            ((1,), False),
            ((4,), False),  # square of prime
            ((9,), False),  # square of prime
            ((25,), False),  # square of prime
            ((13,), True),
            ((15,), False),
        ],
        "roman": [
            ((1,), "I"),
            ((4,), "IV"),
            ((9,), "IX"),
            ((400,), "CD"),  # subtractive hundreds
            ((900,), "CM"),
            ((1994,), "MCMXCIV"),
            ((444,), "CDXLIV"),  # subtractive hundreds
            ((2024,), "MMXXIV"),
        ],
        "running_max": [
            (([1, 3, 2],), [1, 3, 3]),
            (([-3, -1, -2],), [-3, -1, -1]),  # negatives
            (([],), []),  # empty
            (([-5],), [-5]),  # single negative
            (([5, 4, 6, 2],), [5, 5, 6, 6]),
            (([3, 3, 3],), [3, 3, 3]),
            (([-1, -5, -2],), [-1, -1, -1]),  # all negative
            (([2, -1, 5, -3],), [2, 2, 5, 5]),
        ],
    }

    results: dict[str, dict[str, int]] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in components.items():
        fn = getattr(m, name, None)
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
        "task": "bugs",
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
