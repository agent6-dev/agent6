#!/usr/bin/env python3
"""Authoritative hidden grader for rpn. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Imports <worktree-dir>/rpn.py, runs a thorough per-component battery, and
prints one JSON line: cases_passed/cases_total (fine score) and per-component
pass (coarse "did it forget a component" signal). A component counts as passed
only if every one of its cases passes. Each case is a zero-argument predicate
returning True on pass; value cases use _case, error cases use _raises.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any


def _load(worktree: str) -> Any:
    spec = importlib.util.spec_from_file_location("rpn", f"{worktree}/rpn.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load rpn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(fn: Callable[..., Any], args: tuple[Any, ...], expected: Any) -> bool:
    try:
        return fn(*args) == expected
    except Exception:
        return False


def _raises(fn: Callable[..., Any], args: tuple[Any, ...]) -> bool:
    """True iff calling fn(*args) raises ValueError (not some other exception)."""
    try:
        fn(*args)
    except ValueError:
        return True
    except Exception:
        return False
    return False


def grade(worktree: str) -> dict[str, Any]:
    try:
        m = _load(worktree)
    except Exception as exc:
        return {
            "task": "rpn",
            "import_error": str(exc)[:200],
            "cases_passed": 0,
            "cases_total": 1,
            "score": 0.0,
            "components": {},
            "components_passed": 0,
            "components_total": 4,
        }

    tok = getattr(m, "tokenize", None)
    ev = getattr(m, "evaluate", None)
    ee = getattr(m, "evaluate_expr", None)
    calc = getattr(m, "RPNCalculator", None)

    def _calc_records_history() -> bool:
        c = calc()
        c.push("3 4 +")
        return c.history == [("3 4 +", 7.0)]

    def _calc_accumulates() -> bool:
        c = calc()
        c.push("3 4 +")
        c.push("10 2 /")
        return len(c.history) == 2 and c.last() == 5.0

    def _calc_clears() -> bool:
        c = calc()
        c.push("1 2 +")
        c.clear()
        return c.history == [] and c.last() is None

    components: dict[str, list[Callable[[], bool]]] = {
        "tokenize": [
            lambda: _case(tok, ("3 4 +",), ["3", "4", "+"]),
            lambda: _case(tok, ("3   4   +",), ["3", "4", "+"]),  # collapse spaces
            lambda: _case(tok, ("",), []),
            lambda: _case(tok, ("   ",), []),  # whitespace only
            lambda: _case(tok, ("10",), ["10"]),
            lambda: _case(tok, ("3.5 2 *",), ["3.5", "2", "*"]),
            lambda: _case(tok, ("1 2 3 + -",), ["1", "2", "3", "+", "-"]),
        ],
        "evaluate": [
            lambda: _case(ev, (["3", "4", "+"],), 7.0),
            lambda: _case(ev, (["10", "2", "/"],), 5.0),  # division order
            lambda: _case(ev, (["2", "3", "-"],), -1.0),  # subtraction order
            lambda: _case(ev, (["5", "1", "2", "+", "4", "*", "+", "3", "-"],), 14.0),
            lambda: _raises(ev, (["1", "0", "/"],)),  # division by zero
            lambda: _raises(ev, (["1", "foo", "+"],)),  # unknown token
            lambda: _raises(ev, (["1", "+"],)),  # stack underflow
            lambda: _raises(ev, (["1", "2", "3", "+"],)),  # leftover operands
        ],
        "evaluate_expr": [
            lambda: _case(ee, ("3 4 +",), 7.0),
            lambda: _case(ee, ("10 2 /",), 5.0),
            lambda: _case(ee, ("  2   3   -  ",), -1.0),  # extra spaces -> tokenize
            lambda: _case(ee, ("5 1 2 + 4 * + 3 -",), 14.0),
            lambda: _raises(ee, ("1 0 /",)),  # error propagates from evaluate
            lambda: _raises(ee, ("1 2 3 +",)),  # leftover operands
        ],
        "RPNCalculator": [
            lambda: _case(calc().push if calc else None, ("3 4 +",), 7.0),
            _calc_records_history,
            _calc_accumulates,
            lambda: calc().last() is None,
            _calc_clears,
            lambda: _raises(calc().push if calc else None, ("1 0 /",)),  # push propagates
        ],
    }

    results: dict[str, dict[str, int]] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in components.items():
        p = 0
        for case in cases:
            cases_total += 1
            try:
                ok = bool(case())
            except Exception:
                ok = False
            if ok:
                p += 1
                cases_passed += 1
        results[name] = {"passed": p, "total": len(cases)}
        if p == len(cases):
            components_passed += 1

    return {
        "task": "rpn",
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
