#!/usr/bin/env python3
"""Authoritative hidden grader for ledger. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Imports <worktree-dir>/ledger.py, runs a thorough per-component battery, and
prints one JSON line: cases_passed/cases_total (fine score) and per-component
pass (coarse "did it forget a component" signal). A component counts as passed
only if every one of its cases passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any


def _load(worktree: str) -> Any:
    spec = importlib.util.spec_from_file_location("ledger", f"{worktree}/ledger.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load ledger.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raises(fn: Callable[[], Any], exc: type[BaseException]) -> bool:
    """True iff calling fn() raises the expected exception type (not another)."""
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _run(check: Callable[[Any], bool], m: Any) -> bool:
    try:
        return bool(check(m))
    except Exception:
        return False


# --- Account: __init__ + deposit -------------------------------------------


def _acct_init_defaults(m: Any) -> bool:
    a = m.Account("alice")
    return a.name == "alice" and a.balance == 0.0 and a.history == []


def _acct_init_balance(m: Any) -> bool:
    a = m.Account("bob", 100.0)
    return a.balance == 100.0 and a.history == []


def _deposit_adds(m: Any) -> bool:
    a = m.Account("a", 10.0)
    a.deposit(5.0)
    return a.balance == 15.0


def _deposit_history(m: Any) -> bool:
    a = m.Account("a")
    a.deposit(50.0)
    return a.history == ["deposit 50.0"]


def _deposit_accumulates(m: Any) -> bool:
    a = m.Account("a")
    a.deposit(5.0)
    a.deposit(2.5)
    return a.balance == 7.5 and a.history == ["deposit 5.0", "deposit 2.5"]


def _deposit_rejects_zero(m: Any) -> bool:
    return _raises(lambda: m.Account("a").deposit(0), ValueError)


def _deposit_rejects_negative(m: Any) -> bool:
    a = m.Account("a", 10.0)
    return _raises(lambda: a.deposit(-5.0), ValueError) and a.balance == 10.0 and a.history == []


# --- Account.withdraw ------------------------------------------------------


def _withdraw_subtracts(m: Any) -> bool:
    a = m.Account("a", 100.0)
    a.withdraw(30.0)
    return a.balance == 70.0


def _withdraw_history(m: Any) -> bool:
    a = m.Account("a", 100.0)
    a.withdraw(30.0)
    return a.history == ["withdraw 30.0"]


def _withdraw_exact_ok(m: Any) -> bool:
    a = m.Account("a", 10.0)
    a.withdraw(10.0)
    return a.balance == 0.0 and a.history == ["withdraw 10.0"]


def _withdraw_overdraft_raises(m: Any) -> bool:
    a = m.Account("a", 10.0)
    return _raises(lambda: a.withdraw(20.0), ValueError)


def _withdraw_overdraft_noop(m: Any) -> bool:
    a = m.Account("a", 10.0)
    raised = _raises(lambda: a.withdraw(20.0), ValueError)
    return raised and a.balance == 10.0 and a.history == []


def _withdraw_rejects_zero(m: Any) -> bool:
    return _raises(lambda: m.Account("a", 10.0).withdraw(0), ValueError)


def _withdraw_rejects_negative(m: Any) -> bool:
    a = m.Account("a", 10.0)
    return _raises(lambda: a.withdraw(-1.0), ValueError) and a.balance == 10.0 and a.history == []


# --- Account.transfer ------------------------------------------------------


def _transfer_moves(m: Any) -> bool:
    a = m.Account("a", 100.0)
    b = m.Account("b", 0.0)
    a.transfer(b, 40.0)
    return a.balance == 60.0 and b.balance == 40.0


def _transfer_history_out(m: Any) -> bool:
    a = m.Account("a", 100.0)
    b = m.Account("b", 0.0)
    a.transfer(b, 40.0)
    return a.history == ["transfer-out 40.0 to b"]


def _transfer_history_in(m: Any) -> bool:
    a = m.Account("a", 100.0)
    b = m.Account("b", 0.0)
    a.transfer(b, 40.0)
    return b.history == ["transfer-in 40.0 from a"]


def _transfer_overdraft_raises(m: Any) -> bool:
    a = m.Account("a", 10.0)
    b = m.Account("b", 0.0)
    return _raises(lambda: a.transfer(b, 50.0), ValueError)


def _transfer_overdraft_rollback(m: Any) -> bool:
    a = m.Account("a", 10.0)
    b = m.Account("b", 5.0)
    raised = _raises(lambda: a.transfer(b, 50.0), ValueError)
    return raised and a.balance == 10.0 and b.balance == 5.0 and a.history == [] and b.history == []


def _transfer_rejects_nonpositive(m: Any) -> bool:
    a = m.Account("a", 10.0)
    b = m.Account("b", 0.0)
    return _raises(lambda: a.transfer(b, 0), ValueError)


def _transfer_nonpositive_rollback(m: Any) -> bool:
    a = m.Account("a", 10.0)
    b = m.Account("b", 5.0)
    raised = _raises(lambda: a.transfer(b, -3.0), ValueError)
    return raised and a.balance == 10.0 and b.balance == 5.0 and a.history == [] and b.history == []


# --- Ledger ----------------------------------------------------------------


def _open_returns_account(m: Any) -> bool:
    book = m.Ledger()
    acc = book.open_account("a", 100.0)
    return acc.name == "a" and acc.balance == 100.0


def _open_stores(m: Any) -> bool:
    book = m.Ledger()
    book.open_account("a", 100.0)
    return book.get("a").balance == 100.0


def _open_default_zero(m: Any) -> bool:
    book = m.Ledger()
    return book.open_account("b").balance == 0.0


def _open_duplicate_raises(m: Any) -> bool:
    book = m.Ledger()
    book.open_account("a")
    return _raises(lambda: book.open_account("a"), ValueError)


def _get_missing_raises(m: Any) -> bool:
    book = m.Ledger()
    return _raises(lambda: book.get("nope"), KeyError)


def _total_assets_sums(m: Any) -> bool:
    book = m.Ledger()
    book.open_account("a", 100.0)
    book.open_account("b", 50.0)
    return book.total_assets() == 150.0


def _total_assets_empty(m: Any) -> bool:
    return m.Ledger().total_assets() == 0.0


def _total_assets_tracks_ops(m: Any) -> bool:
    book = m.Ledger()
    a = book.open_account("a", 100.0)
    b = book.open_account("b", 0.0)
    a.transfer(b, 30.0)
    a.deposit(10.0)
    return book.total_assets() == 110.0


# --- parse_amount ----------------------------------------------------------


def grade(worktree: str) -> dict[str, Any]:
    try:
        m = _load(worktree)
    except Exception as exc:
        return {
            "task": "ledger",
            "import_error": str(exc)[:200],
            "cases_passed": 0,
            "cases_total": 1,
            "score": 0.0,
            "components": {},
            "components_passed": 0,
            "components_total": 5,
        }

    components: dict[str, list[Callable[[Any], bool]]] = {
        "Account.deposit": [
            _acct_init_defaults,
            _acct_init_balance,
            _deposit_adds,
            _deposit_history,
            _deposit_accumulates,
            _deposit_rejects_zero,
            _deposit_rejects_negative,
        ],
        "Account.withdraw": [
            _withdraw_subtracts,
            _withdraw_history,
            _withdraw_exact_ok,
            _withdraw_overdraft_raises,
            _withdraw_overdraft_noop,
            _withdraw_rejects_zero,
            _withdraw_rejects_negative,
        ],
        "Account.transfer": [
            _transfer_moves,
            _transfer_history_out,
            _transfer_history_in,
            _transfer_overdraft_raises,
            _transfer_overdraft_rollback,
            _transfer_rejects_nonpositive,
            _transfer_nonpositive_rollback,
        ],
        "Ledger": [
            _open_returns_account,
            _open_stores,
            _open_default_zero,
            _open_duplicate_raises,
            _get_missing_raises,
            _total_assets_sums,
            _total_assets_empty,
            _total_assets_tracks_ops,
        ],
        "parse_amount": [
            lambda m: m.parse_amount("$1,234.56") == 1234.56,
            lambda m: m.parse_amount("1234.56") == 1234.56,
            lambda m: m.parse_amount("  42 ") == 42.0,
            lambda m: m.parse_amount("$1,000") == 1000.0,
            lambda m: m.parse_amount("0") == 0.0,
            lambda m: m.parse_amount("$2,500.00") == 2500.0,
            lambda m: _raises(lambda: m.parse_amount("abc"), ValueError),
            lambda m: _raises(lambda: m.parse_amount("-5"), ValueError),
        ],
    }

    results: dict[str, dict[str, int]] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, checks in components.items():
        p = 0
        for check in checks:
            cases_total += 1
            if _run(check, m):
                p += 1
                cases_passed += 1
        results[name] = {"passed": p, "total": len(checks)}
        if p == len(checks):
            components_passed += 1

    return {
        "task": "ledger",
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
