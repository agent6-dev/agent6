"""ledger — small in-memory bank-ledger library. See spec.md.

Implement each class and function. Stub bodies raise NotImplementedError so the
test suite fails until you do.
"""

from __future__ import annotations


class Account:
    def __init__(self, name: str, balance: float = 0.0) -> None:
        raise NotImplementedError

    def deposit(self, amount: float) -> None:
        raise NotImplementedError

    def withdraw(self, amount: float) -> None:
        raise NotImplementedError

    def transfer(self, other: Account, amount: float) -> None:
        raise NotImplementedError


class Ledger:
    def __init__(self) -> None:
        raise NotImplementedError

    def open_account(self, name: str, opening: float = 0.0) -> Account:
        raise NotImplementedError

    def get(self, name: str) -> Account:
        raise NotImplementedError

    def total_assets(self) -> float:
        raise NotImplementedError


def parse_amount(s: str) -> float:
    raise NotImplementedError
