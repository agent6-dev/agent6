# ledger — a small in-memory bank-ledger library

Implement the two classes and one helper in `ledger.py`. The test suite
(`./verify.sh`, which runs `python3 -m unittest`) checks each component
separately. All state lives in memory (no I/O). Use only the Python standard
library.

## Components

1. `Account(name: str, balance: float = 0.0)`
   A single account. Attributes: `name`, `balance`, and `history: list[str]`
   (empty at first). `deposit(self, amount: float) -> None` adds `amount` to
   `balance` and appends `f"deposit {amount}"` to `history`. Raise `ValueError`
   if `amount <= 0`, leaving balance and history unchanged.

2. `Account.withdraw(self, amount: float) -> None`
   Subtract `amount` from `balance` and append `f"withdraw {amount}"` to
   `history`. Raise `ValueError` if `amount <= 0` or if `amount > balance` (no
   overdraft). On a rejected withdraw, balance and history stay unchanged; the
   history line is written on success only. Withdrawing exactly `balance` is
   allowed.

3. `Account.transfer(self, other: Account, amount: float) -> None`
   Move `amount` from `self` into `other` atomically: withdraw from `self`,
   deposit into `other`. On success append `f"transfer-out {amount} to {other.name}"`
   to `self.history` and `f"transfer-in {amount} from {self.name}"` to
   `other.history` (not the plain `deposit`/`withdraw` lines). If the withdraw
   would fail (overdraft or `amount <= 0`), raise `ValueError` and leave NEITHER
   balance changed and NO history written on either side.

4. `Ledger()`
   Holds `accounts: dict[str, Account]`. `open_account(self, name: str, opening: float = 0.0) -> Account`
   creates, stores, and returns a new `Account`; raise `ValueError` if `name`
   already exists. `get(self, name: str) -> Account` returns the stored account;
   raise `KeyError` if it is missing. `total_assets(self) -> float` returns the
   sum of every account's balance (`0.0` when there are none).

5. `parse_amount(s: str) -> float`
   Parse a money string to a float. Strip surrounding whitespace, drop a leading
   `$`, and remove thousands commas. Examples: `"$1,234.56"` -> `1234.56`,
   `"1234.56"` -> `1234.56`, `"  42 "` -> `42.0`, `"$1,000"` -> `1000.0`. Raise
   `ValueError` on a non-numeric string or a negative result (`"0"` is allowed).

## Done when

`./verify.sh` passes (exit 0). Do not edit `test_ledger.py` or `verify.sh`.
