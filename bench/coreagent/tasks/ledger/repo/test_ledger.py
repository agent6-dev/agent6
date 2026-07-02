"""Starter tests for ledger (basics only; runs via ./verify.sh)."""

from __future__ import annotations

import unittest

import ledger


class TestDeposit(unittest.TestCase):
    def test_deposit_adds(self) -> None:
        a = ledger.Account("alice", 100.0)
        a.deposit(50.0)
        self.assertEqual(a.balance, 150.0)
        self.assertEqual(a.history, ["deposit 50.0"])

    def test_deposit_rejects_nonpositive(self) -> None:
        a = ledger.Account("alice")
        with self.assertRaises(ValueError):
            a.deposit(0)


class TestWithdraw(unittest.TestCase):
    def test_withdraw_subtracts(self) -> None:
        a = ledger.Account("a", 100.0)
        a.withdraw(30.0)
        self.assertEqual(a.balance, 70.0)
        self.assertEqual(a.history, ["withdraw 30.0"])

    def test_withdraw_overdraft(self) -> None:
        a = ledger.Account("a", 10.0)
        with self.assertRaises(ValueError):
            a.withdraw(20.0)
        self.assertEqual(a.balance, 10.0)


class TestTransfer(unittest.TestCase):
    def test_transfer_moves_funds(self) -> None:
        a = ledger.Account("a", 100.0)
        b = ledger.Account("b", 0.0)
        a.transfer(b, 40.0)
        self.assertEqual((a.balance, b.balance), (60.0, 40.0))
        self.assertEqual(a.history, ["transfer-out 40.0 to b"])
        self.assertEqual(b.history, ["transfer-in 40.0 from a"])

    def test_transfer_rollback(self) -> None:
        a = ledger.Account("a", 10.0)
        b = ledger.Account("b", 5.0)
        with self.assertRaises(ValueError):
            a.transfer(b, 50.0)
        self.assertEqual((a.balance, b.balance), (10.0, 5.0))
        self.assertEqual(a.history, [])
        self.assertEqual(b.history, [])


class TestLedger(unittest.TestCase):
    def test_open_and_total(self) -> None:
        book = ledger.Ledger()
        book.open_account("a", 100.0)
        book.open_account("b", 50.0)
        self.assertEqual(book.total_assets(), 150.0)

    def test_get_missing(self) -> None:
        book = ledger.Ledger()
        with self.assertRaises(KeyError):
            book.get("nope")


class TestParseAmount(unittest.TestCase):
    def test_dollar_commas(self) -> None:
        self.assertEqual(ledger.parse_amount("$1,234.56"), 1234.56)

    def test_negative_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ledger.parse_amount("-5")


if __name__ == "__main__":
    unittest.main()
