# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pin what the credit ledger and the journal fold keep of a balance series."""

from __future__ import annotations

import decimal
from collections.abc import Iterable

import pytest

from agent6 import credit_usage

R = credit_usage.Reading


def _series(*pairs: tuple[int, float], account: str = "a") -> tuple[credit_usage.Reading, ...]:
    return tuple(R(account, at, balance) for at, balance in pairs)


def test_a_run_of_equal_balances_keeps_its_first_and_last_reading() -> None:
    """Every unchanged balance stayed in the ledger and went to the journal again."""
    ledger = credit_usage.Ledger()
    for at in range(1, 41):
        ledger.observe(_series((at, 10)))
    assert len(ledger) == 2 and ledger.spent_usd == 0
    assert ledger.unpublished.readings == _series((1, 10), (40, 10))
    ledger.observe(_series((41, 7)))
    assert len(ledger) == 3 and ledger.spent_usd == 3


def test_a_reading_adds_its_decline_without_rereading_the_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each reading re-sorted the whole series, quadratic over a run's calls."""
    reread: list[int] = []
    declines = credit_usage._declines  # pyright: ignore[reportPrivateUsage]

    def counted(readings: Iterable[credit_usage.Reading]) -> decimal.Decimal:
        seen = list(readings)
        reread.append(len(seen))
        return declines(seen)

    monkeypatch.setattr(credit_usage, "_declines", counted)
    ledger = credit_usage.Ledger()
    for at in range(1, 41):
        ledger.observe(_series((2 * at, 100 - at)))
    assert len(ledger) == 40 and ledger.spent_usd == 39
    ledger.observe(_series((11, 90)))  # late: 95 fell to 90 before a top-up to 94
    assert ledger.spent_usd == 43 and max(reread) <= 3


def test_a_late_reading_inside_a_run_of_its_balance_adds_nothing() -> None:
    ledger = credit_usage.Ledger()
    ledger.observe(_series((1, 10), (5, 10), (9, 10)))
    ledger.published(ledger.unpublished)
    ledger.observe(_series((1, 10), (3, 10), (5, 10), (9, 10)))
    assert len(ledger) == 2 and ledger.unpublished.readings == ()
    ledger.observe(_series((4, 9)))
    assert ledger.spent_usd == 1 and ledger.unpublished.readings == _series((4, 9))


def test_a_restored_series_charges_only_declines_beyond_its_history() -> None:
    ledger = credit_usage.Ledger()
    ledger.restore(1, "a", set(_series((1, 10), (2, 10), (3, 10), (4, 9))))
    ledger.observe(_series((2, 10), (4, 9), (5, 8)), origins={"a": 1})
    assert len(ledger) == 4 and ledger.spent_usd == 1
    assert ledger.unpublished.readings == _series((5, 8))


@pytest.mark.parametrize(
    ("rows", "kept"),
    [
        ([[(1, 10)], [(2, 10)], [(3, 10)], [(4, 9)]], [(1, 10), (3, 10), (4, 9)]),
        ([[(1, 10), (2, 10)], [(2, 10)], [(3, 10)]], [(1, 10), (3, 10)]),
        ([[(1, 10)], [(9, 10)], [(5, 10)], [(4, 9)], [(6, 8)]], [(1, 10), (4, 9), (6, 8), (9, 10)]),
    ],
)
def test_the_fold_keeps_the_ends_of_each_run_in_time_order(
    rows: list[list[tuple[int, float]]], kept: list[tuple[int, float]]
) -> None:
    series: tuple[credit_usage.Reading, ...] = ()
    for row in rows:
        series = credit_usage.merge(series, _series(*row))
    assert series == _series(*kept)


def test_the_fold_keeps_accounts_apart() -> None:
    series = credit_usage.merge((), _series((1, 10), (2, 10), (3, 10), account="b"))
    series = credit_usage.merge(series, _series((2, 10), (4, 10), account="a"))
    series = credit_usage.merge(series, _series((5, 10), account="a"))
    assert series == (
        *_series((2, 10), (5, 10), account="a"),
        *_series((1, 10), (3, 10), account="b"),
    )
