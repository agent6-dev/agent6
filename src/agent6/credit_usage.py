# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Account balance observations used by the USD cap."""

from __future__ import annotations

import bisect
import dataclasses
import decimal
import itertools
from collections.abc import Collection, Iterable, Mapping


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class Reading:
    """A purchased-credit balance when it was observed, before any child settlement.

    Attributes:
        account: A fingerprint of the account ID, or the provider entry if the ID is absent.
        observed_ns: When the observer received the reading, in Unix nanoseconds.
        balance_usd: The balance in dollars.
    """

    account: str
    observed_ns: int
    balance_usd: float


def _declines(readings: Iterable[Reading]) -> decimal.Decimal:
    balances = (decimal.Decimal(str(r.balance_usd)) for r in sorted(readings))
    return sum((a - b for a, b in itertools.pairwise(balances) if a > b), decimal.Decimal(0))


def _insert(kept: list[Reading], reading: Reading) -> tuple[Reading, ...] | None:
    """Add a reading to an ordered series, each run of equal balances cut to its ends.

    A decline is a difference between neighbours, so a reading inside a run adds none.

    Args:
        kept: The series in order, changed in place.
        reading: The reading to add.

    Returns:
        The end of a run this reading replaced, if any; None when the reading adds nothing.
    """
    at = bisect.bisect_left(kept, reading)

    def same(index: int) -> bool:
        if not 0 <= index < len(kept):
            return False
        other = kept[index]
        return (other.account, other.balance_usd) == (reading.account, reading.balance_usd)

    if (at < len(kept) and kept[at] == reading) or (same(at - 1) and same(at)):
        return None
    if same(at - 1) and same(at - 2):
        at -= 1
    elif not (same(at) and same(at + 1)):
        kept.insert(at, reading)
        return ()
    replaced, kept[at] = kept[at], reading
    return (replaced,)


def merge(kept: tuple[Reading, ...], new: Collection[Reading]) -> tuple[Reading, ...]:
    """Add readings to the ordered series a journal fold keeps, cut as the ledger cuts it.

    Args:
        kept: The series so far, as this function returned it.
        new: The readings to add, in any order.

    Returns:
        The series in account and time order.
    """
    if not new:
        return kept
    series = list(kept)
    for reading in new:
        _insert(series, reading)
    return tuple(series)


@dataclasses.dataclass(slots=True)
class _Series:
    """One execution's observations of one account, in order, and the declines charged to it."""

    kept: list[Reading]
    spent: decimal.Decimal = decimal.Decimal(0)


# A number binds the account. An index map binds only those readings.
type Origins = dict[str, int | dict[int, int]]


@dataclasses.dataclass(frozen=True, slots=True)
class Batch:
    """New balance observations and their origins in the session journal."""

    readings: tuple[Reading, ...]
    origins: Origins

    def scoped(self) -> Iterable[tuple[int | None, Reading]]:
        """Yield each reading with its recovered execution, or None for a live observation."""
        for index, reading in enumerate(self.readings):
            origin = self.origins.get(reading.account)
            yield origin.get(index) if isinstance(origin, dict) else origin, reading


@dataclasses.dataclass(slots=True)
class Ledger:
    """Charge only new declines within each execution's account series."""

    _series: dict[tuple[int | None, str], _Series] = dataclasses.field(default_factory=dict)
    _unpublished: set[tuple[int | None, Reading]] = dataclasses.field(default_factory=set)

    def __len__(self) -> int:
        return sum(len(series.kept) for series in self._series.values())

    @property
    def spent_usd(self) -> float:
        """The additional movement from recovery plus the live execution's declines."""
        return float(sum(series.spent for series in self._series.values()))

    def restore(self, origin: int, account: str, readings: set[Reading]) -> None:
        """Seed a recovered series once, without publishing or charging its history."""
        key = origin, account
        if key not in self._series:
            self._series[key] = _Series(list(merge((), readings)))

    def observe(
        self, readings: Iterable[Reading], *, origins: Mapping[str, int] | None = None
    ) -> None:
        """Charge new evidence to its execution, even when settlement delivers it late."""
        for reading in sorted(readings):
            origin = origins[reading.account] if origins is not None else None
            series = self._series.setdefault((origin, reading.account), _Series([]))
            replaced = _insert(series.kept, reading)
            if replaced is None:
                continue
            at = bisect.bisect_left(series.kept, reading)
            around = series.kept[max(at - 1, 0) : at + 2]
            series.spent += _declines(around) - _declines(k for k in around if k != reading)
            self._unpublished.difference_update((origin, old) for old in replaced)
            self._unpublished.add((origin, reading))

    @property
    def unpublished(self) -> Batch:
        """The observations not yet confirmed by a journal append."""
        rows = sorted(self._unpublished, key=lambda pair: (pair[1], pair[0] or 0))
        origins: Origins = {}
        for account, group in itertools.groupby(
            enumerate(rows), key=lambda item: item[1][1].account
        ):
            if indexed := {i: origin for i, (origin, _) in group if origin is not None}:
                origins[account] = indexed
        return Batch(tuple(r for _, r in rows), origins)

    def published(self, batch: Batch) -> None:
        """Mark only the observations included in a written row as published."""
        self._unpublished.difference_update(batch.scoped())
