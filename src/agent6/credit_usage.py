# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Account balance observations used by the USD cap."""

from __future__ import annotations

import dataclasses
import decimal
import itertools
from collections.abc import Iterable, Mapping


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


def _declines(readings: set[Reading]) -> decimal.Decimal:
    balances = (decimal.Decimal(str(r.balance_usd)) for r in sorted(readings))
    return sum((a - b for a, b in itertools.pairwise(balances) if a > b), decimal.Decimal(0))


@dataclasses.dataclass(slots=True)
class _Series:
    """One execution's observations of one account."""

    seen: set[Reading]
    banked: decimal.Decimal
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
        return sum(len(series.seen) for series in self._series.values())

    @property
    def spent_usd(self) -> float:
        """The additional movement from recovery plus the live execution's declines."""
        return float(sum(series.spent for series in self._series.values()))

    def restore(self, origin: int, account: str, readings: set[Reading]) -> None:
        """Seed a recovered series once, without publishing or charging its history."""
        key = origin, account
        if key not in self._series:
            self._series[key] = _Series(set(readings), _declines(readings))

    def observe(
        self, readings: Iterable[Reading], *, origins: Mapping[str, int] | None = None
    ) -> None:
        """Charge new evidence to its execution, even when settlement delivers it late."""
        for account, group in itertools.groupby(sorted(readings), key=lambda r: r.account):
            origin = origins[account] if origins is not None else None
            series = self._series.setdefault((origin, account), _Series(set(), decimal.Decimal(0)))
            new = set(group) - series.seen
            if new:
                series.seen.update(new)
                series.spent = _declines(series.seen) - series.banked
                self._unpublished.update((origin, r) for r in new)

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
