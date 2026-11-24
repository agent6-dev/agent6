# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Account balance observations used by the USD cap."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable


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


@dataclasses.dataclass(slots=True)
class Ledger:
    """The ordered union of the balances this execution and its children observed."""

    spent_usd: float = 0.0
    _seen: set[Reading] = dataclasses.field(default_factory=set)
    _unpublished: list[Reading] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self._seen)

    def observe(self, readings: Iterable[Reading]) -> None:
        """Count each decline once, including a decline after a top-up."""
        new = set(readings).difference(self._seen)
        if not new:
            return
        self._unpublished.extend(sorted(new))
        self._seen.update(new)
        self.spent_usd = math.fsum(
            max(earlier.balance_usd - later.balance_usd, 0.0)
            for _, series in itertools.groupby(sorted(self._seen), key=lambda r: r.account)
            for earlier, later in itertools.pairwise(series)
        )

    def take_readings(self) -> tuple[Reading, ...]:
        """Return just the observations added since the last budget row."""
        readings = tuple(self._unpublished)
        self._unpublished.clear()
        return readings
