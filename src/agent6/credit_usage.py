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
    _unpublished: set[Reading] = dataclasses.field(default_factory=set)

    def __len__(self) -> int:
        return len(self._seen)

    def observe(self, readings: Iterable[Reading]) -> None:
        """Count each decline once, including a decline after a top-up."""
        new = set(readings).difference(self._seen)
        if not new:
            return
        self._unpublished.update(new)
        self._seen.update(new)
        self.spent_usd = math.fsum(
            max(earlier.balance_usd - later.balance_usd, 0.0)
            for _, series in itertools.groupby(sorted(self._seen), key=lambda r: r.account)
            for earlier, later in itertools.pairwise(series)
        )

    @property
    def unpublished(self) -> tuple[Reading, ...]:
        """The observations not yet confirmed by a journal append."""
        return tuple(sorted(self._unpublished))

    def published(self, readings: Iterable[Reading]) -> None:
        """Mark only the observations included in a written row as published."""
        self._unpublished.difference_update(readings)
