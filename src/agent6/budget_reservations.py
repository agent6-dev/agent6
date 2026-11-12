# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The allowance a parent splits across its lanes, and the record of what it still holds.

Pure arithmetic over the three currencies; `budget.BudgetTracker` owns the lock, the
ledgers and the plan readings.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetAllowance:
    """One allowance in the three currencies, each with the cap's meaning.

    Attributes:
        usd: The USD cap: -1 unlimited, 0 refuse, > 0 the amount.
        tokens: The fallback-token cap, the same way.
        plan_points: The plan percentage-point cap, the same way.
    """

    usd: float
    tokens: int
    plan_points: float


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetAmount:
    """An amount in the three currencies, never a cap: 0 is none.

    Attributes:
        usd: Dollars.
        tokens: Fallback tokens.
        plan_points: Plan percentage points.
    """

    usd: float
    tokens: int
    plan_points: float


def split_allowance(available: BudgetAllowance, lanes: int) -> tuple[BudgetAllowance, ...]:
    """Split an allowance evenly across lanes, never granting more than was available.

    Unlimited (-1) and refuse (0) pass through per currency. Token remainders go to the
    first lanes in order; the float currencies trim the last share so the shares never add
    up to more than the available amount, and a tiny amount still gives every lane a share.

    Args:
        available: The parent's available allowance.
        lanes: How many lanes share it, at least one.

    Returns:
        One allowance per lane, in lane order.

    Raises:
        ValueError: Fewer than one lane.
    """
    if lanes < 1:
        raise ValueError(f"an allowance splits across at least one lane, not {lanes} lanes")
    usd = _split_float(available.usd, lanes)
    points = _split_float(available.plan_points, lanes)
    tokens = _split_int(available.tokens, lanes)
    return tuple(
        BudgetAllowance(usd=usd[i], tokens=tokens[i], plan_points=points[i]) for i in range(lanes)
    )


def _split_float(amount: float, lanes: int) -> list[float]:
    if amount <= 0:
        return [amount] * lanes
    shares = [amount / lanes] * lanes
    over = math.fsum(shares) - amount
    if over > 0:
        shares[-1] = max(0.0, shares[-1] - over)
    return shares


def _split_int(amount: int, lanes: int) -> list[int]:
    if amount <= 0:
        return [amount] * lanes
    base, remainder = divmod(amount, lanes)
    return [base + (1 if i < remainder else 0) for i in range(lanes)]


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetReservation:
    """A parent's grant to one group of lanes, and which lanes have settled.

    Attributes:
        identity: The dispatch the grant belongs to.
        owner: The execution that made it.
        shares: Each lane's session id and share, in lane order.
        settled: The lanes whose usage was charged and whose share was released.
        outstanding: The lanes whose usage is unknown; their shares stay held.
    """

    identity: str
    owner: str
    shares: tuple[tuple[str, BudgetAllowance], ...]
    settled: frozenset[str] = frozenset()
    outstanding: frozenset[str] = frozenset()

    def share_of(self, child: str) -> BudgetAllowance:
        """Return a lane's share.

        Args:
            child: The lane's session id.

        Returns:
            Its share.

        Raises:
            KeyError: The lane is not in this grant.
        """
        for session_id, share in self.shares:
            if session_id == child:
                return share
        raise KeyError(child)

    def held(self) -> BudgetAmount:
        """Return the unsettled shares summed per currency; an outstanding lane's stays held."""
        usd = tokens = points = 0.0
        for session_id, share in self.shares:
            if session_id in self.settled:
                continue
            usd += max(share.usd, 0.0)
            tokens += max(share.tokens, 0)
            points += max(share.plan_points, 0.0)
        return BudgetAmount(usd=usd, tokens=int(tokens), plan_points=points)

    def settle(self, child: str) -> BudgetReservation:
        """Return the grant with the lane settled and no longer outstanding."""
        return dataclasses.replace(
            self, settled=self.settled | {child}, outstanding=self.outstanding - {child}
        )

    def mark_outstanding(self, child: str) -> BudgetReservation:
        """Return the grant with the lane's usage recorded as unknown."""
        return dataclasses.replace(self, outstanding=self.outstanding | {child})
