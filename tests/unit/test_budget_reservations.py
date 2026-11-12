# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The allowance a parent splits across its lanes, and the record of what it still holds."""

from __future__ import annotations

import math

import pytest

from agent6 import budget_reservations as reservations

A = reservations.BudgetAllowance


def test_split_keeps_unlimited_and_refuse_per_currency() -> None:
    shares = reservations.split_allowance(A(usd=-1.0, tokens=0, plan_points=9.0), 3)
    assert [s.usd for s in shares] == [-1.0, -1.0, -1.0]
    assert [s.tokens for s in shares] == [0, 0, 0]
    assert [s.plan_points for s in shares] == pytest.approx([3.0, 3.0, 3.0])


def test_split_lands_the_token_remainder_in_lane_order() -> None:
    shares = reservations.split_allowance(A(usd=10.0, tokens=10, plan_points=-1.0), 4)
    assert [s.tokens for s in shares] == [3, 3, 2, 2]
    assert [s.usd for s in shares] == pytest.approx([2.5] * 4)


@pytest.mark.parametrize("usd", [0.1, 1 / 3, 2.000001, 1e-9, 7.0])
@pytest.mark.parametrize("lanes", [1, 2, 3, 7, 10])
def test_split_never_rounds_the_sum_above_the_available(usd: float, lanes: int) -> None:
    """Tiny shares stay positive and the shares never add up to more than was available."""
    shares = reservations.split_allowance(A(usd=usd, tokens=1, plan_points=usd), lanes)
    assert len(shares) == lanes
    assert all(s.usd > 0.0 and s.plan_points > 0.0 for s in shares)
    assert math.fsum(s.usd for s in shares) <= usd
    assert math.fsum(s.plan_points for s in shares) <= usd
    assert sum(s.tokens for s in shares) == 1


@pytest.mark.parametrize("lanes", [0, -1])
def test_split_refuses_a_bad_lane_count(lanes: int) -> None:
    with pytest.raises(ValueError, match="lanes"):
        reservations.split_allowance(A(usd=1.0, tokens=1, plan_points=1.0), lanes)


def test_a_reservation_holds_its_unsettled_shares() -> None:
    grant = reservations.BudgetReservation(
        identity="p1",
        owner="exec-1",
        shares=(
            ("a", A(usd=2.0, tokens=5, plan_points=-1.0)),
            ("b", A(usd=3.0, tokens=5, plan_points=-1.0)),
        ),
    )
    assert grant.share_of("b") == A(usd=3.0, tokens=5, plan_points=-1.0)
    assert grant.held() == reservations.BudgetAmount(usd=5.0, tokens=10, plan_points=0.0)
    settled = grant.settle("a")
    assert settled.held() == reservations.BudgetAmount(usd=3.0, tokens=5, plan_points=0.0)
    # An outstanding child's share stays held until its usage is known.
    unknown = settled.mark_outstanding("b")
    assert unknown.held() == settled.held()
    assert unknown.settle("b").held() == reservations.BudgetAmount(0.0, 0, 0.0)
    with pytest.raises(KeyError):
        grant.share_of("zed")
