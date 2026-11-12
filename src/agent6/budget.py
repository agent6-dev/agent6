# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Meter one execution's spend and stop it hard at a cap.

A tracker is created per execution and never persists, so a resume gets a full
ceiling again: a circuit breaker against runaway spend, not a ledger across a
task. Once a ledger crosses its cap the next provider call raises
`BudgetExceededError`, and the process exits with its own exit code.

Every call is bounded in one currency. A call the meter can price (a reported
cost, else price times tokens at the model's fetched rates, cache tokens
included) counts against `max_usd`; a call carrying a plan reading counts
percentage points against `max_plan_points`; a call with neither counts its tokens
against `max_tokens_fallback`. Each cap: -1 unlimited, 0 refuse that ledger, >
0 the cap.

A parent that dispatches lanes reserves what it has left (`reserve`): each lane's share is
its own cap, held until the lane settles (`settle_lane`) with its actual usage, charged
whole, its plan readings folded through the same high-water rule as the parent's own.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from agent6 import kinds
from agent6.models import pricing

# No static price table: an unknown price is honest, an outdated hardcoded one is wrong.


# A reported reset time that moves later by more than this is a new window; the backend's
# times jitter by a second between readings.
RESET_SLACK_S = 60.0


class BudgetExceededError(Exception):
    """A configured limit is exceeded; raised by `BudgetTracker.check`."""


@dataclasses.dataclass(frozen=True, slots=True)
class PlanWindow:
    """One rate-limit window of a subscription plan.

    Attributes:
        name: The backend's label: `primary`, `secondary`, a per-model family.
        used_percent: Percent of the included allowance used.
        window_minutes: The window's length; 0 when the backend gave none.
        resets_at: When the window resets, as a Unix time.
    """

    name: str
    used_percent: float
    window_minutes: int
    resets_at: float


# Purchased credits sell in 1,000-credit packs at $40.
_CREDITS_PER_USD = 25.0


@dataclasses.dataclass(frozen=True, slots=True)
class PlanUsage:
    """One plan-usage reading from a subscription provider.

    Every percent-of-plan reading means the binding window, the one closest to its cap.

    Attributes:
        windows: Every window the backend reported, the primary first.
        has_credits: The account holds purchased credits, drawn on past the included
            window: real money.
        credits_unlimited: The credits are unlimited.
        credits_balance: The balance as the backend sent it.
        limit_reached: The backend's own verdict that a window is exhausted.
    """

    windows: tuple[PlanWindow, ...]
    has_credits: bool = False
    credits_unlimited: bool = False
    credits_balance: str = ""
    limit_reached: bool = False

    @classmethod
    def single(
        cls,
        used_percent: float,
        window_minutes: int,
        resets_at: float,
        *,
        secondary_used_percent: float | None = None,
        **rest: Any,
    ) -> PlanUsage:
        """Return a reading with one primary window and an optional secondary.

        Args:
            used_percent: The primary window's used percent.
            window_minutes: The primary window's length.
            resets_at: When the primary window resets.
            secondary_used_percent: The secondary window's used percent, when reported.
            **rest: The remaining fields.

        Returns:
            The reading.
        """
        windows = [PlanWindow("primary", used_percent, window_minutes, resets_at)]
        if secondary_used_percent is not None:
            windows.append(PlanWindow("secondary", secondary_used_percent, 0, resets_at))
        return cls(windows=tuple(windows), **rest)

    @property
    def binding(self) -> PlanWindow:
        """The window closest to its cap: the one that stops the next call."""
        return max(self.windows, key=lambda w: w.used_percent)

    @property
    def used_percent(self) -> float:
        """The binding window's used percent."""
        return self.binding.used_percent

    @property
    def window_minutes(self) -> int:
        """The binding window's length."""
        return self.binding.window_minutes

    @property
    def resets_at(self) -> float:
        """When the binding window resets."""
        return self.binding.resets_at

    @property
    def window_exhausted(self) -> bool:
        """Whether the next call draws past the included allowance."""
        return self.limit_reached or self.used_percent >= 100.0

    @property
    def credits_usd(self) -> float | None:
        """The purchased-credit balance in dollars; a bare number is credits, converted."""
        raw = self.credits_balance.strip()
        if not raw:
            return None
        try:
            if raw.startswith("$"):
                amount = float(raw.lstrip("$").replace(",", ""))
            else:
                amount = float(raw.replace(",", "")) / _CREDITS_PER_USD
        except ValueError:
            return None
        return amount if math.isfinite(amount) and amount >= 0 else None


@dataclasses.dataclass(slots=True)
class ModelUsage:
    """One route's usage totals: the tracker's live counters, and a snapshot row when copied.

    Attributes:
        input_tokens: Input tokens over every call.
        output_tokens: Output tokens over every call.
        cache_read_tokens: Cache-read tokens over every call.
        cache_write_tokens: Cache-write tokens over every call.
        calls: How many calls.
        reported_cost_usd: The sum of provider-reported per-call cost, authoritative
            for the calls that carried one.
        reported_calls: How many calls carried a reported cost or a plan reading.
        unreported_input_tokens: Input tokens of the calls with no reported cost; the
            price table covers exactly this bucket, so no call is priced twice.
        unreported_output_tokens: Output tokens of those calls.
        unreported_cache_read_tokens: Cache-read tokens of those calls.
        unreported_cache_write_tokens: Cache-write tokens of those calls.
        percent_metered: A plan reading metered a call under this route.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0
    reported_cost_usd: float = 0.0
    reported_calls: int = 0
    unreported_input_tokens: int = 0
    unreported_output_tokens: int = 0
    unreported_cache_read_tokens: int = 0
    unreported_cache_write_tokens: int = 0
    percent_metered: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _ModelCost:
    """One route's resolved cost.

    Attributes:
        usd: The figure.
        reported: Provider-reported dollars fed it.
        estimated: A price-table estimate fed it.
        partial: Some calls were priced by neither, so the figure is a lower bound.
        cache_assumed: Cache tokens were priced by the multipliers, not a listed rate.
    """

    usd: float
    reported: bool
    estimated: bool
    partial: bool = False
    cache_assumed: bool = False


def format_usd(usd: float, *, partial: bool = False) -> str:
    """Return a dollar figure as every surface prints it.

    Args:
        usd: The figure.
        partial: The figure is a known under-estimate, marked with a leading "~".

    Returns:
        Cents from one cent up, four decimals below it, so a sub-cent spend is never $0.00.
    """
    mark = "~" if partial else ""
    return f"{mark}${usd:.2f}" if usd >= 0.01 else f"{mark}${usd:.4f}"


def _billed_apart_from_plan(t: ModelUsage) -> bool:
    """Return whether a plan-metered bucket also holds calls that cost money.

    A route can carry both plan readings and dollar charges.

    Args:
        t: The bucket.

    Returns:
        True when it holds a reported cost or unreported tokens.
    """
    return bool(
        t.reported_cost_usd
        or t.unreported_input_tokens
        or t.unreported_output_tokens
        or t.unreported_cache_read_tokens
        or t.unreported_cache_write_tokens
    )


def _model_cost_usd(route: kinds.ModelRoute, t: ModelUsage) -> _ModelCost | None:
    """Return one route's cost, the one owner of the pricing arithmetic.

    The enforced ceiling and the printed summary both read this. Reported cost is
    authoritative for the calls that carried it; the price table prices only the
    unreported calls' tokens. A cache write is priced at the listed rate, else 1.25
    times input; cache reads at the listed rate, else 0.1 times input.

    Args:
        route: The provider and model billed.
        t: Its usage totals.

    Returns:
        The cost, or None when the model has no cached price and reported nothing.
    """
    if t.percent_metered and not _billed_apart_from_plan(t):
        # Included-plan calls are an authoritative $0, never table-priced.
        return _ModelCost(0.0, reported=True, estimated=False)
    reported = t.reported_calls > 0
    price = pricing.lookup_price(route.model, route.provider)
    if price is None:
        if reported:
            # Dropping the reported dollars would zero real spend out of the USD cap.
            return _ModelCost(
                t.reported_cost_usd,
                reported=True,
                estimated=False,
                partial=t.reported_calls < t.calls,
            )
        return None
    write_rate = price.cache_write if price.cache_write is not None else price.input * 1.25
    read_rate = price.cache_read if price.cache_read is not None else price.input * 0.1
    cache_tokens = t.unreported_cache_write_tokens + t.unreported_cache_read_tokens
    in_usd = t.unreported_input_tokens * price.input / 1e6
    cache_write_usd = t.unreported_cache_write_tokens * write_rate / 1e6
    cache_read_usd = t.unreported_cache_read_tokens * read_rate / 1e6
    out_usd = t.unreported_output_tokens * price.output / 1e6
    estimate = in_usd + cache_write_usd + cache_read_usd + out_usd
    return _ModelCost(
        t.reported_cost_usd + estimate,
        reported=reported,
        estimated=estimate > 0.0,
        cache_assumed=cache_tokens > 0 and (price.cache_read is None or price.cache_write is None),
    )


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
class UsageSettlement:
    """What a lane used, as its parent charges it.

    Attributes:
        usd: The lane's metered spend, charged whole.
        usd_partial: Some of the lane's calls were unpriced, so `usd` is a lower bound.
        tokens: The lane's fallback-ledger tokens.
        plan_readings: The provider entry and reading of each plan observation the lane
            made, oldest first: account-global movement the parent folds, not spend.
    """

    usd: float
    usd_partial: bool
    tokens: int
    plan_readings: tuple[tuple[str, PlanUsage], ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class PlanSpend:
    """One subscription plan's latest reading and observed window movement.

    Attributes:
        usage: The latest reading.
        observed_points: Accumulated window movement in percentage points, including other
            consumers.
    """

    usage: PlanUsage
    observed_points: float


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """A point-in-time copy of a tracker's counters.

    Attributes:
        input_total: Input tokens over every call.
        output_total: Output tokens over every call.
        cache_read_total: Cache-read tokens over every call.
        cache_write_total: Cache-write tokens over every call.
        unmetered_tokens: Tokens of the calls the fallback ledger counts.
        max_usd: The USD cap.
        max_tokens_fallback: The fallback token cap.
        max_plan_points: The plan percentage-point cap.
        plans: One entry per provider entry that reported a plan, the latest last.
        exhausted: A ledger crossed its cap.
        exhausted_reason: Why, or "".
        per_model: The usage totals by provider/model route.
        usd_running_lanes: What the lanes still running have spent so far; outside every
            total here until each settles.
        tokens_running_lanes: Their fallback tokens so far, the same way.
        available: What is left to spend in each currency, running lanes taken out, with the
            cap's meaning.
        settled_lanes: Child sessions charged to this execution's cumulative totals.
    """

    input_total: int
    output_total: int
    cache_read_total: int
    cache_write_total: int
    unmetered_tokens: int
    max_usd: float
    max_tokens_fallback: int
    max_plan_points: float
    plans: dict[str, PlanSpend]
    exhausted: bool
    exhausted_reason: str
    per_model: dict[kinds.ModelRoute, ModelUsage]
    usd_running_lanes: float
    tokens_running_lanes: int
    available: BudgetAllowance
    settled_lanes: frozenset[str] = frozenset()

    @property
    def plan_latest(self) -> PlanUsage | None:
        """The last plan reading any provider reported."""
        return next(reversed(self.plans.values())).usage if self.plans else None

    @property
    def plan_observed_points(self) -> float:
        """The movement observed this run on the plan that moved most, what the cap bounds."""
        return max((spend.observed_points for spend in self.plans.values()), default=0.0)


@dataclasses.dataclass(slots=True)
class BudgetTracker:
    """The thread-safe spend accumulator; every call is bounded in one currency.

    The call that brings a ledger to or over its cap trips `BudgetExceededError` on
    the next `check`, so one call may cross the line and no further call is issued.
    `max_plan_points` meters observed movement: the rise in the account's reported used percent
    across this run's readings, accumulated across window resets; a window counts as reset
    when its reported reset time moves later by more than `RESET_SLACK_S`, and a lower
    reading whose reset time has not moved is out of order and counts nothing. The reading
    is account-global, so it includes other runs' spend. `settle` reads each metered plan
    once more, so the run's last calls count.

    Attributes:
        max_usd: The USD cap; required, so the tracker never carries its own default.
        max_tokens_fallback: The fallback token cap.
        max_plan_points: The plan percentage-point cap.
        allow_plan_overage: Whether a call may use purchased credits or extra usage after
            the included plan window is spent; an omitted value is the safe one.
    """

    max_usd: float
    max_tokens_fallback: int
    max_plan_points: float
    allow_plan_overage: bool = False
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _per_model: dict[kinds.ModelRoute, ModelUsage] = dataclasses.field(default_factory=dict)
    _input_total: int = 0
    _output_total: int = 0
    _cache_read_total: int = 0
    _cache_write_total: int = 0
    _unmetered_tokens: int = 0
    _exceeded_reason: str = ""
    # Provider entry to its latest plan reading, the most recently reported last.
    _plans: dict[str, PlanUsage] = dataclasses.field(default_factory=dict)
    # Per (provider entry, window): the high-water reading, and the movement observed this run.
    _plan_last_window: dict[tuple[str, str], PlanWindow] = dataclasses.field(default_factory=dict)
    _plan_observed_by_window: dict[tuple[str, str], float] = dataclasses.field(default_factory=dict)
    # Provider entry to its plan reader, the request the preflight makes.
    _plan_readers: dict[str, Callable[[], PlanUsage | None]] = dataclasses.field(
        default_factory=dict
    )
    _settled: bool = False
    # Purchased credits seen leaving each account this run, in dollars; folds into the USD meter.
    _credits_last_usd: dict[str, float] = dataclasses.field(default_factory=dict)
    _credits_spent_usd: float = 0.0
    # The spend the settled lanes' usage charged, and each running lane's last reading.
    _children_usd: float = 0.0
    _charged_lanes: set[str] = dataclasses.field(default_factory=set)
    _running: dict[str, UsageSettlement] = dataclasses.field(default_factory=dict)

    def record(
        self,
        *,
        route: kinds.ModelRoute,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        cost_usd: float | None = None,
        plan_usage: PlanUsage | None = None,
    ) -> None:
        """Add one provider response's usage to the running totals.

        Args:
            route: The provider and model billed.
            input_tokens: The call's input tokens.
            output_tokens: The call's output tokens.
            cache_read_tokens: The call's cache-read tokens.
            cache_write_tokens: The call's cache-write tokens.
            cost_usd: The provider-reported cost, including zero; None uses a table price,
                or the fallback ledger when there is none.
            plan_usage: A plan reading, which meters the call in percentage points at an
                authoritative $0; a call that would use purchased credits or extra usage
                refuses unless `allow_plan_overage` is set.
        """
        with self._lock:
            # A negative count from a gateway would subtract from the ledger, so signs are clamped.
            input_tokens = max(input_tokens, 0)
            output_tokens = max(output_tokens, 0)
            cache_read_tokens = max(cache_read_tokens, 0)
            cache_write_tokens = max(cache_write_tokens, 0)
            if cost_usd is not None:
                cost_usd = max(cost_usd, 0.0)
            totals = self._per_model.setdefault(route, ModelUsage())
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.cache_read_tokens += cache_read_tokens
            totals.cache_write_tokens += cache_write_tokens
            totals.calls += 1
            if plan_usage is not None:
                totals.reported_calls += 1
                totals.percent_metered = True
                self._note_plan_usage(route.provider, plan_usage)
            elif cost_usd is not None:
                totals.reported_cost_usd += cost_usd
                totals.reported_calls += 1
            else:
                totals.unreported_input_tokens += input_tokens
                totals.unreported_output_tokens += output_tokens
                totals.unreported_cache_read_tokens += cache_read_tokens
                totals.unreported_cache_write_tokens += cache_write_tokens
            self._input_total += input_tokens
            self._output_total += output_tokens
            self._cache_read_total += cache_read_tokens
            self._cache_write_total += cache_write_tokens
            metered = (
                plan_usage is not None
                or cost_usd is not None
                or pricing.lookup_price(route.model, route.provider) is not None
            )
            if not metered:
                self._unmetered_tokens += input_tokens + output_tokens
            if plan_usage is None and metered and self.max_usd == 0.0:
                self._exceeded_reason = (
                    "USD budget is 0: metered calls are refused"
                    f" (route {route.spec!r} is priced; raise [budget].max_usd)"
                )
            if not metered and self.max_tokens_fallback == 0:
                self._exceeded_reason = (
                    f"unmetered call refused: route {route.spec!r} has no reported cost and no"
                    " price data, and [budget].max_tokens_fallback is 0"
                )
            else:
                self._check_fallback_locked()
            self._check_ceilings(route.provider, plan_usage)

    def _check_fallback_locked(self) -> None:
        """Trip the fallback-token ledger at its cap, with the lock held."""
        used = self._unmetered_tokens + self._running_tokens_locked()
        if self.max_tokens_fallback > 0 and used >= self.max_tokens_fallback:
            self._exceeded_reason = (
                f"fallback token budget exhausted: {used} unmetered"
                f" tokens >= {self.max_tokens_fallback}"
            )

    def _running_usd_locked(self) -> float:
        """Return what the running lanes have spent so far, with the lock held."""
        return math.fsum(max(usage.usd, 0.0) for usage in self._running.values())

    def _running_tokens_locked(self) -> int:
        """Return the running lanes' fallback tokens so far, with the lock held."""
        return sum(max(usage.tokens, 0) for usage in self._running.values())

    def available(self) -> BudgetAllowance:
        """Return what is left to spend in each currency, running lanes' spend taken out.

        A lane starts under this as its own caps. Lanes share it: the run stops them once
        their spend together reaches a cap, so the total can pass it by what was in flight.

        Returns:
            The allowance, each currency with its cap's meaning: -1 unlimited, 0 nothing.
        """
        with self._lock:
            return self._available_locked()

    def _available_locked(self) -> BudgetAllowance:
        usd = self.max_usd
        if usd > 0.0:
            spent = self._estimate_usd_locked()[0] + self._running_usd_locked()
            usd = max(0.0, usd - spent)
        tokens = self.max_tokens_fallback
        if tokens > 0:
            tokens = max(0, tokens - self._unmetered_tokens - self._running_tokens_locked())
        points = self.max_plan_points
        if points > 0.0:
            points = max(0.0, points - self._plan_observed_points())
        return BudgetAllowance(usd=usd, tokens=tokens, plan_points=points)

    def note_running(self, child: str, usage: UsageSettlement) -> None:
        """Count a running lane's spend so far against the caps, replacing its last reading.

        The reading is the lane's cumulative usage, its own running lanes included, so each
        call replaces the one before. Its plan readings fold through the window rule the
        run's own readings use.

        Args:
            child: The lane's session id.
            usage: What the lane has used so far.
        """
        with self._lock:
            if child in self._charged_lanes:
                return
            self._running[child] = usage
            provider = ""
            for provider, plan in usage.plan_readings:
                self._note_plan_usage(provider, plan)
            self._check_fallback_locked()
            self._check_ceilings(provider)

    def mark_settled(self, children: Iterable[str]) -> None:
        """Take lanes an earlier execution of the session charged as settled.

        Their usage is inside the session's banked spend, so this tracker charges none of it.

        Args:
            children: The lanes' session ids.
        """
        with self._lock:
            self._charged_lanes.update(children)

    def settle_lane(self, child: str, usage: UsageSettlement | None) -> None:
        """Charge a lane's final usage once; unknown usage leaves its last reading counted.

        Idempotent per lane: a second settlement changes nothing. The plan readings fold
        through the window rule the run's own readings use, so movement two lanes both saw
        counts once.

        Args:
            child: The lane's session id.
            usage: What the lane used; None when its record never came back.
        """
        with self._lock:
            if child in self._charged_lanes or usage is None:
                return
            self._running.pop(child, None)
            self._children_usd += max(usage.usd, 0.0)
            self._unmetered_tokens += max(usage.tokens, 0)
            self._charged_lanes.add(child)
            provider = ""
            for provider, plan in usage.plan_readings:
                self._note_plan_usage(provider, plan)
            self._check_fallback_locked()
            self._check_ceilings(provider)

    def _plan_observed_points(self, route: str = "") -> float:
        """Return the movement observed on the binding window this run, one route when given."""
        return self._plan_observation_binding(route)[2]

    def _plan_observation_binding(self, route: str = "") -> tuple[str, str, float]:
        """Return the provider entry, window and observed points nearest the cap."""
        return max(
            (
                (entry, window, points)
                for (entry, window), points in self._plan_observed_by_window.items()
                if not route or entry == route
            ),
            key=lambda row: row[2],
            default=("", "", 0.0),
        )

    def _note_plan_usage(self, route: str, plan: PlanUsage) -> None:
        """Fold one reading into the window's observed movement and the credits into the USD meter.

        A window's first reading is its baseline. A rise within the window is movement. A
        reset time later by more than `RESET_SLACK_S` is a new window, whose reading counts
        whole. A reading out of order (another seat read the account first) counts nothing:
        a lower percent on the same window, or any reading from a window already left. A
        credit balance that fell is money spent. The lock is held.

        Args:
            route: The provider entry the reading came from.
            plan: The reading.
        """
        for w in plan.windows:
            key = (route, w.name)
            last = self._plan_last_window.get(key)
            if last is None:
                self._plan_last_window[key] = w
                continue
            if last.resets_at - w.resets_at > RESET_SLACK_S:
                continue
            if w.resets_at - last.resets_at > RESET_SLACK_S:
                delta = w.used_percent
            elif w.used_percent >= last.used_percent:
                delta = w.used_percent - last.used_percent
            else:
                continue
            self._plan_observed_by_window[key] = self._plan_observed_by_window.get(key, 0.0) + delta
            self._plan_last_window[key] = w
        balance = plan.credits_usd
        if balance is not None:
            previous = self._credits_last_usd.get(route)
            if previous is not None and balance < previous:
                self._credits_spent_usd += previous - balance
            self._credits_last_usd[route] = balance
        self._plans.pop(route, None)
        self._plans[route] = plan

    def _check_ceilings(self, provider: str, plan_usage: PlanUsage | None = None) -> None:
        """Check the combined USD and plan-point ledgers with the lock held."""
        would_spend_credits = (
            plan_usage is not None
            and plan_usage.has_credits
            and not plan_usage.credits_unlimited
            and plan_usage.window_exhausted
        )
        if plan_usage is not None and would_spend_credits and not self.allow_plan_overage:
            balance = plan_usage.credits_balance or "unknown"
            usd = plan_usage.credits_usd
            if usd is not None and not plan_usage.credits_balance.strip().startswith("$"):
                balance = f"{balance} credits (~${usd:.2f})"
            self._exceeded_reason = (
                "the plan window is exhausted and the account holds purchased"
                f" credits (balance {balance}): continuing would spend them."
                " Set [budget].allow_plan_overage = true to allow that"
            )
        elif would_spend_credits and self.max_usd == 0.0:
            self._exceeded_reason = (
                "USD budget is 0: purchased-credit calls are refused (raise [budget].max_usd)"
            )
        elif plan_usage is not None and self.max_plan_points == 0.0:
            self._exceeded_reason = (
                "percent budget is 0: plan-metered calls are refused"
                f" (provider {provider!r} draws on a subscription plan;"
                " raise [budget].max_plan_points)"
            )
        elif self.max_plan_points > 0.0 and self._plan_observed_points() >= self.max_plan_points:
            route, window, observed = self._plan_observation_binding()
            reading = self._plan_last_window[(route, window)]
            windows = tuple(w for (r, _), w in self._plan_last_window.items() if r == route)
            self._exceeded_reason = (
                f"plan budget exhausted: the plan moved ~{observed:.1f}"
                f" percentage points >= max_plan_points {self.max_plan_points:g}"
                f" ({route}: account at {reading.used_percent:g}% on its"
                f" {format_plan_window(reading, windows)})"
            )
        elif self.max_usd > 0.0:
            running = self._running_usd_locked()
            cost = self._estimate_usd_locked()[0] + running
            if cost >= self.max_usd:
                lanes = f", {format_usd(running)} of it in running lanes" if running else ""
                self._exceeded_reason = (
                    f"USD budget exhausted: ~{format_usd(cost)} >= {format_usd(self.max_usd)}"
                    f" (includes cache_read/cache_write cost and purchased credits{lanes})"
                )

    def register_plan_reader(self, provider: str, reader: Callable[[], PlanUsage | None]) -> None:
        """Keep the request that reads a provider's plan, for `settle`.

        Args:
            provider: The provider entry.
            reader: Returns a fresh reading, or None when the plan cannot be read.
        """
        with self._lock:
            self._plan_readers[provider] = reader

    def settle(self) -> bool:
        """Read each metered plan once more, so the run's last calls count.

        A reading otherwise comes from a response header or the preflight, before the
        calls it meters finish. One read per plan per tracker; a plan with no metered
        call is not read. The requests run outside the lock.

        Returns:
            Whether a reading landed.
        """
        with self._lock:
            if self._settled:
                return False
            self._settled = True
            metered = {route for route, _window in self._plan_last_window}
            readers = [
                (provider, reader)
                for provider, reader in self._plan_readers.items()
                if provider in metered
            ]
        landed = False
        for provider, reader in readers:
            plan = reader()
            if plan is not None:
                landed = True
                with self._lock:
                    self._note_plan_usage(provider, plan)
                    self._check_ceilings(provider, plan)
        return landed

    def record_plan_preflight(self, provider: str, plan: PlanUsage) -> None:
        """Record a reading taken before the first plan-metered call.

        It is the baseline every later delta counts from, and the credit guard's first
        look, so a run that would draw on purchased credits refuses before its first call.

        Args:
            provider: The provider entry.
            plan: The reading.
        """
        with self._lock:
            self._note_plan_usage(provider, plan)
            self._check_ceilings(provider, plan)

    def check(self) -> None:
        """Stop the next call when a ledger update crossed a ceiling.

        Raises:
            BudgetExceededError: A ledger is over its cap.
        """
        with self._lock:
            reason = self._exceeded_reason
        if reason:
            raise BudgetExceededError(reason)

    def is_exhausted(self) -> bool:
        """Return whether a ledger crossed its cap."""
        with self._lock:
            return bool(self._exceeded_reason)

    def fraction_remaining(self) -> float:
        """Return the fraction of the budget still available, against the nearest ceiling.

        A run that burned 90% of one ceiling and 10% of another reports 0.10, the
        figure the harness winds down on. An unlimited or refuse cap contributes
        nothing.

        Returns:
            A number in [0.0, 1.0].
        """
        with self._lock:
            if self._exceeded_reason:
                return 0.0
            used = 0.0
            if self.max_usd > 0.0:
                usd_spent = self._estimate_usd_locked()[0] + self._running_usd_locked()
                used = max(used, usd_spent / self.max_usd)
            if self.max_tokens_fallback > 0:
                tokens = self._unmetered_tokens + self._running_tokens_locked()
                used = max(used, tokens / self.max_tokens_fallback)
            if self.max_plan_points > 0.0:
                used = max(used, self._plan_observed_points() / self.max_plan_points)
        return max(0.0, 1.0 - used)

    def snapshot(self) -> BudgetSnapshot:
        """Return a point-in-time copy of every counter."""
        with self._lock:
            per_model = {
                route: dataclasses.replace(t)
                for route, t in sorted(self._per_model.items(), key=lambda row: row[0].spec)
            }
            return BudgetSnapshot(
                input_total=self._input_total,
                output_total=self._output_total,
                cache_read_total=self._cache_read_total,
                cache_write_total=self._cache_write_total,
                unmetered_tokens=self._unmetered_tokens,
                max_usd=self.max_usd,
                max_tokens_fallback=self.max_tokens_fallback,
                max_plan_points=self.max_plan_points,
                plans={
                    route: PlanSpend(plan, self._plan_observed_points(route))
                    for route, plan in self._plans.items()
                },
                exhausted=bool(self._exceeded_reason),
                exhausted_reason=self._exceeded_reason,
                per_model=per_model,
                usd_running_lanes=self._running_usd_locked(),
                tokens_running_lanes=self._running_tokens_locked(),
                available=self._available_locked(),
                settled_lanes=frozenset(self._charged_lanes),
            )

    def estimate_usd(self) -> tuple[float, bool]:
        """Estimate the cumulative USD spend over every recorded call.

        The live cost meter, the enforced ceiling and the summary's total all read this.

        Returns:
            The total, and whether any call could not be priced, which makes it a lower
            bound.
        """
        with self._lock:
            return self._estimate_usd_locked()

    def _estimate_usd_locked(self) -> tuple[float, bool]:
        """Estimate the USD spend with the lock held.

        Returns:
            The total, and whether any call could not be priced.
        """
        total_usd = self._credits_spent_usd + self._children_usd
        any_unknown = False
        for route, t in self._per_model.items():
            cost = _model_cost_usd(route, t)
            if cost is None:
                any_unknown = True
                continue
            any_unknown = any_unknown or cost.partial
            total_usd += cost.usd
        return total_usd, any_unknown

    def _model_lines(self, snap: BudgetSnapshot) -> tuple[list[str], int, int, bool]:
        """Return one summary line per provider/model route.

        Args:
            snap: The counters to print.

        Returns:
            The lines, how many routes were priced, how many the USD ledger meters (a
            plan's calls cost dollars nowhere), and whether any figure is an estimate.
        """
        lines: list[str] = []
        priced = 0
        metered = 0
        any_estimated = False
        for route, totals in snap.per_model.items():
            cost = _model_cost_usd(route, totals)
            subscription = totals.percent_metered and not _billed_apart_from_plan(totals)
            if cost is None:
                cost_str = "$? (unknown price)"
            else:
                priced += 1
                metered += 0 if subscription else 1
                any_estimated = any_estimated or cost.estimated
                if cost.partial:
                    note = " (reported, some calls unpriced)"
                elif cost.reported and cost.estimated:
                    note = " (reported + estimated)"
                elif subscription:
                    note = " (subscription)"
                elif cost.reported:
                    note = " (reported)"
                else:
                    note = ""
                if cost.cache_assumed:
                    note += " (cache rates assumed: 0.1x read, 1.25x write)"
                cost_str = f"{format_usd(cost.usd)}{note}"
            lines.append(
                f"  {route.spec}: "
                f"in={totals.input_tokens} out={totals.output_tokens} "
                f"cache_r={totals.cache_read_tokens} "
                f"cache_w={totals.cache_write_tokens} "
                f"calls={totals.calls} {cost_str}"
            )
        return lines, priced, metered, any_estimated

    def format_summary(self, *, include_stop_reason: bool = True) -> str:
        """Return the end-of-run token and cost summary.

        Args:
            include_stop_reason: Print the budget stop unless the caller already printed it.

        Returns:
            The summary.

        Attributes:
            route: Each usage line starts with the provider/model route, such as
                `chatgpt/gpt-6-astra`.
            in: Fresh input tokens for the route, or all routes on the TOTAL line.
            out: Output tokens for the route, or all routes on the TOTAL line.
            cache_r: Cache-read tokens for the route.
            cache_w: Cache-write tokens for the route.
            calls: Calls recorded for the route.
            cost: USD spend, reported or estimated from that route's prices.
                The TOTAL includes purchased credits spent.
                `=` marks an exact total, `~` an estimate or lower bound, and `+` unpriced calls.
            of: The USD cap when a route is USD-metered; `unlimited` means no cap.
            unmetered: Tokens counted against the fallback cap, followed by that cap.
            plan: One plan usage line per provider, with its binding window's used percent,
                observed window movement in points, optional cap, reset time and credit balance.
                Account-wide movement includes other consumers.
            STATUS: The budget stop reason, when exhausted and requested by the caller.
        """
        snap = self.snapshot()
        lines = ["Token + cost summary:"]
        # The total is the figure the ceiling enforces, so it carries purchased credits spent.
        total_usd, any_unknown = self.estimate_usd()
        model_lines, priced, metered, any_estimated = self._model_lines(snap)
        lines.extend(model_lines)
        any_unknown = any_unknown or priced < len(snap.per_model)
        approx = "~" if any_unknown or any_estimated else "="
        total = format_usd(total_usd) + ("+" if any_unknown else "")
        # `of <cap>` names what meters this spend; with nothing USD-metered, max_usd meters none.
        cap = ""
        if metered:
            usd_cap = "unlimited" if snap.max_usd == -1 else format_usd(snap.max_usd)
            cap = f" of {usd_cap}"
        budget_line = (
            f"  TOTAL: in={snap.input_total} out={snap.output_total} cost{approx}{total}{cap}"
        )
        if snap.unmetered_tokens:
            fb_cap = (
                "unlimited"
                if snap.max_tokens_fallback == -1
                else f"{snap.max_tokens_fallback:,}"  # as the preflight NOTE prints it
            )
            budget_line += f" (unmetered: {snap.unmetered_tokens}/{fb_cap} fallback tokens)"
        if any_unknown:
            budget_line += " (some models unpriced; figure is a lower bound)"
        lines.append(budget_line)
        if snap.usd_running_lanes or snap.tokens_running_lanes:
            parts = [format_usd(snap.usd_running_lanes)] if snap.usd_running_lanes else []
            if snap.tokens_running_lanes:
                parts.append(f"{snap.tokens_running_lanes} fallback tokens")
            lines.append("  in lanes still running: " + ", ".join(parts))
        for route, spend in snap.plans.items():
            lines.append("  " + format_plan_usage(route, spend, snap.max_plan_points))
        if snap.exhausted and include_stop_reason:
            lines.append(f"  STATUS: {snap.exhausted_reason}")
        return "\n".join(lines)


def format_plan_window(window: PlanWindow, windows: tuple[PlanWindow, ...]) -> str:
    """Label a window by duration, distinguishing equal durations by their keys' suffixes.

    Args:
        window: The window to label.
        windows: The provider's reported windows, including this window.

    Returns:
        The duration with a distinguishing suffix when durations collide.
    """
    minutes = window.window_minutes
    if minutes >= 1440:
        duration = f"{minutes / 1440:g}-day "
    elif minutes >= 60:
        duration = f"{minutes / 60:g}-hour "
    elif minutes > 0:
        duration = f"{minutes}-minute "
    else:
        duration = ""
    names = {w.name for w in windows if w.window_minutes == minutes}
    suffix = ""
    if len(names) > 1:
        parts = [name.split("_") for name in names]
        common = 0
        for words in zip(*parts, strict=False):
            if len(set(words)) != 1:
                break
            common += 1
        distinguishing = "_".join(window.name.split("_")[common:])
        if distinguishing:
            suffix = f" ({distinguishing})"
    return f"{duration}window{suffix}"


def format_plan_reading(route: str, usage: PlanUsage) -> str:
    """Return a plan's reading with no observed movement, as a group prints its lanes' newest.

    Args:
        route: The provider entry's name.
        usage: The reading.

    Returns:
        The line.
    """
    resets_h = max(0.0, (usage.resets_at - time.time()) / 3600)
    return (
        f"plan usage ({route}): {usage.used_percent:g}% of the"
        f" {format_plan_window(usage.binding, usage.windows)} (resets in {resets_h:.0f}h)"
    )


def format_plan_usage(route: str, spend: PlanSpend, max_plan_points: float) -> str:
    """Return the plan-usage line every surface prints for one provider entry.

    Args:
        route: The provider entry.
        spend: Its latest reading and observed window movement.
        max_plan_points: The cap the observed movement is printed against; -1 prints none.

    Returns:
        The line.
    """
    plan = spend.usage
    cap = "" if max_plan_points == -1 else f" of max_plan_points {max_plan_points:g}"
    resets_h = max(0.0, (plan.resets_at - time.time()) / 3600)
    return (
        f"plan usage ({route}): {plan.used_percent:g}% of the"
        f" {format_plan_window(plan.binding, plan.windows)}"
        f" (window moved ~{spend.observed_points:g} points{cap} during this run;"
        f" resets in {resets_h:.0f}h)"
        + (
            f"; purchased credits balance {plan.credits_balance or 'present'}"
            if plan.has_credits and not plan.credits_unlimited
            else ""
        )
    )
