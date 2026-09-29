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
percentage points against `max_percent`; a call with neither counts its tokens
against `max_tokens_fallback`. Each cap: -1 unlimited, 0 refuse that ledger, >
0 the cap.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from typing import Any

from agent6.models import pricing

# No static price table: an unknown price is honest, an outdated hardcoded one is wrong.


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
    """One model's usage totals: the tracker's live counters, and a snapshot row when copied.

    Attributes:
        input_tokens: Input tokens over every call.
        output_tokens: Output tokens over every call.
        cache_read_tokens: Cache-read tokens over every call.
        cache_creation_tokens: Cache-creation tokens over every call.
        calls: How many calls.
        reported_cost_usd: The sum of provider-reported per-call cost, authoritative
            for the calls that carried one.
        reported_calls: How many calls carried a reported cost or a plan reading.
        unreported_input_tokens: Input tokens of the calls with no reported cost; the
            price table covers exactly this bucket, so no call is priced twice.
        unreported_output_tokens: Output tokens of those calls.
        unreported_cache_read_tokens: Cache-read tokens of those calls.
        unreported_cache_creation_tokens: Cache-creation tokens of those calls.
        percent_metered: A plan reading metered a call under this model.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0
    reported_cost_usd: float = 0.0
    reported_calls: int = 0
    unreported_input_tokens: int = 0
    unreported_output_tokens: int = 0
    unreported_cache_read_tokens: int = 0
    unreported_cache_creation_tokens: int = 0
    percent_metered: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _ModelCost:
    """One model's resolved cost.

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

    One model id can reach both a subscription provider and a paid API, and the bucket
    is keyed by the id, so the plan's $0 must not stand for the API dollars.
    """
    return bool(
        t.reported_cost_usd
        or t.unreported_input_tokens
        or t.unreported_output_tokens
        or t.unreported_cache_read_tokens
        or t.unreported_cache_creation_tokens
    )


def _model_cost_usd(model: str, t: ModelUsage, provider: str = "") -> _ModelCost | None:
    """Return one model's cost, the one owner of the pricing arithmetic.

    The enforced ceiling and the printed summary both read this. Reported cost is
    authoritative for the calls that carried it; the price table prices only the
    unreported calls' tokens. Cache creation is priced at the listed rate, else 1.25
    times input; cache reads at the listed rate, else 0.1 times input.

    Args:
        model: The model id.
        t: Its usage totals.
        provider: The provider entry it is routed through.

    Returns:
        The cost, or None when the model has no cached price and reported nothing.
    """
    if t.percent_metered and not _billed_apart_from_plan(t):
        # Included-plan calls are an authoritative $0, never table-priced.
        return _ModelCost(0.0, reported=True, estimated=False)
    reported = t.reported_cost_usd > 0.0
    price = pricing.lookup_price(model, provider)
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
    cache_tokens = t.unreported_cache_creation_tokens + t.unreported_cache_read_tokens
    in_usd = t.unreported_input_tokens * price.input / 1e6
    cache_creation_usd = t.unreported_cache_creation_tokens * write_rate / 1e6
    cache_read_usd = t.unreported_cache_read_tokens * read_rate / 1e6
    out_usd = t.unreported_output_tokens * price.output / 1e6
    estimate = in_usd + cache_creation_usd + cache_read_usd + out_usd
    return _ModelCost(
        t.reported_cost_usd + estimate,
        reported=reported,
        estimated=estimate > 0.0,
        cache_assumed=cache_tokens > 0 and (price.cache_read is None or price.cache_write is None),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PlanSpend:
    """One subscription plan's latest reading and this run's consumption.

    Attributes:
        usage: The latest reading.
        consumed: This run's consumption on the binding window, in percentage points.
    """

    usage: PlanUsage
    consumed: float


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """A point-in-time copy of a tracker's counters.

    Attributes:
        input_total: Input tokens over every call.
        output_total: Output tokens over every call.
        cache_read_total: Cache-read tokens over every call.
        cache_creation_total: Cache-creation tokens over every call.
        unmetered_tokens: Tokens of the calls the fallback ledger counts.
        max_usd: The USD cap.
        max_tokens_fallback: The fallback token cap.
        max_percent: The plan percentage-point cap.
        plans: One entry per provider entry that reported a plan, the latest last.
        exhausted: A ledger crossed its cap.
        exhausted_reason: Why, or "".
        per_model: The usage totals by model id.
    """

    input_total: int
    output_total: int
    cache_read_total: int
    cache_creation_total: int
    unmetered_tokens: int
    max_usd: float
    max_tokens_fallback: int
    max_percent: float
    plans: dict[str, PlanSpend]
    exhausted: bool
    exhausted_reason: str
    per_model: dict[str, ModelUsage]

    @property
    def plan_latest(self) -> PlanUsage | None:
        """The last plan reading any provider reported."""
        return next(reversed(self.plans.values())).usage if self.plans else None

    @property
    def plan_consumed(self) -> float:
        """This run's consumption on the plan that moved most, what `max_percent` caps."""
        return max((spend.consumed for spend in self.plans.values()), default=0.0)


@dataclasses.dataclass(slots=True)
class BudgetTracker:
    """The thread-safe spend accumulator; every call is bounded in one currency.

    The call that brings a ledger to or over its cap trips `BudgetExceededError` on
    the next `check`, so one call may cross the line and no further call is issued.
    `max_percent` meters consumption: the rise in the account's reported used percent
    across this run's readings, accumulated across window resets. The reading is
    account-global, so a concurrent run's spend lands in whichever run reads it next:
    over-counting, never under.

    Attributes:
        max_usd: The USD cap; required, so the tracker never carries its own default.
        max_tokens_fallback: The fallback token cap.
        max_percent: The plan percentage-point cap.
        allow_paid_credits: Whether a call may draw on purchased credits; an omitted
            value is the safe one.
    """

    max_usd: float
    max_tokens_fallback: int
    max_percent: float
    allow_paid_credits: bool = False
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _per_model: dict[str, ModelUsage] = dataclasses.field(default_factory=dict)
    _input_total: int = 0
    _output_total: int = 0
    _cache_read_total: int = 0
    _cache_creation_total: int = 0
    _unmetered_tokens: int = 0
    _exceeded_reason: str = ""
    # Provider entry to its latest plan reading, the most recently reported last.
    _plans: dict[str, PlanUsage] = dataclasses.field(default_factory=dict)
    # Per (provider entry, window): the last reading, and this run's consumption sawtooth.
    _plan_last_percent: dict[tuple[str, str], float] = dataclasses.field(default_factory=dict)
    _plan_consumed_by_window: dict[tuple[str, str], float] = dataclasses.field(default_factory=dict)
    # Purchased credits seen leaving each account this run, in dollars; folds into the USD meter.
    _credits_last_usd: dict[str, float] = dataclasses.field(default_factory=dict)
    _credits_spent_usd: float = 0.0
    # Model id to the provider entry that bills it, so a model two providers list is priced right.
    _routes: dict[str, str] = dataclasses.field(default_factory=dict)

    def note_route(self, model: str, provider: str) -> None:
        """Record which provider entry a model is called through."""
        with self._lock:
            self._routes[model] = provider

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        cost_usd: float = 0.0,
        plan_usage: PlanUsage | None = None,
    ) -> None:
        """Add one provider response's usage to the running totals.

        Args:
            model: The model id.
            input_tokens: The call's input tokens.
            output_tokens: The call's output tokens.
            cache_read_tokens: The call's cache-read tokens.
            cache_creation_tokens: The call's cache-creation tokens.
            cost_usd: The provider-reported cost; 0.0 leaves the call to a table price,
                or to the fallback ledger when there is none.
            plan_usage: A plan reading, which meters the call in percentage points at an
                authoritative $0; a call that would draw on purchased credits refuses
                unless `allow_paid_credits` is set.
        """
        with self._lock:
            # A negative count from a gateway would subtract from the ledger, so signs are clamped.
            input_tokens = max(input_tokens, 0)
            output_tokens = max(output_tokens, 0)
            cache_read_tokens = max(cache_read_tokens, 0)
            cache_creation_tokens = max(cache_creation_tokens, 0)
            cost_usd = max(cost_usd, 0.0)
            totals = self._per_model.setdefault(model, ModelUsage())
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.cache_read_tokens += cache_read_tokens
            totals.cache_creation_tokens += cache_creation_tokens
            totals.calls += 1
            if plan_usage is not None:
                totals.reported_calls += 1
                totals.percent_metered = True
                self._note_plan_usage(self._routes.get(model, model), plan_usage)
            elif cost_usd > 0.0:
                totals.reported_cost_usd += cost_usd
                totals.reported_calls += 1
            else:
                totals.unreported_input_tokens += input_tokens
                totals.unreported_output_tokens += output_tokens
                totals.unreported_cache_read_tokens += cache_read_tokens
                totals.unreported_cache_creation_tokens += cache_creation_tokens
            self._input_total += input_tokens
            self._output_total += output_tokens
            self._cache_read_total += cache_read_tokens
            self._cache_creation_total += cache_creation_tokens
            if plan_usage is not None:
                self._check_plan_ceilings(model, plan_usage)
                return
            metered = (
                cost_usd > 0.0
                or pricing.lookup_price(model, self._routes.get(model, "")) is not None
            )
            if not metered:
                self._unmetered_tokens += input_tokens + output_tokens
            if metered and self.max_usd == 0.0:
                self._exceeded_reason = (
                    "USD budget is 0: metered calls are refused"
                    f" (model {model!r} is priced; raise [budget].max_usd)"
                )
            elif self.max_usd > 0.0:
                cost, _ = self._estimate_usd_locked()
                if cost >= self.max_usd:
                    self._exceeded_reason = (
                        f"USD budget exhausted: ~{format_usd(cost)} >= {format_usd(self.max_usd)}"
                        " (includes cache_read/cache_creation cost)"
                    )
            if not metered and self.max_tokens_fallback == 0:
                self._exceeded_reason = (
                    f"unmetered call refused: model {model!r} has no reported cost and no"
                    " price data, and [budget].max_tokens_fallback is 0"
                )
            elif (
                self.max_tokens_fallback > 0 and self._unmetered_tokens >= self.max_tokens_fallback
            ):
                self._exceeded_reason = (
                    f"fallback token budget exhausted: {self._unmetered_tokens} unmetered"
                    f" tokens >= {self.max_tokens_fallback}"
                )

    def _plan_consumed(self, route: str = "") -> float:
        """Return this run's consumption on its binding window, narrowed to one route when given."""
        return self._plan_consumption_binding(route)[2]

    def _plan_consumption_binding(self, route: str = "") -> tuple[str, str, float]:
        """Return the provider entry, window and consumed points nearest the cap."""
        return max(
            (
                (entry, window, points)
                for (entry, window), points in self._plan_consumed_by_window.items()
                if not route or entry == route
            ),
            key=lambda row: row[2],
            default=("", "", 0.0),
        )

    def _note_plan_usage(self, route: str, plan: PlanUsage) -> None:
        """Fold one reading into the consumption sawtooth and the credit balance into the USD meter.

        A rise since the last reading is consumption; a drop is a window reset, counted
        from zero after it. A window's first reading is its baseline. A credit balance
        that fell is money spent. The lock is held.

        Args:
            route: The provider entry the reading came from.
            plan: The reading.
        """
        for w in plan.windows:
            key = (route, w.name)
            last = self._plan_last_percent.get(key)
            if last is not None:
                delta = w.used_percent - last
                self._plan_consumed_by_window[key] = self._plan_consumed_by_window.get(key, 0.0) + (
                    delta if delta >= 0 else w.used_percent
                )
            self._plan_last_percent[key] = w.used_percent
        balance = plan.credits_usd
        if balance is not None:
            previous = self._credits_last_usd.get(route)
            if previous is not None and balance < previous:
                self._credits_spent_usd += previous - balance
            self._credits_last_usd[route] = balance
        self._plans.pop(route, None)
        self._plans[route] = plan

    def _check_plan_ceilings(self, model: str, plan_usage: PlanUsage) -> None:
        """Apply the plan-metered ceilings, most binding first: credits, zero refusals, the cap."""
        would_spend_credits = (
            plan_usage.has_credits
            and not plan_usage.credits_unlimited
            and plan_usage.window_exhausted
        )
        if would_spend_credits and not self.allow_paid_credits:
            balance = plan_usage.credits_balance or "unknown"
            usd = plan_usage.credits_usd
            if usd is not None and not plan_usage.credits_balance.strip().startswith("$"):
                balance = f"{balance} credits (~${usd:.2f})"
            self._exceeded_reason = (
                "the plan window is exhausted and the account holds purchased"
                f" credits (balance {balance}): continuing would spend them."
                " Set [budget].allow_paid_credits = true to allow that"
            )
        elif would_spend_credits and self.max_usd == 0.0:
            self._exceeded_reason = (
                "USD budget is 0: purchased-credit calls are refused (raise [budget].max_usd)"
            )
        elif self.max_percent == 0.0:
            self._exceeded_reason = (
                "percent budget is 0: plan-metered calls are refused"
                f" (model {model!r} draws on a subscription plan;"
                " raise [budget].max_percent)"
            )
        elif self.max_percent > 0.0 and self._plan_consumed() >= self.max_percent:
            route, window, consumed = self._plan_consumption_binding()
            used_percent = self._plan_last_percent[(route, window)]
            self._exceeded_reason = (
                f"plan budget exhausted: this run consumed ~{consumed:.1f}"
                f" percentage points >= max_percent {self.max_percent:g}"
                f" ({route}: account at {used_percent:g}% on its {window} window)"
            )
        elif self.max_usd > 0.0 and self._credits_spent_usd >= self.max_usd:
            self._exceeded_reason = (
                f"USD budget exhausted: ~{format_usd(self._credits_spent_usd)} of purchased"
                f" credits spent >= {format_usd(self.max_usd)}"
            )

    def record_plan_preflight(self, model: str, plan: PlanUsage) -> None:
        """Record a reading taken before the first plan-metered call.

        It is the baseline every later delta counts from, and the credit guard's first
        look, so a run that would draw on purchased credits refuses before its first call.

        Args:
            model: The model id.
            plan: The reading.
        """
        with self._lock:
            self._note_plan_usage(self._routes.get(model, model), plan)
            self._check_plan_ceilings(model, plan)

    def check(self) -> None:
        """Stop the next call when a prior `record` crossed a ceiling.

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
                usd_spent, _ = self._estimate_usd_locked()
                used = max(used, usd_spent / self.max_usd)
            if self.max_tokens_fallback > 0:
                used = max(used, self._unmetered_tokens / self.max_tokens_fallback)
            if self.max_percent > 0.0:
                used = max(used, self._plan_consumed() / self.max_percent)
        return max(0.0, 1.0 - used)

    def snapshot(self) -> BudgetSnapshot:
        """Return a point-in-time copy of every counter."""
        with self._lock:
            per_model = {
                model: dataclasses.replace(t) for model, t in sorted(self._per_model.items())
            }
            return BudgetSnapshot(
                input_total=self._input_total,
                output_total=self._output_total,
                cache_read_total=self._cache_read_total,
                cache_creation_total=self._cache_creation_total,
                unmetered_tokens=self._unmetered_tokens,
                max_usd=self.max_usd,
                max_tokens_fallback=self.max_tokens_fallback,
                max_percent=self.max_percent,
                plans={
                    route: PlanSpend(plan, self._plan_consumed(route))
                    for route, plan in self._plans.items()
                },
                exhausted=bool(self._exceeded_reason),
                exhausted_reason=self._exceeded_reason,
                per_model=per_model,
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
        total_usd = self._credits_spent_usd
        any_unknown = False
        for model, t in self._per_model.items():
            cost = _model_cost_usd(model, t, self._routes.get(model, ""))
            if cost is None:
                any_unknown = True
                continue
            any_unknown = any_unknown or cost.partial
            total_usd += cost.usd
        return total_usd, any_unknown

    def _model_lines(self, snap: BudgetSnapshot) -> tuple[list[str], int, int, bool]:
        """Return one summary line per model.

        Args:
            snap: The counters to print.

        Returns:
            The lines, how many models were priced, how many the USD ledger meters (a
            plan's calls cost dollars nowhere), and whether any figure is an estimate.
        """
        lines: list[str] = []
        priced = 0
        metered = 0
        any_estimated = False
        for model, totals in snap.per_model.items():
            cost = _model_cost_usd(model, totals, self._routes.get(model, ""))
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
                f"  {model}: "
                f"in={totals.input_tokens} out={totals.output_tokens} "
                f"cache_r={totals.cache_read_tokens} "
                f"cache_c={totals.cache_creation_tokens} "
                f"calls={totals.calls} {cost_str}"
            )
        return lines, priced, metered, any_estimated

    def format_summary(self) -> str:
        """Return the end-of-run token and cost summary."""
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
        if metered or not snap.per_model:
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
        for route, spend in snap.plans.items():
            lines.append("  " + format_plan_usage(route, spend, snap.max_percent))
        if snap.exhausted:
            lines.append(f"  STATUS: BUDGET EXCEEDED ({snap.exhausted_reason})")
        return "\n".join(lines)


def format_plan_usage(route: str, spend: PlanSpend, max_percent: float) -> str:
    """Return the plan-usage line every surface prints for one provider entry.

    Args:
        route: The provider entry.
        spend: Its latest reading and this run's consumption.
        max_percent: The cap the consumption is printed against; -1 prints none.

    Returns:
        The line.
    """
    plan = spend.usage
    minutes = plan.window_minutes
    if minutes >= 1440:
        window = f"{minutes / 1440:g}-day "
    elif minutes >= 60:
        window = f"{minutes / 60:g}-hour "
    elif minutes > 0:
        window = f"{minutes}-minute "
    else:
        window = ""
    cap = "" if max_percent == -1 else f" of max_percent {max_percent:g}"
    resets_h = max(0.0, (plan.resets_at - time.time()) / 3600)
    which = "" if plan.binding.name == "primary" else f" ({plan.binding.name})"
    return (
        f"plan usage ({route}): {plan.used_percent:g}% of the {window}window{which}"
        f" (this run ~{spend.consumed:g} points{cap}; resets in {resets_h:.0f}h)"
        + (
            f"; purchased credits balance {plan.credits_balance or 'present'}"
            if plan.has_credits and not plan.credits_unlimited
            else ""
        )
    )
