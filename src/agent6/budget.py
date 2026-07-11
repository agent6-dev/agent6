# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-invocation token budget tracker with hard-stop enforcement.

**Scope of "per-invocation"**: The BudgetTracker is created fresh at the
start of every `agent6 run` (or `agent6 plan ...`, etc.) command. It
counts tokens used by that invocation only. It does NOT persist across
invocations - a subsequent `agent6 resume <id>` gets a fresh budget
ceiling. This is intentional and matches the "prevent runaway costs
PER USER INTERACTION" design goal: the budget is a circuit breaker
against runaway spend during one invocation, not a long-running
ledger across a multi-day task.

Practical implication for long-running goals: if you `agent6 run X`,
hit the budget cap, then `agent6 resume <id>` to continue, you get
another full ceiling. Across N resumes, total real spend can be N x
the configured budget. The CLI logs a one-line notice on resume to
make this visible.

Budget enforcement is a HARD STOP (not a warning). When
`max_input_tokens` or `max_output_tokens` is exceeded, the next
provider call raises `BudgetExceeded`; the workflow drains and the
process exits with a distinct exit code so resume tooling can
recognise the condition.

USD budgets: configure `[budget].best_effort_usd_limit` to specify a
dollar cap; the config loader converts it to token ceilings at load
time using the worker model's pricing (see `usd_budget_to_tokens` in
this module). Tokens stay the authoritative ceiling because token
counts are exact and provider-returned; the USD cap is a more
operator-friendly knob that translates once at startup.

This module is import-light (stdlib + agent6.models.pricing, which is itself
stdlib + cache-file reads); the AnthropicProvider wires it in via
constructor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agent6.models.pricing import lookup_price

# There is NO static price table. Prices come from the provider's own models
# endpoint, fetched + cached by agent6.models.cache and read back through
# agent6.models.pricing.lookup_price. A model without a published price is reported
# as "$? (unknown price)" and the USD->token budget conversion does not apply
# to it: an unknown price is honest, an outdated hardcoded one is wrong.


def usd_budget_to_tokens(
    max_usd: float,
    *,
    worker_model: str,
) -> tuple[int, int] | None:
    """Convert an operator-friendly USD cap into (max_input_tokens,
    max_output_tokens) for a given worker model's pricing.

    Pricing comes from the provider-fetched cache (agent6.models.pricing). Returns
    None when the model has no known price: the USD->token tightening simply
    does not apply, the operator token ceilings stand as configured, and the
    runtime `max_usd` ceiling still enforces wherever the provider reports
    per-call cost (OpenRouter `usage.cost`) or a cached price exists.

    Each axis is sized so that THAT axis alone reaching its cap costs ~max_usd.
    The authoritative bound is the runtime USD ceiling (BudgetTracker.max_usd):
    this conversion only runs when `best_effort_usd_limit > 0`, so that ceiling
    is always active, it bounds the cache-INCLUSIVE COMBINED spend, and on any
    mixed workload it trips before either axis alone is exhausted. The per-axis
    caps are a backstop (each bounds its own axis to ~max_usd).

    Sizing each axis to the full budget -- rather than splitting it by an
    assumed input:output ratio -- is what lets an output-heavy workload use the
    whole budget. A reasoning model whose reasoning_content dominates output
    (e.g. GLM, Kimi K2.x) spends far more on output than a 5:1 code-edit ratio
    assumes; a ratio-split output cap would halt such a run at a fraction of the
    USD budget (output cap hit while input sat almost untouched) even though the
    USD ceiling had plenty of room.

    Example: at $3/M in, $15/M out, usd_budget_to_tokens(5.0, ...) returns
    (max_input=1_666_666, max_output=333_333) - each axis alone is ~$5.
    """
    if max_usd <= 0:
        raise ValueError(f"max_usd must be positive, got {max_usd}")
    price = lookup_price(worker_model)
    if price is None:
        return None
    in_per_mtok, out_per_mtok = price
    if in_per_mtok <= 0 or out_per_mtok <= 0:
        # A free or provider-unpriced model (OpenRouter has reported 0/0 for
        # some routes, e.g. z-ai/glm-5.2 transiently): a USD budget can't be
        # turned into a token ceiling, and the runtime USD tracker reads the
        # same 0 cost, so there is nothing to convert. Return None like the
        # no-price case (caller keeps the operator token ceilings) instead of
        # dividing by zero.
        return None
    # Clamp to at least one token: an extreme-but-legal tiny USD budget can
    # floor to 0 here, which would synthesize an invalid 0 token ceiling (the
    # BudgetConfig validators require gt=0). The runtime USD ceiling (max_usd
    # via BudgetTracker) still enforces the true dollar bound, so a 1-token
    # floor just means the run stops after a single tiny call.
    max_input_tokens = max(1, int((max_usd * 1_000_000) / in_per_mtok))
    max_output_tokens = max(1, int((max_usd * 1_000_000) / out_per_mtok))
    return max_input_tokens, max_output_tokens


class BudgetExceeded(Exception):
    """Raised by `BudgetTracker.check()` once a configured limit is exceeded."""


@dataclass(slots=True)
class _ModelTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0
    # Sum of provider-reported per-call USD cost. Populated only
    # for routes that surface ``usage.cost`` in the response body (today:
    # OpenRouter). When > 0 it is preferred over the price-table
    # estimate; when 0 we fall back to the table.
    reported_cost_usd: float = 0.0
    reported_calls: int = 0


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Immutable per-model usage totals inside a :class:`BudgetSnapshot`."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    calls: int
    reported_cost_usd: float
    reported_calls: int


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Immutable snapshot of a BudgetTracker's counters at one instant."""

    input_total: int
    output_total: int
    cache_read_total: int
    cache_creation_total: int
    max_input_tokens: int
    max_output_tokens: int
    exhausted: bool
    exhausted_reason: str
    per_model: dict[str, ModelUsage]


@dataclass(slots=True)
class BudgetTracker:
    """Thread-safe token accumulator with a hard ceiling.

    `max_input_tokens` and `max_output_tokens` are exclusive ceilings: a call
    that *brings* the running total to or above the ceiling triggers
    `BudgetExceeded` on the *next* `check()`. This means a single call may
    cross the line, but no subsequent call will be issued.
    """

    max_input_tokens: int
    max_output_tokens: int
    # Estimated-dollar ceiling (0 = off). Set from `[budget]
    # best_effort_usd_limit`. This is the AUTHORITATIVE spend bound when
    # `usd_budget_to_tokens` derives the token caps (it sizes each axis to the
    # full budget on purpose). Unlike the token caps -- which `record`
    # thresholds on fresh input/output only -- this bounds the estimated spend
    # INCLUDING cache_read/cache_creation tokens, which cost real money but
    # never count toward the token caps, and bounds COMBINED in+out so it trips
    # before either full-budget axis cap on a mixed workload. Without it, a
    # heavily-cached run can blow well past `max_usd`.
    max_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _per_model: dict[str, _ModelTotals] = field(default_factory=dict)
    _input_total: int = 0
    _output_total: int = 0
    _cache_read_total: int = 0
    _cache_creation_total: int = 0
    _exceeded_reason: str = ""

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        """Add the usage from a single provider response to the running totals.

        ``cost_usd`` is the provider-reported USD figure for this single
        call when available (OpenRouter surfaces it as ``usage.cost``).
        Pass 0.0 (the default) when no authoritative figure is supplied;
        the price-table estimate will be used at summary time.
        """
        with self._lock:
            totals = self._per_model.setdefault(model, _ModelTotals())
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.cache_read_tokens += cache_read_tokens
            totals.cache_creation_tokens += cache_creation_tokens
            totals.calls += 1
            if cost_usd > 0.0:
                totals.reported_cost_usd += cost_usd
                totals.reported_calls += 1
            self._input_total += input_tokens
            self._output_total += output_tokens
            self._cache_read_total += cache_read_tokens
            self._cache_creation_total += cache_creation_tokens
            if self._input_total >= self.max_input_tokens:
                self._exceeded_reason = (
                    f"input token budget exhausted: {self._input_total} >= {self.max_input_tokens}"
                )
            elif self._output_total >= self.max_output_tokens:
                self._exceeded_reason = (
                    f"output token budget exhausted: "
                    f"{self._output_total} >= {self.max_output_tokens}"
                )
            elif self.max_usd > 0.0:
                cost, _ = self._estimate_usd_locked()
                if cost >= self.max_usd:
                    self._exceeded_reason = (
                        f"USD budget exhausted: ~${cost:.4f} >= ${self.max_usd:.2f}"
                        " (includes cache_read/cache_creation cost)"
                    )

    def check(self) -> None:
        """Raise `BudgetExceeded` if a prior `record()` crossed a ceiling."""
        with self._lock:
            reason = self._exceeded_reason
        if reason:
            raise BudgetExceeded(reason)

    def is_exhausted(self) -> bool:
        with self._lock:
            return bool(self._exceeded_reason)

    def fraction_remaining(self) -> float:
        """Fraction of the budget still available, in ``[0.0, 1.0]``.

        Computed against whichever ceiling is closest to exhaustion, so a run
        that has burned 90% of one ceiling but only 10% of another reports 0.10,
        the conservative, decision-relevant figure. Used by the workflow to
        decide whether a metric plateau is worth quitting on, whether enough
        budget remains to keep pivoting, and when to nudge a graceful wind-down
        (verify + finish_run) before the hard stop.

        The USD ceiling counts too when set: ``max_usd`` is the AUTHORITATIVE
        spend bound and, unlike the token caps, it includes cache_read /
        cache_creation cost (which never counts toward the token caps). On a
        USD-budgeted, cache-heavy run the USD ceiling is what actually
        hard-stops the run, so leaving it out here reported plenty of budget
        left while ``record`` was about to raise ``BudgetExceeded`` -- every
        wind-down threshold stayed un-triggered and the worker was hard-killed
        mid-edit instead of finishing cleanly. An unpriced model contributes $0
        to the estimate (max_usd is unenforceable there anyway), so this stays a
        no-op exactly when the USD bound is a no-op.
        """
        with self._lock:
            input_used = (
                self._input_total / self.max_input_tokens if self.max_input_tokens > 0 else 1.0
            )
            output_used = (
                self._output_total / self.max_output_tokens if self.max_output_tokens > 0 else 1.0
            )
            used = max(input_used, output_used)
            if self.max_usd > 0.0:
                usd_spent, _ = self._estimate_usd_locked()
                used = max(used, usd_spent / self.max_usd)
        return max(0.0, 1.0 - used)

    def snapshot(self) -> BudgetSnapshot:
        """Immutable snapshot of all counters."""
        with self._lock:
            per_model = {
                model: ModelUsage(
                    input_tokens=t.input_tokens,
                    output_tokens=t.output_tokens,
                    cache_read_tokens=t.cache_read_tokens,
                    cache_creation_tokens=t.cache_creation_tokens,
                    calls=t.calls,
                    reported_cost_usd=t.reported_cost_usd,
                    reported_calls=t.reported_calls,
                )
                for model, t in sorted(self._per_model.items())
            }
            return BudgetSnapshot(
                input_total=self._input_total,
                output_total=self._output_total,
                cache_read_total=self._cache_read_total,
                cache_creation_total=self._cache_creation_total,
                max_input_tokens=self.max_input_tokens,
                max_output_tokens=self.max_output_tokens,
                exhausted=bool(self._exceeded_reason),
                exhausted_reason=self._exceeded_reason,
                per_model=per_model,
            )

    def estimate_usd(self) -> tuple[float, bool]:
        """Estimate cumulative USD spend across all recorded calls.

        Returns ``(usd_total, any_unknown)`` where ``any_unknown`` is
        True iff at least one model in the per-model breakdown is missing
        from the pricing table (so the figure is a lower bound).

        Shared between the end-of-run text summary, the live TUI cost
        meter, and the in-record USD ceiling so they all quote the same
        number from the same arithmetic.
        """
        with self._lock:
            return self._estimate_usd_locked()

    def _estimate_usd_locked(self) -> tuple[float, bool]:
        """Cost estimate computed directly from ``self._per_model``.

        Assumes ``self._lock`` is already held (called from both ``record`` --
        under the lock -- and ``estimate_usd``), so it never re-acquires it.
        """
        total_usd = 0.0
        any_unknown = False
        for model, t in self._per_model.items():
            # When the provider returned an authoritative ``usage.cost`` for
            # EVERY call to this model, prefer that sum over the price-table
            # estimate. If even one call lacked the field (mixed-route, transient
            # OpenRouter quirk, etc.) we fall back to the table for the whole
            # model so the numbers are consistent rather than partially-mixed.
            if t.reported_cost_usd > 0.0 and t.reported_calls == t.calls:
                total_usd += t.reported_cost_usd
                continue
            price = lookup_price(model)
            if price is None:
                any_unknown = True
                continue
            # pricing model (Anthropic-accurate):
            #   fresh input:      price[0]         (already excludes cached portion)
            #   cache_creation:   price[0] * 1.25  (5-min cache write surcharge)
            #   cache_read:       price[0] * 0.10  (cache hit discount)
            #   output:           price[1]
            # OpenAI-route models (Kimi etc.) currently report
            # cache_creation_tokens=0 since the chat-completions usage
            # block has no separate write-surcharge field, so the 1.25x
            # branch is a no-op for them.
            in_usd = t.input_tokens * price[0] / 1e6
            cache_creation_usd = t.cache_creation_tokens * (price[0] * 1.25) / 1e6
            cache_read_usd = t.cache_read_tokens * (price[0] * 0.1) / 1e6
            out_usd = t.output_tokens * price[1] / 1e6
            total_usd += in_usd + cache_creation_usd + cache_read_usd + out_usd
        return total_usd, any_unknown

    def format_summary(self) -> str:
        """Human-facing end-of-run summary with USD estimate where known."""
        snap = self.snapshot()
        lines = ["Token + cost summary:"]
        total_usd = 0.0
        any_unknown = False
        for model, totals in snap.per_model.items():
            price = lookup_price(model)
            cost_str: str
            if totals.reported_cost_usd > 0.0 and totals.reported_calls == totals.calls:
                # Provider-authoritative.
                total_usd += totals.reported_cost_usd
                cost_str = f"${totals.reported_cost_usd:.4f} (reported)"
            elif price is None:
                cost_str = "$? (unknown price)"
                any_unknown = True
            else:
                # See estimate_usd for the pricing model rationale.
                in_usd = totals.input_tokens * price[0] / 1e6
                cache_creation_usd = totals.cache_creation_tokens * (price[0] * 1.25) / 1e6
                cache_read_usd = totals.cache_read_tokens * (price[0] * 0.1) / 1e6
                out_usd = totals.output_tokens * price[1] / 1e6
                model_usd = in_usd + cache_creation_usd + cache_read_usd + out_usd
                total_usd += model_usd
                cost_str = f"${model_usd:.4f}"
            lines.append(
                f"  {model}: "
                f"in={totals.input_tokens} out={totals.output_tokens} "
                f"cache_r={totals.cache_read_tokens} "
                f"cache_c={totals.cache_creation_tokens} "
                f"calls={totals.calls} {cost_str}"
            )
        budget_line = (
            f"  TOTAL: in={snap.input_total}/{snap.max_input_tokens} "
            f"out={snap.output_total}/{snap.max_output_tokens} "
            f"cost~${total_usd:.4f}"
        )
        if any_unknown:
            # The figure is a lower bound; at least one model has no cached
            # provider price (see agent6.models.pricing: no static fallback).
            budget_line += "+ (some models unpriced; figure is a lower bound)"
        lines.append(budget_line)
        if snap.exhausted:
            lines.append(f"  STATUS: BUDGET EXCEEDED — {snap.exhausted_reason}")
        return "\n".join(lines)
