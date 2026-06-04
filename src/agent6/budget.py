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

USD budgets: configure `[budget].max_usd` in agent6.toml to specify a
dollar cap; the config loader converts it to token ceilings at load
time using the worker model's pricing (see `usd_budget_to_tokens` in
this module). Tokens stay the authoritative ceiling because token
counts are exact and provider-returned; the USD cap is a more
operator-friendly knob that translates once at startup.

This module is pure stdlib + dataclasses; the AnthropicProvider wires
it in via constructor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# Per-1M-token USD price table. Best-effort, used only for the end-of-run
# human-facing report. Unknown models render as "$? (unknown price)".
# Update as Anthropic changes pricing; this is informational only.
_PRICE_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    # Claude Opus 4.x family (input, output)
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-5-20250929": (15.0, 75.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    # Sonnet 4.x
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    # Haiku 4.x
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # Kimi family via OpenRouter (prices as of 2026-05; check OpenRouter
    # /v1/models endpoint for current rates). Cache_read priced same as
    # input here; OpenRouter rebills cache hits at provider rates which
    # vary - leave the conservative estimate.
    "moonshotai/kimi-k2.6": (0.68, 3.42),
    "moonshotai/kimi-k2.5": (0.40, 1.90),
    "moonshotai/kimi-k2-thinking": (0.60, 2.50),
    "moonshotai/kimi-k2-0905": (0.60, 2.50),
    "moonshotai/kimi-k2": (0.57, 2.30),
    "moonshotai/kimi-latest": (0.73, 3.49),
    # Other open-weights via OpenRouter (smoke).
    # Prices fetched live from openrouter.ai/api/v1/models on 2026-06-02.
    "z-ai/glm-4.6": (0.43, 1.74),
    "minimax/minimax-m2.7": (0.279, 1.20),
    "minimax/minimax-m2": (0.255, 1.00),
    "deepseek/deepseek-v3.2-exp": (0.27, 0.41),
    "deepseek/deepseek-v3.2": (0.23, 0.34),
    "qwen/qwen3-coder": (0.22, 1.80),
    "qwen/qwen3-coder-30b-a3b-instruct": (0.07, 0.27),
}


INPUT_TO_OUTPUT_RATIO_FOR_USD_BUDGET = 5.0
"""Operator-facing USD budgets are converted into separate input/output
token ceilings via this assumed ratio. Empirically code-editing workloads
spend ~5 input tokens for each output token (the worker re-reads more
than it writes). The ratio is intentionally conservative on the output
side because output tokens are 5x the per-token price; underestimating
output capacity is cheaper than underestimating input capacity. Tweak
here if your workload shape is very different - this only affects the
USD-to-tokens conversion, not the runtime token accounting."""


def usd_budget_to_tokens(
    max_usd: float,
    *,
    worker_model: str,
    fallback_input_per_mtok: float = 3.0,
    fallback_output_per_mtok: float = 15.0,
) -> tuple[int, int]:
    """Convert an operator-friendly USD cap into (max_input_tokens,
    max_output_tokens) for a given worker model's pricing.

    Uses `_PRICE_PER_MTOK_USD` if the model is listed (current Anthropic
    Claude family). Falls back to the sonnet-4.5 rate ($3/M in, $15/M
    out) when the model is unknown - typically OpenRouter / Kimi /
    deepseek where listed pricing varies. Operators can override the
    fallback rates per-call.

    Returns (input_tokens, output_tokens) sized so that hitting BOTH
    ceilings simultaneously costs approximately `max_usd`. The runtime
    enforcement is still per-ceiling (input_total >= max_input_tokens
    triggers BudgetExceeded regardless of output usage), so the actual
    spend ceiling is roughly `max_usd` when the workload's I/O ratio
    matches `_INPUT_TO_OUTPUT_RATIO_FOR_USD_BUDGET`, less when the
    workload spends one bucket but not the other.

    Example: usd_budget_to_tokens(5.0, worker_model="claude-sonnet-4-5")
    returns (max_input=1_388_888, max_output=277_777) - 5x more input
    headroom than output, sized so that hitting both at the same time
    is ~$5.
    """
    if max_usd <= 0:
        raise ValueError(f"max_usd must be positive, got {max_usd}")
    price = _PRICE_PER_MTOK_USD.get(worker_model)
    if price is None:
        in_per_mtok = fallback_input_per_mtok
        out_per_mtok = fallback_output_per_mtok
    else:
        in_per_mtok, out_per_mtok = price
    ratio = INPUT_TO_OUTPUT_RATIO_FOR_USD_BUDGET
    # Solve: input_usd + output_usd = max_usd
    #        input_usd = ratio * output_usd  (by ratio of tokens at their
    #                                         respective per-token rates)
    # => (ratio + 1) * output_usd = max_usd
    output_usd = max_usd / (ratio + 1)
    input_usd = max_usd - output_usd
    max_input_tokens = int((input_usd * 1_000_000) / in_per_mtok)
    max_output_tokens = int((output_usd * 1_000_000) / out_per_mtok)
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

        Computed against whichever ceiling is closer to exhaustion (the
        input or output bucket), so a run that has burned 90% of its
        output ceiling but only 10% of its input ceiling reports 0.10 —
        the conservative, decision-relevant figure. Used by the workflow
        to decide whether a metric plateau is worth quitting on or
        whether enough budget remains to keep pivoting.
        """
        with self._lock:
            input_used = (
                self._input_total / self.max_input_tokens if self.max_input_tokens > 0 else 1.0
            )
            output_used = (
                self._output_total / self.max_output_tokens if self.max_output_tokens > 0 else 1.0
            )
        used = max(input_used, output_used)
        return max(0.0, 1.0 - used)

    def snapshot(self) -> dict[str, object]:
        """Immutable snapshot of all counters; safe to JSON-encode."""
        with self._lock:
            per_model: dict[str, dict[str, int | float]] = {
                model: {
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cache_read_tokens": t.cache_read_tokens,
                    "cache_creation_tokens": t.cache_creation_tokens,
                    "calls": t.calls,
                    "reported_cost_usd": t.reported_cost_usd,
                    "reported_calls": t.reported_calls,
                }
                for model, t in sorted(self._per_model.items())
            }
            return {
                "input_total": self._input_total,
                "output_total": self._output_total,
                "cache_read_total": self._cache_read_total,
                "cache_creation_total": self._cache_creation_total,
                "max_input_tokens": self.max_input_tokens,
                "max_output_tokens": self.max_output_tokens,
                "exhausted": bool(self._exceeded_reason),
                "exhausted_reason": self._exceeded_reason,
                "per_model": per_model,
            }

    def estimate_usd(self) -> tuple[float, bool]:
        """Estimate cumulative USD spend across all recorded calls.

        Returns ``(usd_total, any_unknown)`` where ``any_unknown`` is
        True iff at least one model in the per-model breakdown is missing
        from the pricing table (so the figure is a lower bound).

        Shared between the end-of-run text summary and the live TUI cost
        meter so both quote the same number from the same arithmetic.
        """
        snap = self.snapshot()
        per_model = snap["per_model"]
        assert isinstance(per_model, dict)
        total_usd = 0.0
        any_unknown = False
        for model, totals in per_model.items():
            # When the provider returned an authoritative
            # ``usage.cost`` for EVERY call to this model, prefer that
            # sum over the price-table estimate. If even one call lacked
            # the field (mixed-route, transient OpenRouter quirk, etc.)
            # we fall back to the table for the whole model so the
            # numbers are consistent rather than partially-mixed.
            reported = float(totals.get("reported_cost_usd", 0.0))
            reported_calls = int(totals.get("reported_calls", 0))
            if reported > 0.0 and reported_calls == int(totals["calls"]):
                total_usd += reported
                continue
            price = _PRICE_PER_MTOK_USD.get(model)
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
            in_usd = totals["input_tokens"] * price[0] / 1e6
            cache_creation_usd = totals["cache_creation_tokens"] * (price[0] * 1.25) / 1e6
            cache_read_usd = totals["cache_read_tokens"] * (price[0] * 0.1) / 1e6
            out_usd = totals["output_tokens"] * price[1] / 1e6
            total_usd += in_usd + cache_creation_usd + cache_read_usd + out_usd
        return total_usd, any_unknown

    def format_summary(self) -> str:
        """Human-facing end-of-run summary with USD estimate where known."""
        snap = self.snapshot()
        per_model = snap["per_model"]
        assert isinstance(per_model, dict)
        lines = ["Token + cost summary:"]
        total_usd = 0.0
        any_unknown = False
        for model, totals in per_model.items():
            price = _PRICE_PER_MTOK_USD.get(model)
            cost_str: str
            reported = float(totals.get("reported_cost_usd", 0.0))
            reported_calls = int(totals.get("reported_calls", 0))
            if reported > 0.0 and reported_calls == int(totals["calls"]):
                # Provider-authoritative.
                total_usd += reported
                cost_str = f"${reported:.4f} (reported)"
            elif price is None:
                cost_str = "$? (unknown price)"
                any_unknown = True
            else:
                # See estimate_usd for the pricing model rationale.
                in_usd = totals["input_tokens"] * price[0] / 1e6
                cache_creation_usd = totals["cache_creation_tokens"] * (price[0] * 1.25) / 1e6
                cache_read_usd = totals["cache_read_tokens"] * (price[0] * 0.1) / 1e6
                out_usd = totals["output_tokens"] * price[1] / 1e6
                model_usd = in_usd + cache_creation_usd + cache_read_usd + out_usd
                total_usd += model_usd
                cost_str = f"${model_usd:.4f}"
            lines.append(
                f"  {model}: "
                f"in={totals['input_tokens']} out={totals['output_tokens']} "
                f"cache_r={totals['cache_read_tokens']} "
                f"cache_c={totals['cache_creation_tokens']} "
                f"calls={totals['calls']} {cost_str}"
            )
        budget_line = (
            f"  TOTAL: in={snap['input_total']}/{snap['max_input_tokens']} "
            f"out={snap['output_total']}/{snap['max_output_tokens']} "
            f"cost~${total_usd:.4f}"
        )
        if any_unknown:
            budget_line += " (some models unpriced)"
        lines.append(budget_line)
        if snap["exhausted"]:
            lines.append(f"  STATUS: BUDGET EXCEEDED — {snap['exhausted_reason']}")
        return "\n".join(lines)
