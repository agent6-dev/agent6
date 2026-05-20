# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-run token budget tracker with hard-stop enforcement.

Per Locked Decision §3 of the v1.1 plan, budget enforcement is a HARD STOP
(not a warning). When `max_input_tokens` or `max_output_tokens` is exceeded,
the next provider call raises `BudgetExceeded`; the workflow drains and the
process exits with a distinct exit code so resume tooling can recognise the
condition.

This module is pure stdlib + dataclasses; the AnthropicProvider wires it in
via constructor.
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
}


class BudgetExceeded(Exception):
    """Raised by `BudgetTracker.check()` once a configured limit is exceeded."""


@dataclass(slots=True)
class _ModelTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0


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
    ) -> None:
        """Add the usage from a single provider response to the running totals."""
        with self._lock:
            totals = self._per_model.setdefault(model, _ModelTotals())
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.cache_read_tokens += cache_read_tokens
            totals.cache_creation_tokens += cache_creation_tokens
            totals.calls += 1
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

    def snapshot(self) -> dict[str, object]:
        """Immutable snapshot of all counters; safe to JSON-encode."""
        with self._lock:
            per_model: dict[str, dict[str, int]] = {
                model: {
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cache_read_tokens": t.cache_read_tokens,
                    "cache_creation_tokens": t.cache_creation_tokens,
                    "calls": t.calls,
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
            if price is None:
                cost_str = "$? (unknown price)"
                any_unknown = True
            else:
                in_usd = (totals["input_tokens"] + totals["cache_creation_tokens"]) * price[0] / 1e6
                # cache-read tokens are billed at 10% of input price per Anthropic docs
                cache_read_usd = totals["cache_read_tokens"] * (price[0] * 0.1) / 1e6
                out_usd = totals["output_tokens"] * price[1] / 1e6
                model_usd = in_usd + cache_read_usd + out_usd
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
