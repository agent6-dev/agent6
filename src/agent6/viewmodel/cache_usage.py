"""Fold worker cache use and input after compaction from the session journal."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class SummaryAttempt:
    """Keep one summary writer's outcome and reported input.

    Attributes:
        writer: The writer selected for this attempt.
        model_call_id: The matching logical model call.
        fallback_reason: Why the helper took over, if it did.
        outcome: Acceptance, rejection or no recorded outcome.
        uncached_tokens: Fresh plus cache-write input, when reported.
        cached_tokens: Input served from cache, when reported.
    """

    writer: str
    model_call_id: str
    fallback_reason: str
    outcome: str = "no recorded outcome"
    uncached_tokens: int | None = None
    cached_tokens: int | None = None

    def line(self, number: int) -> str:
        """Render the attempt without turning absent measurements into zeroes.

        Args:
            number: The one-based attempt number in the session.

        Returns:
            One line of the receipt.
        """
        fresh = f"{self.uncached_tokens:,}" if self.uncached_tokens is not None else "unreported"
        cached = f"{self.cached_tokens:,}" if self.cached_tokens is not None else "unreported"
        fallback = f"; fallback: {self.fallback_reason}" if self.fallback_reason else ""
        return (
            f"Summary attempt {number}: {self.writer}; {fresh} uncached, {cached} cached "
            f"input tokens; {self.outcome}{fallback}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CacheUsage:
    """Keep session-wide worker usage from logical model results.

    Attributes:
        calls: Worker results, including those without usage.
        reported_calls: Worker results with input and cache-read counts.
        input_tokens: Fresh input on reported calls.
        cache_read_tokens: Input served from cache on reported calls.
        cache_write_tokens: Input written to cache on reported calls.
        post_compaction_calls: Reported worker calls following a history rewrite.
        post_compaction_uncached_tokens: Fresh plus cache-write input on those calls.
        summaries: Summary attempts in journal order, excluded from ordinary worker usage.
    """

    calls: int = 0
    reported_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    post_compaction_calls: int = 0
    post_compaction_uncached_tokens: int = 0
    summaries: tuple[SummaryAttempt, ...] = ()

    def lines(self) -> tuple[str, ...]:
        """Render measured cache use, omitting unreported usage.

        Returns:
            Cache share and post-compaction input, with partial coverage named.
        """
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        attempts = tuple(s.line(i) for i, s in enumerate(self.summaries, start=1))
        if not self.reported_calls or not total:
            return attempts
        partial = (
            f"; {self.reported_calls}/{self.calls} calls reported"
            if self.reported_calls != self.calls
            else ""
        )
        noun = "call" if self.post_compaction_calls == 1 else "calls"
        return (
            f"Worker input cache (session): {100 * self.cache_read_tokens / total:.1f}% "
            f"({self.cache_read_tokens:,}/{total:,} tokens){partial}",
            f"After compaction (session): {self.post_compaction_calls} worker {noun}, "
            f"{self.post_compaction_uncached_tokens:,} uncached input tokens"
            + (" (reported calls only)" if partial else ""),
            *attempts,
        )


@dataclasses.dataclass(slots=True)
class CacheUsageFold:
    """Correlate actual history rewrites with the next logical worker call."""

    usage: CacheUsage = dataclasses.field(default_factory=CacheUsage)
    pending: bool = False
    after: set[str] = dataclasses.field(default_factory=set)
    summary_call_id: str = ""

    def observe(self, row: Mapping[str, Any]) -> None:
        """Fold one journal row without counting helper calls or retries twice.

        Args:
            row: The decoded journal event.
        """
        kind = row.get("type")
        if kind in ("session.start", "session.resume"):
            self.pending = False
            self.after.clear()
            self.summary_call_id = ""
        if self._summary(row):
            return
        if (
            (kind in ("compact.dropped", "compact.deduped") and _count(row, "count"))
            or (kind == "compact.thinking_dropped" and _count(row, "messages"))
            or (kind == "compact.gist.done" and _count(row, "demoted"))
            or kind == "compact.summarise.done"
        ):
            self.pending = True
        elif row.get("role") == "worker":
            call_id = str(row.get("model_call_id", ""))
            if kind == "model.call" and self.pending:
                self.after.add(call_id)
                self.pending = False
            elif kind == "model.result":
                after = call_id in self.after
                self.after.discard(call_id)
                fresh = _count(row, "input_tokens")
                cached = _count(row, "cache_read_tokens")
                written = _count(row, "cache_write_tokens") or 0
                old = self.usage
                if fresh is None or cached is None:
                    self.usage = dataclasses.replace(old, calls=old.calls + 1)
                    return
                self.usage = dataclasses.replace(
                    old,
                    calls=old.calls + 1,
                    reported_calls=old.reported_calls + 1,
                    input_tokens=old.input_tokens + fresh,
                    cache_read_tokens=old.cache_read_tokens + cached,
                    cache_write_tokens=old.cache_write_tokens + written,
                    post_compaction_calls=old.post_compaction_calls + after,
                    post_compaction_uncached_tokens=old.post_compaction_uncached_tokens
                    + (fresh + written if after else 0),
                )

    def _summary(self, row: Mapping[str, Any]) -> bool:
        """Fold the active summary attempt and identify its model rows.

        Args:
            row: The decoded journal event.

        Returns:
            True for model rows to exclude from ordinary worker totals.
        """
        kind = row.get("type")
        attempts = self.usage.summaries
        if kind == "compact.summarise.call" and row.get("writer"):
            self.summary_call_id = str(row.get("model_call_id", ""))
            attempt = SummaryAttempt(
                str(row["writer"]), self.summary_call_id, str(row.get("fallback_reason", ""))
            )
            self.usage = dataclasses.replace(self.usage, summaries=(*attempts, attempt))
        elif self.summary_call_id:
            last = attempts[-1]
            if kind in ("compact.summarise.done", "compact.summarise.failed"):
                outcome = (
                    "accepted"
                    if kind == "compact.summarise.done"
                    else f"rejected: {row.get('error')}"
                )
                last = dataclasses.replace(last, outcome=outcome)
                self.summary_call_id = ""
            elif row.get("model_call_id") == self.summary_call_id:
                if kind == "model.result":
                    fresh = _count(row, "input_tokens")
                    last = dataclasses.replace(
                        last,
                        uncached_tokens=None
                        if fresh is None
                        else fresh + (_count(row, "cache_write_tokens") or 0),
                        cached_tokens=_count(row, "cache_read_tokens"),
                    )
                    self.usage = dataclasses.replace(self.usage, summaries=(*attempts[:-1], last))
                return True
            else:
                return False
            self.usage = dataclasses.replace(self.usage, summaries=(*attempts[:-1], last))
        return False


def _count(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
