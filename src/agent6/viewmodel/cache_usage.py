"""Fold worker cache use and input after compaction from the session journal."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


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
    """

    calls: int = 0
    reported_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    post_compaction_calls: int = 0
    post_compaction_uncached_tokens: int = 0

    def lines(self) -> tuple[str, ...]:
        """Render measured cache use, omitting unreported usage.

        Returns:
            Cache share and post-compaction input, with partial coverage named.
        """
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        if not self.reported_calls or not total:
            return ()
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
        )


@dataclasses.dataclass(slots=True)
class CacheUsageFold:
    """Correlate actual history rewrites with the next logical worker call."""

    usage: CacheUsage = dataclasses.field(default_factory=CacheUsage)
    pending: bool = False
    after: set[str] = dataclasses.field(default_factory=set)

    def observe(self, row: Mapping[str, Any]) -> None:
        """Fold one journal row without counting helper calls or retries twice.

        Args:
            row: The decoded journal event.
        """
        kind = row.get("type")
        if kind in ("session.start", "session.resume"):
            self.pending = False
            self.after.clear()
        elif (
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
                self.usage = CacheUsage(
                    calls=old.calls + 1,
                    reported_calls=old.reported_calls + 1,
                    input_tokens=old.input_tokens + fresh,
                    cache_read_tokens=old.cache_read_tokens + cached,
                    cache_write_tokens=old.cache_write_tokens + written,
                    post_compaction_calls=old.post_compaction_calls + after,
                    post_compaction_uncached_tokens=old.post_compaction_uncached_tokens
                    + (fresh + written if after else 0),
                )


def _count(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
