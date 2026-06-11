# ratelimit

Implement `src/lib.rs` so that `cargo test` passes.
Do NOT modify the tests in `tests/ratelimit_test.rs`.

## Spec

parse_rate(s) parses a rate expression "<events>/<window>" where
events is a positive decimal integer and window is a positive decimal
integer followed by a unit: `ms`, `s`, or `m`.

    parse_rate("10/5s")  -> Rate { events: 10, per_ms: 5000 }
    parse_rate("3/250ms")-> Rate { events: 3,  per_ms: 250 }
    parse_rate("60/1m")  -> Rate { events: 60, per_ms: 60000 }

Errors (RateError, in priority order when several apply):
- BadFormat: not exactly one '/', empty parts, non-digit chars in the
  numeric parts, or a missing/unknown unit suffix.
- ZeroEvents: events parses to 0.
- ZeroWindow: window parses to 0.

TokenBucket::new(rate, burst) creates a bucket holding `burst` tokens
(full at time 0). `allow(&mut self, now_ms: u64) -> bool` consumes one
token if available. Refill is continuous at `rate.events` per
`rate.per_ms` milliseconds using integer math: between calls, the
bucket gains `elapsed_ms * events / per_ms` tokens, capped at `burst`.
To avoid losing fractional progress, track refill in units of
events-per-per_ms (i.e. keep a remainder; do not use floats). `now_ms`
values are non-decreasing; a repeated `now_ms` refills nothing extra.
Fractional-token state must carry across calls: at 1/100ms, calls at
t=0(spend), t=50, t=99 stay denied and t=100 is allowed again.
