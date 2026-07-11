# eventflow: session analytics over an event log

Five small modules process an event log into a per-user report. The pipeline
is `parser` -> `sessions` -> `stats` -> `report`, configured by `config`.
Money is handled in INTEGER CENTS everywhere; floats must never be used for
money arithmetic.

## config.py

- `DEFAULTS`: `{"session_gap_s": 1800, "top_n": 3}`.
- `load_overrides(d)`: update the active settings with `d` and increment
  `config_version()` by 1. Unknown keys raise `KeyError`.
- `get(key)`: current value. `config_version()`: starts at 0.

## parser.py

- `parse_line(line)` parses `"<iso-ts>|<user>|<action>|<value>"`.
  - `iso-ts`: `YYYY-MM-DDTHH:MM:SS`, UTC; convert to epoch seconds (int).
  - `value`: a decimal amount with AT MOST two fraction digits (e.g. `3.5`,
    `0.29`, `12`). Convert to integer cents EXACTLY: `"0.29"` is `29`,
    `"3.5"` is `350`, `"12"` is `1200`. Digit-string arithmetic only; using
    binary floating point here produces wrong cents for values like `0.29`.
  - Whitespace around fields is stripped. A line with anything else wrong
    (field count, bad timestamp, negative value, more than 2 fraction
    digits) raises `ValueError`.
- `parse_lines(lines)` yields events for every line, skipping lines that are
  empty/whitespace-only, propagating `ValueError` from bad lines.

## sessions.py

- `Session` holds `user`, `events` (list). A NEW `Session()` always starts
  with its own EMPTY list: two sessions never share an events list, and each
  session's list contains exactly its own events.
- `build_sessions(events)`: group by user, order each user's events by
  timestamp (stable for equal timestamps), and split into sessions: a new
  session starts when the gap to the previous event is STRICTLY GREATER
  than `config.get("session_gap_s")`. A gap exactly equal to the limit
  stays in the same session. Returns sessions ordered by (user, first ts).

## stats.py

- `user_totals(sessions)`: `{user: total_cents}` summed over all their
  events (integer arithmetic).
- `avg_session_value(sessions, user)`: the user's total cents divided by
  their number of sessions, rounded HALF-UP to a whole cent (e.g. 5 cents
  over 2 sessions is 3, not 2). Integer/decimal arithmetic only.
- Results MUST reflect the current configuration: if `load_overrides`
  changes `session_gap_s`, previously computed values for the old setting
  must not be returned for the new one (recompute or key any cache by
  `config_version()`).

## report.py

- `top_users(totals)`: the top `config.get("top_n")` users as
  `"<user> <dollars>.<cents:02d>"` lines, sorted by total DESCENDING and,
  for EQUAL totals, by user name ASCENDING.

## Verification

`./verify.sh` runs the committed unittest suite. The suite is a subset of
the spec; conformance to THIS DOCUMENT is the requirement, not merely a
green suite.
