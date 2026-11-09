# Session ordering

Reading journal tails preserves recorded activity times without replaying session history to select the newest session.
Full session summaries still replay journals for their other fields.

## Method

Run from a scratch directory on disk, with the checkout's environment installed:

```sh
checkout=$PWD
scratch=$(mktemp -d /var/tmp/session-ordering.XXXXXX)
cd "$scratch"
"$checkout/.venv/bin/python" "$checkout/bench/session_ordering.py"
```

- Linux, disk-backed scratch storage, three sequential replicates per operation and implementation, with no concurrent test run.
- The full-fold comparison loads the unchanged parent `session_mtime` into the listing module; the tail comparison uses the working tree implementation.
- 100 synthetic sessions, each with 10,000 streaming deltas and a final timestamped event: 112,004,300 journal bytes (106.8 MiB).
- Filesystem times run opposite to recorded activity; every lookup must select the session with the latest recorded event.
- `newest` calls the CLI's no-id resolver; `ordering` calls the shared session-directory listing.
- Timings exclude imports and fixture creation; file contents are warm from writing the fixtures, with no cache flush between replicates.
- Read counts come from `/proc/self/io`'s `rchar`: bytes returned by reads, including cached reads and the counter itself, not physical disk traffic.
- Host contention was not controlled; these are synthetic lookup measurements, not full listing-render times.

## Results

Seconds in replicate order; standard deviation is the sample standard deviation.

| Implementation | Operation | Seconds | Median | Standard deviation | Bytes read per lookup |
|---|---|---|---|---|---|
| Full journal fold | newest | 5.546898, 6.316149, 5.135776 | 5.546898 | 0.599173 | 112,006,294 to 112,006,298 |
| Full journal fold | ordering | 7.036269, 7.508972, 6.982545 | 7.036269 | 0.289673 | 112,004,425 |
| Tail timestamp | newest | 0.011764, 0.011184, 0.010887 | 0.011184 | 0.000446 | 821,189 to 821,190 |
| Tail timestamp | ordering | 0.010634, 0.010468, 0.010383 | 0.010468 | 0.000127 | 819,318 |

`tests/unit/test_session_mtime.py` bounds reads to 8 MiB over 100 sessions with more than 100 MiB of journals, across the no-id resolver, directory ordering and newest-with-child lookup.
All three cases failed against the full fold at about 116 MB read and passed with tail lookup.
The test uses Linux read accounting rather than a wall-clock threshold and skips where `/proc/self/io` is absent.

Tail lookup has no fixed byte cap: a large final event, torn suffix or untimed journal can require more reads to preserve the recorded timestamp and legacy fallback rules.
It stops at the last parseable event timestamp and uses the full scanner's timestamp conversion.
