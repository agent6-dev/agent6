# kvstore (debugging task)

This package is an in-memory key-value store with per-key TTL and LRU
eviction. It has bugs: `go test ./...` fails.

Find and fix the bugs in `kvstore.go`. Do NOT modify `kvstore_test.go`.
The test file documents the intended behavior; the README and code
comments are correct descriptions of intent.

Intended behavior:
- Set(key, value, ttl): stores value; ttl <= 0 means no expiry.
  Overwriting a key refreshes its TTL and makes it most-recently-used.
- Get(key): returns the value and true if present and not expired;
  a Get makes the key most-recently-used. Expired keys behave exactly
  like missing keys.
- Len(): number of live (non-expired) keys.
- Capacity: when a Set of a NEW key would exceed capacity, the least
  recently used live key is evicted first. Expired keys are pruned
  before evicting anything live.
- A key expires when now >= insertion_time + ttl.

The clock is injected (`now func() time.Time`) so tests are
deterministic.
