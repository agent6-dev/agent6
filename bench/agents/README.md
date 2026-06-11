# bench/agents: coding-agent comparison

Runs agent6, Claude Code, opencode, and aider headless on small coding
tasks in Go and Rust. The OpenRouter agents share one model so the harness
is the variable. Claude Code only runs Anthropic models, so its rows
measure agent plus model, not the harness alone.

| agent | model | $/MTok in/out |
|---|---|---|
| agent6 | moonshotai/kimi-k2.6 | 0.68 / 3.41 |
| aider | moonshotai/kimi-k2.6 | 0.68 / 3.41 |
| opencode | moonshotai/kimi-k2.6 | 0.68 / 3.41 |
| claude code | claude-haiku-4-5 | 1.00 / 5.00 |

## Tasks

Each task is a fresh git repo with a README spec, a committed test suite,
and a uniform `./verify.sh`. Agents get the same prompt and may not modify
tests. Success means the unmodified test suite passes after the agent
exits. Templates are in `tasks/`; the reference solutions used to validate
the test suites are in `tasks/_references/`.

- `go-logwindow`: implement a log parser and sliding-window aggregator
  from a spec with edge rules (inclusive cutoffs, tie-breaks, whitespace).
- `rust-ratelimit`: implement a rate-expression parser (typed error enum
  with precedence) and an integer-math token bucket (fractional refill
  carry, burst cap).
- `go-kvstore-debug`: a TTL+LRU store with four planted bugs; tests fail
  until all four are fixed.

## Running

```sh
bash bench/agents/run_all.sh                      # all combinations, sequential
bash bench/agents/run_one.sh go-logwindow agent6  # one combination
```

Cost per run is the OpenRouter key-usage delta (runs are sequential for
this reason); Claude Code reports its own `total_cost_usd`. Results land
in `$HOME/agentbench-runs/results.jsonl`.

## Results (2026-06-10, devcontainer, hardened sandbox for agent6)

Every run passed its task. Wall time includes agent startup. Where the
usage delta lagged, the agent's own report is used.

go-logwindow:

| agent | wall | cost | exit |
|---|---|---|---|
| agent6 before fixes | 161s | $0.60, budget exhausted | 3 |
| agent6 after fixes | 42s | $0.037, 4 calls | 0 |
| aider | 302s | $0.066 | 0 |
| opencode | 209s | $0.011 | 0 |
| claude code | 27s | $0.055 | 0 |

rust-ratelimit (agent6 ran with sandbox fixes but not yet the loop fixes):

| agent | wall | cost | exit |
|---|---|---|---|
| agent6 | 233s | $0.62, budget exhausted | 3 |
| aider | 627s | $0.020 | 0 |
| opencode | 172s | $0.011 | 0 |
| claude code | 29s | $0.066 | 0 |

go-kvstore-debug (agent6 ran with all fixes):

| agent | wall | cost | exit |
|---|---|---|---|
| agent6 | 23s | $0.033 | 0 |
| aider | 310s | $0.03 | 0 |
| opencode | 168s | $0.021 | 0 |
| claude code | 25s | $0.059 | 0 |

On go-logwindow the fixes took agent6 from $0.60 and 52 calls to $0.037
and 4 calls on the same task and model. On go-kvstore-debug agent6 made
4 tool calls (read, one edit fixing all four bugs, jailed verify,
finish_run) and had the lowest wall time of the four agents.

## Code quality notes

Read by hand after the runs; tests alone do not separate the agents.

- go-logwindow: agent6, opencode, and claude validated service names with
  ASCII checks per the spec. aider used `unicode.IsLower/IsDigit`, which
  accepts characters the spec forbids. claude allocates a lookup map per
  ParseLine call.
- rust-ratelimit: aider and claude used overflow-safe arithmetic (u128
  widening, checked_mul, map_err). agent6 and opencode have latent
  overflow panics on adversarial input. All four implementations share
  one semantic the tests do not pin: refill credit keeps accruing while
  the bucket is full, so a spend after a long idle refills sooner than a
  strict token bucket would.
- go-kvstore-debug: the three kimi agents produced byte-identical minimal
  fixes, including the same mutating `pruneExpired()` inside `Len()`.
  claude counted live entries without mutating, a better API choice.
- Summary: on well-scoped tasks the model determines the patch (identical
  fixes across three harnesses); the harness determines cost, latency,
  and how cleanly the run ends.

## What the bench exposed in agent6 (fixed in this round)

The first agent6 run passed its tests but spent the whole $0.60 budget:
52 calls, 580k input tokens, 44 run_command calls, zero run_verify calls.
The transcript showed the model fighting the sandbox rather than the task:

- `go test` failed in the jail because there was no writable HOME for
  GOCACHE, and the model then probed the sandbox instead of the task
- a wrong argv path was reported as "jail unavailable", so the model
  treated the sandbox as broken instead of fixing the path
- `cargo` could not build at all: a Landlock ruleset without
  `LANDLOCK_ACCESS_FS_REFER` denies every cross-directory rename or
  hardlink, and cargo hardlinks artifacts between target/ subdirs

The fixes (see the sandbox commit): jailed HOME on the writable /tmp,
Landlock ABI V2 with REFER on rw paths, exec failures reported as rc-127
results, and a prompt rule to run tests only through run_verify_command.

## Cost notes (kimi 2.6 via OpenRouter)

- Prompt caching depends on backend routing and varies day to day: round
  1 saw zero cached tokens on every call, the rerun and a direct probe on
  the same config got real hits. Without caching the growing-transcript
  loop pays full input price every iteration. Watch `cache_r` in run
  summaries; if it stays 0, pin a caching backend via
  `[providers.openrouter] extra_body = { provider = { order = [...] } }`.
- Wall time on kimi is dominated by model latency, not the harness.
