# bench/machine: machine create bench

Measures `agent6 machine create` end to end: attempts, spend, wall time,
and whether the produced bundle passes `machine check` and `machine test`.

## Running

```sh
bash bench/machine/run_create_bench.sh         # 3 runs
RUNS=5 bash bench/machine/run_create_bench.sh  # more runs
```

Each run uses a fresh git repo and a fixed task (poll a feed, classify
with an LLM, record paper trades). Results land in
`$BENCH_ROOT/results.jsonl` (default `/tmp/agent6-create-bench`).

What good looks like: one attempt, a bundle that ships scripts plus mock
tests, both `check` and `test` passing, cost dominated by output tokens
rather than retries.

## Recorded numbers (tweet paper-trader task, one run each)

| date | model | attempts | in/out tokens | cost | wall | check/test |
|---|---|---|---|---|---|---|
| 2026-06-10, before retry and guide fixes | kimi-k2.6 | 3 | ~7k/14k per attempt | $0.18 | ~13 min, 2 dropped connections | test failed on a dry-run bug, since fixed |
| 2026-06-10, after | kimi-k2.6 | 1 | 6.5k/16.7k | $0.083 | ~11 min | both pass |
| 2026-06-10, after | claude-sonnet-4-6 | 1 | 6.6k/6.1k | ~$0.11 | ~5 min | both pass |

Notes:

- Output and reasoning tokens are about 90% of create cost on kimi; the
  prompt side is ~$0.006 per attempt. Cutting attempts is the lever, not
  trimming the guide.
- Carrying the prior scripts in retry prompts plus a guide note on list
  interpolation took the task from 3 attempts to 1 on kimi.
- Headless non-streaming calls dropped twice mid-generation on OpenRouter
  before machine agents switched to always-stream; none after.
- Sonnet writes about a third of kimi's output tokens for the same bundle
  and halves the wall time, at a similar cost at list prices.

## Line counter

One `machine create` task specified counting nonblank lines, writing the count and checking that it was a nonnegative integer.
Both creates used `chatgpt/gpt-6.2-sol` with medium worker effort configured.

| Draft effort sent | Attempts | `machine check` | `machine run` |
|---|---:|---|---|
| `none` | 1 | No draft. HTTP 400: “Unsupported value: 'none' is not supported with the 'gpt-6.2-sol' model. Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'.” | Not run |
| `medium` | 2 | Passed, 4 states and 2 scripts. The first draft lacked `capture` on a tool state with `output_schema`. | Reviewed and committed bundle reached `done` in one transition. `count.txt` contained `2` for two nonblank lines. |
