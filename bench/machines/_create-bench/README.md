# _create-bench: `agent6 machine create` across models

Measures `agent6 machine create` on one fixed task (a poll → classify → act
status monitor) across several providers/models: attempts, spend, wall time, and
whether the drafted bundle passes `machine check` + `machine test`.

Each model runs in its own throwaway repo with an isolated `AGENT6_STATE_HOME`
and its worker model pinned via per-repo config; provider keys come from the
global secrets store. The generated per-model run dirs are gitignored
(reproducible via `run.sh`); only this README, `run.sh`, and `results.jsonl` are
tracked.

## Running

```sh
bash bench/machines/_create-bench/run.sh
```

## Recorded numbers (2026-06-29, one run each, --max-attempts 4)

| provider | model | attempts | spend | wall | scripts | check | test |
|---|---|---|---|---|---|---|---|
| anthropic | claude-haiku-4-5 | 1 | $0\* | 22s | 5 | ok | ok |
| openrouter | moonshotai/kimi-k2.6 | 1 | $0.025 | 42s | 5 | ok | ok |
| anthropic | claude-sonnet-4-6 | 1 | $0\* | 47s | 5 | ok | ok |
| openrouter | z-ai/glm-5.2 | 1 | $0.023 | 178s | 5 | ok | ok |
| openrouter | openai/gpt-oss-120b | 4 | $0.011 | 104s | 0 | FAIL | FAIL |
| openrouter | deepseek/deepseek-v3.2 | 4 | $0.009 | 308s | 0 | FAIL | FAIL |

\* anthropic-direct models are unpriced in the OpenRouter price list agent6
caches, so their spend reads $0 even though tokens were used.

**4/6 authored a valid bundle on the first attempt** (machine + 5 scripts incl.
mock tests, both `check` and `test` green). The two failures are model-protocol
quirks, not agent6 faults; in both, agent6 detected the bad output, re-prompted,
and after 4 attempts failed cleanly with a precise diagnostic:

- **gpt-oss-120b** `silent_finish`ed every attempt: it answered in prose instead
  of calling `finish_run`, so no `result.toml` was ever returned. (The harmony
  format does not reliably emit tool calls in this headless structured-output
  loop.)
- **deepseek-v3.2** double-encoded the payload: it returned `result` as a
  JSON *string* (`'{"toml": "...", "scripts": {...}}'`) rather than a nested
  object, so the trust-boundary validator rejected it as "not a dictionary".

Notes:

- haiku is both the fastest (22s) and, at list prices, the cheapest path to a
  valid bundle; kimi and sonnet are close behind.
- GLM is correct but slow (178s), consistent with its tendency to over-reason.
- The pinned task is the canonical poll → classify → act shape the authoring
  guide is tuned for; harder or more novel control flow would stress the models
  (and the retry loop) more.
