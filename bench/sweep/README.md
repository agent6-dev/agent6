# bench/sweep: cross-model benchmark for agent6

A reproducible, statistically-summarised comparison of agent6 driven by
different worker models on a fixed task suite. The question is narrow and
measurable: **given the same task and the same agent, how do models differ in
success rate and cost?**

**Latest results: [`report.md`](report.md)** — 8 models (Kimi K2.6/K2.7, GLM 5.2,
Qwen3.6 27B-dense + 35B-A3B, DeepSeek V4 Flash, Claude Sonnet 4.6 / Opus 4.8),
330 runs. On this suite all models reach 92–100% success, so cost/latency is the
differentiator; the sweep also surfaced (and agent6 fixed) a real defect — agent6
could not run `claude-opus-4-8` at all.

## Task suite

The tasks are drawn from `bench/realworld/`: each one stubs out a single
function in a pinned open-source repository (or a self-contained module) and
asks the agent to restore the original behaviour until that project's **own
test suite** passes. The default subset spans a difficulty gradient:

| task | source | shape |
|---|---|---|
| `werkzeug-safe-join` | pallets/werkzeug 3.0.1 | path-traversal-safe join |
| `tinydb-search` | msiemens/tinydb v4.8.0 | query/predicate logic |
| `click-unstyle` | pallets/click 8.1.7 | ANSI escape stripping |
| `csv-rfc4180` | self-contained | RFC 4180 CSV parsing |
| `html-strip` | self-contained | HTML tag stripping |
| `url-rfc3986` | self-contained | RFC 3986 URL parsing |

Success is **not** the agent's self-report: after each run the harness
re-executes the task's `verify_command` out-of-band and uses its exit code as
ground truth. A run "passes" only if the upstream tests pass on the code the
agent left behind.

## What is measured, per run

`success` (bool), `cost_usd`, `input_tokens`, `output_tokens`,
`wall_seconds`, `agent_exit`. Cost/tokens come from the run's own usage
accounting; wall-clock from the harness. One JSON sample is written per run.

## Reproducing

API keys are read by agent6 from `~/.config/agent6/secrets.toml`
(`[providers.openrouter].api_key`, `[providers.anthropic].api_key`); no key is
passed on the command line or stored in any sample.

```bash
# show the plan + rough cost, run nothing
python3 bench/sweep/run_sweep.py --plan

# run the full sweep (resumable; re-run to fill only missing cells)
python3 bench/sweep/run_sweep.py --out /tmp/a6sweep --conc 6

# a single model / task while iterating
python3 bench/sweep/run_sweep.py --models kimi-k2.6 --tasks werkzeug-safe-join --reps 3

# summarise into a scientific-style report
python3 bench/sweep/stats.py /tmp/a6sweep/samples --ref sonnet-4-6 --out bench/sweep/report.md
```

Each run gets its own `BENCH_ROOT` (clone + venv, deleted after the sample is
extracted unless `--keep`). Upstream repos are mirrored locally once and git's
`insteadOf` redirects every per-run clone to the mirror, so the sweep never
rate-limits against GitHub.

## Statistics

`stats.py` is pure standard library (no numpy/scipy) so every number is
auditable. It reports:

- **success rate** with a 95% **Wilson score interval** (stable at small n and
  at p = 0 or 1, unlike the Wald interval);
- **cost / tokens / wall-clock** as **median [Q1, Q3]** with a seeded
  **percentile-bootstrap 95% CI** for the median;
- **cost-per-successful-task** = total spend / successes;
- pairwise vs a reference model: **Fisher's exact test** on success counts and
  the **Mann-Whitney U** test (tie-corrected normal approximation) with
  **Cliff's delta** effect size on cost-on-success (conditioning the cost
  comparison on success).

The bootstrap is seeded, so the report is byte-reproducible from the samples.

## Caveats (read before quoting any number)

- **n is small by construction.** Repeated runs characterise variance; they do
  not turn a 6-task suite into a population estimate. Read the intervals, not
  the point estimates.
- **The suite is narrow:** single-function restoration with a ground-truth test
  oracle. It rewards models that can localise and implement a well-specified
  change; it does not measure open-ended design, multi-file refactors, or
  long-horizon planning.
- **Cost depends on pricing** at run time and on agent6's budget/caching
  settings, which are held fixed across models here but are not universal.
- p-values are uncorrected; apply a Holm-Bonferroni correction before claiming
  any single pairwise difference.
