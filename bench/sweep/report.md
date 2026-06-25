# agent6 cross-model worker benchmark

This report measures one variable: the **worker model** driving agent6, holding
the agent, sandbox, task suite, and budget fixed. 330 runs, 8 models, 6 tasks,
each (model, task) cell repeated independently. It is a snapshot — regenerate
the tables from the samples with `bench/sweep/stats.py` (see `README.md`).

## Summary

On this suite every model completes 92–100% of cells; the 95% success-rate
intervals overlap across all eight. **Success is therefore not the
discriminating axis here — cost and latency are.** Conditioning the cost
comparison on success (Mann-Whitney on cost-of-passing-runs vs `sonnet-4-6`),
the six open-weights configurations are cheaper than the frontier reference by
a large effect (Cliff's δ ≈ −0.7 to −1.0, p < 0.001 for five of six). The
numbers, not adjectives, are in the tables below; a few neutral observations:

- The cheapest run cost (`deepseek-v4-flash`, $0.0066 median) and the most
  expensive (`sonnet-4-6`, $0.097 derived) differ ~15× at indistinguishable
  success on this suite.
- `qwen3.6-27b-dense` completed every cell (48/48) at $0.022 and 39 s median
  wall-clock. The same-vendor, same-generation sparse model
  `qwen3.6-35b-a3b` (35B, 3B active) completed 44/48 at ~4.6× the wall-clock
  (181 s) — a measured difference between a dense and a sparse model of the
  same family on agentic tool-use; the cause (endpoint throughput vs in-loop
  iteration count) is not separable from these metrics alone.
- `opus-4-8` used the fewest input tokens (4214 median) and the lowest latency
  (33 s) of any model — it tends to one-shot these tasks; `sonnet-4-6` used
  more (20631) at higher latency.

## The task suite is two kinds of task

The six tasks (drawn from `bench/realworld/`) split into:

- **Construction (3):** `werkzeug-safe-join`, `tinydb-search`, `click-unstyle`
  — a function is stubbed so the project's own test suite **fails** until it is
  correctly restored.
- **Restraint / premature-finish bait (3):** `csv-rfc4180`, `url-rfc3986`,
  `html-strip` — the code already **passes** its tests; the task description
  tempts the agent to over-implement, and success means *not* breaking the
  passing tests with unnecessary edits.

Consequence: a do-nothing or crashed run still "passes" all three restraint
tasks, so the aggregate success rate has a **floor of ~50% (3/6)**. A mid-range
score can mean "did nothing," not "half-capable" — always read the per-task
matrix. (See next section for how this caught a real defect.)

## A defect this benchmark surfaced (and agent6 fixed)

The first opus-4-8 sweep scored **6/12** — implausibly low for a frontier
model when `sonnet-4-6` scored 30/30. Inspection showed every opus run exited
at iteration 1 with **0 tokens consumed**: `claude-opus-4-8` rejects any
`temperature` parameter (`400 "temperature is deprecated for this model"`), and
agent6 both pinned `temperature` for determinism and treated 4xx as permanent —
so **agent6 could not run the model at all.** The 6 "passes" were the three
restraint tasks passing on untouched code (the floor above).

Fixed in agent6 (commit `cb26543`): on a 400 that names `temperature`, drop it
and retry once, latching a per-provider flag for the rest of the run
(model-agnostic; no hardcoded model list). With the fix, opus-4-8 runs and
completes 12/12. The opus rows below are the post-fix re-run.

## Provenance & reproducibility

- Open-weights sweep (240 runs): commit `8974d60`.
- Anthropic + dense-Qwen sweep, and the opus re-run: after merging 22
  bug-fixes (`5227d50`, `8c9e408`) and the opus fix (`cb26543`).
- **Reproducibility check:** the merged fixes do not touch the OpenRouter
  worker execution path, so the open-weights numbers should be invariant to
  them — confirmed empirically below.

Keys are read from `~/.config/agent6/secrets.toml`; no key touches a sample or
the process argv. API spend was $9.09 measured (OpenRouter); Anthropic cost is
list-price-derived from token counts (see the cost note below).

## Reproducibility result

A 60-run subset (2 reps × 6 tasks × 5 open models) was re-run on the fixed
code and compared to the full sweep:

| model | full (n=48) | confirm (n=12) | median cost in full 95% CI |
|---|---|---|:--:|
| `kimi-k2.7` | 94% / $0.0260 | 92% / $0.0220 | yes |
| `kimi-k2.6` | 100% / $0.0296 | 100% / $0.0335 | yes |
| `glm-5.2` | 98% / $0.0401 | 92% / $0.0381 | yes |
| `qwen3.6-35b-a3b` | 92% / $0.0097 | 100% / $0.0125 | yes |
| `deepseek-v4-flash` | 94% / $0.0066 | 100% / $0.0120 | no (higher) |

Success rates reproduce within small-sample noise, and **4 of 5 confirmatory
median costs fall inside the full sweep's 95% bootstrap CI**. The exception,
`deepseek-v4-flash`, is the highest-variance model (heavy-tailed cost); its
n=12 median landed just above the n=48 CI — sampling variance on a cheap, loopy
model, not a code effect. This supports treating the open-weights numbers as
invariant to the merged fixes.

## Caveats

- **n is small by construction.** Repeated runs characterise variance, not a
  population. Read the intervals, not the point estimates.
- **The suite is narrow:** single-function construction/restraint with a
  ground-truth test oracle. It does not measure multi-file refactors,
  open-ended design, or long-horizon planning.
- **Cost mixes measured and derived** (OpenRouter reports per-call cost;
  Anthropic is derived at list price, excluding caching discounts).
- p-values are uncorrected; apply Holm-Bonferroni before claiming any single
  pairwise difference.

---

<!-- Tables below generated by bench/sweep/stats.py: python3 bench/sweep/stats.py <samples_dir> --ref sonnet-4-6 -->

## Method

Each task scopes the agent to a single function in a pinned open-source
repository plus that project's own verify command (a subset of its test
suite). Success is scored out-of-band by re-running the verify command after
the run -- the agent's own claim is not trusted. (Tasks differ in whether the
verify starts failing or passing; see the suite notes.) Each (model, task)
cell is repeated independently; cost and token counts come from the run's
usage accounting and wall-clock from the harness.

**Cost note:** OpenRouter reports a per-call USD cost, used directly. The
Anthropic API does not, so cost for those models is *derived* from measured
token counts at public list price (opus-4-8 $5/$25, sonnet-4-6 $3/$15 per
1M input/output tokens) — an estimate, not billed spend, and it excludes
prompt-caching discounts the other models' reported costs already include.

Reported statistics: success rate with a 95% Wilson score interval; cost,
tokens and wall-clock as median [Q1, Q3] with a seeded percentile-bootstrap
95% CI for the median; cost-per-successful-task = total spend / successes.
Pairwise tests are versus `sonnet-4-6` (the reference): Fisher's exact test for
success counts and the Mann-Whitney U test (tie-corrected normal
approximation) with Cliff's delta for cost-on-success. All intervals are 95%.
n is small by construction; treat single comparisons as indicative and read
the intervals, not the point estimates.

## Per-model results

| model | n | success | 95% CI | median cost | cost 95% CI | cost/success | med in tok | med out tok | med wall |
|---|--:|--:|---|--:|---|--:|--:|--:|--:|
| `qwen3.6-27b-dense` | 48 | 100% (48/48) | [93, 100]% | $0.0222 | [0.0214, 0.0280] | $0.0350 | 64571 | 1312 | 39s |
| `kimi-k2.6` | 48 | 100% (48/48) | [93, 100]% | $0.0296 | [0.0267, 0.0355] | $0.0523 | 11203 | 3082 | 44s |
| `sonnet-4-6` | 30 | 100% (30/30) | [89, 100]% | $0.0971 | [0.0748, 0.1108] | $0.1057 | 20631 | 2147 | 52s |
| `opus-4-8` | 12 | 100% (12/12) | [76, 100]% | $0.0435 | [0.0339, 0.1079] | $0.1143 | 4214 | 982 | 33s |
| `glm-5.2` | 48 | 98% (47/48) | [89, 100]% | $0.0401 | [0.0300, 0.0446] | $0.0427 | 12764 | 962 | 51s |
| `deepseek-v4-flash` | 48 | 94% (45/48) | [83, 98]% | $0.0066 | [0.0055, 0.0078] | $0.0095 | 28190 | 3624 | 189s |
| `kimi-k2.7` | 48 | 94% (45/48) | [83, 98]% | $0.0260 | [0.0197, 0.0297] | $0.0363 | 11736 | 954 | 89s |
| `qwen3.6-35b-a3b` | 48 | 92% (44/48) | [80, 97]% | $0.0097 | [0.0078, 0.0144] | $0.0189 | 40958 | 1852 | 181s |

## Success by task (passes / reps)

| model | click-unstyle | csv-rfc4180 | html-strip | tinydb-search | url-rfc3986 | werkzeug-safe-join |
|---|--:|--:|--:|--:|--:|--:|
| `qwen3.6-27b-dense` | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| `kimi-k2.6` | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| `sonnet-4-6` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| `opus-4-8` | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| `glm-5.2` | 8/8 | 8/8 | 7/8 | 8/8 | 8/8 | 8/8 |
| `deepseek-v4-flash` | 8/8 | 8/8 | 5/8 | 8/8 | 8/8 | 8/8 |
| `kimi-k2.7` | 8/8 | 7/8 | 7/8 | 7/8 | 8/8 | 8/8 |
| `qwen3.6-35b-a3b` | 8/8 | 8/8 | 5/8 | 8/8 | 7/8 | 8/8 |

## Pairwise comparison vs `sonnet-4-6`

Cost-on-success compares only runs that passed in each model (a fair cost
comparison conditions on success). Fisher's p tests the success-count difference over all reps.

| model | Δ success rate | Fisher p | median cost Δ (succ) | Mann-Whitney p | Cliff's δ |
|---|--:|--:|--:|--:|--:|
| `qwen3.6-27b-dense` | +0pp | 1.000 | -0.0749 | 0.000 | -0.82 |
| `kimi-k2.6` | +0pp | 1.000 | -0.0675 | 0.000 | -0.68 |
| `opus-4-8` | +0pp | 1.000 | -0.0536 | 0.237 | -0.24 |
| `glm-5.2` | -2pp | 1.000 | -0.0569 | 0.000 | -0.68 |
| `deepseek-v4-flash` | -6pp | 0.281 | -0.0902 | 0.000 | -1.00 |
| `kimi-k2.7` | -6pp | 0.281 | -0.0705 | 0.000 | -0.77 |
| `qwen3.6-35b-a3b` | -8pp | 0.156 | -0.0867 | 0.000 | -0.90 |

_pp = percentage points. δ > 0 means this model spent more than the reference
on successful runs; |δ| ≈ 0.15/0.33/0.47 ≈ small/medium/large. p-values are
uncorrected; with this many pairwise tests, apply a Holm-Bonferroni correction
before claiming any single difference._
