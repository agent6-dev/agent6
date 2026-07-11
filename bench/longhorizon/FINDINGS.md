# Long-horizon experiments — findings

Day-1 numbers from the harness this directory documents (see README). All
runs: real OpenRouter models, hidden-grader partial credit, records in
`results/`. Adoption rule unchanged from bench/coreagent: strictly better →
default; helps-some → knob; helps-nowhere → scrap.

## 1. Tier-1 compaction losses are real, and they are regime-gated

qwen3-coder-30b, stylebook (10-rule retention probe) and relay (6-stage
pipeline), n=3 per cell:

| task | baseline | window32k | window16k |
|------|----------|-----------|-----------|
| stylebook score | 0.921 ± 0.022 | 0.825 ± 0.033 (−0.10) | **0.425 ± 0.202 (−0.50)** |
| stylebook drops / rereads / iters | 0 / 0.3 / 25 | 0.3 / 4.3 / 29 | 65 / 71.7 (all post-drop) / 98 |
| relay score | 0.975 ± 0.006 | 0.956 ± 0.025 (flat) | 0.994 ± 0.006 (flat) |

- **window32k is nearly inert for a tidy reader**: qwen keeps total
  tool-result volume ~50k chars, under the 58k drop threshold; drops fired
  in 1 of 6 wave-1 runs. The shipped adaptive thresholds on big-window
  models never engage on tasks this size (as bench/coreagent predicted).
  Heavy readers do engage there: kimi-k2.6 crossed it on the same task.
- **window16k (a real local-serving default) is a death spiral on
  retention work**: first drop lands mid-reading (tool call ~9), then
  elision → re-read → more pressure → elision, averaging 65 dropped
  results, ~5 tier-2 restarts, 4x the iterations, and half the score. The
  per-rule retention curve shows broad destruction (every rule ≤ 0.67 at
  n=3; in the mildest rep the early-read rules died first: r04 0.14, r01
  0.33, vs late/structural r06/r08/r09 at 1.00). Every redundant read is
  post-drop.
- **relay is score-immune in the same regime** (0.994, +11% iterations;
  one rep paid +58% with a tier-2 restart and still landed 1.0): code on
  disk is cheaply re-readable, spec nuance is not. Compaction taxes
  retention tasks in CORRECTNESS and implementation tasks in EFFICIENCY.
- **Facts-ledger gate: OPEN, scoped.** A tier-1 mitigation (facts ledger or
  salience-keep) is worth building FOR SMALL-WINDOW DEPLOYMENTS and should
  be A/B'd here under window16k on stylebook (score) + relay (iterations).
  Nothing so far justifies changing behavior for 128k+ windows.
- Task fairness check: kimi-k2.6 produced a PERFECT 1.0 stylebook (all 12
  components) under window32k with a drop — the ceiling is reachable under
  compaction pressure; it just timed out at 2400s before finish_run
  (103k output tokens of reasoning; see methodology).

## 2. Nobody writes memories unprompted

Across all 46 legs / 2 models (qwen3-coder-30b, kimi-k2.6) spanning
orchard, relay, stylebook: `add_memory` calls = **0**. The `<memories>` block's
read side is proven (fake-provider e2e, 2026-07-06), but the write side is
inert as shipped: models with a discovered non-obvious fact in hand (the
orchard generated-file trap) never record it. Consequently baseline vs
`fresh_state` on orchard is flat by construction.

**Next step:** strengthen the write-side nudge (the <memories> header is
the only prompt surface; consider mentioning add_memory at the moment of a
verify-revealed surprise, or in the finish_run gate), then re-run
`mem1` here. Until a model actually writes, memory value on long tasks is
unmeasurable and UNPROVEN, not disproven.

## 3. add_dependency: unused so far

`deps_added = 0` across every leg, including relay (a strict 6-stage
dependency chain, the friendliest possible shape). Standing decision says
rip it out if useless; day-1 evidence leans useless, but n is small and no
weak model (the DAG-heavy population) has run yet. Verdict deferred to a
mistral-small / decompose-on wave.

## 4. The orchard trap catches real behavior

qwen leg-1 (smoke2): hand-edited the generated `data/catalog.tsv` 3 times,
tested around verify, called finish_run believing it was done — hidden
score 0.889 (exactly the unfixed-seed signature). Leg 2 then went through
verify, which regenerated the file and resurfaced the failure; the agent
root-caused `tools/catalog_source.tsv` properly and scored 1.0 including
the half-up-vs-banker's discriminator (F-310 → 909). kimi never falls in
(8/8 legs 1.0, trap_edits 0). The task separates exactly the population
the memory experiment needs.

## Methodology notes

- qwen3-coder-30b is the workhorse: $0.03–0.07 and 1–5 min per run.
  kimi-k2.6 spends 15–25 min inside a single reasoning turn when asked to
  write a precise module in one shot (30k+ output tokens before its first
  edit); use it as the slow second model, not the screen.
- Scores move on component curves, not just means: read `stats.py
  --components` before trusting a delta.
- `run_waves.sh` holds the fuller matrix (kimi compaction cells, more
  reps) for when more coverage is wanted (~$10–25).
