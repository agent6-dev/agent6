# Long-horizon experiments: findings

Measurements from the `bench/longhorizon` harness (see its README): multi-leg tasks over a repository whose conventions a later leg needs (orchard, three legs; ledger, five legs), a retention probe (stylebook, ten rules), a six-stage pipeline (relay); hidden-grader partial credit; records in `results/`.
Conditions: `baseline` (the cross-run memory store persists between legs), `fresh_state` (each leg starts empty), `poisoned` (stale memories planted before leg 2), `window16k` and `window32k` (compaction thresholds forced down).
Adoption rule: better everywhere becomes the default; helps some models becomes a knob; helps nowhere is removed.

## Compaction under pressure

qwen3-coder-30b, n=3 per cell, bare tier-1 placeholders:

| task | baseline | window32k | window16k |
|---|---|---|---|
| stylebook score | 0.921 ± 0.022 | 0.825 ± 0.033 | 0.425 ± 0.202 |
| stylebook drops / re-reads / iterations | 0 / 0.3 / 25 | 0.3 / 4.3 / 29 | 65 / 71.7 / 98 |
| relay score | 0.975 ± 0.006 | 0.956 ± 0.025 | 0.994 ± 0.006 |

- A tidy reader keeps its tool-result volume (about 50k chars) under the 58k drop threshold, so window32k is nearly inert; a heavy reader (kimi-k2.6) crosses it on the same task; the shipped adaptive thresholds on large-window models never engage on tasks this size.
- window16k is a spiral on retention work: the first drop lands mid-reading, elision causes re-reads, re-reads cause more pressure; every rule scores at most 0.67 and the early-read rules die first.
- relay is score-immune in the same regime: code on disk is cheaply re-readable, spec nuance is not; compaction taxes retention tasks in correctness and implementation tasks in efficiency.

Tier-1 gists (`[context].elision_gists`, default on): a large read result decays to a model-written gist before the bare marker, one batched summariser call per drop event.

| stylebook under window16k | qwen gists on | qwen gists off | kimi-k2.6 baseline | kimi gists on | kimi gists off |
|---|---|---|---|---|---|
| score | 0.825 ± 0.040 | 0.533 ± 0.264 | 1.000 | 0.983 ± 0.017 | 0.917 ± 0.083 |
| iterations | 53.7 | 106.0 | 11 | 10 | 103 |
| drops / tier-2 restarts | 19.3 / 2.7 | 36.7 / 5.0 | 0 | 13.7 | 438 |
| re-reads | | | 0.7 | 6.0 | 449.7 |
| cost per leg | $0.069 | $0.128 | $0.29 | $0.37 | $0.82 |

- Gists remove the re-read spiral: qwen's window16k stylebook rises from 0.425 with bare markers to 0.825 against the 0.921 uncompacted baseline, with 6 of 12 components at or above 0.90 and the spread collapsed; kimi with gists sits within noise of its uncompacted baseline on every axis.
- Cost halves for qwen even with the summariser calls (58 gists written across the wave, 10 demoted to bare markers, 0 distiller failures).
- relay never engages the gist path (0 gists across 6 legs) and stays flat: no harm outside the engaged regime.

## Cross-run memory

- Unprompted, no model records a memory: 0 writes across 46 legs on qwen3-coder-30b and kimi-k2.6, with a discovered non-obvious fact in hand (orchard's generated-file trap).
- Two write-side nudges ship: an advisory at the first red-to-green verify flip, and a once-deferred `finish_session` after such a recovery when nothing was recorded. Writes went from 0.0 to 0.5 to 0.8 per leg (qwen, n=4 per cell); the flip advisory converted about half the writers and the finish backstop the rest; decliners finished cleanly on the second call. Of 9 stores written, 8 are the durable trap facts, 1 a confidently wrong rationalisation, 0 task-progress junk.
- Read-side value is real on a weak model. Orchard's third leg re-probes the generated-file rule and the half-up rounding rule with fresh discriminators the shipped tests cannot see (qwen, n=6 per cell): baseline 0.922 ± 0.038 vs fresh_state 0.750 ± 0.097, against a same-wave leg-1 calibration delta of -0.019; components rounding 0.78 vs 0.33, api 1.00 vs 0.83, regen 0.91 vs 0.78; trap edits 0.0 vs 0.7 per leg; iterations flat (30.8 vs 31.2). Memory buys correctness here, not speed.
- What the memory says predicts what transfers: a store that spelled out "rounding half-up" in words carried to a 1.0; a store that only encoded the catalog formula (half-up implicit in `(x+50)//100`) carried the generated-file fact and truncated the new computation. A memory records the rule, not the instance.
- Capable models have no headroom: claude-haiku-4-5 and claude-sonnet-5 score 1.0 on all three orchard legs in both conditions and rediscover the conventions every leg.
- The ledger campaign on gpt-5.6-sol (3 reps x 3 conditions, 45 sessions): score 1.000 on every leg and component in every condition. With the shared store, 1.7 writes and 0.8 reads per leg (2.7 memory files by the end); under fresh_state 1.5 writes and 0.2 reads. Memory does not make later legs cheaper for this model: tokens in 37.9k vs 35.3k on later legs, tool calls 33.7 vs 31.2, re-reads flat.
- Poisoned: the score is unchanged (the model verifies the repo over the stale memory every time), later legs cost +15% tokens in (43.4k), and the planted memories were rewritten in 1 of 12 legs and never removed. Two confidently wrong memories were observed across waves; no model removed or corrected one.

## Tools measured

- `add_dependency`: unused by qwen and kimi across every leg, including relay's strict chain; mistral-small-3.2 (decompose on) calls it unprompted, 1.7 edges per leg on orchard-weekend and 0.3 on relay, with sensible investigate-first fan-outs. Kept: a weak-model affordance, free when unused.
- The orchard trap catches real behaviour: qwen hand-edits the generated `data/catalog.tsv`, tests around verify, finishes believing it is done (hidden score 0.889, the unfixed-seed signature) and root-causes it in leg 2 when verify regenerates the file; kimi never falls in (8/8 legs at 1.0); mistral-small falls in 3/3 and never recovers. With the store wiped the trap re-fires on the same model within one sequence (qwen fresh_state hand-edited the clearance feed in 2 of 6 leg-3 reps, baseline in 0 of 6).
- `trap_edits` matches the edit call's `path` when present; repeated hand-edit attempts each count, so a faller's leg-1 count runs 4 to 6.

## Provider findings

- The Anthropic SSE idle watchdog killed sonnet-5 on hard tasks: adaptive thinking emits only ping heartbeats for over 45 s, and the 45 s mid-stream budget aborted every long think (stylebook 0/3, provider_error at 3 to 4 iterations). A thinking block now gets a patient idle budget; the same cell runs 3/3 at 1.0 (10 iterations, $0.39, about 205 s). Before and after: `results/sb-sonnet.jsonl`, `results/sb-sonnet-fixed.jsonl`.
- Anthropic thinking is configured per model family: adaptive thinking on the models that refuse `budget_tokens`.

## Methodology notes

- qwen3-coder-30b is the workhorse ($0.03 to 0.07 and 1 to 5 min per run); kimi-k2.6 spends 15 to 25 min inside one reasoning turn when asked to write a precise module in one shot (30k+ output tokens before its first edit): the slow second model, not the screen.
- Read `stats.py --components` before trusting a delta.
- Bench waves cap memory (`MemoryMax` on the unit plus `ulimit -v` under it): a leg's sandboxed process once allocated 5.3 GB on an 8 GB box with no swap and took the host down; a resumed wave restarts rep ids, so cells stay balanced by count.
- `--provider anthropic` runs token-capped (Anthropic is unpriced here); `--timeout-scale 2.0` lets kimi's 15 to 70 min stylebook legs finish.
- `run_waves.sh` holds the fuller matrix (about $10 to 25).
- The grader is validated per task: the reference scores 1.0 on every leg; the untouched seed 0.0 on later legs; a hand-edited generated file loses exactly `regen`; a banker's-rounding mutant loses exactly `rounding`.
