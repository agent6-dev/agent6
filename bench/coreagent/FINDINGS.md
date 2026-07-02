# Core-agent experiments — findings

Two thrusts on the agent6 core loop — front-loaded task **decomposition** and
keep-last-K-verbatim **compaction** — plus one opportunistic provider fix found
along the way. All numbers from the `bench/coreagent` harness (see README):
multi-component, hidden-grader-scored tasks; `score` = fraction of grader cases
passing (partial credit, low variance); `solved` = score == 1.0.

Adoption rule: strictly-better-everywhere → default; helps-some/hurts-others →
off-by-default config knob; helps-nowhere → scrap (no cruft "just in case").

## Thrust 2 — task decomposition (`[prompt].decompose`) → SHIPPED as off-by-default knob

**Hypothesis:** small models finish multi-part tasks better when forced to break
the task into DAG subtasks first and work one at a time (the existing
surface-current-task + finish-gate machinery then walks the frontier).

**Result: helps a small model with headroom on every task; pure overhead on
capable models. → off-by-default config knob.**

Δscore (decompose − baseline), by model × task (mistral at n=10 where deepened):

| model | textkit | rpn | ledger | bugs | verdict |
|---|---|---|---|---|---|
| **mistral-small-3.2-24b** | **+0.53** (n=10) | **+0.13** (n=10) | +0.18 (n=4) | -0.00 (n=10) | win on implement-many |
| qwen3-coder-30b | 0.00 | 0.00 | 0.00 | 0.00 | ceiling, overhead |
| qwen3.6-35b | 0.00 | 0.00 | — | — | ceiling, overhead |
| claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 | ceiling, overhead |

- **mistral-small-3.2-24b** is the one model that prematurely finishes / drops
  components. Decompose + the finish-gate force it to address every component →
  large robust win on **textkit** (5 independent functions; baseline 0.18 → 0.71,
  ±0.12, n=10) and a solid win on **rpn** (0.85 → 0.98). On rpn it also cut
  iterations (~31 → 22, n=10): the finish-gate keeps it on-task instead of
  thrashing.
- The win is mechanism-specific, not universal across task TYPE: on **bugs**
  (a debug task) decompose shows NO reliable win — n=10 mean is flat (−0.00), and
  that average hides two disagreeing batches (deep n=6 −0.14, screen n=4 +0.21),
  i.e. high variance, not a clean effect either way. Fixing 7 seeded bugs isn't a
  "forgot a component" failure, so up-front decomposition has no clear handle.
  Decompose helps where the failure mode is dropping or under-finishing
  INDEPENDENT components, which is exactly its design intent.
- **Capable models** (qwen3-coder-30b, qwen3.6-35b, haiku) sit at the score
  ceiling both ways; decompose only adds overhead. haiku is the cleanest demo:
  score 1.00 → 1.00 but iterations **6 → 29 (rpn), 8 → 29 (bugs)** — a ~4x tax for
  zero gain, because it dutifully creates 5-8 subtasks for work it would have done
  directly. This 2-4x overhead is exactly why decompose must default off.
- `solved` (full credit) moves less than `score`: decompose reliably gets MORE
  components right, but a weak model still misses some edge cases.

**Decision:** off-by-default `[prompt].decompose`. Not a default (overhead on
capable models rules out strictly-better); not scrapped (mistral's consistent
wins rule out helps-nowhere). Documented in docs/config.md, visible in
`config show`. Turn it on when driving a small/open model on a multi-component
implementation task. (Refinement worth noting: the original `<dag-rules>` block
already nudged toward decomposition; making it the explicit first step plus the
finish-gate is what converts mistral's premature finishes into full coverage.)

## Thrust 1 — keep-last-K-verbatim compaction (`[context].keep_recent_chars`) → SCRAPPED

**Hypothesis (COMPACTION_RESEARCH.md top idea):** agent6's tier-2 restart to
`[task, summary]` is the field's most aggressive choice; every other agent
(Anthropic, Codex, Cursor, LangChain) keeps a recent verbatim tail. Keeping the
last K balanced turns should cut post-compaction re-reads and cold-restart stalls.

**Result: structurally inert in agent6's compaction dynamics → scrapped.**

I implemented it cleanly (a `select_verbatim_tail` helper + a hybrid
`_summarise_and_restart`, `keep_recent_chars=0` = the historical hard restart,
6 unit tests, default-off). Then I measured it, and it does nothing:

- agent6 compaction is **two tiers**: tier-1 elides oldest tool_results (fires
  constantly, keeps context bounded); tier-2 summarises + restarts (what
  `keep_recent_chars` changes). `keep_recent_chars` touches ONLY tier-2.
- **Tier-2 almost never fires.** Because tier-1 keeps the context bounded *below*
  the tier-2 trigger, tier-2 only fires on pathologically long runs (a 100+-turn
  weak-model spiral), and there it is confounded with the model's own variance.
  In a forced-compaction A/B — kimi on the `needle` task (read five ~6k-char ref
  files, retain all five buried rules), aggressive 18k/36k thresholds, n=6 per
  condition — **tier-2 fired 0 times in all 18 runs.** keep_recent_chars was
  completely inactive; the hard-vs-hybrid score gap (0.91 vs 0.96) is noise (the
  hybrid even had MORE re-reads, 22 vs 15).
- **The real compaction cost is tier-1, which keep-last-K doesn't address.** Tight
  tier-1 elision dropped the ref files and the model re-read them 12-43x per run,
  knocking kimi from 1.00 (no compaction) to ~0.91. That is a tier-1 phenomenon,
  and only at artificially tight thresholds — at the shipped adaptive thresholds
  (~45%/80% of a 128k-256k window) neither tier fires on tasks this size.
- agent6's durable **task DAG + restart notice + surface-current-task** already
  carry task state across a restart, which is the continuity keep-last-K provides
  for memoryless chat agents. agent6 isn't memoryless, so the premise doesn't hold.

**Decision:** scrap (reverted from the tree; implementation preserved in
`scratchpad/thrust1_keeplastk_SCRAPPED.diff` and described here). No measurable
benefit in any regime I could produce → no cruft.

**Where the real compaction lever is (future work):** the cost lives in **tier-1**
(elision causing re-reads on long runs) and in summary fidelity, not in the tier-2
tail. The high-value, A/B-able candidates from the research are tier-1 salience-keep
(don't elide a file the worker is still editing) and a deterministic facts ledger
(record `path:line` + verify outcomes, no LLM call) — see COMPACTION_RESEARCH.md
#1/#6. Both need a genuinely long-horizon benchmark to validate (these short
test-graded tasks don't trigger realistic compaction); building that is the right
next step before touching compaction again.

## Opportunistic fix — Gemini/Gemma `tool_code` tool calls

gemma-3-27b scored ~0 on every task: it emits tool calls as a fenced
```` ```tool_code\n[read_file(path='spec.md')]\n``` ```` block (Gemini family) in
`content` with empty native `tool_calls`, which agent6 didn't parse →
`silent_finish` after 1 iteration. Fix: parse that block with `ast` (never
executed) in `providers/openai.py` `_coerce_text_tool_calls`, the same recovery
path as the Qwen `<function=>` XML form. **Live-validated:** post-fix, gemma parses
a 5-call ```tool_code add_task plan and drives the DAG (0 → real engagement). It
still fails the tasks, but for unrelated reasons — heavy 429 rate-limiting on
OpenRouter's free DeepInfra backend, and an occasional UNFENCED `[fn(...)]` list
(a known remaining limit; the fenced form is the documented, unambiguous one).
A strict improvement for the whole Gemini/Gemma family. Shipped with 3 regression
tests.

## Reproduce

```bash
cd bench/coreagent
python3 run.py --model <m> --provider <p> --tasks textkit,rpn,ledger,bugs \
    --conditions baseline,decompose --reps 4 --parallel 6 --label myrun
python3 stats.py results/myrun.jsonl
```
Result JSONL for every experiment is committed under `results/`.
