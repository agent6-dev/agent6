# Core-loop experiments: findings

Measurements from the `bench/coreagent` harness (see its README): multi-component tasks scored by a hidden grader; `score` is the fraction of grader cases passing (partial credit, low variance), `solved` is score 1.0.
Adoption rule: better everywhere becomes the default; helps some models and hurts others becomes an off-by-default knob; helps nowhere is removed.
Result JSONL for every experiment is under `results/`.

## Reproduce

```bash
cd bench/coreagent
python3 run.py --model <m> --provider <p> --tasks textkit,rpn,ledger,bugs \
    --conditions baseline,decompose --reps 4 --parallel 6 --label myrun
python3 stats.py results/myrun.jsonl
```

Read `stats.py --components` before trusting a delta: scores move on component curves, not only means, and n=4 cells disagreed with pooled n=12 cells on one task.

## Task decomposition (`[prompt].decompose`)

Score delta (decompose minus baseline) by model and task:

| model | textkit | rpn | ledger | bugs |
|---|---|---|---|---|
| mistral-small-3.2-24b | +0.53 (n=10; 0.18 -> 0.71 ± 0.12) | +0.13 (n=10; 0.85 -> 0.98, iterations 31 -> 22) | +0.18 (n=4) | -0.00 (n=10; batches of -0.14 and +0.21) |
| qwen3-coder-30b | 0.00 | 0.00 | 0.00 | 0.00 |
| qwen3.6-35b | 0.00 | 0.00 | n/a | n/a |
| claude-haiku-4-5 | 0.00 | 0.00 | 0.00 | 0.00 |

- The win is on tasks whose failure is dropping or under-finishing independent components (textkit's five functions); a debug task shows no reliable effect.
- Capable models sit at the ceiling both ways and pay 2 to 4x the iterations (haiku: 6 -> 29 on rpn, 8 -> 29 on bugs) creating subtasks for work they would do directly.
- State: `[prompt].decompose` defaults to `auto`, on per worker model from the capability registry; `--decompose` forces it for one run.

## Spiral guards and their thresholds

The constants in `workflows/_nudges.py` and the observations behind them:

| guard | thresholds | evidence |
|---|---|---|
| no-progress on verify (the same normalized error while the worker keeps editing) | nudge 4, escalate 7, stop 10 | mistral-small: nine identical verify failures burned a third of a run; on the guard waves (n=14) the detector fired on exactly the doomed runs (77-iteration score-0 spirals) and never on a healthy one; nudges rescued none, so the stop ends the run as resumable `no_progress` at 40 to 59 iterations, about a third of a doomed run's cost saved |
| tool-error spiral (repeated identical tool errors) | nudge 3, escalate 5, stop 8 | kimi re-issuing a malformed grep (a 117 KB pattern truncated by the output-token ceiling) until the run timed out |
| verify-broken detection | a verify that exits instantly without running its tests is flagged, not passed on as a red | `python -m pytest` with pytest absent, exit 1 in 0.0 s, on sympy testbeds across three models |
| verify-settled completion and the low-budget wrap-up | | Kimi K2.6 ran to 128 iterations on a task done at about 45, re-running read-only commands; another run solved the task, never re-ran verify and never called finish_session |
| task finish gate | patience 3 | a weak model quitting at silent_finish on iteration 7 with 7 subtasks open |
| early prose finish | bounced back to the tools at most twice, in the first iterations | kimi answered the problem statement in prose at iteration 2 and the loop accepted it as a finish |

Related fixes: back-to-back identical tool results are deduplicated (a re-read spiral had grown context to 125K tokens); compaction placeholders no longer instruct an identical re-call; the no-progress guard defers to metric runs (it would have truncated a budgeted optimisation search); a `run_command` spiral on a binary that exists on the host is named as a sandbox-reachability problem with the fixes (install, `--dangerously-disable-sandbox`, `extra_read_paths`); a perf wall timeout bounds a model that evades the budget cap with cheap unbounded reasoning.

kimi-k2.7-code on the SWE-bench Verified random-12 subset ($1 per instance, official evaluator), before and after the guards: resolved 5/12 -> 7/12, empty-patch spiral-outs 4 -> 2, mean cost per run flat at about $0.57.
The two gained instances are the two that had spiralled to an empty patch; the resolve move is inside single-run noise, the empty count is the signal.
On the same subset claude-sonnet-5 resolved 7/12 and claude-haiku-4-5 6/12 at about a seventh of sonnet's cost.

## Compaction

- agent6 compacts in two tiers: tier 1 elides the oldest tool results and keeps the context bounded; tier 2 summarises and restarts.
- A keep-last-K-verbatim tail for tier 2 measured inert on these tasks: under forced 18k/36k thresholds on the `needle` task (kimi, n=6 per condition) tier 2 fired 0 times in 18 runs, because tier 1 keeps the context below the tier-2 trigger; the 0.91 vs 0.96 score gap was noise.
- The measurable compaction cost is tier 1: tight elision dropped the reference files and the model re-read them 12 to 43 times per run (kimi 1.00 -> 0.91); at the shipped adaptive thresholds neither tier fires on tasks this size.
- State: a tier-2 restart keeps the last 80,000 characters verbatim (`[context].keep_recent_chars`); tier-1 gists are measured in `bench/longhorizon/FINDINGS.md`.

## Provider fixes found here

- Gemini and Gemma models emit tool calls as a fenced `tool_code` block in `content` with empty native `tool_calls`; the openai provider parses that block with `ast` (never executed), the same recovery path as the Qwen `<function=>` form.

## Head-to-head (`bench/agents`, shared models)

- Go tasks: agent6 is competitive on wall and cost (kvstore-debug with kimi: 30 s and $0.015, against Claude Code with haiku at 26 s and $0.055).
- rust-ratelimit: agent6 spent 202 to 350 s and $0.42 to 0.88 where Claude Code spent 23 to 44 s and $0.04 to 0.19, hunting for cargo through 32 commands; the cause was the bench box's jail-hostile rustup proxy install (no system rustc), not the verify cadence; the jail runs a system rust unmodified.
- agent6 cost cells use the agent's own accounting; the key-usage delta method needs an exclusive key.

## gpt-5.6-sol effort tiers on the dev slice

Three arms of the same 20 instances (bugs and ledger, the ChatGPT plan), differing only in `[models.worker].effort`:

| arm | n | score | wall per run | iterations | output tokens per run |
|---|---|---|---|---|---|
| low | 20 | 1.000 | 25 s | 4.0 | 886 |
| medium | 20 | 1.000 | 29 s | 4.0 | 1041 |
| max | 20 | 1.000 | 46 s | 4.3 | 1919 |

Every arm solves every instance; max costs +84% wall and +117% output tokens over low, so effort is a cost axis on this slice.

## Measured nulls

| lever | measurement | result | state |
|---|---|---|---|
| style riders (a detectable "end every prose reply with MOOSE" instruction) | qwen3-coder-30b and mistral-small, every channel (system base appended and prepended, user prompt), delivery byte-verified | 0 compliance in 0/12, 0/13, 0/13, 0/27, 0/30 prose turns; mistral obeyed only in the user prompt and overcomplied (128 emissions, score 0.0 vs 0.93); kimi-k2.7 ignored it, glm-5.2 emitted it once | style prose is not a lever in the loop |
| terseness rules (a one-line "be concise"; a 180-word ruleset) | 2 tasks x 2 models x n=3 | qwen narration rate 0.94 to 0.96 in every arm; mistral baseline already about 0; kimi emits no prose between calls | nothing shipped |
| an irrelevant 14-skill index (1.3 KB) vs length-matched padding | mistral textkit n=6 each | baseline 0.644 / $0.056 / 39.8 iterations; padding 0.444 / $0.095 / 52.3; index 0.350 / $0.092 / 54.5; qwen flat | mistral-small is fragile to any system-prompt addition; docs/config.md records that models almost never invoke a skill from the passive index |
| skill delivery (a file baked into the prompt, `[skills.state]` always, index + `use_skill`) | bugs, both models, n=3 | scores at ceiling; `use_skill` called 0 times in 6 runs with the index present | small models get skills through `always`, `/name` and `--skill`; the passive index is a capable-model affordance |
| the session-bootstrap pattern (`using-superpowers = "always"`) | bugs, n=3 per model | qwen: zero `use_skill` calls, identical scores; mistral: 3/3 invocations, score 0.64 vs 0.98 baseline, cost 1.9x | not recommended |
| a debugging-methodology skill | bugs | baselines 0.988 (qwen) and 0.976 to 1.0 (mistral): no headroom | untestable here |
| a compact operating brief as the system prompt (2.2k chars) | qwen3.6-35b and mistral-small, n=12; kimi n=4 | ledger -0.50 (6/12 -> 0/12 solved) and -0.39; rpn +0.17 and +0.21; bugs and textkit flat; kimi at ceiling | not shipped |
| a native spec-recheck bounce at finish | three models, n=6 per arm | no score gain beyond noise, a drop on mistral-small, +38 to 88% cost | removed |
| terse rules on capable models | kimi-k2.7-code and glm-5.2, textkit, n=4 | scores 1.0 in every arm; kimi output tokens 3901 ± 2851 vs 2566 ± 1997 (overlapping) | no action |

The ledger loss under the brief has a tool-shape cause: qwen rewrites the stub whole with `apply_edit kind="create"` on an existing file (refused), then narrates the fix without calling anything until the went-quiet guard ends the run; mistral thrashes small edits into a syntax error.
There is no whole-file overwrite move: `create` refuses an existing file and `replace` needs the byte-exact old text.

## Failure mining

Of 19 genuine unresolved SWE-bench and rust runs diagnosed against gold patches (34 agents, adversarially verified): 11 were hidden-test near-misses (the right file, a plausible fix discriminated only by a hidden test), 4 kimi degeneracy spirals, 3 a dead-verify cluster, 1 a cutoff.
For Claude-tier models the binding constraint is capability; harness work bounds waste and catastrophic outcomes.

## Methodology notes

- A wave counts a run only when its last provider transcript shows every tool call answered, the grader non-degenerate and usd > 0.
- mistral serving stalls (700 to 850 s) inflate its cost variance.
- The data dir (`XDG_DATA_HOME`) is isolated per run, so host-installed skills cannot leak into an arm; conditions interpolate `{ROOT}` and `USER_SUFFIX` for prompt-file and user-channel arms.
