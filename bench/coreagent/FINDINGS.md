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

The settled stop on steered runs (gpt-6-astra, n=4 per cell): 16 runs per arm on four tasks where a steer after the first edit asks for seven read-only steps before the work continues, 12 runs per arm on three tasks with no steer.
The task set is ad hoc; its scripts and raw rows are outside the tree.

| | stop at 6 idle turns | stop at 6, a steer restarts the count | armed only by a green verify | stop at 12 |
|---|---|---|---|---|
| steered: ended "passed" with the task unbuilt | 16 of 16 | 16 of 16 | 0 of 16 | 0 of 16 |
| steered: task done | 0 of 16 | 0 of 16 | 16 of 16 | 16 of 16 |
| steered: longest idle streak | 6 | 6 to 7 | 8 to 12 | 7 to 10 |
| no steer: task done | 12 of 12 | 12 of 12 | 12 of 12 | 12 of 12 |

- At 6 idle turns the stop ends every steered run in the middle of the operator's seven steps, on a green gate.
- Restarting the count at the steer moves each event one turn later and changes no outcome.
- With no steer no run reached 6 idle turns in any arm (longest 4), so the control shows no side effect of any arm.
- The stop at 12 ships: it keeps the net for a worker that re-runs read-only commands after its work is done, which arming on a green verify alone would remove under `verify_when = "finish"`.

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

## The no-route-out statement

gpt-6-astra, n=4 per cell: 16 runs per arm on four tasks where a package the task seems to want is absent from the jail, 12 runs per arm on three control tasks.
The task set is ad hoc; its scripts and raw rows are outside the tree.
Arms: no statement; one sentence in the `run_command` description; one run-prompt rule, present only when the run's commands have no route off the machine.

| | no statement | tool description | prompt rule |
|---|---|---|---|
| install or download attempts per run | 1.44 ± 1.09 | 1.06 ± 0.77 | 0.31 ± 0.48 |
| pass, missing-package tasks | 9 of 16 | 11 of 16 | 15 of 16 |
| turns, missing-package tasks | 10.38 ± 4.59 | 9.81 ± 2.99 | 10.31 ± 2.47 |
| pass, control tasks | 12 of 12 | 12 of 12 | 12 of 12 |
| turns, control tasks | 6.92 ± 1.56 | 7.58 ± 1.73 | 7.00 ± 1.54 |

- The prompt rule lowers install attempts by 1.12 per run (twice the standard error is 0.60) and raises the pass rate from 0.56 to 0.94 (twice the standard error is 0.29).
- The tool-description sentence shows no measured difference on either.
- Turns show no measured difference in either group, and the control pass rate is unchanged.
- Every pass on the two tasks that need a third-party import came from copying the package out of the system site-packages, which the jail reads on the measuring host, so the pass counts depend on that host.
- The prompt rule ships; the tool-description sentence does not.

## A memory write and the settled count

gpt-6-astra, four fix-the-failing-test tasks, n=4 per task and arm in each of two batches (64 runs); every task text asks for one memory fact once the harness suggests it, since without that sentence 0 of 2 smoke runs wrote one.
Arms: a memory-store write counts as an idle turn; a memory-store write restarts the idle count.

| | counts as idle | restarts the count |
|---|---|---|
| runs with the settled nudge at or after the first memory write, first batch | 4 of 16 | 0 of 15 |
| the same, second batch | 5 of 16 | 0 of 16 |
| turns, both batches | 9.6 ± 1.2 | 9.6 ± 1.2 |
| a memory fact written | 32 of 32 | 32 of 32 |
| ended by `finish_session`, exit 0 | 32 of 32 | 32 of 32 |

- The nudge counts cover the runs the advisory reached; it missed one, in the restart arm's first batch.
- The nudge after the harness's own advisory goes from 0.28 of runs to 0.00 (twice the standard error is 0.16).
- Turns, tokens out, memory writes and the end show no measured difference.
- The hidden check failed in 2 and 4 of 32 runs, all on one task where the worker rewrote the red test before the arms diverge; with a strengthened test allowed it passes 32 of 32 in both arms.
- Not exercised: a run that keeps writing memory restarts the count each time, so the twelve-turn stop never reaches it.
- The restart is adopted.

## The gate before the first turn

gpt-6-astra, n=4 per task and arm, two tasks per family (48 runs); no task text asks for a verify first.
Arms: no gate run of the harness's own before the first turn; one gate run before the first model call, recorded as the baseline, its verdict sent with the first turn when red.
The gate is `python3 -m unittest` over 21 tests, 0.5 s.

| family | measure | no start gate | start gate |
|---|---|---|---|
| an unrelated red test beside a feature task | ends `finished · gate was already red` | 0 of 8 | 8 of 8 |
| | `finish_session` returned as not green, per run | 2.0 | 0.0 |
| | turns | 13.0 | 8.9 |
| | tokens out | 1022 | 645 |
| | wall seconds | 104 | 77 |
| the red test is the task | turns | 7.2 | 5.8 |
| | hidden check passes | 8 of 8 | 8 of 8 |
| a green base, a feature task | turns | 8.8 | 9.5 |
| | gate runs per run | 1.62 | 2.88 |

- With an unrelated red test every difference exceeds twice its standard error (turns 0.5, tokens out 64, wall 13.7).
- Where the red test is the task, turns fall by 1.5 (twice the standard error is 1.0); on a green base no measure differs.
- Without the start gate no feature run verified before its first edit, so none recorded a baseline.
- The start gate adds one gate run to a run on a green base; a slow gate was not measured.
- Two defects in the measured arm: a green start gate armed the settled count from the first turn, and a red one made every fix a red-to-green flip, so the memory advisory fired in 8 of 8 fix runs against 2 of 8.
- Not adopted: the arm is on no release branch.

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
| the base-red notice's last clause ("those failures predate the task") | gpt-6-astra, two fix-the-failing-test tasks and two add-a-feature tasks beside an unrelated red test, n=4 per task and arm (32 runs); every task asks for one verify before the first edit, so the notice fired in 32 of 32 (without that sentence, 0 of 2 smoke runs: both edited before their first verify); a second batch of the fix tasks, 16 runs | hidden checks 8 of 8 in every cell with and without the clause; 0 of 16 feature runs edited the unrelated red test; feature-task turns 10.2 ± 0.9 with the clause, 9.4 ± 1.5 without (twice the standard error is 1.2); fix-task turns 8.5 ± 1.5 with, 6.6 ± 0.7 without in the first batch (lower by 1.9, twice the standard error is 1.2) and 7.1 ± 1.6 with, 7.0 ± 1.3 without in the second (twice the standard error is 1.5) | the clause is removed: no measured regression |

The ledger loss under the brief has a tool-shape cause: qwen rewrites the stub whole with `apply_edit kind="create"` on an existing file (refused), then narrates the fix without calling anything until the went-quiet guard ends the run; mistral thrashes small edits into a syntax error.
There is no whole-file overwrite move: `create` refuses an existing file and `replace` needs the byte-exact old text.

## Failure mining

Of 19 genuine unresolved SWE-bench and rust runs diagnosed against gold patches (34 agents, adversarially verified): 11 were hidden-test near-misses (the right file, a plausible fix discriminated only by a hidden test), 4 kimi degeneracy spirals, 3 a dead-verify cluster, 1 a cutoff.
For Claude-tier models the binding constraint is capability; harness work bounds waste and catastrophic outcomes.

## Methodology notes

- A wave counts a run only when its last provider transcript shows every tool call answered, the grader non-degenerate and usd > 0.
- mistral serving stalls (700 to 850 s) inflate its cost variance.
- The data dir (`XDG_DATA_HOME`) is isolated per run, so host-installed skills cannot leak into an arm; conditions interpolate `{ROOT}` and `USER_SUFFIX` for prompt-file and user-channel arms.
