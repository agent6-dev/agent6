# agent6 on SWE-bench: findings

Measurements of agent6 on SWE-bench Verified and SWE-rebench with the official scorers, and what they showed about the agent.
Numbers are stated with their split, n and date; a sample is never the full set.

## The harness (`bench/swebench/`)

- `run_sweep.py` pulls each official instance image, runs agent6 inside it and takes the working tree's diff as the prediction, with test-file diffs stripped (the agent can never touch the graded tests).
- `in_container.sh` installs agent6 from a locally built wheel (`AGENT6_JAIL_TARGET=musl`, so the jail binary runs on the images' older glibc), writes the run config and runs the task.
- `score.py` wraps the unmodified `swebench.harness.run_evaluation` (FAIL_TO_PASS plus PASS_TO_PASS).
- The container is the isolation: runs use `sandbox.isolation = "none"` inside SWE-bench's own Docker images (the jail has its own tests).
- Splits: `sample_50.json` (Verified, seed 20260623, held out, never tuned on); `dev_slice_25.json` (25 of the 450 non-eval instances, the tuning slice); random draws by registered seed for coverage; SWE-rebench 2026_03 (110 instances, `fetch_rebench.py`, prebuilt images).
- Caps: $1 of tokens per instance at list price, 1200 s wall, effort medium, concurrency 1 to 6 on 4 to 16 cores.
- Verify is auto-detected per repo (pytest `-q -x`; django `runtests.py --parallel 2`), `verify_timeout_s = 240`; `AGENT6_SB_VERIFY=none` runs gateless through `[workflow].verify_infer = false` (without it, adoption re-arms an inferred gate).
- Every instance gets the same four-line AGENTS.md steering derivation over upstream-fix recall; the same file is planted for every agent compared.
- `AGENT6_DETACHED_AWAY=deny`: an approval nobody can answer is denied instead of parking the run until the wall.
- The patch is `git add -A` minus the files untracked at the base commit and agent6's own files, so a module the run created is submitted (four of the rebench 110 need one).
- A failed image pull writes no prediction (`pull_failed`); Docker Hub's anonymous window refuses at about 100 pulls an hour, so sweeps run in batches of 12 with images pruned after scoring; a full Docker root fails pulls the same way, so `docker image prune -af` runs between chains; a scorer killed mid-run leaves its `sweb.eval.*` container, and the next scoring of that run id fails with 409 until it is removed.
- Anti-cheat: no benchmark detection in product code, test diffs stripped, the unmodified scorer, the dev/eval split registered before tuning.
- The plan meter was checked against provider billing: 0.0% error over five runs.

## Headline results

| split | config | resolved | date |
|---|---|---|---|
| SWE-bench Verified, 487 distinct instances across every draw | gpt-5.6-sol medium, $1/1200 s (verify-on 82/110, gateless 294/377) | 376/487 = 77.2% [73.3, 80.7] | 2026-08-23 |
| Verified eval-50, held out | kimi-k3, veto seat, fast verify | 32/50 = 64.0%, $0.58 per solve, 1 empty (baseline wheel 31/50, 8 empty) | 2026-08 |
| Verified dev-slice-25, same model and materials | kimi-k3 | mini-swe-agent 2.4.6 23/25, pi 21/25, agent6 21/25 (pre-fix build 19/25) | 2026-08 |
| SWE-rebench 2026_03, 110 instances | gpt-5.6-sol medium, finish gate, fixed staging | 63/110 and 66/110, mean 64.5/110 = 58.6% | 2026-08-30, 08-31 |

Reference rows: SWE-bench Verified full-500 vendor tables mini-swe-agent 67.3, KimiCode 67.5, Claude Code 73.7; the SWE-rebench board on the 2026_03 window (mean of five runs, SEM) Junie 61.6 ± 0.64, Codex 60.4 ± 1.37, Claude Code 59.6 ± 1.98, Cursor 53.0 ± 0.53, GLM-5.2 51.1 ± 1.13, Kimi K2.6 46.5 ± 1.27.
The board runs one minimal ReAct scaffold uncapped, 128K context, five runs per problem; a mean of two at 58.6 (SEM about 2.1) sits at Claude Code's figure, within one SEM of Codex and below Junie.

Variance: a single run on 110 ids has a standard deviation of about 3 resolves (43 ids never resolve, 41 always, 26 flip); retrying a miss resolves it about 15% of the time, so a total that substitutes reruns for misses is a ratchet, not a score.
The dev slice is easier than the full set (every agent scores far above its full-set number); cross-slice comparison is invalid.

Other reads: the dev slice by model, kimi-k3 21/25, claude-sonnet-5 16/20 (first 20), claude-opus-5 4/4 (health check); Verified coverage by condition, verify-on 82/110 = 74.5% [65.6, 81.8] and gateless tranches 79/100, 77/100, 80/100, 58/77.
The first n=6 pilot ran with verify broken (the jail binary could not exec in the containers): opus-4.8 4/6, sonnet-4-6 3/6, GLM-5.2 3/6, kimi-k2.6 2/6, lower bounds without test feedback.

## What moved resolve rate

| change | measurement | result |
|---|---|---|
| a working verify | Verified, 18 per arm | +2/18, django wins recovered |
| the finish gate (`verify_when = finish`, `verify_retries = 2`) | rebench 110, one run per side | 48/110 gateless -> 53/110; empties 7 -> 12 (11 of them wall timeouts) |
| the finish gate | Verified 77 paired, and the cert30 subset | resolve-neutral (58/77 both; 17/30 vs 16/30); PASS_TO_PASS regressions 4 vs 6 and 5 vs 14; 1.3 to 1.5x the calls, 2 to 2.5x the wall |
| headless approvals deny instead of parking | rebench, the 12 ids that had parked on an off-list `fetch` approval | 6/12 resolved on rerun; the same-config total 53 -> 56/110, empties 12 -> 5 |
| the patch carries created files | rebench 110 | 61/110 -> 63/110; the five ids needing a new module all resolve |
| a stagnation notice (wall clock with zero edits and zero verifies) and a 240 s verify with `-x` | eval-50 and the dev slice | empty patches 8 -> 1 (attempt rate 84% -> 98%); every prior empty-patch class gone |
| the verify rule softened from "tests only through the gate" to targeted `run_command` tests | hard-30 paired, verify on | 17/30 -> 19/30, targeted tests 103 -> 206, gate runs 65 -> 40, wall 2:23 -> 2:01 |
| the bare fact in place of permission prose ("run_verify_command runs the operator's gate; a passing run auto-commits the step") | hard-30 | 19/30, 177 self-tests, 39 gate runs: the permission sentence is inert |
| every run-mode string recast to facts | hard-30 | 19/30 (gained django-13344, lost sympy-13974), zero empties |
| model choice | n=6 and 18 | opus converges in 12 to 20 turns; glm runs to about 49 mean turns with several at 96 to 118; turn efficiency is model-bound |

The eval-50 gain of +2 is within run variance (±2 on identical configs); the structural change is the attempt rate.
Wrong patches, not empties, are the frontier on every split.

## Measured nulls

| lever | split, n | result | state |
|---|---|---|---|
| four run-mode prompt nudges against re-reads | Verified, 18 per arm, glm-5.2 | 7/18 = 7/18, mean turns 49.0 vs 49.6 | reverted |
| a same-model `correctness` veto seat before finish | dev-25 | 21/25, every finish approved; 17 wrong patches passed | shipped as a preset only |
| a cross-model sonnet-5 seat | dev-25 | 22/25, zero vetoes in 22 reviews | not default |
| a distinct-model review panel with a quorum gate | n=6 | 3/6 = 3/6, zero gate events | not default |
| the structural repo priors (outline, hot symbols, co-change) | dev-25 (the last section) | 22 vs 20, one real instance | removed |
| escalation to a strong model on a stall | two evals, n=6 (qwen3.6-27b -> glm-5.2; qwen3-coder-30b -> opus-4.7) | zero fail-to-resolve conversions; the `ever_edited` gate blocks the never-edited failures a strong model could help | reverted |
| the heal ladder | paired 30 | fired 0 of 29 runs (base miss rate about 7% of runs) | kept, guarded by unit pins |
| reasoning replay | paired 30 | 17/30 vs the 15/30 baseline, 2 gained, 0 lost | kept |
| the contract-examples step (input -> output examples derived before the first edit) | rebench 110 | 48 -> 46/110, calls +25% | removed |
| a probe test written before the first edit | rebench 110 | 53 -> 52/110, empties 12 -> 8 | not default |
| a deadline steer at T-120 s | rebench 110 | 53 -> 54/110; fired in 3 of 110 legs | not default |
| caps raised from $1/1200 s to $3/2400 s | 30 seeded ids | 0/15 -> 2/15 misses, 15/15 -> 13/15 controls, net 0 | caps unchanged |
| apply_patch friction | 56 misses vs 60 resolved | error rates 19% vs 22%, zero abandonments, no discrimination | unchanged |
| a "never weaken a test" prompt line | 8 broke-P2P ids | every leg still edited tests with the line in its prompt | not shipped |
| interface-layer prose ("drive the change through the layer the issue describes") | 23 zero-F2P ids | 2 conversions, the same 2 a plain retry converts | not shipped |
| a run-end notice naming importers the diff left untouched | 12 legs | fired 2 of 12, converted 0 of 7, 1 of 5 controls lost | not shipped |
| issue-derived literal-contract checks | the 54 rebench misses, three extractors | 0 truthful fires on a failing contract; the oracle ceiling is 4/54 | not built |
| `chattr +i` over test files in the container | 8 legs | unsupported on the images' overlayfs | dead for this harness |
| the `ultra` and `paranoid` presets | 5 ids, smoke | 1/5 and 3/5 with every end reviewed by the panel | a settled end passes the same gates as a finish |

A retry control prices the arms: the same 54 misses on the plain wheel converted 8/54 against the combined pilot's 10/54, 6 ids in both; a 2-conversion spread at n=54 is noise.

## Failure profile

Verified, every scored gpt-5.6-sol draw (211/267 resolved): none of the 56 misses is an empty prediction; every miss is a wrong patch.
Their newest transcripts: 51 ended `gate_stale`, django 26 of 56; 12 verify-on misses timed out every verify at 240 s (the django suite cannot finish under the cap); 40 of 56 hit a tool error; 47 of 56 ran targeted tests through `run_command` (median 2); median 12 iterations and 20 tool calls; no compactions, so context was never the binding constraint.

SWE-rebench 2026_03 misses (a run can be in more than one class): 22 near-miss (some FAIL_TO_PASS passing), 30 zero-F2P (none passing), 14 broke PASS_TO_PASS; 53 of the 55 wrong-patch runs finished under their own power, so the gap is not cap pressure.

| class | n | tool calls | read the F2P file | ran the F2P file | read a test before the first edit |
|---|---|---|---|---|---|
| resolved | 48 | 22.3 | 40/48 | 39/48 | 44/48 |
| near-miss | 15 | 24.4 | 13/15 | 15/15 | 14/15 |
| broke-P2P | 14 | 25.4 | 12/14 | 12/14 | 14/14 |
| zero-F2P | 26 | 25.9 | 18/26 | 15/26 | 25/26 |

| class | F2P tests | added by the hidden test patch | instances with every F2P test new |
|---|---|---|---|
| resolved (48) | 139 | 115 (83%) | 38/48 |
| near-miss (25) | 169 | 120 (71%) | 18/25 |
| broke-P2P (14) | 48 | 39 (81%) | 11/14 |
| zero-F2P (30) | 71 | 68 (96%) | 27/30 |

- The model behaves the same way in every class and finds the file the hidden contract lands in; a zero-F2P miss is a contract that exists only in tests not yet written, so no search over the existing suite can surface it.
- Zero-F2P patches are subsets of the gold patch: 16 of 23 never touched a gold code file, stopping one layer short of the interface the graded tests drive (a model fixed but not the responses layer that calls it; a function registered everywhere but never written).
- Near-misses are stable contracts: two independent legs (different wheel, different prompt) missed the same assertions on 11 of 14 ids; several fail one test (24/25, 14/15, 9/10) on an exact literal (a rendered attribute, an error string) or a pre-existing red test the run never ran.
- Broke-P2P legs saw a red verify in-run and 7 of 8 edited an existing test file until the gate went green; the stripped test files are restored at grading and the break resurfaces.
- Gold patches often touch changelog fragments (about 13 misses), but every failing graded test is a behaviour test.
- Wall-timeout empties were mostly approval parks (19 legs across three fleets on the same 12 ids), not reasoning latency; with them denied, a full-110 run has 0 or 1 empties.

## Sandbox usability

| environment | effective auto | `check sandbox` | resolution |
|---|---|---|---|
| unprivileged docker | hardened | FAIL (the `/etc`-write probe escaped) | unsandboxed opt-in |
| privileged docker | strict | all probes pass | strict |
| podman rootless | strict | all probes pass | strict |

- `sandbox.isolation = "none"` is an operator-only config value with a loud startup warning; `auto` never resolves to it on Linux.
- strict exposes `extra_read_paths` at `/ro<src>`; a granted interpreter's real path is absent and the `/ro` path works.
- podman rootless is not detected as a container, and strict warns once on a `/proc` remount EPERM there.
- verify inference falls back to `python3` on PATH when no `.venv` exists (the harness passes the conda interpreter's absolute path).

## Repo priors on the dev slice: the block cost every start and moved one instance (2026-09-14)

Arms: the structural `<repo-priors>` block (symbol outline, hot symbols, co-change pairs) on and off; the dev slice (`dev_slice_25.json`, n=25 each, never a headline number); gpt-5.6-sol at effort medium on the ChatGPT plan; verify on; 1 plan point per instance; the same wheel; the official scorer.

| arm | resolved | empty | mean iterations | mean uncached input tokens | session.start to first model call |
|---|--:|--:|--:|--:|--:|
| priors on | 22/25 | 1 | 21.3 | 38,376 | 19.9 s (2.8 to 40.1) |
| priors off | 20/25 | 2 | 21.2 | 35,326 | 0.3 s (0.1 to 0.7) |

- 20 instances resolved in both arms; none only without the priors; two only with them.
- sympy-13877 without the priors ended `budget_exhausted` at iteration 3: the 1-point cap read as spent when the plan's whole-percent meter ticked over mid-run, not a capability result.
- django-16950: both arms patched it; the patch without the priors fails the tests.
- Identical patch bytes on 12 of 25 instances; the same end reason on 19.
- Replicates of the two disagreeing instances, three per arm: with the priors 6/6, without them 5/6 (sympy-13877 3/3; django-16950 2/3, so 4/4 with the priors and 2/4 without over every run).

Mechanism: the block needed the tree-sitter index of the whole tree before the first call, 20 s on django and 40 s on sympy in the container and minutes on larger trees, and added up to 8k characters to every call's system prompt, which the plan wire does not cache.

State: the structural block is removed. `<repo-priors>` keeps the repo map, AGENTS.md and the recent commits; the nav tools (`outline`, `find_definition`, `find_references`) build the index on their first call.
