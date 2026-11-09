# coreagent bench: decomposition & compaction experiments

A harness to measure agent6 *core-agent* changes (prompt/DAG/compaction) on
objective, test-graded tasks, across models, with enough reps to separate signal
from noise. Built for two thrusts:

- **Thrust 2, task decomposition** (`[prompt].decompose`): whether
  front-loading a subtask plan helps small models finish multi-part tasks.
- **Thrust 1, context compaction** (`[compaction].keep_recent_chars`): whether
  a recent verbatim tail at tier-2 beats the hard `[task, summary]` restart.

## What it measures

Each run drives the installed `agent6` binary on a task from a throwaway git repo
with a private `XDG_STATE_HOME`, then grades the result with the task's
**authoritative hidden grader** (`tasks/<t>/grade.py`, never shipped into the
agent's repo, so it can't be tampered or over-fit). Metrics per run (from the
run's `logs.jsonl` + the grader):

- `score`: fraction of grader CASES passing (partial credit -> low variance).
  `solved` = score >= 0.999.
- `components_passed`: coarse "did it forget a whole component" signal.
- `n_subtasks` / `n_subtasks_passed`: DAG decomposition actually performed.
- `compactions` / `drops`: tier-2 / tier-1 compaction events.
- `redundant_reads` (+ `_post_compact`): read-shaped tool calls repeating an
  earlier `(name,args)`. The primary compaction-quality signal.
- `iterations` (loop turns), `usd`, `tokens_in/out`, `wall_s`, `stop_reason`, `tampered`.
- `verification`: the final `session.stop.verification` word, not a guess from the grader.
- `task_rewritten`: whether `loop.prompt_revision.result` was recorded (the rewrite is used automatically in `revise_auto`).
  A started or failed revision alone does not count.

## Tasks (`tasks/<name>/`)

Stdlib-only Python, multi-component so they decompose naturally. `repo/` is the
agent's starting tree (spec.md + stub + a STARTER unittest + `verify.sh` running
`python3 -m unittest`). `grade.py` is the hidden superset grader.

| task | shape | components | headroom |
|------|-------|-----------|----------|
| `textkit` | implement 5 independent string fns | 5 | low (capable models ceiling) |
| `rpn` | implement RPN calc, sequential deps | 4 | low-med |
| `ledger` | implement stateful bank ledger | 5 | med |
| `bugs` | DEBUG: fix 7 seeded bugs | 7 | high (buggy stub = 0.64) |

Validate a task's grader against its reference solution before trusting it
(reference solutions live outside the repo, in the session scratchpad).

## Running

```bash
# one (model x conditions x tasks x reps) matrix -> results/<label>.jsonl
python3 run.py --model qwen/qwen3.6-35b-a3b --provider openrouter \
    --tasks textkit,rpn,ledger,bugs --conditions baseline,decompose \
    --reps 4 --parallel 6 --budget 0.60 --label myrun

python3 stats.py results/myrun.jsonl          # per-cell mean+/-se + treatment deltas
```

Providers and secrets come from the layered global `~/.config/agent6` config.
The harness writes `[models.default]` with `model = "<provider>/<model>"` and the optional `effort` pin, plus `harness.verify_command = ["bash", "verify.sh"]`.
Keep global role overrides unset and other experiment settings at their defaults so every role inherits the tested route and only the condition changes.
Each live run checks its merged config with `agent6 config show` before spending.
Anthropic models (unpriced) get token caps instead of `--max-usd`.
ChatGPT routes use the host's plan cap; `--budget` caps dollars, not plan use.

### G12 and B15 on ChatGPT

From the repository root, with the installed binary on PATH (or `AGENT6_BIN` set):

```bash
# G12: model-only verification versus the harness-run finish gate.
python3 bench/coreagent/run.py --provider chatgpt --model <model> --effort high \
    --tasks textkit,rpn,ledger,bugs --conditions baseline,verify_finish \
    --reps 3 --parallel 2 --label g12
python3 bench/coreagent/stats.py bench/coreagent/results/g12.jsonl

# B15: original task versus an automatic helper rewrite.
python3 bench/coreagent/run.py --provider chatgpt --model <model> --effort high \
    --tasks textkit,rpn,ledger,bugs --conditions baseline,revise_auto \
    --reps 3 --parallel 2 --label b15
python3 bench/coreagent/stats.py bench/coreagent/results/b15.jsonl
```

Replace `<model>` with the model id, without the `chatgpt/` prefix.
Both commands write the route `chatgpt/<model>`.
Add `--dry-run` to build the same repos and configs without launching agent6 or the grader.
`COREAGENT_RUNS` selects the repo and private state directory parent (default `~/coreagent-runs`).
Dry runs print each config path and do not append result rows.

## Conditions (`CONDITIONS` in run.py)

| Condition | Setting | Comparison |
|-----------|---------|------------|
| `baseline` | Shipped defaults: `harness.verify_when = "never"`, `prompt.revise_prompt = "off"` | G12 and B15 control |
| `verify_finish` | `harness.verify_when = "finish"` | G12 treatment: harness runs the finish gate; both arms retain the same tools and verify command |
| `revise_auto` | `prompt.revise_prompt = "auto"` | B15 treatment: helper rewrites the task before the worker loop |
| `decompose` | `prompt.decompose = "on"` | Front-loaded decomposition |
| `compact_tight` | Lower compaction thresholds | Force compaction on moderate tasks |

The other prompt and skill conditions are listed in `CONDITIONS`.
The historical `spec_recheck` condition is unavailable because `harness.spec_recheck_on_finish` is not a supported config key.

Add a condition by adding a config-TOML fragment to the dict. The task prompt is
deliberately NEUTRAL about decomposition, so any DAG-use difference comes from the
condition's config, not the prompt.

## Reproducibility

Results are append-only JSONL under `results/`. Each record carries the model,
condition, task, rep, session_id, and every metric, so a run is fully reproducible
from `(model, condition, task)`. `COMPACTION_RESEARCH.md` is the literature
survey behind Thrust 1. Findings + adopt/scrap calls: see `FINDINGS.md`.
