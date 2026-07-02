# coreagent bench — decomposition & compaction experiments

A harness to measure agent6 *core-agent* changes (prompt/DAG/compaction) on
objective, test-graded tasks, across models, with enough reps to separate signal
from noise. Built for two thrusts:

- **Thrust 2 — task decomposition** (`[prompt].decompose`): does front-loading a
  subtask plan help small models finish multi-part tasks?
- **Thrust 1 — context compaction** (`[context].keep_recent_chars`): does keeping
  a recent verbatim tail at tier-2 beat the hard `[task, summary]` restart?

## What it measures

Each run drives the installed `agent6` binary on a task from a throwaway git repo
with a private `XDG_STATE_HOME`, then grades the result with the task's
**authoritative hidden grader** (`tasks/<t>/grade.py`, never shipped into the
agent's repo, so it can't be tampered or over-fit). Metrics per run (from the
run's `logs.jsonl` + the grader):

- `score` — fraction of grader CASES passing (partial credit -> low variance).
  `solved` = score >= 0.999.
- `components_passed` — coarse "did it forget a whole component" signal.
- `n_subtasks` / `n_subtasks_passed` — DAG decomposition actually performed.
- `compactions` / `drops` — tier-2 / tier-1 compaction events.
- `redundant_reads` (+ `_post_compact`) — read-shaped tool calls repeating an
  earlier `(name,args)`. The primary compaction-quality signal.
- `iterations`, `usd`, `tokens_in/out`, `wall_s`, `end_reason`, `tampered`.

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

Providers + secrets come from the layered global `~/.config/agent6` config; the
harness only pins the three roles to `--model` and wires the verify command.
Anthropic models (unpriced) get token caps instead of `--max-usd`.

## Conditions (`CONDITIONS` in run.py)

- `baseline` — shipped defaults.
- `decompose` — `[prompt].decompose = true`.
- `compact_tight` — aggressive `[context]` thresholds to force tier-2 on a
  moderate task.
- `decompose_tight` — both.

Add a condition by adding a config-TOML fragment to the dict. The task prompt is
deliberately NEUTRAL about decomposition, so any DAG-use difference comes from the
condition's config, not the prompt.

## Reproducibility

Results are append-only JSONL under `results/`. Each record carries the model,
condition, task, rep, run_id, and every metric, so a run is fully reproducible
from `(model, condition, task)`. `COMPACTION_RESEARCH.md` is the literature
survey behind Thrust 1. Findings + adopt/scrap calls: see `FINDINGS.md`.
