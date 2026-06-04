# Real-world benchmark for agent6

These tasks run agent6 against shallow clones of real OSS projects at
fixed commits. The methodology is SWE-bench-Lite style: a small piece
of working code is "gutted" (its body replaced with `raise
NotImplementedError`), and agent6 is asked to restore the
implementation. Scoring is automatic — the project's own test suite
gives ground truth.

We pin commits and use `--depth 200` clones (enough git
history for the co-change-pairs planner prior to surface; ~2MB extra
per click-class repo). Nothing in this directory is committed beyond
the task definitions; the actual checkouts land under `$BENCH_ROOT`
(default `/tmp/agent6-realworld`).

## Layout

```
bench/realworld/
  README.md                 (this file)
  run_realworld.sh          (harness)
  tasks/
    <task>.json             (one per task)
```

## Task schema (JSON)

| key              | description                                          |
|------------------|------------------------------------------------------|
| `name`           | identifier (matches the filename)                    |
| `repo_url`       | git URL to clone                                     |
| `commit`         | tag or commit hash (passed to `--branch`)            |
| `install`        | list of argv lists to bootstrap the project's venv   |
| `break`          | list of `{path, find, replace}` text substitutions   |
| `task_md`        | the prompt written to TASK.md                        |
| `verify_command` | list of argv elements for `[workflow] verify_command`|
| `description`    | short label for the results table                    |

## Running

```bash
ANTHROPIC_API_KEY=... bash bench/realworld/run_realworld.sh
```

Per-task budget is capped via `[budget]` in the generated `agent6.toml`
and via agent6's own internal accounting. Total spend is bounded by the
number of tasks × per-task cap.

For an A/B comparison of tools-on vs tools-off, set
`AGENT6_REALWORLD_TOOLSET=baseline` or `=index` and run twice; the
results file name includes the toolset.

### Filtering tasks + routing controls

```bash
# Run a single task (substring match on task name)
AGENT6_REALWORLD_TASK_FILTER=click-rename-split-opt \
  bash bench/realworld/run_realworld.sh

# Enable worker_loop (multi-turn architect + editor fusion path). When
# unset (default false), routing uses single-shot architect_decide +
# editor_translate per step - the cheapest path when the planner's
# co-change priors already enumerate all relevant files.
AGENT6_WORKER_LOOP_ENABLED=true \
  bash bench/realworld/run_realworld.sh

# Each run goes to its own BENCH_ROOT so multiple configurations can
# be compared side-by-side without dir collisions:
BENCH_ROOT=/tmp/rw-config-a \
  bash bench/realworld/run_realworld.sh
```

### Aggregating results

```bash
# Print a side-by-side markdown table for multiple BENCH_ROOTs
bench/realworld/summarize.py /tmp/rw-config-a /tmp/rw-config-b
```

## Why this is a fair benchmark

- The original code shipped to PyPI and was tested by maintainers; the
  test suite encodes the intended behaviour, not the candidate model's.
- The candidate sees only TASK.md and the (broken) repository. It does
  not see the original implementation diff.
- The verify command is identical to the project's CI suite for the
  targeted tests — no special-casing.

## Recent results (guidance only — N = 1)

A single head-to-head pass over the 11 tasks (worker model
`claude-sonnet-4-5` on both sides; per-task budget cap $1.00):

| runner       | verify       | total cost | wall    |
|--------------|--------------|-----------|---------|
| agent6       | 11 / 11 pass | ~$2.60    | —       |
| claude-code  | 11 / 11 pass | $3.96     | 1443.5s |

### agent6 across worker models (`network = "provider_only"`, N = 1)

Two further single passes of the same 11 tasks driven entirely through
agent6, varying only the worker model. Both ran under
`[sandbox] network = "provider_only"` (strict profile), so every task
also exercised the rootless egress broker end-to-end:

| worker model         | verify       | total cost | wall      |
|----------------------|--------------|-----------|-----------|
| `claude-sonnet-4-5`  | 11 / 11 pass | $8.45     | 2519.6s   |
| `moonshotai/kimi-k2.6` (OpenRouter) | 11 / 11 pass | $1.20 | 3342.6s |

Both worker models solved the entire suite. Cost and wall time are not
comparable head-to-head here: the Sonnet pass above used a tighter
per-task profile, while these two passes let each task optimise toward
the per-task cap. The open-weights Kimi run is much cheaper per token and,
since agent6 now stops optimising a metric once it reaches a provable
ceiling (e.g. a grader that prints `SCORE: 27/27`), no longer burns wall
time re-deriving solved tasks — its slowest task is now `tinydb-search`
(~1351s), a genuine search over the project. Per-task breakdowns land in
`$BENCH_ROOT/results_index.md` for each run.

These are all **single runs (N = 1)** — not an average or median, and we
have not measured variance. Per-task cost on a stochastic worker swings
widely run-to-run, so treat this only as directional ("all of them solve
the suite"). The claude-code per-task breakdown lives in
`$BENCH_ROOT/results_claude.md` after running `run_realworld_claude.sh`.
Re-run the harnesses to measure for yourself rather than quoting these
numbers as headline figures.

