# Real-world benchmark for agent6

These tasks run agent6 against shallow clones of OSS projects at fixed tags, or self-contained fixtures.
The restoration tasks replace working code with `raise NotImplementedError` and ask agent6 to restore the implementation.
The project's tests score the final tree; tasks can also declare a metric.

Clones use `--depth 200` for the run's recent-commit context (about 2MB extra per click-class repo).
Checkouts, logs and results land under `$BENCH_ROOT` (default `/tmp/agent6-realworld`), outside this directory.

## Layout

```text
bench/realworld/
  README.md                 (this file)
  run_realworld.sh           (agent6 harness)
  run_realworld_claude.sh    (claude-code harness)
  summarize.py              (result aggregation)
  tasks/
    <task>.json             (one per task)
```

## Task schema (JSON)

| key | description |
|---|---|
| `name` | identifier (matches the filename) |
| `repo_url` | optional git URL to clone |
| `commit` | tag or branch name passed to `git clone --branch` when cloning |
| `files` | `{relative_path: content}` mapping for a self-contained task without `repo_url` |
| `install` | optional list of argv lists to bootstrap the project's venv |
| `break` | optional list of `{path, find, replace}` text substitutions |
| `task_md` | the prompt written to `TASK.md` |
| `verify_command` | argv list for `[harness] verify_command` |
| `metric` | optional `{command: argv, pattern: regex, goal: "minimize" or "maximize"}` for `[harness.metric]` and independent scoring |
| `description` | short label printed before the task runs |

## Running

```bash
bash bench/realworld/run_realworld.sh
```

The worker is the one your agent6 config names (`agent6 config show models.worker`).
The harness writes each task's `agent6.toml` and passes it as `--config`, layering over your global and per-repository config.
It writes no provider or model tables; an enabled reviewer also comes from your config.

| variable | effect | default |
|---|---|---|
| `AGENT6_REALWORLD_MODEL` | `[PROVIDER/]MODEL`, passed as `run --model` | your configured worker |
| `PER_TASK_BUDGET_USD` | `[budget] max_usd` for each task | `1.00` |
| `PER_TASK_MAX_PERCENT` | `[budget] max_percent` for each task (subscription providers) | your config |
| `AGENT6_REALWORLD_TASK_TIMEOUT_S` | wall-clock bound on one task's `agent6 run`, in seconds | `3600` |
| `AGENT6_REALWORLD_TASK_FILTER` | comma-separated substrings of task names | all tasks |
| `AGENT6_REALWORLD_TOOLSET` | `index` (symbol tools on) or `baseline` (off) | `index` |
| `BENCH_ROOT` | where checkouts, logs and results land | `/tmp/agent6-realworld` |

Metered runs use the dollar cap; subscription runs use the plan-points cap ([budget reference](../../docs/config.md#budget)).
The plan reading is account-global, so concurrent runs can affect it.
The timeout sends SIGTERM, then SIGKILL after 30 seconds if needed.
It bounds only `agent6 run`, not setup, the final verify, or the metric.
Unattended approvals default to `AGENT6_DETACHED_AWAY=deny`; an explicit away-mode is preserved.
The harness leaves jail selection to agent6, including any explicit `AGENT6_JAIL_BIN` override.

### Filtering tasks

```bash
# Run a single task (substring match on task name).
AGENT6_REALWORLD_TASK_FILTER=click-rename-split-opt \
  bash bench/realworld/run_realworld.sh

# Use a separate root for each configuration.
BENCH_ROOT=/tmp/rw-config-a \
  bash bench/realworld/run_realworld.sh
```

## Results

Each task ends in one of three states:

- `PASS`: a scored run's verify command passes on the final tree.
- `FAIL`: the run ended on the agent's side (finished, gave up, crashed, or hit its budget) and the verify command fails.
- `UNSCORED: <reason>`: setup failed, agent6 refused to start, the provider failed, the verify command could not run in the jail, the run hit its timeout, or its end could not be classified.

Unscored runs stay unscored even when the final tree passes verify.
The pass rate counts scored tasks only; the totals line gives the unscored count.
Exits 0, 3, 4 and 5 are scored; exit 1 is scored unless the session reports a provider error, an unexecutable verify command, or no readable reason.
Other exits are unscored.
The harness does not exit nonzero merely because a row is FAIL or UNSCORED.

Each task's `result.json` and logs are under `$BENCH_ROOT/_logs/<task>_<toolset>/`.
The record keeps the observed `verify_pass` separate from the `unscored` reason, along with the session id, model, end reason, dollar cost, plan points and tokens.
`agent6 sessions show <id> --json` supplies the end reason, model and plan points; the printed budget summary supplies dollar cost and tokens.
The results header names the last model reported by a session, or the requested model or operator config when none was reported.

Use a fresh `BENCH_ROOT` per run: `results_<toolset>.md` includes stored rows from earlier runs, but its totals count only the current run.

### Aggregating results

```bash
# Print a side-by-side Markdown table for multiple BENCH_ROOTs.
bench/realworld/summarize.py /tmp/rw-config-a /tmp/rw-config-b
```

The aggregator also excludes unscored rows from its pass rate.

## Why this is a fair benchmark

- The restoration tasks' original code shipped to PyPI and was tested by maintainers; the test suite encodes the intended behaviour, not the candidate model's.
- The candidate reads `TASK.md`, `AGENTS.md` and the broken repository, not the original implementation diff.
- The restoration tasks' verify command runs the project's targeted tests, with no special-casing.

## Recorded results (guidance only: N = 1)

A single head-to-head pass over 11 tasks used worker model `claude-sonnet-4-5` on both sides.
The agent6 per-task cap for this recorded pass is not confirmed.

| runner | verify | total cost | wall |
|---|---|---|---|
| agent6 | 11 / 11 pass | ~$2.60 | not recorded |
| claude-code | 11 / 11 pass | $3.96 | 1443.5s |

These are single runs (N = 1), not averages or medians; variance has not been measured.
Per-task cost on a stochastic worker varies between runs, so treat these numbers as guidance.
The claude-code per-task breakdown lives in `$BENCH_ROOT/results_claude.md` after running `run_realworld_claude.sh`.
Re-run the harnesses to measure for yourself rather than quoting these numbers as headline figures.
