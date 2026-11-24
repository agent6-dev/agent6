# Tool measurements

Each comparison uses the same committed tree, route (`chatgpt/gpt-6.2-sol`), effort (`medium`), tasks and caps on both sides.
`control` offers the shipped tools and text.

| Comparison with `control` | Development | Evaluation | Question |
|---|---|---|---|
| `no-patch` | `textkit` | `bugs`, `eventflow` | Does offering `apply_patch` change failed edit calls, turns, tokens or passes? |
| `no-docs` | `machine-test` | `plan-cap` | Does offering `agent6_docs` in `ask` change correct answers, turns or tokens? |
| `short` | `textkit` | `bugs`, `eventflow` | Does shortening the definitions of `apply_edit`, `ask_user` and `parallel` together change the editing results? |

**Fixed split.** Use development for the single-run smoke and any tuning.
Freeze the patches before opening evaluation journals.
Run three replicates per evaluation task and arm, sequentially on an otherwise idle account.
Use the same control samples for both editing comparisons.

**Editing tasks.** [Coreagent](../coreagent/README.md) supplies the prompts, starting trees and hidden graders.
`textkit` exercises implementation from a stub.
`bugs` needs several edits in one file; `eventflow` needs fixes across modules.
A pass means the original `grade.py` reported every case passing, not that the starter suite passed or the model claimed success.

**Ask tasks.** The [machine bench](../machine/README.md) tests generated bundles offline.
Ask for the command that runs a bundle's mock tests (`agent6 machine test`).
The [coreagent bench](../coreagent/README.md) distinguishes a dollar budget from a ChatGPT plan cap.
Ask for the per-run flag that limits plan usage (`--max-plan-points`).
Ask in an empty repository and request only the command or flag.
Match the answer after removing surrounding whitespace and inline code ticks.

**Reporting.** Report the split, task names and observed n per arm verbatim.
Include failed and capped runs, not just successful ones.
Show the spread across replicates before attributing a difference to a tool.
Record every comparison, including no observed difference, with the mechanism tested.
These tasks do not separate the effects of the three shorter descriptions.

## Run sets

Use one clean checkout and the matching executable static `agent6-jail` on a host with full isolation.
The runner requires full isolation and refuses sandbox settings in the profile.
Prepare Python 3.13 and uv's offline cache with the wheel build and runtime dependencies first.
The runner archives `HEAD`, applies the arm's patch there, copies the launcher and installs that source into its own venv.
The control patch is empty.

Set `base` to a private directory on a real disk outside the checkout and `launcher` to the binary's absolute path.
Sign in to a dedicated profile and skip the default-model picker.
Its `config.toml` may contain only provider settings and `agent6.config_version`.
Do not share this profile with another process: each run returns its refreshed `secrets.toml` to the profile.
The jail hides the source profile and the arm build from both the submission and the grader.

```sh
XDG_CONFIG_HOME="$base/signin" agent6 connect chatgpt

# From the checkout root:
runset() {
    python3.13 -m bench.tools.run --launcher "$launcher" --profile "$base/signin/agent6" \
        --model chatgpt/gpt-6.2-sol --effort medium --max-plan-points 2 "$@"
}
runset --arm control --tasks textkit --reps 1 --out "$base/smoke"
```

Inspect the smoke row and journal before spending on the matrix.
Use a new output directory for each set.
Record the run-set order because provider load and cache hits can vary with time.

```sh
runset --arm control  --tasks bugs eventflow plan-cap --reps 3 --out "$base/control"
runset --arm no-patch --tasks bugs eventflow          --reps 3 --out "$base/no-patch"
runset --arm no-docs  --tasks plan-cap                --reps 3 --out "$base/no-docs"
runset --arm short    --tasks bugs eventflow          --reps 3 --out "$base/short"
python3.13 -m bench.tools.table "$base/control/results.jsonl" "$base/no-patch/results.jsonl" \
    "$base/no-docs/results.jsonl" "$base/short/results.jsonl"
```

**Bounds.** Each run gets a new repository and private config, state, data and cache homes.
The build finds `git` and `uv` on the caller's `PATH`.
Trial processes on the host use `PATH=/usr/bin:/bin` and invoke the arm by absolute path.
Commands inside the jail use [agent6's jail PATH](../../docs/security.md#2-sandbox).
The plan cap measures observed account movement in percentage points and does not stop an in-flight call.
The $1 cap counts priced spend and observed account credit declines.
Tokens count toward the 100,000-token fallback when a call has no plan reading, reported cost or known price.
Editing arms allow jailed commands and offer `parallel` so all three definitions reach the model.
Ask withholds commands; a headless approval request is denied.

**Grading.** After an editing session, `agent6 exec --new` uses a fresh `grading/` repository as its writable workspace.
Its own `.git` stays protected.
A read-only grant covers the sibling submission, `repo/`, including its `.git`.
The runner sends the archived scorer unchanged to `python3 -I -` through an anonymous pipe.
`bugs` and `textkit` import the submission in the scorer process.
Checking the report format and totals does not independently verify its claims.

One two-minute deadline covers sending the scorer, draining the pipes and waiting for `exec` to exit.
The runner accepts up to 64 KiB each on stdout and stderr.
Exceeding either limit leaves the run unscored.
Neither stream reaches the terminal; the runner keeps the accepted stderr in `grader.stderr`.
The pipe limits do not cap `agent.log` or memory used inside a scorer.
`eventflow` captures stdout and stderr from each case subprocess in memory with a 30-second timeout and no byte cap.
Use a host memory limit for the run set if a hard memory bound is needed.

**Evidence.** `results.jsonl` has one row per attempted task and replicate.
The task-replicate directory keeps `agent.log`, the repository and the journals.
Turns are the root `session.stop.turns`; the ask answer is its `summary` when the reason is `answered`.
Failed/completed edit counts come from `tool.result.ok`, including children.
Input (fresh, cache read and cache write) and output tokens sum `model.result` once per logical call across those journals.
This counts the final attempt's reported usage and excludes earlier attempts.
The table shows pooled failures/calls and the median [min,max] per run for failures, turns and tokens.
It gives the count of known readings when some are missing and prints `?` when none are known.
A missing root journal, root stop or grader result is a measurement error, not a zero score.
Grader stdout must contain one JSON report with the task's case and component totals.
The official `bugs` and `textkit` import-error reports count as scored failures.
A malformed, extra or oversized report leaves that run unscored; the set continues.

## Results

Evaluation split on `chatgpt/gpt-6.2-sol` at `medium`, three replicates per task and arm.
Every run passed; none was capped or errored, and no `apply_patch` or `apply_edit` call failed.
Call counts are totals over the three runs; other quantities are the median [minimum, maximum].

| Task | Arm | n | Pass | `apply_patch` calls | `apply_edit` calls | Turns | Input tokens | Output tokens |
|---|---|---|---|---|---|---|---|---|
| `bugs` | `control` | 3 | 3/3 | 1 | 2 | 5 [4, 5] | 43,570 [36,287, 43,573] | 664 [659, 702] |
| `bugs` | `no-patch` | 3 | 3/3 | 0 | 3 | 4 [4, 5] | 35,617 [35,551, 42,446] | 703 [631, 716] |
| `bugs` | `short` | 3 | 3/3 | 3 | 0 | 5 [5, 5] | 42,411 [42,375, 42,530] | 615 [599, 620] |
| `eventflow` | `control` | 3 | 3/3 | 3 | 0 | 5 [5, 6] | 40,001 [34,221, 45,657] | 2,668 [2,306, 2,979] |
| `eventflow` | `no-patch` | 3 | 3/3 | 0 | 15 | 6 [5, 6] | 43,135 [33,025, 43,786] | 2,535 [2,228, 2,724] |
| `eventflow` | `short` | 3 | 3/3 | 3 | 0 | 5 [5, 7] | 33,460 [32,976, 50,579] | 2,334 [2,202, 2,490] |
| `plan-cap` | `control` | 3 | 3/3 | 0 | 0 | 2 [2, 2] | 16,690 [16,686, 16,692] | 29 [29, 38] |
| `plan-cap` | `no-docs` | 3 | 3/3 | 0 | 0 | 1 [1, 1] | 1,082 [1,081, 1,083] | 9 [9, 9] |

- **`no-patch`:** no observed difference in passes, turns or tokens.
  On `eventflow` the worker made 15 `apply_edit` calls where `control` made 3 `apply_patch` calls.
- **`short`:** no observed difference; the ranges overlap on both tasks.
- **`no-docs`:** both arms answered correctly.
  `control` took two turns and 16,690 input tokens, `no-docs` one turn and 1,082.
  The set holds one question.
