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
A pass requires every case in the original `grade.py`, not a green starter suite or a model's claim.

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
Use partial isolation only with a local scripted provider and no credentials.
Prepare Python 3.13 and uv's offline cache with the wheel build and runtime dependencies first.
The runner archives `HEAD`, applies the arm's patch there, copies the launcher and installs that source into its own venv.
The control patch is empty.

Set `base` to a private directory on a real disk outside the checkout and `launcher` to the binary's absolute path.
Sign in to a dedicated profile and skip the default-model picker.
Keep only provider settings in its `config.toml`.
Do not share this profile with another process: each run returns its refreshed `secrets.toml` to the profile.

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
The plan cap bounds observed account movement, not dollars or an in-flight call.
Priced calls have a $1 cap; unpriced calls have a 100,000-token fallback cap.
Editing arms allow jailed commands and offer `parallel` so all three definitions reach the model.
Ask withholds commands; a headless approval request is denied.
The hidden grader runs after the session via `agent6 exec --new`, not against the host Python environment.

**Evidence.** `results.jsonl` has one row per attempted task and replicate.
The task-replicate directory keeps `agent.log`, the repository and the journals.
Turns are the root `session.stop.turns`; the ask answer is its `summary` when the reason is `answered`.
Failed/completed edit counts come from `tool.result.ok`, including children.
Input (fresh, cache read and cache write) and output tokens sum `model.result` once per logical call across those journals.
This counts reported usage on the final attempt, not unreported retry spend.
The table shows pooled failures/calls and the median [min,max] per run for failures, turns and tokens.
It gives the count of known readings when any are missing.
A missing journal or grader result is a measurement error, not a zero score.
