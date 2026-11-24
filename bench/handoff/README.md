# Handoff by task id or copied text

Does handing a listed task to a child by id change how much work the parent does at the same task acceptance?
Both arms use the same build of agent6 with the [parallel tool](../../docs/usage.md#the-parallel-tool).
Only one instruction in the [parent brief](job.py) differs.

| Arm | Instruction to the parent |
|---|---|
| `text` | Copy each listed title verbatim into `parallel.task`. Do not name a task id in the dispatch. |
| `id` | Give `parallel` the listed `task_id` instead of text. |

**Job.** Six independent tasks ask for the code in a file, in `ash`, `beech`, `cedar`, `elm`, `fir`, `oak` order.
The runner queues all six through the operator task queue before the first turn.
Each task's title is its whole brief, so a verbatim copy and a handoff by id give the child the same work.
The parent must dispatch one default worker per task and accept each returned report on the original task if needed.
Both arms carry the pin `Parent: keep 2 children running on distinct listed tasks while work remains.`
The ceiling is two children, not two attempts at one task.

**Controls.** Each run starts at the same commit in a fresh repository with private config, state, data, cache and HOME.
The parent and children use `git.control = "model"`, the selected route and `medium` effort.
The parent has 48 turns; the children use the run's remaining budget.
Each run has a $1 cap, a 2-point plan cap and a 100,000-token fallback.
The job needs no edits, commands, merge, verify gate or review panel.
Command tools that execute shell commands are withheld.

## Run set

Use a clean checkout with its matching CLI installed in the project environment.
Use a private directory on a real disk outside the checkout for `base`.
The output directory must not already exist.

**Scripted check.** This command needs no credentials and makes no paid calls.
It runs the CLI and its detached children against a loopback provider.
The scripted children read the files and return their codes.
This split tests the method, not model behaviour.

```sh
uv run python -m bench.handoff.run --scripted --reps 1 --out "$base/scripted"
```

**Live evaluation.** Use a host with full isolation and the matching `agent6-jail` binary.
Sign in to a dedicated profile, for example with `XDG_CONFIG_HOME="$base/signin" agent6 connect chatgpt`, skipping the model picker.
Its `agent6/config.toml` may contain only provider settings and `agent6.config_version`.
The runner uses the [tool benchmark's private homes](../tools/README.md#run-sets) and returns any refreshed credentials to that profile.
Do not use the profile elsewhere during the set.

```sh
uv run python -m bench.handoff.run --profile "$base/signin/agent6" \
    --model chatgpt/gpt-6.2-sol --reps 3 --out "$base/live"
```

Run the set sequentially on a quiet account.
Each replicate includes both arms, text first on odd replicates and id first on even ones.
`--reps 1` runs one pair for a smoke in a separate directory.
Freeze the job and settings before evaluation; do not pool the smoke or scripted runs with live replicates.

## Journal measures

`results.jsonl` keeps one line per run, including failed and capped runs.
Each line names the revision, seed commit, arm, replicate, parent journal, CLI exit code and stop reason.
The run directory keeps the config, checkout, console log and journals.

| Measure | Evidence in the parent journal |
|---|---|
| Stuck notices | Count `nudge.task_stuck`, not mentions of being stuck in text. |
| Parent turns | Read `session.stop.turns`. |
| Parent tokens | Sum fresh input, cache reads, cache writes and output separately from `model.attempt`, including retries. Exclude child usage and do not add `model.result` again. |
| All accepted | Look up the six `task.queued` ids in the last `graph.update`. All six must be `passed`. A passed copy, a passed root or a finish summary is not evidence about the originals. |
| Started twice or never | Match `parallel.child_activity` starts to exact queued titles. Count how many of the six started at least twice, and how many had no start. A refused call is not a start. |
| Wait below two | For each task, accumulate seconds from queueing until first start, first observed non-open status or parent stop, only while fewer than two children ran. `under_two_s` counts those intervals once even when several tasks waited. |
| Off-arm calls | Count `parallel` calls whose arguments do not use the assigned id or verbatim text form. Keep these runs under the assigned arm. |

Child activity timestamps measure when the parent observed starts and ends, not CPU time or the instant a worker process exited.
`wait_by_task_s` and `starts_by_task` retain the per-task readings.
An unmatched start, a missing endpoint or an unreadable journal leaves the affected measures unknown.
An unreported token category is `null`, not zero.
Graph acceptance measures coordination, not the correctness of the codes in a live summary.

The runner prints the table at the end of the set.
To print it again without running a provider:

```sh
uv run python -m bench.handoff.table "$base/live/results.jsonl"
```

Report the split, observed run count, all-accepted runs over known outcomes, and median [minimum,maximum] per arm.
A partial column shows its coverage; `?` means no known value.
Check the stop reasons, off-arm calls and missing evidence before comparing turns, tokens or wait at equal acceptance.
Report no observed difference when the arms do not separate.
