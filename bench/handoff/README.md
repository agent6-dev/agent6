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
Each run has a $1 cap, a plan cap (`--max-plan-points`, default 2 points) and a 100,000-token fallback.
The job needs no edits, commands, merge, verify gate or review panel.
With `harness.parallel.judges = []`, a second attempt at one task starts no ranking reviewer.
Command tools that execute shell commands are withheld.

## Run set

Run from the root of a clean checkout with its matching CLI installed in the project environment.
Use a private directory on a real disk outside the checkout for `base`.
The output directory must not already exist.

**Scripted check.** This command needs no credentials and makes no paid calls.
It runs the CLI and its detached children against a loopback provider.
The scripted children read the files and return their codes.
This split tests the method, not model behaviour.

```sh
uv run python -m bench.handoff.run --scripted --reps 1 --out "$base/scripted"
```

**Live evaluation.** Use a host with full isolation.
A run uses the `agent6-jail` bundled in the project environment (`src/agent6/sandbox/_bin/`); the runner's private environment drops `AGENT6_JAIL_BIN`.
Sign in to a dedicated profile, for example with `XDG_CONFIG_HOME="$base/signin" agent6 connect chatgpt`, skipping the model picker.
Its `agent6/config.toml` may contain only provider settings and `agent6.config_version`.
The runner uses the [tool benchmark's private homes](../tools/README.md#run-sets) and returns any refreshed credentials to that profile.
Do not use the profile elsewhere during the set.

```sh
handoff() {
    uv run python -m bench.handoff.run --profile "$base/signin/agent6" \
        --model chatgpt/gpt-6.2-sol "$@"
}
handoff --reps 1 --out "$base/smoke"
handoff --reps 3 --out "$base/live"
```

The first command is the smoke: one pair, which is two parent runs and twelve children.
Inspect its rows, journals and each `agent.log` before the measured set of six parent runs and 36 children.
The plan cap counts a parent and its children together.
On a plan route `agent.log` reports the run's plan movement: give the measured set a `--max-plan-points` above what a smoke run moved.
Run the set sequentially on a quiet host and account: host load delays child starts, which the wait measure includes.
Each replicate includes both arms, text first on odd replicates and id first on even ones.
Freeze the job and settings before evaluation; do not pool the smoke or scripted runs with live replicates.

## Journal measures

`results.jsonl` keeps one line per run, including failed and capped runs.
Each line names the revision, seed commit, arm, replicate, plan cap, parent journal, CLI exit code and stop reason.
The run directory keeps the config, checkout, console log and journals.

| Measure | Evidence in the parent journal |
|---|---|
| Stuck notices | Count `nudge.task_stuck`, not mentions of being stuck in text. |
| Parent turns | Read `session.stop.turns`. |
| Parent tokens | Sum fresh input, cache reads, cache writes and output separately from `model.attempt`, including retries. Exclude child usage and do not add `model.result` again. |
| All accepted | Look up the six `task.queued` ids in the last `graph.update`. All six must be `passed`. A passed copy, a passed root or a finish summary is not evidence about the originals. |
| Started twice or never | Match `parallel.child_activity` starts to exact queued titles. Count how many of the six started at least twice, and how many had no start. A refused call is not a start. |
| Wait below two | For each task, accumulate seconds from queueing until first start, first observed non-open status or parent stop, only while fewer than two children ran. `under_two_s` counts those intervals once even when several tasks waited. |
| Off-arm calls | Count `parallel` calls, refused ones included, whose arguments do not use the assigned id or verbatim text form. Keep these runs under the assigned arm. |

Child activity timestamps measure when the parent observed starts and ends, not CPU time or the instant a worker process exited.
`wait_by_task_s` and `starts_by_task` retain the per-task readings.
An unmatched start, a missing endpoint or an unreadable journal leaves the affected measures unknown.
An unreported token category is `null`, not zero.
A started child whose own journal records another provider or model is named in `issues`; the run stays in its arm.
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

## Results

Live split on `chatgpt/gpt-6.2-sol` at `medium`, three replicates per arm.
Every run was accepted, with no issue, stuck notice, second start, unstarted task or off-arm call.
Quantities are the median [minimum, maximum] from the parent's journal; cache writes were unreported.

| Arm | n | All accepted | Parent turns | Fresh input | Cache read | Output | Wait below two (s) |
|---|---|---|---|---|---|---|---|
| `text` | 3 | 3/3 | 19 [18, 21] | 38,199 [33,524, 39,503] | 68,864 [62,720, 90,624] | 607 [501, 647] | 44.2 [36.9, 57.2] |
| `id` | 3 | 3/3 | 15 [14, 16] | 35,802 [34,762, 41,354] | 43,776 [43,008, 59,648] | 331 [329, 371] | 28.1 [27.2, 41.3] |

The `id` arm used fewer parent turns and fewer output tokens: those two ranges do not overlap.
Fresh input, cache reads and the wait overlap between the arms.
The smoke, one run per arm and not pooled, read 19 against 14 turns and 601 against 297 output tokens.

## Queue arrivals and a blocking child question

Two detached runs used `chatgpt/gpt-6.2-sol` at medium, full isolation, away mode `wait` and a three-child limit.
Each root received three operator tasks through the task queue before startup and three `/task` steers after the first child started.
The original task IDs join `task.queued` to the final `graph.update`; the child start detail identifies the assignment.
Sessions show and attach ran in separate CLI processes.
Read times include polling and CLI startup, so they bound visibility from above.

### Run 1

| Task | Queued | Child | Report | Final task state |
|---|---|---|---|---|
| 0002: Choose a file | Before start | queue-p1-l1 | blue-491 | passed |
| 0003: Ash | Before start | queue-p2-l1 | ash-497 | passed |
| 0004: Birch | Before start | queue-p3-l1 | birch-500 | passed |
| 0005: Cedar | During children | queue-p4-l1 | cedar-503 | passed |
| 0006: Oak | During children | queue-p5-l1 | oak-506 | passed |
| 0007: Missing file | During children | queue-p6-l1 | Absent | passed |

From the child's question to the root journal: 1.86 s.
`sessions show` returned `waiting · question` at +6.44 s; `attach --json` returned an unanswered blocking question at +9.35 s.
The child received the CLI answer at +11.85 s and reported the code in blue.txt.

### Run 2

| Task | Queued | Child | Report | Final task state |
|---|---|---|---|---|
| 0002: Choose a file | Before start | queue-p1-l1 | blue-491 | passed |
| 0003: Ash | Before start | queue-p2-l1 | ash-497 | passed |
| 0004: Birch | Before start | queue-p3-l1 | birch-500 | passed |
| 0005: Cedar | During children | queue-p4-l1 | cedar-503 | passed |
| 0006: Oak | During children | queue-p5-l1 | oak-506 | passed |
| 0007: Missing file | During children | queue-p6-l1 | Absent | passed |

From the child's question to the root journal: 0.86 s.
`sessions show` returned `waiting · question` at +6.10 s; `attach --json` returned an unanswered blocking question at +9.73 s.
The child received the CLI answer at +12.46 s and reported the code in blue.txt.

Neither run lost an ID.
Both marked the missing file passed despite its failed report.
Both refused the parent’s attempt to change passed to failed.
The final summaries named that failure.

## Questions in a recorded parent queue

Count one row per question in `question.prompt` in 56 child run journals, checked against `ask_user` tool calls.
Match `question.answer` by session and question ID.
Exclude nine questions the parent asked on its own behalf.
Classify by who held the evidence, not by who received the notification.

| Answer source | Questions | Example |
|---|---:|---|
| Parent's records | 1 | A child without access to its predecessor's review asked for seven findings and the reviewers' summaries. |
| Operator only | 0 | None |
| Child's repository | 0 | None |

The single question was queued without waiting and had no recorded answer, so median time to answer is undefined.
The parent could read the earlier child's journal and ultimately recovered the findings; the record does not show it had read all seven before the question.
Its attempted steer reached an already ended child and delivered nothing.
This sample cannot supply five examples per class or estimate savings from a parent answering questions.
