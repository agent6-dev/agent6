# Compaction settings experiment

The six-module task measures what a worker keeps after a requested summary.
For a measurement that opens only a journal and never starts a worker, use the [offline counter](compaction-offline.md).
Recorded sessions are not live inputs.
Neither experiment changes a product setting.

## Six-module task

Each set creates a disposable Git repository from [twelve rules](six_modules/spec.md).
A scripted opening reads `spec.md`, `pairs.py` and `limits.py`.
It delivers a labelled steer after the first request and stops at three turns.
Each trial resumes at turn 4 with the same checkout, state and conversation from that opening.
It uses the same absolute workspace path and a new conversation cache key.
There is no session-directory input to `small`.

For a smoke with **no model or credentials**, run this from the bench source checkout with an installed agent6:

```sh
umask 077
work=$(mktemp -d /var/tmp/six-modules.XXXXXXXX)
export XDG_CONFIG_HOME="$work/config" XDG_CACHE_HOME="$work/cache"
export XDG_STATE_HOME="$work/state" XDG_DATA_HOME="$work/data"
mkdir -p "$XDG_CONFIG_HOME/agent6"
printf '[sandbox]\nrun_commands = "allow"\n' > "$XDG_CONFIG_HOME/agent6/config.toml"
bench() { uv run python -m bench.compaction_bench "$@"; }
bench small --pilot pointers --scripted --out "$work/smoke"
bench table "$work/smoke/runs.jsonl"
```

The loopback worker edits one module and adds a passing unittest.
The expected external score is **1/6** and the task does not pass.
Its token counts come from scripted responses.

For model measurements, use a fresh private directory on a real disk and a separate profile.
Run `uv run agent6 connect chatgpt` there. Use this `$XDG_CONFIG_HOME/agent6/config.toml` for the trials:

```toml
[providers.chatgpt]
backend = "chatgpt"
[budget]
max_plan_points = 5.0
max_usd = 5.0
```

Choose the limits before running a set.
`max_plan_points` caps observed subscription-plan percentage points per execution.
`max_usd` caps dollar-metered usage, not ordinary subscription usage.
Neither caps the whole set. Leave plan overage disabled.
Use your normal user outside another agent6 run on a host with full command isolation.
Under the default command policy, answer command prompts in the terminal.
The bench asks separately before running the external scorer.
Do not weaken a sandbox or approve a command you would otherwise deny.

```sh
bench small --pilot baseline --out "$work/pilot-small"
bench small --out "$work/sol-small"
bench table "$work/sol-small/runs.jsonl"
# K1 and K2 are your two distinct finalists.
bench small --finalists "$K1" "$K2" --out "$work/astra-small"
```

The screen runs five replicates per arm on `chatgpt/gpt-6.2-sol`.
Confirmation runs three per finalist on `chatgpt/gpt-6.2-astra`.
Compare pass and exact recall first, then rereads, restarts, turns and tokens on like tasks.
Record the reason for the choice before confirmation.
Do not count pilots as replicates.

## Keep settings

Each arm starts from all operator messages, unlimited turns and an 80,000-character recent tail.
The caps apply to optional history.

| Arm | Change from `baseline` |
| --- | --- |
| `baseline` | None |
| `no-operator` | `keep_operator_messages = 0` |
| `chars-20000` | `keep_recent_chars = 20000` |
| `turns-1` | `keep_recent_turns = 1` |
| `operator-only` | `keep_recent_turns = 0` |
| `pointers` | Charge the baseline tail, then replace its tool results with the drop step's short pointer. Preserve operator answers. |

Keep trials require a requested worker summary before the first ordinary request.
Both worker and helper use `medium` effort.
There is no review, parallel tool or elision gist.
The turn cap is 100 on resume.
The small task drops at 8,000 characters and summarizes at 100,000.

## Reading the table

Each new set directory holds `runs.jsonl` with one flushed row per attempt.
A measurement error has an `error` field and causes a nonzero set exit.
Inspect the journal under `trial/artifacts` before interpreting a missing count.
The table groups by stage, model, task, checkpoint and arm.
It prints successes per observed boolean and min..max with the number observed for quantities.
Missing evidence is `null`, not zero.

| Field | Evidence after the saved journal boundary |
| --- | --- |
| `passed` | A `finish_session` stop with a full passing verification. Gateless, scoped or unknown verification is not a pass. A small pass also needs 6/6 groups on the final checkout. |
| `groups` | Number of modules the six-group checker reports passing. Each group checks two functions and their error cases separately from the worker's own tests. |
| `turns` | Distinct worker turns with an ordinary inference request. Summary calls and retries do not add worker turns. |
| `rereads_6_calls`, `rereads_10_turns` | Successful `read_file` dispatches on repository-relative paths already read. Seed the set with completed pre-boundary reads. Count repeats within the window too. The windows are the first six tool dispatches and turns `[restart, restart + 10)`. Shell reads and inferred accesses do not count. |
| `second_restart` | A subsequent accepted summary after the first ordinary worker request. Size checks, failed attempts and dropped results alone do not count. |
| `exact_recall` | The expected string is a whole, exact line of the actual finish summary. A labelled finish without the line is false. No label or no finish answer is unavailable. This measures the answer, not how the worker found it. |
| `first_request_input_tokens`, `first_worker_input_tokens` | Total input of the first inference attempt and the first ordinary worker attempt respectively. A cold helper can go first. Each has a separate cache-read field. |
| `uncached_input_tokens`, `total_tokens` | Sum settled attempt usage through stop, including summary calls and retries. Uncached is fresh input plus reported cache writes when a cache split is available. Total adds reported cache reads and output to fresh input and writes. A missing attempt bill makes the affected sum unavailable. |

A path preview ending in the journal's `… (N chars)` marker does not identify a file.
If a reread count depends on such a preview in the frozen prefix or the resumed window, the row reports `null` for that count.

The measured tree, source session ID, restart turn, model, arm, replicate and checkpoint digest identify a comparison in the trial artifacts and row.
The small scorer starts through isolated Python in the bench directory.
It uses the `agent6 exec --new` sandbox with the policy captured before the worker ran, without loading config from the finished checkout.
Both the model-backed worker and its scorer require `full` isolation at launch.
Only the scripted fixture may use automatic isolation.
The runner allows 120 seconds and 64 KiB per output stream for one report of six boolean verdicts.
A refused, malformed or oversized report is a measurement error, not a score.
The submitted modules run inside the interpreter returning the verdicts: the report is not proof against a submission written to falsify it.

Artifacts contain the conversation and final checkout.
Keep each private set directory for inspection, then remove it as a unit.
