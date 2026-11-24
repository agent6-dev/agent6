# Compaction settings experiment

The six-module task measures what a worker keeps after a requested summary.
For a measurement that opens only a journal and never starts a worker, use the [offline counter](compaction-offline.md).
The harness refuses supplied sessions as worker inputs, including sessions from disposable checkouts.
A saved session does not establish which files a new worker or its commands could access.
There is no command to register or relocate a recording.
Neither the keep comparison nor the cold-start comparison runs on recorded sessions.
Neither experiment changes a product setting.

## Six-module task

Each set creates a disposable Git repository from [twelve rules](six_modules/spec.md).
A scripted opening reads `spec.md`, `pairs.py` and `limits.py`.
It delivers a labelled steer after the first request and stops at three turns.
Each trial resumes at turn 4 with the same checkout, state and conversation from that opening.
It uses the same absolute workspace path and a new conversation cache key.
There is no session-directory input to `small`.

For a smoke with **no model or credentials**, run this from the bench source checkout with an installed agent6 and its `agent6-jail` launcher:

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
In six scripted pilots (one per keep arm), all six worker gates passed and all six external scores were **1/6**, so no task passed.
Each pilot took three resumed turns, repeated the label exactly, made no rereads in either window and had no second restart.
The scripted responses reported 33,052 input tokens on the first ordinary worker request, 960 uncached input tokens through stop and 132,664 total tokens.
These fixed replies test counting and grading, not which setting works best.

The [request test](../tests/unit/test_compaction_arm_requests.py) uses a generated checkpoint with six long tool results.
It checks what reaches the first ordinary loopback worker request under all six keep and five cold settings.
It does not measure a cold provider cache.

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

## Screen results

The screen ran five attempts per arm on `chatgpt/gpt-6.2-sol` at `medium`.
Five attempts ended in the measurement error `no accepted requested worker restart before the first worker request` and have no score.
Every scored run passed with 6/6 groups and exact recall, and none had a second restart.
Quantities are min..max over the scored runs.

| Arm | Attempts | Errors | Pass | Turns | Rereads (6 calls) | Uncached input | Total tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 5 | 2 | 3/3 | 17..18 | 0..0 | 41,924..58,426 | 134,027..142,624 |
| `no-operator` | 5 | 1 | 4/4 | 18..20 | 0..0 | 38,772..64,980 | 140,852..162,507 |
| `chars-20000` | 5 | 0 | 5/5 | 17..20 | 0..0 | 34,246..50,637 | 131,510..157,966 |
| `turns-1` | 5 | 1 | 4/4 | 21..25 | 2..2 | 55,400..88,725 | 152,840..184,031 |
| `operator-only` | 5 | 0 | 5/5 | 21..25 | 2..2 | 55,939..85,400 | 158,868..186,817 |
| `pointers` | 5 | 1 | 4/4 | 18..21 | 0..3 | 49,178..60,189 | 138,507..159,131 |

`turns-1` and `operator-only` reread two files in every scored run and took 21 to 25 turns against the baseline's 17 to 18.
`no-operator`, `chars-20000` and `pointers` overlap the baseline on turns and tokens.
No arm separated from the baseline on pass or recall.
The pilot, one `baseline` run and not a replicate, passed 6/6 with exact recall in 18 turns.

## Confirmation on the six-module task

The finalists were `chars-20000` and `no-operator`.
Both had exact recall and zero rereads on every scored screen run, with five and four scores respectively against three for the baseline.
Each finalist resumed the same generated opening three times on `chatgpt/gpt-6.2-astra` at `medium`.
All six passed the worker gate and all six groups, repeated the exact label, and needed no second restart.

| Arm | Rep | Turns | Rereads (6 calls / 10 turns) | First worker input | Uncached input | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chars-20000` | 1 | 17 | 0 / 0 | 4,900 | 40,407 | 134,796 |
| `no-operator` | 1 | 17 | 0 / 0 | 4,906 | 34,299 | 134,963 |
| `chars-20000` | 2 | 17 | 0 / 0 | 4,946 | 34,987 | 134,577 |
| `no-operator` | 2 | 17 | 0 / 0 | 4,881 | 37,256 | 135,728 |
| `chars-20000` | 3 | 17 | 0 / 0 | 4,887 | 34,840 | 135,595 |
| `no-operator` | 3 | 17 | 0 / 0 | 4,887 | 36,596 | 134,584 |

The two settings did not separate on correctness, recall, resumed turns or rereads.
Their total token ranges overlap.

There is no live result on a long recorded session.
The worker accepts only the generated opening, since a supplied session does not establish its effective file access policy.
The [tier-1 replay](compaction-steps.md) reads recorded requests without starting a worker.
Neither it nor the offline retention counter can measure whether a resumed worker rereads a file or finishes correctly.

## Cold resume with a lookup from an earlier turn

Each first execution read three 100-row sheets linked by filename and stopped at a three-turn cap.
After the two-second provider idle limit, the second execution received a steer to write row 037 of the first sheet to a file.
Every journal recorded the requested cold action; one kept turn, no elision gists, and the helper summary writer separate the four treatments on a short history.
Inputs include cache reads; the first request can be the helper’s, and the total includes both executions and any helper.
Rereads count resumed `read_session` calls and reads of files seen in the opening; correctness compares the output file with the generated value.

| Model (medium) | Arm | Rep | First input | First worker input | Rereads | Correct | Total input |
|---|---|---:|---:|---:|---:|---|---:|
| gpt-6.2-sol | drop | 1 | 5,441 | 5,441 | 1 | Yes | 29,027 |
| gpt-6.2-sol | drop | 2 | 5,438 | 5,438 | 1 | Yes | 29,008 |
| gpt-6.2-sol | drop | 3 | 5,442 | 5,442 | 1 | Yes | 29,033 |
| gpt-6.2-sol | resend | 1 | 6,397 | 6,397 | 0 | Yes | 25,242 |
| gpt-6.2-sol | resend | 2 | 6,396 | 6,396 | 0 | Yes | 25,236 |
| gpt-6.2-sol | resend | 3 | 6,399 | 6,399 | 0 | Yes | 25,251 |
| gpt-6.2-sol | summarize | 1 | 464 | 4,446 | 1 | Yes | 26,514 |
| gpt-6.2-sol | summarize | 2 | 464 | 4,476 | 1 | Yes | 26,604 |
| gpt-6.2-sol | summarize | 3 | 464 | 4,473 | 1 | Yes | 26,625 |
| gpt-6.2-sol | truncate | 1 | 4,155 | 4,155 | 1 | Yes | 31,764 |
| gpt-6.2-sol | truncate | 2 | 4,159 | 4,159 | 1 | Yes | 31,792 |
| gpt-6.2-sol | truncate | 3 | 4,157 | 4,157 | 2 | Yes | 48,597 |
| gpt-6.2-astra | resend | 1 | 6,400 | 6,400 | 0 | Yes | 25,256 |
| gpt-6.2-astra | resend | 2 | 6,396 | 6,396 | 0 | Yes | 25,236 |
| gpt-6.2-astra | summarize | 1 | 464 | 4,446 | 1 | Yes | 28,451 |
| gpt-6.2-astra | summarize | 2 | 464 | 4,445 | 1 | Yes | 26,517 |

All twelve sol runs ended right.
The two arms with the lowest median total input were `resend` and `summarize`.
On astra, both passed twice and `resend` again used fewer total input tokens.
`truncate` reduced first input relative to `drop` on sol, but used more total input in every run, so this lookup gives no reason to replace the `drop` default with `truncate`.
This is one task with one kept turn, not a comparison of cache latency, billing or long sessions.
