# Compaction writers

## Host smoke

Use the cheapest model on the connected plan at its lowest supported effort for both roles.
Substitute its `provider/model` and effort below.
Use a model with a known context window for the helper.
For a supported Anthropic model, set `server_compaction = false` to compare the two local writers.
`claude_code` exercises the direct helper fallback, not a worker-written summary.

```sh
work=$(mktemp -d /tmp/compaction-writers-XXXXXX)
mkdir "$work/repo" && cd "$work/repo"
git init -q
python3 - <<'PY'
from pathlib import Path
for letter in 'abcdefghz':
    Path(f'{letter}.txt').write_text(letter * (48000 if letter == 'z' else 1000) + '\n')
PY
git add -- *.txt && git commit -qm 'seed read targets'
```

Save this overlay as `$work/worker.toml`:

```toml
[models.worker]
model = "provider/model"
effort = "low"
[models.helper]
model = "provider/model"
effort = "low"
[harness]
max_turns = 20
[harness.verify]
infer = false
at_start = false
when = "never"
[harness.prompt]
decompose = "off"
[review]
when = "never"
[compaction]
drop_at_chars = 6000
summarise_at_chars = 52000
keep_recent_chars = 1200
elision_gists = false
summariser = "worker"
```

```sh
sed 's/^summariser = "worker"/summariser = "helper"/' "$work/worker.toml" > "$work/helper.toml"
task='Read a.txt through h.txt in order, then z.txt. Use read_file once per turn, one whole file per call. Do not edit files or run commands. Finish with the letters you read.'
for writer in worker helper; do
    agent6 run --no-commands --session-id "$writer" --config "$work/$writer.toml" "$task"
    agent6 sessions receipt "$writer"
    journal=$(agent6 sessions dir "$writer")/logs.jsonl
    jq -c 'select((.type | startswith("compact.")) or .type == "model.result") | del(.summary)' "$journal"
done
```

- `compact.dropped.count` counts old tool results replaced before the final read.
- `compact.summarise.call` names the `writer` and `model_call_id`.
  Match that id to `model.result`: uncached input is `input_tokens` plus any `cache_write_tokens`; cached input is `cache_read_tokens`.
- `compact.summarise.done` names the accepted `writer` and `fallback_reason`.
  A rejected attempt has `compact.summarise.failed.error`.
- The receipt prints `Summary attempt N: WRITER; U uncached, C cached input tokens; accepted` for the summary itself.
  `After compaction (session)` measures a different call.
  A helper fallback adds `; fallback: REASON` to its attempt line.

## Stand-in observations

Three CLI runs used a local scripted OpenAI endpoint and fresh config, state, data and cache directories under `/tmp`.
Each run dropped four results in one step and accepted one summary.
Each response reported 300 uncached and 700 cached input tokens by construction.

| Setting | Requests | Receipt outcome |
| --- | ---: | --- |
| `worker` | 11 | worker accepted |
| `helper` | 11 | helper accepted |
| `worker` with a summary tool call | 12 | worker rejected; helper accepted; `fallback: summary contains a tool call` |

No live cache rate or quality score was measured on the stand-in.

## Live observations

One run per setting used `chatgpt/gpt-5.6-luna` at `low` effort for both roles, with the overlay above.
Each run made 11 model calls, dropped four results in one step, accepted its first summary and finished with `abcdefghz`.

| Setting | Summary call input (uncached / cached) | Summary length | First worker call after the restart (uncached / cached) | Run input (uncached / cached) |
| --- | ---: | ---: | ---: | ---: |
| `worker` | 24,438 / 5,120 | 9 chars | 24,264 / 3,328 | 59,906 / 37,120 |
| `helper` | 531 / 0 | 834 chars | 24,456 / 3,328 | 37,341 / 30,464 |

- The worker summary call read the earlier request prefix from the cache; its uncached input is the `z.txt` result, which no earlier call held.
- The retained tail holds that result again, so the `worker` setting sent it uncached twice and the `helper` setting once.
- The worker summary was the task's answer (`abcdefghz`), without the four numbered parts the instruction asks for.
- The helper summary said the letters could not be determined and that `z.txt` was not read: its input has no tool output and stops before the retained tail.
- One call in each run read nothing from the cache although its system text and tool definitions matched the previous request.

**Limits:** n = 1 per setting on a nine-file read task; the last result is 48,000 characters by construction, so the repeated uncached input is at its largest.
Recall on real work is unmeasured: read `compact.summarise.done.summary` in ordinary runs.

## Idle compaction findings

Two CLI runs used the local scripted OpenAI endpoint and four fresh XDG directories each.
The worker read a 12,000-character file, then called `run_command` with `sleep 6; echo command-finished`.
The idle limit was 2 seconds, the floor was 8,000 characters and the check-in was 30 seconds.
The endpoint supplied the cache counts.

| Receipt: accepted worker summary | Gap from request start | Command result | Cold-cache notice |
|---|---:|---|---|
| 100 uncached, 900 cached input tokens | 2.01 s | exit 0, `command-finished` after 6.09 s | none |
| 1,000 uncached, 0 cached input tokens | 2.00 s | exit 0, `command-finished` after 6.24 s | once in each of the log, journal and receipt |

- Both journals recorded `tool.call`, `compact.summarise.done` with `reason: idle`, then `tool.result`.
- Each summary request contained the previous request's messages plus the summary instruction and the same tools.
- Each run made four model calls, including the summary.
- The cold receipt said: `Idle summary 2.0s after the last request began: 1,000 uncached, 0 cached input tokens. providers.test.idle_compact_s = 2 may be longer than the cache lasts.`
- Neither run measured a live cache lifetime or price.

**Host smoke.** Use a throwaway repo with a configured HTTP provider and its normal command isolation.
Create a read target with `python -c 'print("observation\n" * 1000)' > notes.txt` and commit it.
Set `providers.<name>.idle_compact_s = 2` and `providers.<name>.idle_compact_min_chars = 8000` on the worker entry.
Keep the check-in above 6 seconds.

```sh
agent6 run --session-id idle-smoke 'Read notes.txt. Run sleep 6 with run_command in the foreground. Finish with its return code.'
agent6 sessions receipt idle-smoke
jq -c 'select(.name == "run_command" or (.type | startswith("compact."))) | del(.summary)' \
    "$(agent6 sessions dir idle-smoke)/logs.jsonl"
```

Compare the summary attempt's counts with its `gap_s`, not with the command duration.
If the model takes longer than two seconds to answer, the idle limit is already due when the wait begins.

## Idle compaction, live observation

One run used `chatgpt/gpt-5.6-luna` at `low` effort with `providers.chatgpt.idle_compact_s = 12`, a floor of 8,000 characters, a 120-second check-in and a foreground `sleep 30`.

| Event | Seconds after the worker request began | Input (uncached / cached) |
| --- | ---: | ---: |
| Worker request that called `run_command` | 0 | 3,223 / 3,072 |
| Idle summary call, worker-written, accepted | 12.0 | 6,512 / 0 |
| `run_command` result, exit 0 | 31.6 | |
| Next worker call | 31.8 | 438 / 3,072 |

- The journal order was `tool.call`, `compact.summarise.done` with `reason: idle` and `kept_messages: 1`, then `tool.result`.
- The summary request had the previous request's cache key, instructions and tools, and read nothing from the cache 12 seconds after a request that read 3,072 tokens from it.
- The cold-cache notice appeared once in the log, the journal and the receipt.

**Limits:** n = 1 on a three-call conversation. This model also missed its cache on one call in each run of the writer comparison above, so one cold summary call does not measure the cache lifetime.

## Signed endpoint compaction findings

Two headless CLI runs used the loopback [Messages stand-in](compaction_messages.py) with `server_compaction = true` and `standin/claude-haiku-5-5` at high effort.
Each used a fresh git repository and fresh config, state, data and cache directories under `/tmp`.
The first read returned a 13,201-character file.
Both runs used `drop_at_chars = 1000`.

| Trigger | Worker request | Summary | First request after adoption |
| --- | --- | --- | --- |
| Size, 25,000 characters | 5 messages, clearing requested | Same 5 messages, no clearing | Block, signed reply, read result, task and pin |
| Idle, 1 second, floor 8,000 characters | 5 messages, clearing requested | Same 5 messages at 1.00 s, no clearing | Block, signed reply, command result, task and pin |

- Both runs finished in four model calls, including one accepted `endpoint` summary.
- The stand-in checked the system text, tools and messages against the preceding ordinary request.
- It checked the block and the kept reply for exact equality on adoption.
- The idle journal orders `tool.call`, `compact.summarise.done` with `kept_messages: 1`, then `tool.result`.
  The foreground `sleep 4` returned exit 0 after 4.07 seconds under partial isolation.
- Both receipts reported 60 uncached and 33,050 cached input tokens on the summary, zero thinking blocks dropped and two tool results cleared.
  These counts come from the captured replies, not a measurement of cache performance or signature verification.

**Host smoke with a live key.** Use the throwaway read targets from the first smoke and an Anthropic direct entry with no custom URL.
Select a supported model such as `anthropic/claude-haiku-5-5` at `effort = "high"`.
Use `drop_at_chars = 1000` and `summarise_at_chars = 25000` for size.
For idle, raise the size limit to `300000` and set `idle_compact_s = 2` and `idle_compact_min_chars = 8000` on the provider entry.
Run the commands below with the corresponding config files and the host's ordinary command isolation.

```sh
agent6 run --config "$work/size.toml" --no-commands --session-id signed-size \
    'Read a.txt, b.txt and z.txt, one file per turn. Report the letters.'
agent6 run --config "$work/idle.toml" --session-id signed-idle \
    'Read z.txt. Run sleep 6 in the foreground. Report the return code.'
for id in signed-size signed-idle; do
    agent6 sessions receipt "$id"
    jq -c 'select(.type == "compact.summarise.done" or .type == "model.result") | del(.summary,.text)' \
        "$(agent6 sessions dir "$id")/logs.jsonl"
done
```

Check for `writer: endpoint`, the expected `reason`, and reported transformations on the next worker call.
