# Compaction writers

## Host smoke

Use the cheapest model on the connected plan at its lowest supported effort for both roles.
Substitute its `provider/model` and effort below.
Use a model with a known context window for the helper.
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
- The cold receipt said: `providers.test.idle_compact_s = 2 is longer than the cache lasts. Idle summary 2.0s after the last request began: 1,000 uncached, 0 cached input tokens.`
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
