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

No live cache rate or quality score was measured.
The live token counts are the operator's to fill in from the host receipts.
The operator can read `compact.summarise.done.summary` in ordinary runs to assess what survived.
