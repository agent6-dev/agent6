# Worker summary request prefixes

Six of seven rebuilt ChatGPT requests preserve the complete recorded input prefix.
The seventh preserves the recorded fields and item order, but the transcript omits four image payloads.
All seven preserve the system instructions, tool definitions, reasoning settings and `prompt_cache_key`.

| Session | Summary time (UTC) | Previous request prefix |
| --- | --- | --- |
| children-merge | 18:48:21 | 287 items match |
| children-merge | 19:27:28 | 288 items match |
| children-merge | 20:05:43 | 215 items match |
| children-merge | 20:42:11 | 262 items match |
| children-merge | 21:31:35 | 273 items match |
| web-followups | 18:51:06 | 470 items match on recorded fields; full prefix null (4 images omitted) |
| web-followups | 19:32:08 | 317 items match |

`cache-review` has no tier-2 point.
The recorded thresholds were `drop_at_chars = 489600` and `summarise_at_chars = 1022464`.
The replay uses every journaled summary point rather than predicting when a new run would compact.

## Reproduction

```sh
uv run python bench/summary_prefix_replay.py RECORDINGS
```

Each session under `RECORDINGS` supplies `logs.jsonl`, `transcripts/` and `objects/`.
On either side of `compact.summarise.call`, the replay selects `model.result` rows and transcripts with `actor` set to `worker` and joins them by `model_call_id`.
It rebuilds the conversation from the previous request, its reply and the tool results and notices in the retained tail.
It takes the worker settings and conversation key from the following request.
It passes the conversation through the worker summary call builder and the ChatGPT adapter without sending the request.

The comparison checks the previous `input` items in order and every other request body field for equality as JSON values.
The final item must be the summary instruction.
For an omitted image, its path, media type, byte count and recorded hash stand in for the bytes.
The replay does not open the image path.

**Limits:** The replay checks request structure, not a provider cache hit.
It made zero model calls and incurred no provider charge.
Cached tokens, uncached tokens and recall scores are unmeasured at every point.
See [Compaction writers](compaction-writers.md) for the two-writer host smoke and its receipt fields.
