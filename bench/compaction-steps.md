# Tier-1 size replay

**Dropping to half the bound cut tier-1 drops from 21 to 4 across three recorded sessions.**
The stepped rule held 6% to 25% fewer tool-result characters on average.

## Recorded inputs

All three sessions are the evaluation set; no threshold was tuned on them.
They ran on the ChatGPT plan backend with `compaction.drop_at_chars` resolved to 489,600.
There are 709 worker request transcripts in total.
The recorded drops count `compact.dropped` events, not individual elided results or tier-2 restarts.

| session | worker requests | recorded drops | replay old drops | replay stepped drops | old mean chars | stepped mean chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache-review | 19 | 12 | 11 | 2 | 403,501 | 300,860 |
| children-merge | 368 | 7 | 6 | 1 | 289,309 | 271,103 |
| web-followups | 322 | 7 | 4 | 1 | 306,265 | 275,591 |

The children-merge recording was still running when exported.
The replay's old-rule counts differ from the recorded ones because it leaves out dedup and gists.

## What a drop costs

From the same sessions' journals (`model.result` rows of the worker with a prompt over 20,000 tokens; `input_tokens` is the uncached part):

| session | calls | first call after a `compact.*` event | uncached input in those | uncached input in the others |
| --- | ---: | ---: | ---: | ---: |
| cache-review | 17 | 12 | 1,466k of 1,563k (94%) | 114k of 497k (23%) |
| children-merge | 366 | 12 | 1,107k of 1,245k (89%) | 890k of 38,766k (2.3%) |
| web-followups | 320 | 9 | 1,184k of 1,268k (93%) | 898k of 37,885k (2.4%) |

The `compact.*` events include tier-2 restarts, whose next request is small.
A call with no compaction since the one before read from the provider's cache in 902 of 908 cases, idle gaps up to 9.6 minutes included.

## Reproduction

Each session directory holds its `transcripts/` and `objects/`.
`BOUND` is the recording's resolved `compaction.drop_at_chars`, which the run prints at its start.

```sh
uv run python bench/compaction_replay.py RECORDINGS --drop-at-chars BOUND
```

The input directory contains one directory per session.
The script reads Responses worker requests only and makes no model calls.
Missing objects produce `unavailable` cells, a diagnostic naming the object and exit status 1.

## Method

- Each tool result keeps the character count from its first full appearance, ignoring the recording's later elision markers.
- Both arms follow the recorded request cadence and result membership, including the recorded tier-2 boundaries.
- The old arm drops oldest-first to the bound; the stepped arm drops to half the bound.
- Both arms keep newly introduced results, the newest two results and operator answers.
- Both arms use the current bare placeholder sizes; dedup, gist text, edited-path priority and thinking are outside this size-only comparison.
- A drop counts one request with at least one replacement; the mean weights each worker request equally and counts retained tool-result characters after the pass, including placeholders.

This deterministic replay isolates the size rule, not model quality, provider cache behavior or a counterfactual tier-2 schedule.
It needs no stochastic replicates; unit fixtures check repeated growth and recorded restart boundaries.
A live comparison of cache use and task quality remains unmeasured.
