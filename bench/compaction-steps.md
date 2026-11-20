# Tier-1 size replay

**The recorded-session comparison is blocked by missing transcript objects.**
Every worker request input item is a `$object` reference, and none of the three exports contains its `objects/` directory.
Journal previews and file byte counts do not recover complete tool-result character counts.
The replay refuses these exports instead of substituting estimates.

## Recorded inputs

All three sessions are the evaluation set; no threshold was tuned on them.
There are 709 worker request transcripts in total.
The recorded drops below count `compact.dropped` events, not individual elided results or tier-2 restarts.

| session | worker requests | recorded drops | replay old drops | replay stepped drops | old mean chars | stepped mean chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache-review | 19 | 12 | unavailable | unavailable | unavailable | unavailable |
| children-merge | 368 | 7 | unavailable | unavailable | unavailable | unavailable |
| web-followups | 322 | 7 | unavailable | unavailable | unavailable | unavailable |

No comparison completed, so there is neither a measured improvement nor a measured null result.
The children-merge recording was still running when exported.

## Reproduction

Restore each session's `objects/` beside `transcripts/`, or supply hydrated transcript JSON.
Use the recording's resolved `compaction.drop_at_chars` as `BOUND`; the export does not include that setting.

```sh
uv run python bench/compaction_replay.py RECORDINGS --drop-at-chars BOUND
```

The input directory contains one directory per session.
The script reads Responses worker requests only and makes no model calls.
Missing objects produce `unavailable` cells, a diagnostic naming the object and exit status 1.
The three supplied exports each produced that diagnostic before any size could be measured.

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
