# Later fixes on reviewed lines

For each of 20 reviews, take nonblank additions and replacements visible in the first recorded request’s diff.
Locate the merged head by its unique commit subject and map unchanged runs of text onto that version of each file.
Walk subsequent commits on the touched files, updating line numbers and dropping replaced lines.
Inspect `fix` commits that replace a tracked line or insert between two of them; count a defect only when the correction addresses behaviour present in the review, not a neighbouring import, fixture or feature.
Count a symptom once per head and credit it if either lean reply at that head described it, regardless of severity.

| Review | Reported | Missed |
|---|---:|---:|
| p1/1 | 1 | 8 |
| p1/2 | 1 | 8 |
| p2/1 | 0 | 2 |
| p2/2 | 0 | 0 |
| p3/1 | 2 | 2 |
| p3/2 | 2 | 1 |
| p4 | 0 | 0 |
| p5/1 | 2 | 1 |
| p5/2 | 0 | 1 |
| p6/1 | 2 | 2 |
| p6/2 | 1 | 1 |
| p10 | 0 | 0 |
| p12 | 1 | 0 |
| p13/1 | 0 | 5 |
| p13/2 | 0 | 2 |
| p14 (unmerged) | N/A | N/A |
| p15/1 | 3 | 3 |
| p15/2 | 2 | 0 |
| p16 | 0 | 0 |
| p17 | 3 | 1 |
| Total (19 merged heads) | 20 | 37 |

The unmerged prompt had no later commit on its recorded line of development.
The two Files reviews illustrate the limit: the earlier diff exposed two later fixes (landscape editor controls and unrepresentable paths); neither reply reported them.
Zero means no qualifying fix found, not a clean review.
This is a lower bound: moves, fixes elsewhere, fixes without a `fix` subject and defects not fixed in the observed history escape the method.
The same defect can count at consecutive heads.
Which changes received fixes depends partly on what the original reviewers reported.

### Fixes counted

| Review | Lean replies | Correction |
|---|---|---|
| p1/1 | Reported | distinguish task authors and keep the review contract once |
| p1/1 | Missed | keep tool answers out of endpoint fallback transcripts |
| p1/1 | Missed | reserve operator words before the newest turn |
| p1/1 | Missed | preserve operator answers omitted by tool payload limits |
| p1/1 | Missed | keep the latest planner notice across local restarts |
| p1/1 | Missed | preserve the opening task author across signed summaries |
| p1/1 | Missed | reserve the run's whole task as the oldest message at every local restart |
| p1/1 | Missed | state a pinned instruction once at a restart |
| p1/1 | Missed | quote a steered skill by its name and arguments at a restart |
| p1/2 | Reported | keep replayed answers paired with their calls |
| p1/2 | Missed | measure cold restarts with the notices the worker receives |
| p1/2 | Missed | keep the latest planner notice across local restarts |
| p1/2 | Missed | preserve the opening task author across signed summaries |
| p1/2 | Missed | keep interrupted answers with their returned results |
| p1/2 | Missed | reserve the run's whole task as the oldest message at every local restart |
| p1/2 | Missed | state a pinned instruction once at a restart |
| p1/2 | Missed | quote a steered skill by its name and arguments at a restart |
| p1/2 | Missed | resend a cold start that would replace only an earlier restart notice |
| p2/1 | Missed | keep the editor footer reachable in landscape |
| p2/1 | Missed | keep paths without a Files URL in the tree |
| p3/1 | Reported | distinguish a settled merge failure from a child launch refusal |
| p3/1 | Reported | keep each child status label on one line |
| p3/1 | Missed | record reviewers that returned before a launch failure |
| p3/1 | Missed | show a stopped child's status word once |
| p3/2 | Reported | distinguish a settled merge failure from a child launch refusal |
| p3/2 | Reported | keep returned reviewers when a panel launch or watch fails |
| p3/2 | Missed | show a stopped child's status word once |
| p5/1 | Reported | report unknown command liveness after supervisor loss |
| p5/1 | Reported | keep owned command output readable after denial |
| p5/1 | Missed | report no duration for a background command first found ended |
| p5/2 | Missed | report no duration for a background command first found ended |
| p6/1 | Reported | reload the roster after a reviewer removal attempt |
| p6/1 | Reported | report working tree status with the comparison |
| p6/1 | Missed | reserve Config toolbar space when settings have focus |
| p6/1 | Missed | reload Config after an entry removal attempt |
| p6/2 | Reported | reload Config after an entry removal attempt |
| p6/2 | Missed | reserve Config toolbar space when settings have focus |
| p12 | Reported | match the listed model before looking up its rates |
| p13/1 | Missed | start a new finding request after a refusal |
| p13/1 | Missed | keep the commit snapshot when changing diff mode |
| p13/1 | Missed | keep the resolution form beside live finding history |
| p13/1 | Missed | scrub task titles in the tree and retirement prompt |
| p13/1 | Missed | keep the cursor on its task when the tree rebuilds |
| p13/2 | Missed | scrub task titles in the tree and retirement prompt |
| p13/2 | Missed | keep the cursor on its task when the tree rebuilds |
| p15/1 | Reported | scrub terminal controls from review replay errors |
| p15/1 | Reported | confine review replay reads to the selected state |
| p15/1 | Reported | keep review IDs in one replay table column |
| p15/1 | Missed | identify reviewer recordings without a focus |
| p15/1 | Missed | refuse competing effort controls before replay |
| p15/1 | Missed | exclude backticked placeholders from reader counts |
| p15/2 | Reported | confine review replay reads to the selected state |
| p15/2 | Reported | keep review IDs in one replay table column |
| p17 | Reported | order budget snapshots and appends across writers |
| p17 | Reported | recover the credit series from the interrupted execution |
| p17 | Reported | charge credit declines within each parent execution |
| p17 | Missed | keep a credit series bounded and out of session payloads |
