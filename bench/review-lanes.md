# Review lane execution comparison

## Registered corpus

This comparison measures execution parity and startup cost, not model quality.
The corpus has 10 diff cases and 10 repo cases, each repeated three times per executor with identical prompts, repository contents, model settings and scripted responses.
The route is a local OpenAI-compatible endpoint, temperature 0.0, no reasoning override.
Each successful call reports 10 input tokens, 20 output tokens and USD 0.001; HTTP failures report no usage.
Diff reviewers allow one call with 1,500 output tokens; repo reviewers allow three calls with 2,000 output tokens each.

| Case | Diff response | Repo response trace |
| --- | --- | --- |
| pass | Pass JSON | Pass JSON |
| block | Grounded block JSON | Grounded block JSON |
| risk | Pass with risk text | Pass with risk text |
| risk-cut | Pass with 1,100-character risk | Pass with 1,100-character risk |
| junk | Non-JSON text | Non-JSON text |
| invalid | Invalid verdict JSON | Invalid verdict JSON |
| truncated | Partial JSON, length stop | Partial JSON, length stop |
| empty | Empty response | Empty response |
| provider-error | HTTP 400 | HTTP 400 |
| tools | Pass JSON beside a read call | Read call, read call, pass JSON beside a read call |

The base contains `README.md` with `base`; the reviewed tree adds `changed` on line 2.
The blocking finding cites `README.md:2` on the head side.
Measurements retain elapsed dispatch-to-verdict time, process CPU time, provider request count, reported usage, verdict, risk and grounded findings.
## Results

The baseline and lane arms each contain 60 runs (10 diff cases and 10 repo cases, three repetitions per case).
The arms ran sequentially, not interleaved; pairs share scope, case and repetition.
Raw results are [baseline](review-lanes-baseline.jsonl) and [lanes](review-lanes-new.jsonl).
Both arms were measured with `review.scope` selecting a one-call diff reviewer or a tool-reading repo reviewer.
`uv run python -m bench.review_lanes` runs the repo trace through the one reviewer shape and writes `review-lanes-rerun.jsonl`; it uses only the local scripted endpoint in `tests/review_traces.py`.

| Scope | Arm | n | Dispatch-to-verdict median (s) | Min to max (s) | Parent CPU median (s) | Requests | Reported USD |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| diff | in-process baseline | 30 | 0.0508 | 0.0237 to 0.4150 | 0.0380 | 30 | 0.027 |
| diff | read-only lane | 30 | 1.5084 | 1.5003 to 3.5226 | 0.0494 | 30 | 0.027 |
| repo | in-process baseline | 30 | 0.0865 | 0.0283 to 0.3246 | 0.0517 | 36 | 0.033 |
| repo | read-only lane | 30 | 3.5024 | 1.5018 to 3.5260 | 0.0527 | 36 | 0.033 |

| Scope | Paired latency increase median (s) | Min to max (s) |
| --- | ---: | --- |
| diff | 1.4709 | 1.3737 to 3.4554 |
| repo | 3.3970 | 1.3911 to 3.4742 |

Every pair matched on verdict, grounded findings, summary, diagnostic, risk text, truncation flag, provider request count and reported USD.
The lane record additionally names its real child session.
Diff reviewers made one request each; the repo tool-boundary case made three, including its final verdict beside an unexecuted tool request.
Each lane run started one additional read-only session process; the baseline started none.

**Measurement limits:** dispatch timing includes clone, installed bootstrap, provider calls, wait and import, but excludes preparation of the retained target and local HTTP server.
Startup and join were not timed separately.
Parent CPU excludes the reviewer process; the raw lane data also records reaped child CPU, not a complete process-tree CPU total.
Reported USD comes from scripted usage, not paid requests.
The extra diff-reviewer startup cost is visible in the table.
These deterministic traces test execution parity, not model quality or production-provider latency.
