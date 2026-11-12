# Adversarial review panel: validation

agent6 runs reviewer subagents over a diff and aggregates their verdicts through the in-loop `[review]` config.
The `review` preset enables a three-seat `finish` panel.
The configuration reference documents its triggers, roster and blocking rules.
`agent6 ask "/review <range>"` is a separate markdown review, not a structured panel.

This file records why the panel does not reintroduce the false-blocking that got
the pre-0.0.4 `reviewer.py` deleted, and the runs behind enabling it.

## Why it does not false-block

The old `reviewer.py` blocked correct, green-verify work because its grounding was
prose ("do not block on speculation"), and models rationalize around prose. The
panel makes the grounding executable in the aggregator instead. A reviewer's
`block` gates only if a machine check passes: the cited `file_line` is actually
present in the diff the seat was shown, and the category is in the allowed-block
set (`security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`,
`verify-uncovered-correctness`). A block that fails either check is downgraded to
`warn` before any blocking-model counting. Taste, naming, "should add a test", and
uncited claims cannot gate.

## Validation

Primary gate: the false-block and stall rate on correct work must stay at the
panel-off baseline. Secondary: real-bug catch rate, added tokens and USD, wall
time.

### Post-hoc, planted diffs (`--reviewers 3`, kimi-k2.6, real API)

| Diff | Result |
|------|--------|
| Hardcoded API key + path traversal | 2 grounded blocks (deduped from 3 seats); both real, cited at real lines |
| `token ==` compare + stray `.err` files | 3 grounded blocks: timing side-channel + 2 off-topic-edit |
| Clean (constant-time compare, env secret) | PASS under veto; the lone finding was a non-gating `warn` |

The grounded aggregator caught real, cited, block-eligible defects and did not
false-block clean work with the veto on. About $0.03 to $0.06 per panel; 3
concurrent seats run in about 2 minutes.

### In-loop veto, full run

A `read_doc(name)` task with a path-traversal trap (`finish`, 3 seats,
veto). The worker's naive `open("files/" + name)` was blocked by all 3 seats. Its
first fix (an abspath check) still let symlink traversal through, and security
blocked again. It switched to `os.path.realpath` and the panel passed. Three
rounds, no stall, final code symlink-safe.

### A/B, off vs veto (3 seats, kimi)

| Task | off | veto |
|------|-----|------|
| traversal | vulnerable (4 it) | secure via basename (9 it) |
| token compare | vulnerable (4 it) | secure via env + compare_digest (12 it) |
| correct math | correct (4 it) | correct (4 it, no churn) |
| correct string | correct (4 it) | correct (4 it, no churn) |

Both real vulnerabilities the baseline shipped were caught and fixed under veto,
with no false-blocks on clean work (veto matched off's turn count). A clean
run adds about one panel call; a buggy run costs the turns to fix it.

Keep `XDG_STATE_HOME` out of the repo (the default). An earlier A/B put it
inside, so agent6's own logs were committed and the panel correctly blocked them
as off-topic edits.

### Explore tier, cross-file

`parse(s)` gained a required argument, breaking a caller in `main.py` that was not
in the diff. A diff-tier seat passed (it cannot see the caller). An explore-tier
seat (read-only `read_file` + `find_references`) found `main.py` and blocked with
`verify-uncovered-correctness`, cited at the diff line. The explore tier adds
value over the diff tier for cross-file impact; the diff tier is the default.

## Risk-field prompt A/B registration

The panel prompt asks each seat what could fail and what the supplied checks do not establish.
The reply adds a `risk` string of at most 1,000 characters.
The parser keeps the first 1,000 characters unchanged and marks truncation.
Missing or non-string risk reads “not recorded” without discarding the verdict or findings.
This is a prompt change, not evidence of better review behavior.

- **Arms:** A omits the risk instruction and JSON field; B includes them.
- **Cases:** `cases.json` freezes two development cases and four evaluation cases before tuning.
- **Protocol:** run both arms on each case, with three independent repetitions and three seats per panel, using identical model routes, temperatures, token caps and supplied checks.
- **Measures:** grounded true blocks, false blocks on clean cases, abstentions, missing risk, risk-length violations, tokens, cost and elapsed time.
- **Reporting:** publish each split and its case count, per-case outcomes, repetition ranges and paired arm differences.
- **Status:** measured 2026-11-05 on `chatgpt/gpt-6-astra` at effort high, three seats, three repetitions, all six cases (`results/b14.jsonl`): no false block and no abstention in either arm on the clean cases, every arm-B seat returned a risk within the bound, 69 more input tokens and 84 to 151 more output tokens per seat, one to three seconds more per panel; the planted cases' `grounded_true_blocks` stay null until each block is adjudicated against the frozen oracle.
No behavior improvement is claimed.

### Running B14

**Entry-point limitation:** `/review` does not write panel records, even though it reviews a diff.
`run_ab.py` refuses live execution without `--panel-driver` rather than measuring that different prompt as B14.
The explicit opt-in uses `panel_driver.py` to call agent6's existing `review_gate.run_panel`, route providers and grounded aggregator.
It records native `loop.review.start`, `loop.review.seat` and `loop.review.panel` events and derives a receipt with `agent6.viewmodel.receipt`.
It does not drive the installed `agent6 ask` command or change `/review` behavior.

From the repository root, using the Python environment that contains agent6:

```bash
# Offline preparation, no agent, grader or model calls.
python3 bench/review/run_ab.py --dry-run

# Set MODEL to the ChatGPT model id, without the chatgpt/ prefix.
uv run python bench/review/run_ab.py --panel-driver \
    --model "chatgpt/$MODEL" --effort high --label b14 \
    --output bench/review/results/b14.jsonl
python3 bench/review/stats.py bench/review/results/b14.jsonl
```

Providers, credentials and budget caps come from the host's layered config.
The runner pins `[models.default]` and all three roster entries to the selected qualified route, temperature and effort.
Each cell has a two-commit throwaway repo and a private sibling `XDG_STATE_HOME` and `XDG_DATA_HOME`.
`--runs-root` selects their parent (default `bench/review/results/runs`).
Dry runs print all 36 config paths and write no measurement rows.
The default dry-run route is `chatgpt/DRY-RUN`; pass `--model` to render the intended route instead.

- **Matrix:** development n=2 cases, evaluation n=4 cases, three repetitions, both arms, three seats per panel (36 panels, 108 seats).
- **Scope:** diff-only seats, 1,500 output tokens per call, temperature 0.0 unless explicitly set, and fixed security, correctness and data-loss focuses.
- **Checks:** each fixture runs its supplied checks before the panel, with the same command and result in both arms.
  The cross-file fixture retains `main.py` unchanged and supplies its call in the registered check description.
  The clean-token fixture imports `hmac` in both trees.
- **Isolation:** neither the oracle nor the config is in the reviewed diff.
- **Order:** arms alternate A/B and B/A across repetitions.
- **Risk switch:** `AGENT6_REVIEW_RISK=off` selects A; `on` selects B.
  `prompts.review.panel_prompt` reads it for both diff and repo seats.
  Only `off` removes the JSON risk field and its instruction; an unset variable keeps the shipped prompt.
  The parser, grounding and receipt are unchanged.

### Reading B14 results

Each JSONL row identifies the arm, case, split, repetition, route, effort, temperature and case-file hash.
It retains the panel, block findings, receipt path, token counts, reported or estimated dollars, partial-cost flag, plan points and elapsed seconds.
A failed cell writes an error row and makes the runner exit nonzero.
Missing panels are errors, not zero findings.
ChatGPT plan points are separate from dollars; zero dollars does not mean no plan use.

`grounded_blocks` counts deduplicated grounded block findings, not distinct model routes.
`false_blocks` counts those findings on the registered clean cases.
`abstentions`, `missing_risk` and `risk_length_violations` count seats, including abstentions in missing risk.
A risk-length violation uses the parser's `risk_truncated` flag because retained text is already capped.
Missing risk in arm A is expected.

**Oracle adjudication:** a grounded citation does not prove a true defect.
On defective cases with blocks, `grounded_true_blocks` stays null until each block is judged against the frozen oracle.
The summary accepts a JSON map of run IDs to finding IDs and boolean judgments (`true` means the finding matches the planted defect):

```json
{"<run_id>": {"<finding_id>": true, "<other_finding_id>": false}}
```

```bash
python3 bench/review/stats.py bench/review/results/b14.jsonl \
    --adjudications bench/review/results/adjudications.json
```

The summary reports each split and case, means, repetition ranges, known/total counts and matched B-minus-A differences.
Unjudged true blocks remain missing rather than becoming zeros.
Duplicate cells are refused; use a distinct label for another batch.
