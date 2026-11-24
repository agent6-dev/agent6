# Review panel validation

agent6 runs reviewer subagents over a diff and counts reviewers with new grounded block findings through the in-loop `[review]` config.
The `review` preset enables a three-reviewer `finish` panel.
The configuration reference documents its triggers, roster and blocking rules.
With the panel off, `agent6 ask "/review TARGET"` returns the worker's Markdown review.
With the panel on, reviewers record findings on the target before the worker answers.

[Recorded review replay](replay.md) compares a panel reviewer's first reply under two system prompts.
[Lean review category counts](categories.md) compares labels for repeated defects.

The measurements below used the prompt with a reviewer vote and fixed categories.
They do not measure the prompt with free labels and no vote.

## Which findings can block finish

The pre-0.0.4 `reviewer.py` blocked correct work that passed verification despite its instruction not to block on speculation.
The panel drops findings that do not cite the diff.
A reviewer blocks on a new grounded `block` finding, regardless of its category or the verify result.
Each finding may include a suggested fix for the worker and the operator.
With `review.block_on = "none"`, block findings prevent neither finish nor merge.
A finding that passes these checks can still be wrong.

## Validation

Primary gate: the false-block and stall rate on correct work must stay at the
panel-off baseline. Secondary: real-bug catch rate, added tokens and USD, wall
time.

### Post-hoc, planted diffs (`--reviewers 3`, kimi-k2.6, real API)

| Diff | Result |
|------|--------|
| Hardcoded API key + path traversal | 2 grounded blocks (deduped from 3 reviewers); both real, cited at real lines |
| `token ==` compare + stray `.err` files | 3 grounded blocks: timing side-channel + 2 off-topic-edit |
| Clean (constant-time compare, env secret) | PASS under veto; the lone finding was a non-gating `warn` |

The grounded aggregator caught real, cited, block-eligible defects and did not
false-block clean work with the veto on. About $0.03 to $0.06 per panel; 3
concurrent reviewers run in about 2 minutes.

### In-loop veto, full run

A `read_doc(name)` task with a path-traversal trap (`finish`, 3 reviewers,
veto). The worker's naive `open("files/" + name)` was blocked by all 3 reviewers. Its
first fix (an abspath check) still let symlink traversal through, and security
blocked again. It switched to `os.path.realpath` and the panel passed. Three
rounds, no stall, final code symlink-safe.

### A/B, off vs veto (3 reviewers, kimi)

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

### Cross-file review

`parse(s)` gained a required argument, breaking a caller in `main.py` that was not in the diff.
In this experiment the diff-only reviewer missed the caller.
The reviewer with read tools (`read_file` and `find_references`) found it and blocked with `verify-uncovered-correctness`, cited at the diff line.

## The risk-field prompt A/B (measured, then removed)

The panel prompt once asked each reviewer what could fail and what the supplied checks do not establish, returned as a `risk` string of at most 1,000 characters.

- **Arms:** A omitted the risk instruction and JSON field; B included them.
- **Cases:** two development cases and four evaluation cases, frozen before tuning.
- **Protocol:** both arms on each case, three repetitions, three diff-scope reviewers per panel, identical routes, temperatures, token caps and supplied checks (36 panels, 108 reviewers).
- **Result:** measured 2026-11-05 on `chatgpt/gpt-6-astra` at effort high.
  Neither arm blocked or abstained on the clean cases.
  Every arm-B reviewer returned a risk within the bound.
  Arm B cost 69 more input tokens and 84 to 151 more output tokens per reviewer, and one to three seconds more per panel.
  The planted cases' blocks were not adjudicated against the oracle.
- **Outcome:** no behavior improvement was shown, so the risk field and its instruction were removed with the A/B's runner.
  A reviewer reports a concern as a finding.
