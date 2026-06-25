# Adversarial review panel ("ensemble")

Status: DESIGN v2 (dev-0.0.12), revised after a 4-axis red-team. Opt-in, default
off. The post-hoc path may ship; the in-loop GATE ships only after validation.

## Goal

Let a user configure N adversarial reviewer subagents that scrutinize the
worker's work (like Claude Code "ultracode" multi-agent verification), fitted to
agent6 so it does NOT reintroduce the false-blocking that got the pre-0.0.4
`reviewer.py` deleted.

End state ("C"): independent reviewers, distinct models/personas per seat, run
concurrently, judged by a grounded aggregator — usable in-loop (self-correction)
and post-hoc (`agent6 review`). We reach it in stages, each a real increment.

## Why this beats the deleted reviewer.py (the one idea that matters)

reviewer.py false-blocked correct, green-verify work because its grounding was
**prose** ("verify already passed, don't block on speculation") and models
rationalize around prose. The history (21e9324, 2e34bc9, the giant GREEN-VERIFY
escape-hatch in eaabb8f^) shows three attempts to fix it in the prompt; it was
deleted anyway.

**Fix: make grounding executable in the aggregator, not requested in the prompt.**
A reviewer's `block` is only allowed to gate if a machine check passes:

- its `file_line` citation refers to a line **actually present in the diff it
  was shown**, AND
- its `category` is in the **allowed-block set**:
  `security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`,
  `verify-uncovered-correctness`.

Any `block` failing either check is mechanically **downgraded to `warn`** before
any quorum/veto counting. `warn`/`nit` never gate. Taste, naming, "should add a
test", "should raise X", and uncited claims cannot block — by construction, not
by please-don't.

## Non-goals

- The WORKER stays single-loop (one provider, one history, one edit loop). No
  planner→worker→reviewer handoff. Reviewers are read-only; never edit/commit.
- No verbatim third-party prompts (CL4R1T4S is AGPL + unlicensable leaked IP +
  injection payloads). Original wording only.
- NOT backward-compatible with today's critic. Today's critic sees a 6-message
  transcript tail and a failure-seeking prompt with no diff/verify. The panel
  REPLACES it with a diff+verify-grounded contract; we A/B against THAT.

## Operational model (corrected: synchronous fan-out, no new state infra)

The panel is a **synchronous, in-memory, read-only fan-out run at a trigger**,
exactly like the existing `_run_critic` — NOT a set of DAG sub-tasks.

- agent6's only resumable unit is the worker's `loop_state.json` snapshot,
  written before each worker LLM call. The panel runs *inside* one worker
  iteration, after that snapshot, and folds its result into `messages` before
  the iteration returns. So it is **idempotent-by-recompute**: on resume the
  worker re-emits `finish_run`, the panel simply re-runs in full. There is no
  "seat 2 of 4 done" to cache and no `run_dir/subagents/` state tree.
- Durable state the panel DOES need (persist in the snapshot; bump
  `SNAPSHOT_VERSION`): `review_rejections_total: int` and
  `last_panel: {trigger, iteration, blocked} | None`. Without this, the
  anti-stall counter resets to 0 on every resume and silently disarms.
- Concurrency does not change the recorded order: each seat **returns** its
  `ReviewVerdict` + a buffered event list; after join the **parent** emits
  `loop.review.seat.*` in fixed seat order, then `loop.review.panel`, each
  stamped `(panel_id, seat, seq)`. Seat threads never touch the event sink or
  the worker `messages`. Panel is synchronous w.r.t. the worker loop, so worker
  and review events never interleave.
- The DAG/curator is NOT involved (it owns worker TaskNodes, flock-serialized;
  it is not a fan-out executor). The only post-hoc artifact is a single
  `run_dir/review/<panel_id>.json`, write-only, not read on resume.

(We kept the useful half of the "DAG sub-task" idea — deterministic merged
ordering — and dropped the half that needed nonexistent subagent journaling.)

## Contracts

```
Severity = "block" | "warn" | "nit"
Finding  = { category, severity, file_line, title, detail }
            # category in: security|sandbox-bypass|off-topic-edit|data-loss|
            #              verify-uncovered-correctness|test-gap|style|over-eng|other

ReviewContext = {
    task, agents_md,
    diff,            # the working-tree delta since the last accepted finish
    diff_files,      # set of "path:line" present in diff (for grounding checks)
    verify_ok,       # bool | None (None = no verify configured)
    verify_output,   # tail of the last verify run
    persona,         # seat stance, e.g. "security"
    prior_findings,  # findings already injected & unresolved (dedup, not re-count)
}
ReviewVerdict = { seat, model, verdict: "pass"|"block", findings: [Finding],
                  summary, error: str|None }   # error => abstain (not a pass)

PanelResult = { panel_id, blocked: bool, decision, merged_findings: [Finding],
                per_seat: [ReviewVerdict], n_block, n_abstain, skipped_reason }
```

A **seat** is just `Callable[[ReviewContext], ReviewVerdict]`. Stage 1 ships ONE
implementation (`diff` tier: a structured single call). The `explore` tier
(read-only tool loop) is a *different Callable* slotted in later behind the same
boundary — so deferring it reshapes nothing.

## Aggregation (`aggregate_verdicts`, pure + heavily unit-tested)

1. **Ground every finding** against `ReviewContext`: a `block` survives as a
   block only if `file_line ∈ diff_files` AND `category ∈ allowed_block_set`
   AND (if verify_ok is False) it points inside `verify_output`. Otherwise →
   `warn`. Findings outside the delta diff are dropped.
2. **Dedup** vs `prior_findings` and across seats by `(file_line, category)`;
   severity = max.
3. **Decide** per `decision`:
   - `advisory` (DEFAULT) — never blocks; merged findings injected as guidance.
   - `veto` — any surviving block blocks.
   - `quorum` — `≥ quorum` surviving blocks block, **counting at most one block
     per distinct model** (correlated same-model seats can't fake a quorum).
   - `all` — only if every non-abstaining seat has a surviving block.
4. Abstains (errors / budget-skips) reduce the denominator and never pass/block.

## Anti-stall (so a gating panel can't burn the run)

- Per-run `review_rejections_total` (persisted). On a panel block, +1; on a
  pass, **decay −1** (not reset, so oscillation between seats still trends up).
- When `review_rejections_total ≥ review_max_total_rejections` (default 4) the
  panel **downgrades to advisory for the rest of the run** (reported via the
  `disarmed` boolean on the `loop.review.panel` event). The veto can bite real
  bugs (a higher cap is safe because grounding already blocks taste), but can
  never stall indefinitely.

## In-loop wiring (reuse the critic trigger enum; default advisory)

`workflow.critic` stays the trigger selector. The single critic call becomes a
panel run:

- `before_finish`: on `finish_run`, run the panel over the delta diff +
  verify_ok/output. If `blocked` (only possible when decision≠advisory and not
  disarmed), revoke the finish, inject merged findings as a `[review]` user
  message, +1 rejection. Else accept; still inject `warn` findings as advisory.
- `on_verify_fail`: panel is **advisory-only here** (verify-red is already the
  hard signal); cap to top-1 finding/seat, seats may only point at the failing
  symbol shown in `verify_output`.
- `periodic`: advisory injection every `critic_period` iterations.

## Post-hoc mode (ships first, safest)

`agent6 review --reviewers N [--personas a,b,c]` fans `run_panel` over the diff,
read-only, prints merged findings + per-seat verdicts. Same panel + aggregation
code, zero loop risk — the place we validate the grounded structured review on
real diffs with real API spend before anything touches the worker.

## Budget / events

- Seats draw from the run's shared `BudgetTracker`. A per-trigger sub-budget
  `min(panel_cap, review_budget_fraction × remaining)` is checked **before and
  between** seats; a seat skipped for budget abstains with
  `loop.review.skipped(reason=budget)` (≠ pass). Per-seat usage is folded into
  ONE `budget.update` the parent emits after the panel.
- Each seat provider is an `_InstrumentedProvider(role=f"review:{seat}", ...)`
  on the shared budget + sink — same shape as `_build_critic_provider`, per seat.

## Config (flat, round-trippable by the config editor — NO array-of-tables)

```toml
[workflow]
critic = "before_finish"        # trigger: off|on_verify_fail|before_finish|periodic
review_panel_size = 3           # N seats (sugar; ignored if review_seats set)
review_personas = ["security", "correctness", "tests"]   # cycled across seats
review_decision = "advisory"    # advisory(default)|veto|quorum|all
review_quorum = 2               # K for quorum (distinct-model blocks)
review_max_total_rejections = 4 # per-run gate-stall cap, then auto-advisory
review_budget_fraction = 0.25   # max run-budget fraction the panel may spend
# Explicit seats (overrides size/personas), FLAT strings "persona@provider/model":
review_seats = ["security@anthropic/claude-opus-4-8", "correctness@openrouter/x/kimi-k2"]
```

`[models.reviewer]` is the default route when a seat names no model. (No `tier`
field until the explore stage is validated.)

## Module layout (tach-clean)

- `workflows/_panel.py` — `Finding`/`ReviewVerdict`/`PanelResult`/`ReviewContext`,
  `aggregate_verdicts` (pure), `run_panel` (orchestration). tach: workflows.
- `agents/code_review.py` — add `structured_review(provider, ctx) -> ReviewVerdict`
  (grounding-first system prompt; emits a strict JSON verdict). Keep freeform
  `code_review()` for the human `agent6 review` text rendering.
- `cli/providers.py` — `_build_review_seat_providers(cfg) -> list[Provider]`.
- `config.py` — the flat fields above + parse/validate `review_seats` strings.
- `workflows/loop.py` — replace `_run_critic` with `run_panel` (Stage 2).

## Build stages (toward C; each a working increment, gated by evidence)

1. **Pure core + post-hoc.** `_panel.py` types + `aggregate_verdicts` (executable
   grounding) — TDD, no network. `structured_review` diff-seat. `run_panel`
   sequential. Config fields. `agent6 review --reviewers N`. Real-API smoke on
   real diffs. **Cannot touch the worker loop.**
2. **In-loop, ADVISORY only.** Wire panel into the triggers as advisory
   injection; persist the rejection counter in the snapshot. Real-task A/B
   (below). Only if it clears the false-block gate do we enable veto/quorum.
3. **Concurrency.** `review_concurrency > 1` (thread pool; ordered merge).
4. **Explore tier.** Read-only tool-using seats (explicit read/grep/outline/list
   allowlist; assert dag_*/curator handlers excluded), distinct models per seat.
   Only if Stage 2 proved the panel pays off. Completes "C".

## Validation (pre-registered; PRIMARY gate = false-block rate, panel default off)

The 0.0.4 removal is the null hypothesis. Before enabling any GATE:

- Held-out suite of real tasks: injected-bug fixes, a feature, a refactor, and —
  critically — **correct-work tasks that should NOT be blocked**.
- Paired A/B: same tasks at `panel ∈ {off, advisory-3, veto-3}`, fixed seed.
- **Primary metric (gate): false-block / stall rate on the correct-work tasks
  must stay at the panel-off baseline.** Secondary: real-bug catch rate, extra
  tokens/USD, wall time.
- Ship order: post-hoc `agent6 review` first; in-loop advisory next; the in-loop
  GATE (veto/quorum) only after it clears the primary gate. If it can't, we stop
  at advisory — and that is an acceptable, honest outcome.

## Residual risks tracked

- Grounding check precision (file_line matching to a real diff line; fuzzy
  citations). Unit tests cover citation-not-in-diff, wrong-category, delta
  boundary, dedup, distinct-model quorum, abstain accounting.
- Correlated same-model seats (mitigated: distinct-model quorum + advisory
  default + validation).
- Cost (mitigated: sub-budget, fraction guard, size dial, default off).
- explore-tier wall-clock (mitigated: per-seat `deadline_s` abstain + `max_iters`
  + the panel budget-fraction guard).
- ReDoS via the read-only `grep` tool a model-supplied regex can drive (the
  worker's existing `grep` shares this). Contained in `tools/dispatch`: a
  pattern-length cap, a static screen that rejects the nested-unbounded-quantifier
  shape (`(a+)+` …), and a wall-clock bound on the whole grep. Residual: a novel
  single-line catastrophic pattern that evades the screen is bounded only at the
  aggregate wall-clock level (CPython's `re` can't be interrupted mid-match).

## Validation log

### Stage 1 (post-hoc `agent6 review --reviewers 3`, kimi-k2.6, real API)

Four runs on planted diffs (3 seats: security/correctness/edge-cases):

| Diff | Result | Notes |
|------|--------|-------|
| Hardcoded API key + path traversal | 2 grounded `block`s (deduped from 3 seats) | both real bugs caught, cited at real lines |
| `token ==` compare + stray `.err` files in tree | 3 grounded `block`s | timing side-channel + 2 off-topic-edit (real junk in the diff) |
| Clean: constant-time compare, env secret, docstrings | **PASS under `veto`** | the one finding (TypeError edge) was `warn:other` → non-gating; `n_block=0` |

Conclusion: the grounded aggregator catches real, cited, block-eligible defects
and does NOT false-block correct work even with the gate (veto) on — the failure
mode that deleted the pre-0.0.4 reviewer. Cost ~$0.03–0.06/panel; 3 concurrent
seats ≈ 2 min wall (vs ~3× sequential). Gate proceeds to Stage 2 (in-loop).

## Config profiles (deferred to the end; design locked)

To tame the knob count, a profile is a NAMED bundle of overrides (spanning
[models], [budget], [workflow], and the review_* knobs) selected with
`--profile <name>` or `[workflow] profile = "<name>"`. Resolution:
`builtin defaults < profile < user config (global/repo) < CLI flags` -- a profile
is a smart default you can still override per-setting.

Built-in profiles:
- `quick`     -- review off, tight output budget (fast/cheap).
- `standard`  -- the pre-feature baseline (today's defaults). The default.
- `ultra`     -- the "ultracode" tier: a 3-seat grounded panel, before_finish
                 veto (the 3 seats share one model, so quorum>1 is unreachable;
                 veto gates on any surviving grounded block), more budget.
- `paranoid`  -- 5 explore-tier seats, before_finish veto, bigger output budget; max
                 budget.
- custom: unlimited user `[profiles.<name>]` tables (like custom [providers.*]).

Built AFTER the panel core (incl. explore tier) so a profile bundles stable
knobs. The review_* fields remain the low-level surface; profiles are the front.
NOT the pre-0.0.4 auto-triage profiles (those drove multi-workflow dispatch,
gone with the single loop); these are pure user-selected config presets.

### Stage 2 (in-loop, veto) — live full-cycle run

Task: "Create io_util.py with read_doc(name) that opens files/<name>." (a
path-traversal trap), run with critic=before_finish, 3 seats, decision=veto:

- iter 5: worker's naive `open("files/"+name)` -> all 3 seats BLOCK (traversal)
  -> finish revoked.
- iter 9: worker adds an abspath containment check -> correctness/tests pass but
  SECURITY still blocks: abspath doesn't resolve symlinks, so symlink traversal
  still bypasses it (a subtle second-order catch on the worker's OWN fix)
  -> finish revoked.
- iter 13: worker switches to os.path.realpath -> all seats PASS -> finish.

Final code is symlink-safe (realpath containment). The panel materially hardened
the output, caught the subtle bug in the first fix, and converged in 3 rounds
with no stall (the disarm never fired; grounding kept every block real). This is
the qualitative proof; the A/B quantifies catch-rate vs added iterations/cost.

### A/B validation (off vs veto, 3 seats, kimi, state OUTSIDE the repo)

| task        | off (baseline)        | veto (panel)            |
|-------------|-----------------------|-------------------------|
| V_traversal | vulnerable (4 it)     | SECURE via basename (9 it) |
| V_token     | vulnerable (4 it)     | SECURE via env+compare_digest (12 it) |
| C_math      | correct (4 it)        | correct (4 it, no churn) |
| C_string    | correct (4 it)        | correct (4 it, no churn) |

Result: 2/2 real vulnerabilities the baseline shipped were caught AND fixed under
veto; 0/2 false-blocks on clean work (veto matched off's iteration count exactly
-- no stall). The pre-0.0.4 false-block failure does NOT reproduce; the executable
grounding holds. Cost: a clean run adds ~one panel call (~$0.005-0.01); a
genuinely-buggy run costs the real iterations to fix it (2-3x).

IMPORTANT harness lesson (and a real edge): an EARLIER A/B run put AGENT6_STATE_HOME
INSIDE each repo, so agent6's own logs/transcripts got auto-committed and the panel
(correctly) blocked them as off-topic-edit -> churn. Fixed by keeping state out of
the repo (the default). General note: the panel reviews the whole committed diff,
so stray committed files trigger off-topic blocks -- a non-issue in the default
layout, worth a guard if a repo commits generated artifacts.

Conclusion: the DIFF-tier panel is validated as a genuine improvement for
single-file changes. The explore tier (broader repo reading) was NOT exercised by
these single-file tasks -- its value (cross-file impact) needs multi-file tasks to
justify, so it stays a documented option rather than a speculative build.

### Explore tier validation (cross-file break, diff-tier vs explore-tier)

Controlled experiment: `parser.parse(s)` -> `parse(s, base)` (added required arg);
`main.py` still calls `parse(input())` (one arg, now broken) but is NOT in the diff.

- diff-tier (sees only the diff): PASS -- it cannot know the caller is broken.
- explore-tier: investigated (read_file + find_references -> found main.py), then
  BLOCKED with verify-uncovered-correctness cited at parser.py:1 (the diff line,
  so it grounds + gates), detail naming the broken caller main.py:3.

First attempt the explore reviewer investigated but judged "diff correct in
isolation" (passed) -- fixed by an explicit prompt rule: a diff that breaks an
existing caller you find IS a defect of this diff, cite it at the DIFF line that
caused the break (never the other file's line -- only diff lines gate). Cost
~$0.008/seat. The explore tier adds real value over diff-tier for cross-file
impact, completing the "C" end state (independent reviewers that read the repo,
distinct models, concurrent, grounded judge).
