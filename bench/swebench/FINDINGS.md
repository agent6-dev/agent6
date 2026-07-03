# agent6 × SWE-bench Verified — findings & decisions

Durable record of the SWE-bench work on `dev-0.0.12`: what we measured, the
decisions it drove, and the reasoning — so the "why" survives the commit squash.
This is the design narrative; the commits are the mechanism.

## TL;DR

agent6 **resolves real SWE-bench Verified instances** at a SWE-agent-comparable
budget. Building the benchmark surfaced two real agent6 issues (both fixed) and
one efficiency finding (fix shipped, effect not yet rigorously measured).

## The harness (`bench/swebench/`)

- `run_sweep.py` — orchestrator. For each (model, instance): pull the official
  SWE-bench instance image, run agent6 inside it, take `git diff` as the
  prediction. **Source-only**: test-file diffs are stripped so the agent can
  never touch the gold grading tests.
- `in_container.sh` — installs agent6 from a locally-built wheel (`uv tool
  install`, Python 3.14), writes the run config, runs the agent on the problem.
- `score.py` — wraps the unmodified `swebench.harness.run_evaluation`
  (FAIL_TO_PASS + PASS_TO_PASS), so numbers are leaderboard-comparable.
- Sample: random-50 across all 500 (seed 20260623), `sample_50.json`.

**Decision: drive the agent inside SWE-bench's own Docker images, not agent6's
jail.** This benchmarks the agent's *capability*, not the sandbox (we have other
tests for the jail). Non-privileged `sudo docker`; the container is the isolation.

## Results (random-6 pilot, ~$1/instance budget)

| model | resolved | budget notes |
|---|--:|---|
| GLM-5.2 | 3/6 | hard-capped ~$1.2 (priced, USD cap enforced) |
| kimi-k2.6 | 2/6 | hard-capped ~$1.2 |
| sonnet-4-6 | 3/6 | ran UNCAPPED (see issue 1) — ~$2 avg, up to $6 |
| opus-4-8 | 0/6 | Anthropic credit exhausted mid-run, not a capability result |

Headline: **open models match sonnet at a fraction of the cost.** django
instances are the hard ones (none resolved). n=6 is noisy — directional only.

**Decision: calibrate budget to SWE-agent's ~$1/instance.** If a capable model
can't resolve on a comparable budget, that's an agent6 issue to fix — not a
reason to raise the budget. Current Anthropic list prices (opus $5/$25, sonnet
$3/$15) verified against the API docs.

## Issue 1 — USD-budget enforcement is a no-op for unpriced providers

`best_effort_usd_limit` converts to token ceilings via the worker's price
(`config.py:_apply_usd_budget_override`). Anthropic publishes **no pricing**, so
the conversion returns `None` and the limit silently does nothing — only the
(large) token ceilings bound spend. A SWE-bench run set a $1 limit on sonnet and
it ran to ~$6 on one instance, draining ~$11.80 of credit.

**Decision: warn, don't guess or kill.** (commit `c06f17e`) Run startup now warns
clearly when the USD cap can't be enforced (unpriced worker), naming the model
and pointing at the token ceilings. We deliberately do **not** guess a price (a
wrong guess could terminate a run mid-task — Eric: "I don't want to risk things
being terminated early") nor auto-convert. The `--max-usd` *flag* was already
guarded by `_explicit_usd_flag_error`; this closes the TOML-config path.

**For the benchmark specifically** (where we want an accurate $1): the harness
derives token caps from `$ × list price` directly (`in_container.sh`), since that
*can* be accurate when we know the price out-of-band.

## Issue 2 — turn (not token) inefficiency

agent6 took 29–77 turns to converge vs SWE-agent's ~15. Diagnosed from a real
transcript (glm/django-14155):

- `read_file` called 13× across only **3 distinct files** (same test file 8×,
  same source 4×); 21 explore turns : 3 edit turns.
- **84% prompt-cache hit** → re-reads are cheap on *tokens*; the waste is
  round-trip **turns**. The $6 sonnet blowup was many turns × a large
  accumulated context, not cache failure.

**Root cause (confirmed in source):** run mode's system prompt had **no**
anti-re-read guidance — that discipline existed only in PLAN mode's
`<be-decisive>`, which the worker never sees.

**Decision: four text-only prompt nudges** (commit `4e68f1a`), each grounded and
adversarially verified against the source by a review workflow:
1. run-mode `<be-decisive>`: cached context is authoritative; don't re-read
   content already above; post-edit/post-command re-reads still allowed.
2. `<budget-awareness>`: reframe a re-fetch as a full round-trip *turn*, not just
   token cost.
3. `read_file` description: outline-first + `offset`/`limit` for large files.
4. elision placeholder: stop actively inviting a same-args re-call after
   compaction.

**Rejected** (medium-risk code levers, by the adversarial pass): read-dedup
short-circuits and stale-mtime read caching (can feed stale content → corrupt
edits), loop-guard broadening (false positives on legitimate paging), per-turn
budget heartbeats / lowered nudge thresholds (can rush premature `finish_run`).
All four chosen levers are text-only — they cannot corrupt an edit or break a
call path, and current behavior is their floor.

**Validation: measured NULL → reverted.** A rigorous **18-instance/arm** A/B
(glm-5.2, old-prompt wheel vs new-prompt wheel, bounded $1) showed **no effect**:

| | resolved | mean turns (12 fresh instances) |
|---|--:|--:|
| old prompt | 7/18 | 49.0 |
| new prompt (nudges) | 7/18 | 49.6 |

Identical resolve rate, identical mean turns; per-instance turn swings were large
in both directions (django-12125 113→32 but django-14053 63→118) — pure noise,
no systematic shortening. The promising 6-instance signal (astropy-13579 50→9)
was one draw from that noise. The nudges are sound and harmless (zero resolve
regression) but deliver **no measured efficiency gain**, so they were **reverted**
(`git revert`, not history-rewrite — the squash drops them; revert-the-revert
stays available if a larger or multi-model measurement later shows an effect).

**The real finding: turn-efficiency is model-bound, not harness-bound.** opus
converges in 12–20 turns and resolves 4/6; glm thrashes to ~49 mean turns
(several instances 96–118 or `budget_exhausted`) and resolves ~39% — *regardless
of the prompt*. "agent6 takes more turns than SWE-agent" is really "a weaker model
thrashes; a strong one doesn't." **agent6 + a capable model is already efficient**;
no prompt nudge closes glm's gap to opus. A genuine harness-side efficiency win, if
one exists, lives deeper (loop/compaction/tooling) — not in run-mode prose.

## Fair bounded-$1 comparison (done)

Re-ran sonnet + opus at the bounded $1 token budget (new wheel) for a clean
comparison; the token-budget enforcement held (sonnet/django-15629 hit
`budget_exhausted` at 29 turns instead of the prior ~$6 runaway) and the unpriced
startup warning fired live. Ordering: **opus 4/6 > sonnet 3/6 ≈ glm 3/6 > kimi
2/6** (n=6, noisy). opus was the only model to crack a django instance
(django-14155). Open models hold their own with the frontier on cost.

## ⚠️ The big one: verify was broken in every run above

Every resolve rate above was achieved with `run_verify_command` **non-functional**
— the agent never ran a test. The benchmark wheel was built locally (`uv build`)
without `AGENT6_JAIL_TARGET=musl`, so the jail binary linked against this VM's
glibc 2.39 and could not exec in the glibc-2.35 containers (`GLIBC_2.39 not
found`). So **all the numbers above are lower bounds, achieved blind** — opus
solving django-14155 by pure reasoning is more impressive than it looked, and the
true verify-enabled rates are almost certainly higher.

Fixing it was a cascade, each layer masked by the prior one:
1. **glibc** — rebuild with `AGENT6_JAIL_TARGET=musl` (static binary; CI already
   does this, local builds must too).
2. **verify inference hardcoded `.venv/bin/python`** — real agent6 bug, broke
   verify in any container/system-python env. Fixed (`8f4b3d3`): fall back to
   `python3` on PATH when no `.venv`.
3. **jail PATH `/usr/bin:/bin`** excludes the conda interpreter — harness uses its
   absolute path.
4. **jail couldn't exec the interpreter in-container** — strict bind-mounts
   extra_read_paths at `/ro<src>` (real path denied); hardened denied child exec
   entirely in the SWE-bench image. → see sandbox matrix below.

## Sandbox usability matrix (the real deliverable)

The benchmark's real value was forcing agent6's sandbox to work in real container
setups. Validated empirically (`agent6 check sandbox` + a direct jail-exec probe):

| environment | effective auto | `check sandbox` | resolution |
|---|---|---|---|
| **unprivileged docker** | hardened | ❌ FAIL (etc-write escaped) | **unsandboxed** — new opt-in |
| **privileged docker** | strict | ✅ all probes pass | strict works |
| **podman rootless** | strict | ✅ all probes pass | strict works |

**New: careful unsandboxed opt-in.** `profile = "none"` runs the agent
unsandboxed with a loud startup warning. It is self-authorizing (an
operator-only, LLM-unreachable config value); the per-invocation forms are
`--dangerously-disable-sandbox` / `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`.
**`auto` never resolves to none on Linux** — opting out is always an explicit,
typed choice. The SWE-bench harness sets `profile = "none"`, and **verify works
end-to-end** (validated: 0 jail errors, real patches, the agent running tests).

**Minor sandbox findings (maintainer follow-ups):**
- hardened in unprivileged docker FAILED the `/etc`-write boundary probe (isolation
  gap, possibly root-related) — another reason unsandboxed is right there.
- podman rootless is NOT detected as a container (`in_container=False`) — the
  unsandboxed opt-in would need the env override there; strict (the desired podman
  profile) is unaffected.
- strict exposes extra_read_paths at `/ro<src>`, so a granted conda interpreter's
  real path is absent — use the `/ro`-prefixed path, or agent6 could expose the
  real path.
- non-fatal `/proc` remount EPERM warning under podman strict.

## Capability: it's model + verify, not scaffolding (3 nulls)

Three independent A/Bs all came back null/negative on resolve rate, which is the
strongest signal of the project:

| lever tested | result |
|---|---|
| run-mode prompt nudges (anti-re-read) | null (7/18 = 7/18) → reverted |
| Fugu distinct-model review panel (quorum gate) | null (3/6 = 3/6, 0 gate events) |
| structural priors (hot symbols / outline / co-change) | null/negative (ON 3/6 @ $1.22 vs OFF 4/6 @ $1.68) |

What DID move the needle: **working verify** (+2/18, recovered django wins) and
**model choice** (opus 4/6 / 12-20 turns vs glm ~39% / ~49 turns). Notably the
structural-prior A/B shows we are *ahead of aider* on repo context (ranked hot
symbols + tree-sitter outline + git co-change) yet it doesn't help SWE-bench — so
upfront context is not the bottleneck. New `prompt.structural_priors=false` gives
a leaner/cheaper prompt with no measured resolve cost.

**Conclusion for "harder problems":** spend on the model and on verify quality,
NOT on prompt/review/context scaffolding. Escalation (cheap worker, auto-bump to a
strong model on stall) is now built and its mechanism verified, but the sample
below couldn't measure its resolve value (see next section).

## Escalation: built, evaluated twice, REVERTED (no measured value)

`[models.escalation]` (opt-in role) swapped the worker provider to a stronger model
ONE-WAY once a single run STALLED (degenerate loop, or post-edit no-progress, gated on
`ever_edited`). Built in `d659cd2` + `7cf8f9e`, **reverted in `65351be`** after two
evals showed no value and a feasibility check killed the only variant worth having.

**Eval 1** (qwen3.6-27b -> glm-5.2, 6 instances, verify on, OpenRouter): all three
arms (cheap-alone / escalate / strong-alone) resolve the IDENTICAL set 3/6
{astropy-13579, pytest-7205, sympy-19346}. Escalation fired on 4/6 (glm calls: pytest
14, astropy-14365 5, sympy 5, astropy-13579 4) yet the escalate arm is byte-identical
to cheap-alone. No discriminating instance (none where qwen fails but glm succeeds), so
the lever had no room.

**Eval 2** (bigger gap: qwen3-coder-30b-a3b $0.07/M -> opus-4.7 $5/M, same 6): still no
valid discriminator. The lone candidate django-14155 (the only instance an earlier
opus-4.8 cracked) does NOT reproduce -- opus-4.7-ALONE also fails it, and so does the
escalate arm. weak-alone and escalate both score 2/6 but on DIFFERENT instances (weak
{astropy-13579, pytest}, escalate {pytest, sympy}); escalate even REGRESSED
astropy-13579 to an empty patch. That swap is the cheap worker's run-to-run
nondeterminism, not a strong-model conversion. **Net fail->resolve conversions across
both evals: zero.**

**Why reverted, not kept-off-by-default:**
- **No discriminator, and the gate blocks the one case that would matter.** The
  `ever_edited` gate suppresses escalation on exactly the never-edited / empty-patch
  failures (django-14155, django-15629) where a strong model would most plausibly help.
- **The genuinely useful variant -- long plans that de-escalate per subtask -- is not
  cleanly feasible in a single run.** No loop-visible subtask boundary exists: the task
  DAG (`add_task`/`set_cursor`) is optional, self-managed worker bookkeeping the loop
  never reads for control flow; `set_cursor` is unreliable (workers skip/late/never-
  clear it), so resetting on a cursor move would mis-reset mid-task and re-trigger the
  escalate->stall->escalate cycle the one-way latch avoids. Redesign = HIGH complexity
  across 5+ files incl. resume-state persistence.
- **That win already exists in machine mode.** Each state runs as its own subprocess
  with its own per-state `model`/`provider` (`machine_cmds.py` spawns `machine_agent`
  per state; `engine.py` selects per-state model). A long job modeled as a DAG/state
  machine gets per-subtask model tiering for FREE -- assign a cheap model to easy states
  and a strong one to the hard state. That IS "escalate the hard subtask, de-escalate
  the next," with a true process boundary and no mid-task mis-reset risk.

Per "if it doesn't show worth, don't carry the cruft": reverted (not rewritten, so the
attempt stays visible). Revive only if a discriminating-sample eval (cheap reliably
fails, strong reliably resolves) shows a real cost-per-resolve win -- and if so,
redesign around the `ever_edited` gap.

## Open / next

- **Re-run the benchmark WITH verify** (`profile = "none"`) for the TRUE rates --
  supersedes the blind numbers above. The user owns this round.
- **Per-subtask model tiering: use machine mode, not escalation.** Each agent state
  takes its own `model`/`provider`, which subsumes "strong model for the hard subtask,
  cheap for the rest" with a real process boundary. Escalation was reverted (`65351be`);
  revive only on a discriminating-sample cost-per-resolve win, redesigned around the
  `ever_edited` gap.
- Deeper turn-efficiency (if pursued): loop/compaction/tooling, replicated
  resolve-rate -- prompt nudges were a measured null.
