# AGENTS.md: instructions for coding agents working on this repo

This file is read by coding agents (including agent6 itself) operating in this
repository. It keeps two concerns distinct: how we develop agent6, and what
agent6 offers. Principles live here; detail lives in `docs/`.

## Hard rules (a PR never weakens these)

The load-bearing invariants, collected; each is detailed below or in `docs/`.

- No `git push`, `--force`, history rewrite, or `reset --hard`. `git_ops.py`
  refuses them unconditionally; don't add overrides. `branch -D` is refused for
  the model (the `_git_guard` run_command boundary is absolute), with ONE
  operator-only exception: `runs prune --delete-squashed` force-deletes a run
  branch the manifest confirms was squash-merged into its base (content-safe,
  the commit survives in the reflog). That path never touches LLM output.
- Every child process whose argv depends on LLM output goes through
  `run_in_jail`; never a direct `subprocess` of model output
  (audit: `rg 'subprocess\.(run|Popen)' src/agent6/`).
- Adding a tool (`tools/schema.py`), loosening a security default, or dialling a
  host not derived from a provider `base_url` each require a `Security review
  note:` in the commit message.
- Secrets stay `0600`, never printed by `config show`, never written to
  transcripts, never mounted into the jail.
- Keep the suite green (`ruff check` + `ruff format --check` + `pyright` +
  `tach check` + `pytest`); don't skip the verify command. `tach check` stays
  clean by updating `tach.toml` to mirror the design, never by contorting code
  to fit the old graph (see Architecture).
- Rip out wrong shapes: no backward-compat shims or migrations. No
  `Co-Authored-By` lines.

## How we develop agent6

### Design principles

The standing rationales behind small decisions. State them here once; don't
re-justify them per command in code or docs.

We follow the **Zen of Python** (`python -c 'import this'`): one obvious way,
explicit over implicit, simple over complex, special cases aren't special
enough to break the rules, errors never pass silently. The agent6 concretions,
and the principles the Zen doesn't cover:

- **Ask, don't over-decide.** These rules are guardrails against reflexive
  mistakes, not licence to make judgement calls alone. When a task forks (a
  behaviour tradeoff, a maybe-not-worth-it edge case, growing scope, more than
  one reasonable design, a new dependency), ask the operator; a one-line
  question is cheaper than shipping the wrong or over-built thing. Default to
  the simplest fix for the actual request; name the edges you skip instead of
  chasing every one a review surfaces. The inverse holds: when these
  principles already decide, act; don't ask permission to follow them or
  offer options that violate them.
- **Evidence over churn.** When measurement shows something is better, adopt
  it and delete the old shape; no backward-compat shims, deprecation aliases,
  or migrations until `1.0.0` brings semantic versioning and real migrations.
  A change whose value is a claim about model behaviour, prompts, or
  performance ships only with a measured A/B (replicates, variance) against a
  demonstrated baseline failure; a null result is reported, not shipped.
  Unmeasured tuning is superstition.
- **One obvious way.** One well-named command, not near-duplicate aliases:
  `agent6 connect`, not also `agent6 auth login`. One knob per behaviour:
  never a second config surface controlling what an existing one already
  controls.
- **Explicit.** Defaults are real, readable values `agent6 config show` prints
  with their origin; no behaviour keyed off hidden state; errors never pass
  silently (see Errors).
- **Least surprise.** A command does the boring, expected thing. Config writes
  default to the global config, `--repo` (and `--machine-file FILE`) to
  redirect. The same target selection everywhere; set-valued config merges
  last-overlay-wins.
- **Consistency.** New subcommands mirror existing ones: positional core args,
  `--repo`/`--machine-file` target flags, argcomplete on fixed-choice values.
- **Simplicity.** Less is more: less code beats more, and no speculative
  abstraction or indirection for a future that hasn't arrived. A reviewer
  should read a module top to bottom in one sitting; inline a one-caller
  helper, make a stateless class a function, and if it's hard to explain it's
  a bad idea.
- **Fix the root cause, never the symptom.** No hacks, workarounds, blind
  retries, or special cases that hide the real defect. Prefer rethinking and
  deleting over adding: removing a wrong shape beats guarding against it. A
  problem the operator keeps hitting has a systematic cause; correlate every
  occurrence before concluding "transient". When you cannot find the root
  cause, say so rather than paper over it.
- **Right-shaped data.** Get the data structure right and the code around it
  stays small. A field that can never be half-set belongs in one frozen type,
  not two parallel dicts; when code keeps converting between shapes, fix the
  shape.
- **Decompose proactively.** When a module grows past ~600 lines or a method
  past a few hundred, split it before it ossifies (exemplars:
  `workflows/loop.py` with its `_prompt_blocks` / `_metric` / `_compaction`
  siblings, the `ui/cli` split of `run.py`). The rules:
  - Lift cohesive pure-helper groups into sibling `_name.py` modules. Move
    verbatim; give moved symbols public names and import them back aliased
    (`foo as _foo`) so call sites don't change. A mechanical move with the
    suite green is the proof there's no regression.
  - Give a large stateful method's cross-iteration bookkeeping ONE mutable
    state dataclass so each phase becomes a method taking `state`; never a
    9-parameter helper or a multi-value tuple return.
  - An extraction that shifts a module boundary adds the edge to `tach.toml`.
    pyright allows importing `_name` only from a `_`-prefixed module; a symbol
    shared across a non-private boundary goes public.
  - One module decomposed per commit.
- **Secure by default.** Every new knob ships with the safe value as its
  default and stays visible through `agent6 config show`. Widening a security
  boundary is opt-in and carries a security review note. The operator can
  loosen everything; the agent can never loosen its own sandbox.

### Architecture

- **Layering** is `ui -> app -> workflows -> tools -> sandbox`;
  workflows never import each other, and the engine (`app` and below) never
  imports the UI. `app/` holds the application pipelines that compose the engine
  but are not a front-end -- the run/resume/fork/machine-agent lifecycles and
  the `--parallel` fan-out -- taking the presentation, process-spawn, and
  run-dir bridge callables the front-end injects (`RunFrontend`, `LaneRuntime`),
  and printing only through the injected two-channel `Reporter`.
  `ui/` is the presentation layer and the composition root: the three
  front-ends (`ui/cli`, `ui/tui`, `ui/web`) plus `ui/spawn.py` and
  `ui/notify.py`, over the shared headless read-model fold
  (`viewmodel`). `ui/cli` is the entry point that wires a run.
  [tach](https://docs.gauge.sh/) (`tach.toml`) checks it.
- **`tach.toml` mirrors the design.** Write the right design, then update
  `tach.toml` to match; never contort code (or add an indirection) to satisfy
  tach or strict pyright. After a change, audit the boundaries it produced; if
  they look complex, redesign rather than accept the complexity.

### Validation and reporting

Structural validation (a green suite) is not perceptual validation; the
operator dogfoods daily and feels what tests can't.

- Judge UX by rendering and reading the real output (pty capture, screenshot,
  live run) against the product bar; never declare polish from code
  inspection.
- Report exactly what was and wasn't exercised. Never claim "fixed" or
  "validated" beyond what you observed end-to-end; if tests fail, say so with
  the output.
- Don't flag-and-skip. Surface pre-existing breakage early as a decision, not
  in a final summary as "out of scope". Fix clear bounded breakage properly;
  for a large risky restructure, propose a concrete shape instead.

### Writing style

Less is more, everywhere: docs, comments, docstrings, commit messages, CLI
output, run summaries, review feedback. Terse and to the point: the shortest
version that still carries the point wins.

- Lead with the point. State the fact or instruction first; add rationale
  only when a reader could not reconstruct it.
- Cut noise. If a sentence still works without a word, drop the word; drop
  sentences that restate the one before. Walls of text don't get read, they
  bury the lead.
- Plain punctuation: commas, colons, parentheses, periods. An em dash flags an
  overstuffed sentence to recast, not punctuation to swap.
- Concrete over abstract: name the command, the field, the number. Write
  "retries twice, then fails the run", not "robustly handles failures".
- One idea per sentence, one topic per paragraph. Prefer short bullets to
  prose when listing facts.
- Comments state what the code cannot: a constraint, an invariant, a measured
  number, a link to a decision. Never narrate the next line.
- Commit messages: imperative subject; a body only for a non-obvious why, in
  point form.
- Keep documents flat: a heading plus short paragraphs or bullets. Bold is for
  lead-in labels and load-bearing caveats. Skip intensifiers and marketing
  adjectives.

### Project conventions

- **Language**: Python 3.12+. Every `.py` file starts with
  `from __future__ import annotations`. Strict pyright.
- **Layout**: src layout under `src/agent6/`. Tests under `tests/`. Rust crate
  for the sandbox launcher under `src/agent6/jail/`.
- **Style**: ruff is the only formatter and linter. Line length 100. Run
  `uv run ruff check` and `uv run ruff format` before committing.
- **Typing**: pydantic v2 only at trust boundaries (config, LLM I/O, tool
  schemas, IPC). For internal value types use
  `@dataclass(frozen=True, slots=True)`. Do not mix pydantic into hot
  paths.
- **Imports**: absolute only (`from agent6.x import y`).
- **Errors**: fail loudly. No bare `except:`, no swallowed errors. Custom
  exception classes per subsystem.
- **Versioning**: `__version__` in `src/agent6/__init__.py` is the single
  source of truth; never hardcode the version anywhere else.
- **No new runtime dependencies** without explicit discussion. Current list:
  `pydantic`, `httpx2`, `argcomplete`, the `tree-sitter` pair backing the
  symbol-navigation tools, `textual` (live dashboard), and `ruff` + `ty`
  (validate scripts that `machine create` generates). Build dep is
  `hatchling`; `pyright` stays dev-only.
- **Touch only what the task needs.** Do not add comments or annotations to
  code you did not change, and do not refactor surrounding code in
  passing. Scope creep is a review blocker.
- **Keep docs in sync.** A change affecting the architecture, config, security
  model, or state machines updates the matching file (`docs/architecture.md`,
  `docs/config.md`, `docs/security.md`, `docs/state-machines.md`, `README.md`,
  this file).

### Git and commit practices

- Commit messages follow
  [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(scope):`, `fix(scope):`, `ci:`, `docs:`, `bench:`. The scope matches a
  directory under `src/agent6/` or a top-level area.
- One concern per commit; individual commits are worth keeping. Squash only
  iterative churn: fix-ups (bug/regression fixes) to unpushed work.
- Never push; the operator signs and pushes from another machine. For the same
  reason, never reference commit hashes (signing changes them) or branch names
  (transient) in messages or docs.
- Never rewrite pushed history. Rewrite unpushed commits only when asked, and
  never force-push.
- Stage named files only, never `git add -A`; never commit scratch notes,
  session artifacts, or generated output.
- A release squashes only that churn, keeps clean commits as-is, and verifies
  zero diff after; master only advances (fast-forward, never rewritten). A
  squashed body preserves the decisions and what was tried and rejected;
  durable design reasoning goes to docs.

### Verify command

The repo's `verify_command`; agent6 infers it from this fenced block when
none is configured (a pipeline is wrapped as `sh -c`):

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

All five must pass.

### Self-review

agent6 reviews its own source via `agent6 review`. Reviews live under the
per-repo state directory (`$XDG_STATE_HOME/agent6/<repo-id>/reviews/`), never
in the repo. When working on a module, read its review there if present; it
records real findings and which were acted on.

## What agent6 offers

### Product bar

- **Surfaces tell the truth.** A failed run never renders as "done", a dead
  pane never looks busy, errors keep their reason. Hiding failure is a bug
  wherever it appears.
- **Every surface is at least as polished as the leading agentic coding
  tools.** Parity is the floor, not the goal.
- **The web UI is at least as polished as the TUI.** Web tooling is better, so
  a rougher web UI is backwards.

### Security invariants (do not weaken)

The threat model, defense layers, and rationale live in `docs/security.md`;
these are the invariants a change must preserve.

- The LLM tool surface is the fixed set in `src/agent6/tools/schema.py`, plus
  tools from operator-configured MCP servers when `[mcp].enabled` is set
  (default off). Adding a tool requires a security review note explaining the
  threat model.
- All child processes whose argv depends on LLM output go through
  `agent6.sandbox.jail.run_in_jail`. Modules that shell out with fixed argv
  depending only on operator input may call `subprocess` directly; the
  per-module allowlist with each module's rationale lives in
  `docs/security.md` and is pinned by
  `tests/security/test_subprocess_allowlist.py`. Audit:
  `rg 'subprocess\.(run|Popen)' src/agent6/`.
- Config is secure by default: every field has a default, and
  security-sensitive fields default to the safe value
  (`sandbox.agent_network = "providers"`, `sandbox.tool_network = "block"`,
  `sandbox.run_commands = "ask"`, `sandbox.protect_git = true`,
  `git.allow_* = false`). Every leaf is auditable via `agent6 config show`;
  `Config` stays `extra="forbid", frozen=True`. Loosening a security default
  gets the same scrutiny as adding a tool.
- Secrets (provider API keys) live in `$XDG_CONFIG_HOME/agent6/secrets.toml`,
  enforced `0600` and owner-only. They are never written to transcripts, never
  printed by `config show`, never mounted into the jail. `agent6 connect`
  never executes anything a remote returns (OAuth/paste only).
- Running as root requires explicit opt-in (`--allow-root` /
  `AGENT6_ALLOW_ROOT=1`); the jail, not the uid, is the boundary.
- Agent egress is bounded to the hosts derived from configured
  `[providers.*]` `base_url`s, unioned with operator-set `sandbox.allow_urls`
  (default empty); see `app/egress.py`. Never add a code path that dials a
  host not derived from them. Operator-initiated CLI fetches before any agent
  runs (`agent6 connect` OAuth, `agent6 skills install <url>`) are the
  operator dialling a host they typed, outside that boundary.
- The `agent6-jail` Rust binary is part of the security boundary. Changes to
  `src/agent6/jail/src/main.rs` need at minimum a review note covering: mount
  points changed, Landlock rules changed, seccomp syscalls added or removed,
  and `/dev` nodes exposed.
