# AGENTS.md: instructions for coding agents working on this repo

This file is read by coding agents (including agent6 itself) operating in
this repository. It is the authoritative map of project conventions and the
list of invariants that PRs are not allowed to weaken.

## Hard rules (a PR never weakens these)

The load-bearing invariants, collected; each is detailed in its section below.

- No `git push`, `--force`, history rewrite, `reset --hard`, or `branch -D`.
  `git_ops.py` refuses them unconditionally; don't add overrides.
- Every child process whose argv depends on LLM output goes through
  `run_in_jail`; never a direct `subprocess` of model output
  (audit: `rg 'subprocess\.(run|Popen)' src/agent6/`).
- Adding a tool (`tools/schema.py`), loosening a security default, or dialling a
  host not derived from a provider `base_url` each require a `Security review
  note:` in the commit message.
- Secrets stay `0600`, never printed by `config show`, never written to
  transcripts, never mounted into the jail.
- Keep the suite green (`ruff check` + `ruff format --check` + `pyright` +
  `tach check` + `pytest`); don't skip the verify command.
- Rip out wrong shapes: no backward-compat shims or migrations. No
  `Co-Authored-By` lines.

## Architecture and changes

- **Layering** is `cli -> workflows -> agents -> tools -> sandbox`; workflows
  never import each other, and agents never import workflows or the CLI.
  [tach](https://docs.gauge.sh/) (`tach.toml`) checks it.
- **`tach.toml` mirrors the design.** Write the right design, then update
  `tach.toml` to match; never contort code (or add an indirection) to satisfy
  tach or strict pyright. After a change, audit the boundaries it produced and ask
  whether they still look right; if they look complex, redesign rather than accept
  the complexity.
- **Changes are decided by evidence and tests, not churn.** When measurement
  shows something is better, adopt it and delete the old shape; no backward-compat
  shims, deprecation aliases, or migrations standing as artificial roadblocks.
  (`1.0.0` will add semantic versioning with stable args/config/interfaces and
  migrations; until then there is none of that ceremony. The version in
  `src/agent6/__init__.py` is the single source of truth.)
- **Keep docs in sync.** When a change affects the architecture, config, security
  model, or state machines, update the matching file (`ARCHITECTURE.md`,
  `CONFIG.md`, `SECURITY.md`, `STATE_MACHINES.md`, `README.md`, this file) so the
  docs never drift from the code.

## Design principles

The standing rationales behind small decisions. State them here once; don't
re-justify them per command in code or docs.

We follow the **Zen of Python** (`python -c 'import this'`): one obvious way,
explicit over implicit, simple over complex, special cases aren't special enough
to break the rules, errors never pass silently. The agent6 concretions, and the
principles the Zen doesn't cover:

- **One obvious way.** One well-named command, not near-duplicate aliases:
  `agent6 connect`, not also `agent6 auth login`.
- **Explicit.** Defaults are real, readable values `agent6 config show` prints
  with their origin; no behavior keyed off hidden state; errors never pass
  silently (see Errors).
- **Least surprise.** A command does the boring, expected thing. Config writes
  default to the global config, `--repo` (and `--machine FILE`) to redirect. The
  same target selection everywhere; set-valued config merges last-overlay-wins.
- **Consistency.** New subcommands mirror existing ones: positional core args,
  `--repo`/`--machine` target flags, argcomplete on fixed-choice values.
- **Simplicity.** No speculative abstraction or indirection for a future that
  hasn't arrived. A reviewer should read a module top to bottom in one sitting;
  inline a one-caller helper, make a stateless class a function, and if it's hard
  to explain it's a bad idea.
- **Right-shaped data.** Get the data structure right and the code around it
  stays small. A field that can never be half-set belongs in one frozen type, not
  two parallel dicts; when code keeps converting between shapes, fix the shape.
- **Decompose proactively; don't let debt accumulate.** When a module
  grows past ~1000 lines or a method past a few hundred, split it before
  it ossifies, rather than threading one more local or piling on another
  branch. The patterns we use (see `workflows/loop.py` + its `_prompts` /
  `_metric` / `_compaction` / `_critic` / `_symbol_outline` siblings, the
  `cli/_steer|_ask|_repl` split of `run.py`, the `tools/_edit_diag` /
  `_agent6_docs` / `_result_format` split of `dispatch.py`, and
  `machine/model.py` -> `_semantics.py`):
  - Lift cohesive pure-helper / constant groups into sibling `_name.py`
    modules. Move verbatim; give the moved symbols public names (drop the
    leading underscore) and import them back aliased (`foo as _foo`) so call
    sites and behaviour don't change. A pure mechanical move with the test
    suite green is the proof there's no regression.
  - For a large stateful method (the agent loop), give its cross-iteration
    bookkeeping ONE mutable state dataclass (`workflows.loop._LoopState`) so
    each phase becomes a method taking `state`. Never a 9-parameter helper
    or a multi-value tuple return — that is the spaghetti we are avoiding.
  - When an extraction shifts a module boundary, add the new edge to `tach.toml`
    (see Architecture and changes).
  - pyright `reportPrivateUsage` allows importing `_name` only from a
    `_`-prefixed module; when a symbol must be shared across a non-private
    module boundary, make it public.
  - One source module decomposed is one commit. A different module
    decomposed is a separate commit.
- **Secure by default.** Every new knob ships with the safe value as its
  default and stays visible through `agent6 config show`. Widening a
  security boundary is opt-in and carries a security review note in the
  commit message.

## Writing style

Applies to everything: docs, comments, docstrings, commit messages, CLI
output, run summaries, and review feedback.

- Lead with the point. State the fact or instruction first; add rationale
  only when a reader could not reconstruct it.
- Cut filler. If a sentence still works without a word, drop the word.
  Drop sentences that summarize the sentence before them.
- Plain punctuation: commas, colons, parentheses, periods. No em dashes.
- Concrete over abstract: name the command, the field, the number. Write
  "retries twice, then fails the run", not "robustly handles failures".
- One idea per sentence, one topic per paragraph. Prefer short bullets to
  prose when listing facts.
- Comments state what the code cannot: a constraint, an invariant, a
  measured number, a link to a decision. Never narrate the next line.
- Keep documents flat: a heading plus short paragraphs or bullets. Bold
  is for lead-in labels and load-bearing caveats, not mid-sentence
  emphasis of ordinary words. Skip intensifiers and marketing
  adjectives.

## Project conventions

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
- **No new runtime dependencies** without explicit discussion. Current list:
  `pydantic`, `httpx`, `argcomplete`, the `tree-sitter` pair backing the
  symbol-navigation tools, `textual` (live dashboard), and `ruff` + `ty`
  (validate scripts that `machine create` generates). Build dep is
  `hatchling`; `pyright` stays dev-only.
- **Touch only what the task needs.** Do not add comments or annotations to
  code you did not change, and do not refactor surrounding code in
  passing. Scope creep is a review blocker.

## Verify command

Also what `verify_command` should be in this repo's agent6 config when
running the agent on this repo (agent6 parses this fenced block to infer the
verify command when none is configured — a shell pipeline is wrapped as
`sh -c`):

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

All five must pass; keep the suite green.

## Security invariants (do not weaken)

- The tool surface given to the LLM is the fixed set declared in
  `src/agent6/tools/schema.py`. Adding a tool requires a security review
  note in the commit message explaining the threat model.
- All child processes whose argv depends on LLM output go through
  `agent6.sandbox.jail.run_in_jail`. No direct `subprocess.run` of
  LLM-provided commands anywhere. Modules that shell out with fixed
  argv depending only on operator input may call `subprocess.run`
  / `subprocess.Popen` directly: `git_ops.py`, `detect.py`,
  `graph/curator.py`, `graph/client.py`, `sandbox/jail.py` (the launcher
  itself), `providers/token_command.py` (the operator-configured
  `[providers.*].token_command` that mints a provider bearer; argv comes
  from config, never from LLM output), and a small set of `cli/` helpers (TUI spawn, `$EDITOR` for
  plan editing, `git diff/log` for the review subcommand, `rg` for history
  search, `cli/scriptcheck.py` running ruff/ty with fixed argv to
  statically read generated scripts, which only ever execute via
  `run_in_jail`, and the `machine run` supervisor that spawns each agent
  state as a fixed-argv `python -m agent6.cli.machine_agent` subprocess
  whose request travels in a temp file, never on argv). Audit with
  `rg 'subprocess\.(run|Popen)' src/agent6/`.
- Config is secure by default: every field has a default, and
  security-sensitive fields default to the safe value
  (`sandbox.agent_network = "providers"`, `sandbox.tool_network = "block"`,
  `sandbox.run_commands = "ask"`, `sandbox.protect_git = true`,
  `git.allow_* = false`). Config is layered (built-in defaults < global
  `$XDG_CONFIG_HOME/agent6/config.toml` < per-repo config under
  `$XDG_STATE_HOME/agent6/<repo-id>/config.toml` < `--config FILE`) and
  every leaf is auditable via `agent6 config show`.
  `Config` stays `extra="forbid", frozen=True`; loosening a security
  default requires the same scrutiny as adding a tool.
- Secrets (provider API keys) live in
  `$XDG_CONFIG_HOME/agent6/secrets.toml`, enforced `0600` and owner-only
  (see `secrets.py`). They are never written to transcripts, never printed
  by `config show`, and never mounted into the jail. `agent6 connect` must
  never execute anything a remote returns (OAuth/paste only).
- Running as root requires explicit opt-in (`--allow-root` /
  `AGENT6_ALLOW_ROOT=1`); under sudo, agent6 reads the real user's
  config/secrets and chowns new per-repo state-dir files back to them. It
  does not drop privileges in-process; the jail is the boundary.
- Configured `[providers.*]` endpoints are the only network destinations
  the agent may talk to. The egress allow-list is derived uniformly from each
  provider's effective `base_url` host (every `api_format` and `deployment`
  carries the dialled host there; the deployment profile only appends
  path/model) — see `cli/egress.py:_provider_endpoints`. `base_url` is
  operator-controlled config for both api_formats, so an operator chooses the
  destinations and the credential sent there; do not add a code path that dials
  a host not derived from a provider's `base_url`.
- The `agent6-jail` Rust binary is part of the security boundary. Changes
  to `src/agent6/jail/src/main.rs` need at minimum a review note covering:
  what mount points changed, what Landlock rules changed, what seccomp
  syscalls were added or removed, and what `/dev` nodes are exposed.

## Workflow expectations

- One PR per concern. Split bench, docs, and security fixes into separate
  commits even on the same branch.
- Commit messages follow
  [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(scope):`, `fix(scope):`, `ci:`, `docs:`, `bench:`. The scope
  matches a directory under `src/agent6/` or a top-level area.
- Imperative subject; a body only for non-obvious why, in point form. No
  `Co-Authored-By` lines.
- Security-relevant commits include a `Security review note:` paragraph
  explaining what surface changed.
- Do not push. `git.allow_push = false` is the user-side enforcement; the
  social rule is the same.

## Self-review

agent6 reviews its own source via `agent6 review`. Reviews are written under the
per-repo state directory (`$XDG_STATE_HOME/agent6/<repo-id>/reviews/`), outside the
checkout, never in the repo. When working on a module, read its review there if
present; it records real findings and which were acted on.
