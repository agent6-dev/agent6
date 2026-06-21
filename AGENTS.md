# AGENTS.md: instructions for coding agents working on this repo

This file is read by coding agents (including agent6 itself) operating in
this repository. It is the authoritative map of project conventions and the
list of invariants that PRs are not allowed to weaken.

## Stability policy (pre-1.0)

agent6 is pre-1.0; the version in `src/agent6/__init__.py` is the single
source of truth. Until `1.0.0`, treat every public shape (config TOML, IPC
messages, on-disk graph format, CLI flags, transcript layout) as liquid.
Prefer breaking a shape cleanly over carrying it:

- No backward-compat shims, deprecation warnings, or aliased field names.
- No migration code or `config_version` translators. Bump `config_version`
  only when the new shape genuinely improves the user's error message.
- If something is wrong, rip it out. If you are tempted to write a
  compatibility branch, delete the old shape instead.

## Design principles

These are the standing rationales behind the small decisions. State them
here once; do not re-justify them per command in code comments or docs.

- **One obvious way** (Zen of Python). Prefer a single, well-named
  command over near-duplicate aliases. We have `agent6 connect`, not also
  `agent6 auth login`.
- **Explicit is better than implicit** (Zen). Defaults are real, readable
  values (`agent6 config show` prints every one and its origin); no
  behavior keyed off hidden state; errors never pass silently (see
  Errors below).
- **Least surprise.** A command does the boring, expected thing. Config
  writes default to the global config, with `--repo` (and `--machine
  FILE` where relevant) to redirect; the same target selection
  everywhere. Set-valued config merges last-overlay-wins like every
  other field.
- **Consistency** (special cases aren't special enough to break the
  rules). New subcommands mirror the shape of existing ones (positional
  core args, `--repo`/`--machine` target flags, argcomplete on
  fixed-choice values).
- **Simplicity** (simple is better than complex). No speculative
  abstraction, no plugin layers, no indirection that exists for a future
  that has not arrived. A reviewer should be able to read a module top to
  bottom in one sitting; if the implementation is hard to explain, it's a
  bad idea. If a helper has one caller, inline it; if a class has no
  state, make it a function.
- **Right-shaped data.** Get the data structure correct first and the
  code around it stays small. A field that can never be half-set belongs
  in one frozen type, not two parallel dicts. When code keeps converting
  between shapes, fix the shape instead of adding a converter.
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
- **Module boundaries** are enforced by [tach](https://docs.gauge.sh/):
  `cli -> workflows -> agents -> tools -> sandbox`. Workflows never import
  each other; agents never import workflows or the CLI. Needing to reach
  across a boundary usually means the design is wrong.

## Verify command

Also what `verify_command` should be in this repo's agent6 config when
running the agent on this repo:

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
- `src/agent6/git_ops.py` refuses `push`, `--force`, history rewrite,
  `reset --hard`, and `branch -D` unconditionally. Do not add overrides.
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

agent6 reviews its own source via `agent6 review`. Reviews live under
`.agent6-self-review/` (gitignored). When working on a module, read the
corresponding review file if present; it records real findings and which
were acted on.
