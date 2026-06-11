# AGENTS.md — instructions for coding agents working on this repo

This file is read by coding agents (including agent6 itself) operating in
this repository. It is the authoritative map of project conventions and the
list of invariants that PRs are not allowed to weaken.

## Stability policy (pre-1.0)

agent6 is pre-1.0; the version in
`src/agent6/__init__.py` is the single source of truth. Until `1.0.0`,
treat every public shape — config TOML, IPC messages, on-disk graph
format, CLI flags, transcript layout — as **liquid**. Prefer breaking
a shape cleanly over carrying it:

- No backward-compat shims, deprecation warnings, or aliased field names.
- No migration code or `config_version` translators. Bump
  `config_version` only when the new shape genuinely improves the user's
  error message, never as a substitute for "just make it the right shape".
- No "low-churn" framing in design discussions. If something is wrong,
  rip it out.

Once we ship 1.0 the rules change. Until then: if you're tempted to
write a compatibility branch, delete the old shape instead.

## Design principles

These are the standing rationales behind the small decisions. State them
here once; do not re-justify them per-command in code comments or docs.

- **One obvious way (Zen of Python).** Prefer a single, well-named command
  over near-duplicate aliases. We have `agent6 connect`, not also
  `agent6 auth login` — agent6 stores an API key, it does not run an auth
  flow or a session, so there is nothing to "log in"/"log out" of.
- **Principle of least surprise.** A command does the boring, expected
  thing. Config writes default to the **global** config with `--repo`
  (and, where relevant, `--machine FILE`) to redirect — the same target
  selection everywhere (`connect`, `model`, `config set/...`). Set-valued
  config merges **last-overlay-wins**, like every other list: the most
  specific layer that sets it wins, so a repo can tighten a global value
  without a surprising union.
- **Consistency.** New subcommands mirror the shape of existing ones
  (positional core args, `--repo`/`--machine` target flags, argcomplete on
  fixed-choice values) rather than inventing per-command conventions.
- **Secure by default.** Every new knob ships with the safe value as its
  default (egress closed, network confined), and stays auditable through
  `agent6 config show`. Widening a security boundary is opt-in and carries
  a security-review note in the commit message.

## Project conventions

- **Language**: Python 3.12+. Every `.py` file starts with
  `from __future__ import annotations`. Strict pyright.
- **Layout**: src layout under `src/agent6/`. Tests under `tests/`.
  Rust crate for the sandbox launcher under `src/agent6/jail/`.
- **Style**: ruff is the only formatter and linter. Line length 100. Run
  `uv run ruff check` and `uv run ruff format` before committing.
- **Typing**: pydantic v2 ONLY at trust boundaries (config, LLM I/O, tool
  schemas, IPC). For internal value types use
  `@dataclass(frozen=True, slots=True)`. Do not mix pydantic into hot paths.
- **Imports**: absolute imports only (`from agent6.x import y`). No
  relative imports.
- **Errors**: fail loudly. No bare `except:`. No swallowed errors. Custom
  exception classes for each subsystem.
- **No new runtime dependencies** without explicit discussion. The
  `pyproject.toml` dep list is intentionally small: `pydantic`, `httpx`,
  `argcomplete`, the `tree-sitter` pair (`tree-sitter` +
  `tree-sitter-language-pack`) that backs the symbol-navigation tools, and
  `textual` (the live dashboard, shipped by default). Build dep is `hatchling`.
- **Comments / docstrings**: don't add them to code you didn't change.
  When you do comment, one line on the non-obvious — never restate the
  code. Don't add type annotations to functions you didn't modify. Don't refactor
  surrounding code "while you're there." Scope creep is a review-blocker.
- **Module boundaries** are enforced by [tach](https://docs.gauge.sh/).
  `cli → workflows → agents → tools → sandbox`. Workflows never import
  each other; agents never import workflows or the CLI. If you need to
  reach across a boundary you are usually doing the wrong thing.

## Verify command

This is also what `verify_command` should be set to in this repo's agent6
config (`.agent6/config.toml`, or a `--config` file) when running the
agent on this repo:

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

All five must pass; keep the suite green.

## Security invariants (do not weaken)

- The tool surface given to the LLM is the fixed, audited set declared in
  `src/agent6/tools/schema.py`. Adding a tool requires a security review
  note in the commit message explaining the threat model.
- All child processes whose argv depends on LLM output go through
  `agent6.sandbox.jail.run_in_jail`. No direct `subprocess.run` of
  LLM-provided commands anywhere. Modules that shell out with fixed,
  audited argv depending only on operator input are allowed to call
  `subprocess.run` / `subprocess.Popen` directly: `git_ops.py`,
  `detect.py`, `graph/curator.py`, `graph/client.py`, `sandbox/jail.py`
  (the launcher itself), and a small set of `cli/` helpers (TUI spawn,
  `$EDITOR` for plan editing, `git diff/log` for the review subcommand,
  `rg` for history search, the `cli/scriptcheck.py` validator that runs
  `ruff`/`ty` with fixed argv to STATICALLY read — never execute — the scripts
  `machine create` generates (those scripts only ever EXECUTE via `run_in_jail`),
  and the `machine run` supervisor that spawns each agent state as a fixed-argv
  `python -m agent6.cli.machine_agent` subprocess — its request, including the
  prompt, travels in a temp file, never on argv). Audit with
  `rg 'subprocess\.(run|Popen)' src/agent6/`.
- `src/agent6/git_ops.py` refuses `push`, `--force`, history rewrite,
  `reset --hard`, and `branch -D` unconditionally. Do not add overrides.
- Config is **secure by default**: every field has a default, and
  security-sensitive fields default to the *safe* value (`sandbox.agent_network
  = "providers"`, `sandbox.tool_network = "block"`, `sandbox.run_commands =
  "ask"`, `sandbox.protect_* = true`,
  `git.allow_* = false`). Config is layered (built-in defaults < global
  `$XDG_CONFIG_HOME/agent6/config.toml` < per-repo `.agent6/config.toml`
  < `--config FILE`) and every leaf is auditable via `agent6 config show`.
  `Config` stays `extra="forbid", frozen=True`; loosening a security
  default requires the same scrutiny as adding a tool.
- Secrets (provider API keys) live in `$XDG_CONFIG_HOME/agent6/secrets.toml`,
  enforced `0600` + owner-only (see `secrets.py`). They are never written
  to transcripts, never printed by `config show`, and never mounted into
  the jail. `agent6 connect` must NEVER execute anything a remote returns
  (OAuth/paste only) — the opencode RCE class of bug.
- Running as root requires explicit opt-in (`--allow-root` /
  `AGENT6_ALLOW_ROOT=1`); under sudo, agent6 reads the real user's
  config/secrets and chowns new `.agent6/` files back to them. It does NOT
  drop privileges in-process — the jail is the boundary.
- `[providers.anthropic]` and `[providers.openai]` are the only network
  endpoints the agent is allowed to talk to. New providers go through the
  same Landlock + jail audit; do not bypass.
- The `agent6-jail` Rust binary is part of the security boundary. Changes
  to `src/agent6/jail/src/main.rs` need at minimum a review note covering: what mount
  points changed, what landlock rules changed, what seccomp syscalls were
  added or removed, and what `/dev` nodes are exposed.

## Workflow expectations

- One PR per concern. Split bench + docs + security fixes into separate
  commits even on the same branch.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(scope):`, `fix(scope):`, `ci:`, `docs:`, `bench:`, etc. The
  scope (e.g. `review`, `sandbox`, `graph`) matches a directory under
  `src/agent6/` or a top-level area.
- Keep messages concise: imperative subject; a body only for non-obvious
  *why*, in point form not prose. No `Co-Authored-By` lines.
- Security-relevant commits include a `Security review note:` paragraph
  explaining what surface changed.
- Do not push. `git.allow_push = false` is the user-side enforcement;
  the social rule is the same.

## Self-review

agent6 was used to review its own source via `agent6 review`. The reviews
live under `.agent6-self-review/` (gitignored). When working on a module,
read the corresponding review file if present — it captures real findings
plus rationale for which were acted on and which were rejected as
speculative.
