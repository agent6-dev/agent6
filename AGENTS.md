# AGENTS.md — instructions for coding agents working on this repo

This file is read by coding agents (including agent6 itself) operating in
this repository. It is the authoritative map of project conventions and the
list of invariants that PRs are not allowed to weaken.

## Stability policy (pre-1.0)

agent6 has never cut a release; the version in `src/agent6/__init__.py`
is `0.0.1`. Until `1.0.0`, treat every public shape — config TOML, IPC
messages, on-disk graph format, CLI flags, transcript layout — as
**liquid**. Prefer breaking a shape cleanly over carrying it:

- No backward-compat shims, deprecation warnings, or aliased field names.
- No migration code or `config_version` translators. Bump
  `config_version` only when the new shape genuinely improves the user's
  error message, never as a substitute for "just make it the right shape".
- No "low-churn" framing in design discussions. If something is wrong,
  rip it out.

Once we ship 1.0 the rules change. Until then: if you're tempted to
write a compatibility branch, delete the old shape instead.

## Project conventions

- **Language**: Python 3.12+. Every `.py` file starts with
  `from __future__ import annotations`. Strict pyright.
- **Layout**: src layout under `src/agent6/`. Tests under `tests/`.
  Rust crate for the sandbox launcher under `jail/`.
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
  `pyproject.toml` dep list is intentionally tiny (`pydantic`, `httpx`,
  nothing else). Build dep is `hatchling`.
- **Comments / docstrings**: don't add them to code you didn't change.
  Don't add type annotations to functions you didn't modify. Don't refactor
  surrounding code "while you're there." Scope creep is a review-blocker.
- **Module boundaries** are enforced by [tach](https://docs.gauge.sh/).
  `cli → workflows → agents → tools → sandbox`. Workflows never import
  each other; agents never import workflows or the CLI. If you need to
  reach across a boundary you are usually doing the wrong thing.

## Verify command

This is also what `verify_command` should be set to in `agent6.toml` when
running the agent on this repo:

```bash
uv run ruff check && uv run ruff format --check && \
  uv run pyright && uv run tach check && uv run pytest
```

All five must pass. 208 tests currently green.

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
  (the launcher itself), `tools/code_index.py` (LSP from config), and a
  small set of `cli.py` helpers (TUI spawn, `$EDITOR` for plan editing,
  `git diff/log` for the review subcommand, `rg` for history search).
  Audit with `rg 'subprocess\.(run|Popen)' src/agent6/`.
- `src/agent6/git_ops.py` refuses `push`, `--force`, history rewrite,
  `reset --hard`, and `branch -D` unconditionally. Do not add overrides.
- Config has no implicit defaults. Every field is required at load time
  (`extra="forbid", frozen=True`). Adding a default to `Config` requires
  the same scrutiny as adding a tool.
- `[providers.anthropic]` and `[providers.openai]` are the only network
  endpoints the agent is allowed to talk to. New providers go through the
  same Landlock + jail audit; do not bypass.
- The `agent6-jail` Rust binary is part of the security boundary. Changes
  to `jail/src/main.rs` need at minimum a review note covering: what mount
  points changed, what landlock rules changed, what seccomp syscalls were
  added or removed, and what `/dev` nodes are exposed.

## Workflow expectations

- One PR per concern. Split bench + docs + security fixes into separate
  commits even on the same branch.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(scope):`, `fix(scope):`, `ci:`, `docs:`, `bench:`, etc. The
  scope (e.g. `review`, `sandbox`, `graph`) matches a directory under
  `src/agent6/` or a top-level area.
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
