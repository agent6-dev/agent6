# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init` — write starter agent6.toml, AGENTS.md, and .gitignore entries.

Per Round-2 Locked Decision §2: never overwrite existing files. If the target
exists, write a `.suggested` sibling and tell the user to diff. The bundled
templates are deliberately short and opinionated; the user is expected to
edit them.
"""

from __future__ import annotations

from pathlib import Path

_STARTER_TOML = """\
# agent6 configuration. Every field is REQUIRED.

[agent6]
config_version = 1

[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

# Add additional providers as needed; the table key (e.g. "openrouter",
# "ollama") is the name you reference from [models.<role>] below. See
# agent6.example.toml for OpenAI-compatible examples (OpenAI, OpenRouter,
# Ollama, vLLM, llama.cpp).

# Route each sub-agent role to one of the providers above. Mix and match
# freely (e.g. Anthropic planner + OpenRouter worker + Anthropic reviewer).
[models.planner]
provider = "anthropic"
model = "claude-opus-4-5"

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.critic]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-opus-4-5"

[models.summarizer]
provider = "anthropic"
model = "claude-haiku-4-5"

[sandbox]
# "auto" picks the strongest profile this kernel + container can run.
# See agent6.example.toml in the agent6 source tree for full notes.
profile = "auto"
network = "provider_only"
run_commands = "ask"
# Make .git/ and agent6.toml/.agent6/ read-only from the child's view
# so a worker cannot rewrite history or forge transcripts from inside
# a run_command. Works in both profiles (strict: bind-remount-RO;
# hardened: Landlock carve-out). See agent6.example.toml.
protect_git = true
protect_agent6 = true

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
default = "implement"
# What "step succeeded" means in your repo. EDIT THIS.
verify_command = ["uv", "run", "pytest", "-x"]

[budget]
# Hard stop. The run is resumable from the persistent task graph.
max_input_tokens = 2000000
max_output_tokens = 200000
"""


_STARTER_AGENTS_MD = """\
# AGENTS.md

This file tells coding agents (including agent6) how to work in this repo.
Agents are instructed to read it before planning and to update it when they
change a project convention, build command, dependency, or security invariant.

## Project conventions

<!-- EDIT: language, framework, style, type-check, formatter, naming rules -->

## Verify command

The command agent6 runs to decide whether a step "succeeded". Must match
the `verify_command` in `agent6.toml`.

```bash
# EDIT: replace with your actual verify pipeline.
uv run pytest -x
```

## Security invariants (do not weaken)

<!-- EDIT: things an agent must NEVER do, e.g. -->
- No new runtime dependencies without explicit review.
- No bypassing pre-commit hooks (no `--no-verify`).
- No pushing to remote branches; agent6 enforces this in `git_ops` already.

## Things not to do

<!-- EDIT: idiomatic anti-patterns specific to this codebase. -->
"""


_GITIGNORE_ENTRIES = (
    ".env",
    ".env.*",
    ".envrc",
    "secrets/",
    "*.pem",
    "*.key",
    ".agent6/",
)


def _write_or_suggest(path: Path, content: str, *, force: bool) -> str:
    """Write `content` to `path`. If `path` exists and not force, write
    `path.with_suffix(path.suffix + '.suggested')` instead.

    Returns a one-line status message describing what happened.
    """
    if path.exists() and not force:
        suggested = path.with_name(path.name + ".suggested")
        suggested.write_text(content, encoding="utf-8")
        return f"  exists, wrote suggested: {suggested.name}  (diff against {path.name})"
    path.write_text(content, encoding="utf-8")
    verb = "overwrote" if path.exists() and force else "created"
    return f"  {verb}: {path.name}"


def _update_gitignore(root: Path) -> str:
    """Append any missing entries from `_GITIGNORE_ENTRIES` to `.gitignore`.

    Idempotent: if the file already contains every entry (line-equal match
    after strip), no write happens. Existing content is never reordered or
    removed.
    """
    gi = root / ".gitignore"
    existing_lines: set[str] = set()
    existing_text = ""
    if gi.exists():
        existing_text = gi.read_text(encoding="utf-8")
        existing_lines = {line.strip() for line in existing_text.splitlines()}
    missing = [e for e in _GITIGNORE_ENTRIES if e not in existing_lines]
    if not missing:
        return "  .gitignore: already has all agent6 entries"
    block = ["", "# agent6 (added by `agent6 init`)", *missing, ""]
    new_text = existing_text
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(block)
    gi.write_text(new_text, encoding="utf-8")
    return f"  .gitignore: appended {len(missing)} entries ({', '.join(missing)})"


def init_workspace(root: Path, *, force: bool) -> int:
    """Write starter files into `root`. Returns a CLI exit code."""
    root = root.resolve()
    print(f"agent6 init: {root}")
    print(_write_or_suggest(root / "agent6.toml", _STARTER_TOML, force=force))
    print(_write_or_suggest(root / "AGENTS.md", _STARTER_AGENTS_MD, force=force))
    print(_update_gitignore(root))
    print()
    print("Next:")
    print("  1. Edit agent6.toml: set `verify_command` for your repo.")
    print("  2. Edit AGENTS.md: fill in the EDIT markers.")
    print("  3. export ANTHROPIC_API_KEY=...")
    print("  4. agent6 check-config agent6.toml")
    return 0
