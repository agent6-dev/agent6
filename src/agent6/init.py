# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init` — write a starter per-repo config, AGENTS.md, and .gitignore.

Writes ``.agent6/config.toml`` (the per-repo override, layered on top of the
global ``~/.config/agent6/config.toml`` and secure built-in defaults),
``AGENTS.md``, and appends agent6 entries to ``.gitignore``. Never overwrites
existing files: if a target exists, write a ``.suggested`` sibling and tell
the user to diff. Templates are deliberately short — providers, models, and
API keys normally live in the global config (set via ``agent6 connect`` /
``agent6 model``), so a repo only needs its ``verify_command``.
"""

from __future__ import annotations

from pathlib import Path

from agent6.paths import repo_config_path

_STARTER_TOML = """\
# agent6 per-repo config (.agent6/config.toml).
#
# Layered on top of: built-in secure defaults < your global config
# (~/.config/agent6/config.toml) < this file. Run `agent6 config show` to see
# every effective value and where it comes from. agent6 is secure by default,
# so this file only needs the few things that are specific to THIS repo.

[workflow]
# What "a step succeeded" means in this repo. EDIT THIS to your real pipeline.
verify_command = ["uv", "run", "pytest", "-x"]

# Providers, models, and API keys usually live in your GLOBAL config:
#   agent6 connect                       # add a provider + API key
#   agent6 model --role worker ...       # pick the model for a role
# Override per-repo by uncommenting, e.g.:
# [models.worker]
# provider = "anthropic"
# model = "claude-sonnet-4-5"
"""


# Per-profile overrides for the [workflow].verify_command line + a hint at
# the top of AGENTS.md. The TOML scaffolding above is otherwise identical
# across profiles; profiles are deliberately a tiny convenience, not a
# divergence point.
_PROFILE_VERIFY_COMMANDS: dict[str, list[str]] = {
    "py": ["uv", "run", "pytest", "-x"],
    "rust": ["cargo", "test", "--quiet"],
    "node": ["npm", "test", "--silent"],
}

_PROFILE_AGENTS_HINTS: dict[str, str] = {
    "py": "uv run pytest -x",
    "rust": "cargo test --quiet",
    "node": "npm test --silent",
}


def _render_starter_toml(profile: str) -> str:
    """Substitute the verify_command line for the chosen profile."""
    cmd = _PROFILE_VERIFY_COMMANDS.get(profile)
    if cmd is None:
        raise ValueError(f"unknown init profile: {profile!r}")
    rendered = ", ".join(f'"{p}"' for p in cmd)
    return _STARTER_TOML.replace(
        'verify_command = ["uv", "run", "pytest", "-x"]',
        f"verify_command = [{rendered}]",
    )


def _render_starter_agents_md(profile: str) -> str:
    hint = _PROFILE_AGENTS_HINTS.get(profile, "uv run pytest -x")
    return _STARTER_AGENTS_MD.replace(
        "# EDIT: replace with your actual verify pipeline.\nuv run pytest -x",
        f"# EDIT: replace with your actual verify pipeline.\n{hint}",
    )


_STARTER_AGENTS_MD = """\
# AGENTS.md

This file tells coding agents (including agent6) how to work in this repo.
Agents are instructed to read it before planning and to update it when they
change a project convention, build command, dependency, or security invariant.

## Project conventions

<!-- EDIT: language, framework, style, type-check, formatter, naming rules -->

## Verify command

The command agent6 runs to decide whether a step "succeeded". Must match
the `verify_command` in `.agent6/config.toml`.

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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def init_workspace(root: Path, *, force: bool, profile: str = "py") -> int:
    """Write starter files into `root`. Returns a CLI exit code."""
    root = root.resolve()
    print(f"agent6 init: {root}  (profile={profile})")
    starter_toml = _render_starter_toml(profile)
    starter_agents_md = _render_starter_agents_md(profile)
    print(_write_or_suggest(repo_config_path(root), starter_toml, force=force))
    print(_write_or_suggest(root / "AGENTS.md", starter_agents_md, force=force))
    print(_update_gitignore(root))
    print()
    print("Next:")
    print("  1. agent6 connect                 # add a provider + API key (global)")
    print("  2. agent6 model --role worker ... # pick your worker model (or set it globally)")
    print("  3. Edit .agent6/config.toml: set `verify_command` for this repo.")
    print("  4. agent6 config show             # audit the effective config")
    print("  5. agent6 check                   # sandbox + config pre-flight")
    print('  6. agent6 run "<task>"')
    return 0
