# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init`, write a starter per-repo config, AGENTS.md, and .gitignore.

Writes the per-repo config (the per-repo override, out of the workspace under
the state dir, layered on top of the global ``~/.config/agent6/config.toml``
and secure built-in defaults), ``AGENTS.md`` in the repo, and appends
build-artifact + secret entries to the repo ``.gitignore`` (so a verify
command's output is not swept into agent6's per-step commits). Never
overwrites existing files: if a target exists, write a ``.suggested`` sibling
and tell the user to diff. Templates are deliberately short, providers,
models, and API keys normally live in the global config (set via
``agent6 connect`` / ``agent6 model``), so a repo only needs its
``verify_command``.
"""

from __future__ import annotations

from pathlib import Path

from agent6.paths import repo_config_path

_STARTER_TOML = """\
# agent6 per-repo config (per-machine, stored under your state dir, NOT in the
# repo). Run `agent6 config show` to see its path.
#
# Layered on top of: built-in secure defaults < your global config
# (~/.config/agent6/config.toml) < this file. Run `agent6 config show` to see
# every effective value and where it comes from. agent6 is secure by default,
# so this file only needs the few things that are specific to THIS repo.

[workflow]
# What "a step succeeded" means in this repo. EDIT THIS to your real pipeline.
# Make it a REAL pass/fail (build + tests), not a syntax check: the harness uses
# a green verify as a completion cue, once it passes and the worker stops making
# changes, agent6 wraps the run up instead of letting it spin.
#
# IMPORTANT: this runs INSIDE the sandbox, not your shell. The jailed command
# sees PATH=/usr/bin:/bin plus this repo directory only -- no $HOME, no
# network. So a toolchain installed under your home (uv, cargo, nvm-managed
# node) and a uv-created `.venv/bin/python` (a symlink into ~/.local/share/uv)
# will NOT resolve. Point this at a jail-visible interpreter: a stdlib
# `python -m venv .venv` (whose .venv/bin/python links into /usr) or
# /usr/bin/python3 directly. Examples:
#   [".venv/bin/python", "-m", "pytest", "-x"]    # stdlib venv (the default)
#   ["/usr/bin/python3", "-m", "pytest", "-x"]
verify_command = [".venv/bin/python", "-m", "pytest", "-x"]

# Providers, models, and API keys usually live in your GLOBAL config:
#   agent6 connect                       # add a provider + API key
#   agent6 model worker <provider> <model>  # pick the model for a role
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
    "py": [".venv/bin/python", "-m", "pytest", "-x"],
    "rust": ["cargo", "test", "--quiet"],
    "node": ["npm", "test", "--silent"],
}

_PROFILE_AGENTS_HINTS: dict[str, str] = {
    "py": ".venv/bin/python -m pytest -x",
    "rust": "cargo test --quiet",
    "node": "npm test --silent",
}

# Build artifacts that the verify command can drop into the workspace; agent6
# commits the whole worktree per step, so without these in .gitignore the test
# run's bytecode/output gets committed alongside the agent's edits.
_PROFILE_GITIGNORE: dict[str, tuple[str, ...]] = {
    "py": ("__pycache__/", "*.pyc", ".pytest_cache/"),
    "rust": ("target/",),
    "node": ("node_modules/",),
}


def _render_starter_toml(profile: str) -> str:
    """Substitute the verify_command line for the chosen profile."""
    cmd = _PROFILE_VERIFY_COMMANDS.get(profile)
    if cmd is None:
        raise ValueError(f"unknown init profile: {profile!r}")
    rendered = ", ".join(f'"{p}"' for p in cmd)
    return _STARTER_TOML.replace(
        'verify_command = [".venv/bin/python", "-m", "pytest", "-x"]',
        f"verify_command = [{rendered}]",
    )


def _render_starter_agents_md(profile: str) -> str:
    hint = _PROFILE_AGENTS_HINTS.get(profile, ".venv/bin/python -m pytest -x")
    return _STARTER_AGENTS_MD.replace(
        "# EDIT: replace with your actual verify pipeline.\n.venv/bin/python -m pytest -x",
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
the `verify_command` in your per-repo agent6 config.

```bash
# EDIT: replace with your actual verify pipeline.
.venv/bin/python -m pytest -x
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


def _update_gitignore(root: Path, *, profile: str = "py") -> str:
    """Append any missing secret + build-artifact entries to `.gitignore`.

    Idempotent: if the file already contains every entry (line-equal match
    after strip), no write happens. Existing content is never reordered or
    removed. ``profile`` adds that ecosystem's build artifacts (e.g.
    ``__pycache__/``) so the verify command's bytecode/output is not swept
    into agent6's per-step commits. (Run state lives out of the workspace, so
    there is nothing agent6-specific to ignore.)
    """
    entries = (*_GITIGNORE_ENTRIES, *_PROFILE_GITIGNORE.get(profile, ()))
    gi = root / ".gitignore"
    existing_lines: set[str] = set()
    existing_text = ""
    if gi.exists():
        existing_text = gi.read_text(encoding="utf-8")
        existing_lines = {line.strip() for line in existing_text.splitlines()}
    missing = [e for e in entries if e not in existing_lines]
    if not missing:
        return "  .gitignore: already has all agent6 entries"
    verb = "appended to" if existing_text else "created"
    block = ["", "# agent6 (added by `agent6 init`)", *missing, ""]
    new_text = existing_text
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(block)
    gi.write_text(new_text, encoding="utf-8")
    return f"  .gitignore: {verb} {len(missing)} entries ({', '.join(missing)})"


def _ask(prompt: str, default: bool) -> bool:
    """Yes/no prompt. Returns *default* on EOF or empty input."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


def init_workspace(
    root: Path,
    *,
    force: bool,
    profile: str = "py",
    repo_config_target: Path | None = None,
    interactive: bool = False,
) -> int:
    """Write starter files into `root`. Returns a CLI exit code.

    ``repo_config_target`` is the per-repo config path to write; defaults to
    the per-repo config under the state dir (out of the workspace). When
    ``interactive`` and stdin is a TTY, prompt before writing the config and
    before amending an existing AGENTS.md / .gitignore.
    """
    root = root.resolve()
    cfg_path = repo_config_target or repo_config_path(root)
    print(f"agent6 init: {root}  (profile={profile})")
    starter_toml = _render_starter_toml(profile)
    starter_agents_md = _render_starter_agents_md(profile)

    if interactive and not force:
        # 1. Config scope: a repo config is optional, providers/models/keys
        #    usually live in the global config (agent6 connect / agent6 model).
        if _ask(
            f"Write a starter repo config at {cfg_path}?"
            " (out of the repo; providers/keys usually live in your global config)",
            default=True,
        ):
            print(_write_or_suggest(cfg_path, starter_toml, force=force))
        else:
            print("  skipped repo config (using global config + defaults)")
        # 2. AGENTS.md: create, or offer to leave an existing one untouched.
        agents = root / "AGENTS.md"
        if agents.exists():
            if _ask("AGENTS.md exists — write a .suggested sibling to diff?", default=False):
                print(_write_or_suggest(agents, starter_agents_md, force=False))
            else:
                print("  kept existing AGENTS.md")
        else:
            print(_write_or_suggest(agents, starter_agents_md, force=force))
        # 3. .gitignore: always offered (create or amend; idempotent).
        if _ask("Add secret + build-artifact entries to .gitignore?", default=True):
            print(_update_gitignore(root, profile=profile))
        else:
            print("  skipped .gitignore")
    else:
        print(_write_or_suggest(cfg_path, starter_toml, force=force))
        print(_write_or_suggest(root / "AGENTS.md", starter_agents_md, force=force))
        print(_update_gitignore(root, profile=profile))

    print()
    print("Next:")
    print("  1. agent6 connect                 # add a provider + API key (global)")
    print(
        "  2. agent6 model worker <provider> <model> # pick your worker model (or set it globally)"
    )
    print(f"  3. Edit {cfg_path}: set `verify_command` for this repo.")
    print("  4. agent6 config show             # audit the effective config")
    print("  5. agent6 check                   # sandbox + config pre-flight")
    print('  6. agent6 run "<task>"')
    return 0
