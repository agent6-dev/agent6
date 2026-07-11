# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 init` -- a granular, idempotent setup wizard.

init is OPTIONAL: agent6 runs with a global config + secure defaults, and
`agent6 run` infers a verify command on its own. This wizard just makes the
per-repo niceties easy and explicit. It is safe to run on a FRESH repo or one
already using agent6: each step says what it will do, warns before overriding
anything you already set, and you can skip any of them. It NEVER writes a
blanket ``.suggested`` file or clobbers an existing AGENTS.md / config.

Steps, in order:
  1. create the per-repo config file if it's missing (else leave it);
  2. set ``workflow.verify_command`` if unset -- inferred from the repo
     (AGENTS.md / package.json / Makefile / pyproject / Cargo / go.mod);
  3. add secret + build-artifact entries to ``.gitignore`` (idempotent);
  4. create AGENTS.md, or append a ``## Verify command`` section if missing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from agent6.config_layer import (
    effective_leaf,
    load_effective,
    repo_config_path_for,
    set_config_value,
)
from agent6.verify_infer import infer_verify_command

_EMPTY_CONFIG = """\
# agent6 per-repo config (per-machine, stored under your state dir, NOT in the
# repo). Layered on top of: built-in secure defaults < your global config
# (~/.config/agent6/config.toml) < this file. Run `agent6 config show` to see
# every effective value and where it comes from. agent6 is secure by default,
# so this file only needs the few things specific to THIS repo. `agent6 init`
# and `agent6 config set <key> <value>` write here for you.
"""

_STARTER_AGENTS_MD = """\
# AGENTS.md

This file tells coding agents (including agent6) how to work in this repo.
Agents are instructed to read it before planning and to update it when they
change a project convention, build command, dependency, or security invariant.

## Project conventions

<!-- EDIT: language, framework, style, type-check, formatter, naming rules -->

## Verify command

The command agent6 runs to decide whether a step "succeeded". agent6 reads this
section to infer its verify_command when one is not configured -- keep it a real
pass/fail (build + tests). It runs in the sandbox (PATH=/usr/bin:/bin plus the
standard bin dirs, ephemeral $HOME, no network), so `uv run ...` works (it uses
the already-synced venv); a stdlib `.venv/bin/python` or `/usr/bin/python3` is
also fine.

```bash
{verify}
```

## Security invariants (do not weaken)

<!-- EDIT: things an agent must NEVER do, e.g. -->
- No new runtime dependencies without explicit review.
- No bypassing pre-commit hooks (no `--no-verify`).

## Things not to do

<!-- EDIT: idiomatic anti-patterns specific to this codebase. -->
"""

_VERIFY_SECTION = """\

## Verify command

The command agent6 runs to decide whether a step "succeeded" (agent6 reads this
to infer its verify_command when one is not configured).

```bash
{verify}
```
"""

_GITIGNORE_ENTRIES = (".env", ".env.*", ".envrc", "secrets/", "*.pem", "*.key")

# Per-ecosystem build artifacts to ignore so a verify run's bytecode/output is
# not swept into agent6's per-step commits.
_PROFILE_GITIGNORE: dict[str, tuple[str, ...]] = {
    "py": ("__pycache__/", "*.pyc", ".pytest_cache/"),
    "rust": ("target/",),
    "node": ("node_modules/",),
}

_VERIFY_HEADING = re.compile(r"^#{1,6}\s*verify\b", re.IGNORECASE | re.MULTILINE)


def _detect_profile(root: Path) -> str:
    """Best-effort ecosystem guess for the .gitignore artifacts ("" if unknown)."""
    if any((root / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg")):
        return "py"
    if (root / "Cargo.toml").is_file():
        return "rust"
    if (root / "package.json").is_file():
        return "node"
    return ""


# A yes/no prompter: (prompt, default) -> bool. `_ask` prompts; `_accept_default`
# is the non-interactive stand-in that just takes each step's default.
_Ask = Callable[[str, bool], bool]


def _ask(prompt: str, default: bool) -> bool:
    """Yes/no prompt. Returns *default* on EOF or empty input."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
    except EOFError:
        return default
    return default if not ans else ans in ("y", "yes")


def _accept_default(_prompt: str, default: bool) -> bool:
    return default


def _read_agents_md(root: Path) -> str:
    p = root / "AGENTS.md"
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _update_gitignore(root: Path, *, profile: str) -> str:
    """Append any missing secret + build-artifact entries to `.gitignore`.

    Idempotent: a line-equal match (after strip) is never re-added; existing
    content is never reordered or removed."""
    entries = (*_GITIGNORE_ENTRIES, *_PROFILE_GITIGNORE.get(profile, ()))
    gi = root / ".gitignore"
    existing_text = gi.read_text(encoding="utf-8") if gi.is_file() else ""
    existing_lines = {line.strip() for line in existing_text.splitlines()}
    missing = [e for e in entries if e not in existing_lines]
    if not missing:
        return ".gitignore already has all agent6 entries"
    verb = "appended to" if existing_text else "created"
    block = ["", "# agent6 (added by `agent6 init`)", *missing, ""]
    new_text = existing_text
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(block)
    gi.write_text(new_text, encoding="utf-8")
    return f".gitignore: {verb} {len(missing)} entries ({', '.join(missing)})"


def _setup_verify_command(root: Path, *, profile: str, ask: _Ask) -> None:
    """Set workflow.verify_command if unset, inferring it from the repo. Warns
    (and asks) before overriding a command already set in any layer."""
    leaf = effective_leaf(load_effective(root), "workflow.verify_command")
    value, source = leaf or ((), "default")
    already = bool(value)
    if already:
        print(f"  verify_command already set ({source}): {' '.join(value)}")
        if not ask("  Re-infer and replace it?", False):
            return
    inferred = infer_verify_command(root, _read_agents_md(root))
    if inferred is None:
        print(
            "  no verify command could be inferred from this repo. `agent6 run`"
            " will infer one (LLM) at run time or run gateless; set"
            " workflow.verify_command later to pin one."
        )
        return
    shown = " ".join(inferred.argv)
    warn = " (OVERRIDES the current value)" if already else ""
    if not ask(f"  Set workflow.verify_command to `{shown}` (from {inferred.source}){warn}?", True):
        print("  skipped verify_command.")
        return
    err = set_config_value(
        root, "workflow.verify_command", json.dumps(list(inferred.argv)), to_repo=True
    )
    if err:
        print(f"  ERROR setting verify_command: {err}")
    else:
        print(f"  set workflow.verify_command = {list(inferred.argv)}")


def _setup_agents_md(root: Path, *, profile: str, ask: _Ask) -> None:
    """Create a starter AGENTS.md, or append a Verify-command section if the
    existing one lacks it. Never overwrites existing content."""
    agents = root / "AGENTS.md"
    inferred = infer_verify_command(root, _read_agents_md(root))
    verify_hint = " ".join(inferred.argv) if inferred else "# EDIT: your verify pipeline"
    if not agents.is_file():
        if ask("Create a starter AGENTS.md (how agents should work in this repo)?", True):
            agents.write_text(_STARTER_AGENTS_MD.format(verify=verify_hint), encoding="utf-8")
            print("  created AGENTS.md")
        else:
            print("  skipped AGENTS.md")
        return
    text = agents.read_text(encoding="utf-8", errors="replace")
    if _VERIFY_HEADING.search(text):
        print("  AGENTS.md already documents a verify command; leaving it.")
        return
    if ask("AGENTS.md has no '## Verify command' section; append one?", False):
        suffix = "" if text.endswith("\n") else "\n"
        agents.write_text(
            text + suffix + _VERIFY_SECTION.format(verify=verify_hint), encoding="utf-8"
        )
        print("  appended a '## Verify command' section to AGENTS.md")


def init_workspace(
    root: Path,
    *,
    profile: str = "",
    repo_config_target: Path | None = None,
    interactive: bool = False,
) -> int:
    """Run the granular setup wizard. Returns a CLI exit code.

    ``interactive`` prompts each step; otherwise every step takes its default.
    Either way nothing existing is overwritten. ``profile`` overrides ecosystem
    auto-detection.
    """
    root = root.resolve()
    cfg_path = repo_config_target or repo_config_path_for(root)
    detected = profile or _detect_profile(root)
    # Non-interactive: take every step's default answer.
    ask: _Ask = _ask if interactive else _accept_default

    print(f"agent6 setup: {root}")
    print(f"  per-repo config: {cfg_path}  (out of the repo, under your state dir)")
    print()

    # 1. Per-repo config file.
    if cfg_path.is_file():
        print(f"  config exists ({cfg_path.name}); leaving it in place.")
    elif ask(f"Create the per-repo config file at {cfg_path}?", True):
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(_EMPTY_CONFIG, encoding="utf-8")
        print(f"  created {cfg_path}")
    else:
        print("  skipped; using the global config + built-in defaults.")

    # 2. verify_command (optional; inferred).
    _setup_verify_command(root, profile=detected, ask=ask)

    # 3. .gitignore (idempotent).
    if ask("Add secret + build-artifact entries to .gitignore?", True):
        print("  " + _update_gitignore(root, profile=detected or "py"))
    else:
        print("  skipped .gitignore")

    # 4. AGENTS.md.
    _setup_agents_md(root, profile=detected, ask=ask)
    # The CLI wrapper (`cli.init_cmds`) prints the "Next:" pointers after its
    # git-setup offer, so the advertised commands come last and actually work.
    return 0
