# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the `agent6 init` setup wizard: granular, idempotent, optional.

Each step says what it will do, warns before overriding anything set, and can
be skipped; nothing existing is overwritten. In order: the per-repo config
file, `harness.verify_command` inferred from the repo, the secret and
build-artifact `.gitignore` entries, and AGENTS.md or its verify section.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable

from agent6 import errors, paths, verify_infer
from agent6.config import layer, write

_EMPTY_CONFIG = """\
# agent6 per-repo config (per-machine, stored under your state dir, never in
# the repo). Layered on top of: built-in secure defaults < your global config
# (~/.config/agent6/config.toml) < this file. Run `agent6 config show` to see
# every effective value and where it comes from. agent6 is secure by default,
# so this file only needs the few things specific to this repo. `agent6 init`
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

# Build artifacts a verify run leaves, kept out of the per-step commits.
_ECOSYSTEM_GITIGNORE: dict[str, tuple[str, ...]] = {
    "py": ("__pycache__/", "*.pyc", ".pytest_cache/"),
    "rust": ("target/",),
    "node": ("node_modules/",),
}

_VERIFY_HEADING = re.compile(r"^#{1,6}\s*verify\b", re.IGNORECASE | re.MULTILINE)


def _detect_ecosystem(root: pathlib.Path) -> str:
    """Return the ecosystem the repo's manifests suggest; "" when unknown."""
    if any((root / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg")):
        return "py"
    if (root / "Cargo.toml").is_file():
        return "rust"
    if (root / "package.json").is_file():
        return "node"
    return ""


# A yes/no prompter; `_accept_default` is the non-interactive stand-in.
_Ask = Callable[[str, bool], bool]


def _ask(prompt: str, default: bool) -> bool:
    """Ask a yes/no question.

    Returns:
        The answer; the default on EOF or empty input.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
    except EOFError:
        return default
    return default if not ans else ans in ("y", "yes")


def _accept_default(_prompt: str, default: bool) -> bool:
    """Return the default without asking."""
    return default


def _read_agents_md(root: pathlib.Path) -> str:
    """Return the AGENTS.md text; "" when absent or unreadable."""
    p = root / "AGENTS.md"
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _missing_gitignore_entries(root: pathlib.Path, *, ecosystem: str) -> list[str]:
    """Return the secret and build-artifact entries `.gitignore` lacks.

    Raises:
        OSError: The file cannot be read.
        UnicodeDecodeError: The file is not UTF-8.
    """
    entries = (*_GITIGNORE_ENTRIES, *_ECOSYSTEM_GITIGNORE.get(ecosystem, ()))
    gi = root / ".gitignore"
    existing_text = gi.read_text(encoding="utf-8") if gi.is_file() else ""
    existing_lines = {line.strip() for line in existing_text.splitlines()}
    return [e for e in entries if e not in existing_lines]


def _append_gitignore(root: pathlib.Path, missing: list[str]) -> str:
    """Append the missing entries to `.gitignore` under an agent6 comment.

    Returns:
        The line to print.
    """
    gi = root / ".gitignore"
    existing_text = gi.read_text(encoding="utf-8") if gi.is_file() else ""
    verb = "appended to" if existing_text else "created"
    block = ["", "# agent6 (added by `agent6 init`)", *missing, ""]
    new_text = existing_text
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(block)
    gi.write_text(new_text, encoding="utf-8")
    return f".gitignore: {verb} {len(missing)} entries ({', '.join(missing)})"


def _setup_verify_command(
    root: pathlib.Path, *, ecosystem: str, ask: _Ask, config_path: pathlib.Path | None = None
) -> None:
    """Set `harness.verify_command` from the repo when unset, asking before an override."""
    leaf = layer.effective_leaf(layer.load_effective(root, config_path), "harness.verify_command")
    value, source = leaf or ((), "default")
    already = bool(value)
    if already:
        print(f"  verify_command already set ({source}): {' '.join(value)}")
        if not ask("  Re-infer and replace it?", False):
            return
    inferred = verify_infer.infer_verify_command(root, _read_agents_md(root))
    if inferred is None:
        print(
            "  no verify command could be inferred from this repo. `agent6 run`"
            " will infer one (LLM) at run time or run gateless; set"
            " harness.verify_command later to pin one."
        )
        return
    shown = " ".join(inferred.argv)
    warn = " (OVERRIDES the current value)" if already else ""
    if not ask(f"  Set harness.verify_command to `{shown}` (from {inferred.source}){warn}?", True):
        print("  skipped verify_command.")
        return
    try:
        err = write.set_config_value(
            root, "harness.verify_command", json.dumps(list(inferred.argv)), to_repo=True
        )
    except errors.OperatorError as exc:
        err = str(exc)  # an unwritable repo config skips this step, never the whole init
    if err:
        print(f"  ERROR setting verify_command: {err}")
    else:
        print(f"  set harness.verify_command = {list(inferred.argv)}")


def _setup_agents_md(root: pathlib.Path, *, ecosystem: str, ask: _Ask) -> None:
    """Create a starter AGENTS.md, or append a verify section when the existing one lacks it."""
    agents = root / "AGENTS.md"
    inferred = verify_infer.infer_verify_command(root, _read_agents_md(root))
    verify_hint = " ".join(inferred.argv) if inferred else "# EDIT: your verify pipeline"
    if not agents.is_file():
        if ask("Create a starter AGENTS.md (how agents should work in this repo)?", True):
            agents.write_text(_STARTER_AGENTS_MD.format(verify=verify_hint), encoding="utf-8")
            print("  created AGENTS.md")
        else:
            print("  skipped AGENTS.md")
        return
    try:
        # A strict read: the text is written back, and a lossy decode would rewrite bytes.
        text = agents.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"  AGENTS.md could not be read ({exc}); leaving it alone.")
        return
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
    root: pathlib.Path,
    *,
    ecosystem: str = "",
    repo_config_target: pathlib.Path | None = None,
    interactive: bool = False,
    config_path: pathlib.Path | None = None,
) -> int:
    """Run the setup wizard.

    Args:
        root: The repository.
        ecosystem: Overrides the detected ecosystem.
        repo_config_target: Overrides the per-repo config path.
        interactive: Prompt each step; otherwise every step takes its default.
        config_path: The explicit config the effective config is read under.

    Returns:
        The CLI exit code.
    """
    root = root.resolve()
    cfg_path = repo_config_target or paths.repo_config_path(root)
    detected = ecosystem or _detect_ecosystem(root)
    ask: _Ask = _ask if interactive else _accept_default

    print(f"agent6 setup: {root}")
    print(f"  per-repo config: {cfg_path}  (out of the repo, under your state dir)")
    print()

    # 1. Per-repo config file.
    if cfg_path.is_file():
        print(f"  config exists ({cfg_path.name}); leaving it in place.")
    elif ask(f"Create the per-repo config file at {cfg_path}?", True):
        paths.mkdir_for_real_user(cfg_path.parent)
        cfg_path.write_text(_EMPTY_CONFIG, encoding="utf-8")
        print(f"  created {cfg_path}")
    else:
        print("  skipped; using the global config + built-in defaults.")

    # 2. verify_command (optional; inferred).
    _setup_verify_command(root, ecosystem=detected, ask=ask, config_path=config_path)

    # 3. .gitignore: the question names the entries it would add.
    try:
        missing = _missing_gitignore_entries(root, ecosystem=detected)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"  .gitignore could not be read ({exc}); leaving it alone")
    else:
        if not missing:
            print("  .gitignore already has all agent6 entries")
        elif ask(f"Add {len(missing)} entries to .gitignore ({', '.join(missing)})?", True):
            print("  " + _append_gitignore(root, missing))
        else:
            print("  skipped .gitignore")

    # 4. AGENTS.md.
    _setup_agents_md(root, ecosystem=detected, ask=ask)
    return 0
