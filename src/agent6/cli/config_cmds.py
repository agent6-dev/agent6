# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config` subcommands (show/fill/path/get/set/unset/add/remove)."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.cli.toml_io import (
    _parse_cli_value,
    _read_toml_file,
    _read_toml_leaf,
    _remove_toml_leaf,
    _upsert_toml_leaf,
)
from agent6.config import (
    ConfigError,
)
from agent6.config_layer import (
    effective_leaf,
    format_value,
    load_effective,
    load_effective_with_overlay,
    materialize,
    render_show,
    repo_config_path_for,
)
from agent6.paths import (
    chown_to_real_user,
    effective_user,
    global_config_path,
    secrets_path,
)


def _cmd_config_show(config_path: Path | None, *, as_json: bool) -> int:
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    print(render_show(eff, as_json=as_json), end="")
    return 0


def _cmd_config_path() -> int:
    user = effective_user()
    gp = global_config_path(user)
    rp = repo_config_path_for(Path.cwd())
    sp = secrets_path(user)
    for label, p in (("global config", gp), ("repo config  ", rp), ("secrets      ", sp)):
        note = "" if p.is_file() else "  (not present)"
        print(f"{label}: {p}{note}")
    return 0


def _cmd_config_fill(config_path: Path | None, *, to_repo: bool, force: bool) -> int:
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    target = repo_config_path_for(Path.cwd()) if to_repo else global_config_path()
    if target.is_file() and not force:
        print(
            f"ERROR: {target} already exists. Re-run with --force to overwrite.",
            file=sys.stderr,
        )
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(materialize(eff.config, for_repo=to_repo), encoding="utf-8")
    chown_to_real_user(target.parent)
    chown_to_real_user(target)
    print(f"Wrote fully-resolved config to {target}")
    return 0


def _config_write_target(*, repo: bool, machine: Path | None) -> tuple[Path, str]:
    """Resolve the file + dotted-key prefix a config write should target.

    Global by default; ``--repo`` writes the in-repo config; ``--machine FILE``
    edits that machine's ``[config]`` overlay (so keys are prefixed ``config.``
    and land in ``[config.<section>]``). ``--repo`` and ``--machine`` together
    are ambiguous and rejected.
    """
    if machine is not None:
        if repo:
            raise ValueError("use either --repo or --machine, not both")
        return machine, "config."
    if repo:
        return repo_config_path_for(Path.cwd()), ""
    return global_config_path(), ""


def _reject_machine_providers(key: str, machine: Path | None) -> str | None:
    """Error string if *key* touches ``providers.*`` in a machine overlay."""
    if machine is not None and (key == "providers" or key.startswith("providers.")):
        return "machine [config] overlays must not set providers.* (endpoints/keys are global-only)"
    return None


def _revalidate_config(target: Path, prior_text: str | None, *, machine: Path | None) -> str | None:
    """Re-validate the config after a write; restore *prior_text* on failure.

    Returns a ready-to-print error message when the edit produced an invalid
    config (so the caller fails loud and the file is left untouched), else None.
    """
    try:
        if machine is not None:
            overlay = _read_toml_file(target).get("config", {})
            load_effective_with_overlay(Path.cwd(), overlay if isinstance(overlay, dict) else {})
        else:
            load_effective(Path.cwd(), None)
    except ConfigError as exc:
        if prior_text is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(prior_text, encoding="utf-8")
        return str(exc)
    return None


def _cmd_config_get(key: str, *, machine: Path | None) -> int:
    """Print a leaf's effective value + the layer that set it."""
    try:
        if machine is not None:
            overlay = _read_toml_file(machine).get("config", {})
            eff = load_effective_with_overlay(
                Path.cwd(), overlay if isinstance(overlay, dict) else {}
            )
        else:
            eff = load_effective(Path.cwd(), None)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    found = effective_leaf(eff, key)
    if found is None:
        print(f"ERROR: {key!r} is not a config leaf (see `agent6 config show`).", file=sys.stderr)
        return 2
    value, source = found
    print(f"{key} = {format_value(value)}  [{source}]")
    return 0


def _cmd_config_set(key: str, value: str, *, repo: bool, machine: Path | None) -> int:
    """Set a scalar leaf in the target file (global / repo / machine overlay)."""
    if err := _reject_machine_providers(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        target, prefix = _config_write_target(repo=repo, machine=machine)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    parsed = _parse_cli_value(value)
    try:
        _upsert_toml_leaf(target, prefix + key, parsed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if err := _revalidate_config(target, prior, machine=machine):
        print(f"ERROR: {key} = {value!r} is not valid:\n{err}", file=sys.stderr)
        return 2
    chown_to_real_user(target.parent)
    chown_to_real_user(target)
    print(f"Set {key} = {format_value(parsed)} in {target}")
    return 0


def _cmd_config_unset(key: str, *, repo: bool, machine: Path | None) -> int:  # noqa: PLR0911
    """Remove a leaf so it reverts to the next-lower layer / built-in default."""
    if err := _reject_machine_providers(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        target, prefix = _config_write_target(repo=repo, machine=machine)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not target.is_file():
        print(f"ERROR: {target} does not exist; nothing to unset.", file=sys.stderr)
        return 2
    prior = target.read_text(encoding="utf-8")
    try:
        removed = _remove_toml_leaf(target, prefix + key)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not removed:
        print(f"{key} is not set in {target}; nothing to unset.")
        return 0
    if err := _revalidate_config(target, prior, machine=machine):
        print(f"ERROR: unsetting {key} left an invalid config:\n{err}", file=sys.stderr)
        return 2
    chown_to_real_user(target)
    print(f"Unset {key} in {target}")
    return 0


def _config_list_edit(  # noqa: PLR0911
    key: str, value: str, *, repo: bool, machine: Path | None, add: bool
) -> int:
    """Shared body for `config add` / `config remove` on a list field."""
    if err := _reject_machine_providers(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        target, prefix = _config_write_target(repo=repo, machine=machine)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    current = _read_toml_leaf(_read_toml_file(target), prefix + key)
    if current is None:
        current = []
    if not isinstance(current, list):
        print(f"ERROR: {key} is not a list field in {target}.", file=sys.stderr)
        return 2
    parsed = _parse_cli_value(value)
    items = list(current)
    if add:
        if parsed in items:
            print(f"{format_value(parsed)} already in {key}.")
            return 0
        items.append(parsed)
    else:
        if parsed not in items:
            print(f"{format_value(parsed)} not in {key}.")
            return 0
        items = [x for x in items if x != parsed]
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    try:
        _upsert_toml_leaf(target, prefix + key, items)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if err := _revalidate_config(target, prior, machine=machine):
        print(f"ERROR: {value!r} is not valid for {key}:\n{err}", file=sys.stderr)
        return 2
    chown_to_real_user(target.parent)
    chown_to_real_user(target)
    verb, prep = ("Added", "to") if add else ("Removed", "from")
    print(f"{verb} {format_value(parsed)} {prep} {key} in {target}")
    return 0


def _cmd_config_add(key: str, value: str, *, repo: bool, machine: Path | None) -> int:
    return _config_list_edit(key, value, repo=repo, machine=machine, add=True)


def _cmd_config_remove(key: str, value: str, *, repo: bool, machine: Path | None) -> int:
    return _config_list_edit(key, value, repo=repo, machine=machine, add=False)
