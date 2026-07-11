# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config` subcommands (show/fill/path/get/set/unset/add/remove)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agent6 import models_cache
from agent6.config import (
    ConfigError,
)
from agent6.config.io import (
    parse_cli_value,
    read_toml_file,
    read_toml_leaf,
    remove_toml_leaf,
    upsert_toml_leaf,
)
from agent6.config.layer import (
    effective_leaf,
    format_value,
    load_effective,
    load_effective_with_overlay,
    materialize,
    render_show,
    repo_config_path_for,
)
from agent6.machine import MachineError, load_machine
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
    resolved = models_cache.resolved_adaptive_values(eff.config)
    print(render_show(eff, as_json=as_json, resolved=resolved), end="")
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

    Global by default; ``--repo`` writes the in-repo config; ``--machine-file FILE``
    edits that machine's ``[config]`` overlay (so keys are prefixed ``config.``
    and land in ``[config.<section>]``). ``--repo`` and ``--machine-file``
    together are ambiguous and rejected.
    """
    if machine is not None:
        if repo:
            raise ValueError("use either --repo or --machine-file, not both")
        return machine, "config."
    if repo:
        return repo_config_path_for(Path.cwd()), ""
    return global_config_path(), ""


def _reject_machine_protected(key: str, machine: Path | None) -> str | None:
    """Error string if *key* touches an operator-only table in a machine overlay.

    Mirrors ``MachineSpec._forbid_protected_overlay_tables``: a machine
    ``[config]`` overlay must not carry ``providers.*`` (endpoints/secrets) or
    ``sandbox.*`` (jail policy, network egress incl. allow_urls, run_commands,
    .git protection). Those are operator decisions, set in the
    global/repo config, never in a (possibly untrusted) machine file.
    """
    if machine is None:
        return None
    for table in ("providers", "sandbox"):
        if key == table or key.startswith(f"{table}."):
            return (
                f"machine [config] overlays must not set {table}.*:"
                " connections/secrets and sandbox policy are operator-only (global/repo config)"
            )
    return None


def _machine_is_valid(text: str | None) -> bool:
    """True iff *text* parses as a complete, valid machine spec.

    Used to decide whether a `config set --machine-file` edit BROKE a working machine
    (block + roll back) versus merely touched an already-incomplete one (allow).
    """
    if text is None:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".asm.toml", delete=True, encoding="utf-8") as tf:
        tf.write(text)
        tf.flush()
        try:
            load_machine(Path(tf.name))
        except MachineError:
            return False
    return True


def _revalidate_config(target: Path, prior_text: str | None, *, machine: Path | None) -> str | None:
    """Re-validate the config after a write; restore *prior_text* on failure.

    Returns a ready-to-print error message when the edit produced an invalid
    config (so the caller fails loud and the file is left untouched), else None.
    """
    err: str | None = None
    try:
        if machine is not None:
            data = read_toml_file(target)
            overlay = data.get("config", {})
            load_effective_with_overlay(Path.cwd(), overlay if isinstance(overlay, dict) else {})
            # Validate the WHOLE machine spec too (not just the [config] overlay)
            # so `config set --machine-file` can't BREAK a runnable machine. We only
            # block when the edit made a previously-VALID machine invalid -- a
            # machine that was already invalid (or a brand-new stub) is left for
            # the author to finish; `machine check` is the gate for runnability.
            if "states" in data and _machine_is_valid(prior_text):
                load_machine(target)
        else:
            load_effective(Path.cwd(), None)
    except ConfigError as exc:
        err = str(exc)
    except MachineError as exc:
        err = "; ".join(exc.problems)
    if err is not None:
        if prior_text is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(prior_text, encoding="utf-8")
        return err
    return None


def _cmd_config_get(key: str, *, machine: Path | None) -> int:
    """Print a leaf's effective value + the layer that set it."""
    try:
        if machine is not None:
            overlay = read_toml_file(machine).get("config", {})
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
    if err := _reject_machine_protected(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        target, prefix = _config_write_target(repo=repo, machine=machine)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    parsed = parse_cli_value(value)
    try:
        upsert_toml_leaf(target, prefix + key, parsed)
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
    if err := _reject_machine_protected(key, machine):
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
        removed = remove_toml_leaf(target, prefix + key)
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


def _schema_says_not_a_list(key: str) -> bool:
    """True when the config schema knows *key* and its value is not a list.

    Guards `config add/remove` on keys the target file does not set yet: the
    effective (defaults-included) value reveals the leaf's shape, so a scalar
    like sandbox.agent_network fails with "not a list field" instead of a
    contradictory revalidation error. Unknown keys and unloadable configs
    return False; revalidation still rejects those."""
    try:
        eff = load_effective(Path.cwd(), None)
    except ConfigError:
        return False
    leaf = effective_leaf(eff, key)
    # List leaves surface as list or tuple depending on the field's type. A
    # None effective value is an UNSET optional field (e.g. the list-valued
    # providers.*.token_command, default None); it doesn't prove the leaf is a
    # scalar, so fall through and let revalidation reject a genuine scalar.
    return leaf is not None and leaf[0] is not None and not isinstance(leaf[0], (list, tuple))


def _config_list_edit(  # noqa: PLR0911
    key: str, value: str, *, repo: bool, machine: Path | None, add: bool
) -> int:
    """Shared body for `config add` / `config remove` on a list field."""
    if err := _reject_machine_protected(key, machine):
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        target, prefix = _config_write_target(repo=repo, machine=machine)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    current = read_toml_leaf(read_toml_file(target), prefix + key)
    if current is None:
        if _schema_says_not_a_list(key):
            print(f"ERROR: {key} is not a list field.", file=sys.stderr)
            return 2
        current = []
    if not isinstance(current, list):
        print(f"ERROR: {key} is not a list field in {target}.", file=sys.stderr)
        return 2
    parsed = parse_cli_value(value)
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
        upsert_toml_leaf(target, prefix + key, items)
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
