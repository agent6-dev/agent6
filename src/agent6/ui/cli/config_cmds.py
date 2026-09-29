# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 config` subcommands: show, fill, fix, path, presets, get, set, unset, add, remove."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from agent6.app.confine import resolved_config_values
from agent6.config import (
    ConfigError,
)
from agent6.config.io import (
    parse_cli_value,
    read_toml_file,
    read_toml_leaf,
    remove_toml_leaf,
    remove_toml_table,
    upsert_toml_leaf,
)
from agent6.config.layer import (
    EffectiveConfig,
    InvalidEntry,
    effective_leaf,
    find_invalid_entries,
    flatten_leaves,
    load_effective,
    load_effective_with_overlay,
    load_global_only,
    materialize,
    preset_catalog,
)
from agent6.config.write import (
    keep_or_rollback,
    merged_config_error,
    resolved_write_path,
    revalidate_write,
    set_config_value,
    unknown_key_error,
    unset_config_value,
    writing_config,
)
from agent6.errors import OperatorError, read_operator_file
from agent6.machine import (
    MachineError,
    load_machine,
    protected_overlay_error,
    protected_overlay_key_error,
)
from agent6.paths import (
    cache_dir,
    chown_to_real_user,
    data_dir,
    effective_user,
    global_config_path,
    mkdir_for_real_user,
    repo_config_path,
    secrets_path,
    state_base,
    state_dir,
)
from agent6.portable import atomic_write, locked_file
from agent6.ui.cli._common import error, warn
from agent6.viewmodel.config_view import format_value, render_key_detail, render_show


def _require_machine_file(machine: Path | None) -> None:
    """Refuse a missing machine file instead of treating it as empty.

    Raises:
        OperatorError: The file does not exist.
    """
    if machine is not None and not machine.is_file():
        raise OperatorError(f"no such machine file: {machine}")


def _read_machine_overlay(machine: Path | None) -> dict[str, object] | None:
    """Read a machine file's `[config]` overlay.

    Returns:
        The overlay; None when no file is named.

    Raises:
        OperatorError: The file is missing or sets an operator-only key.
    """
    if machine is None:
        return None
    _require_machine_file(machine)
    overlay = read_toml_file(machine).get("config", {})
    if not isinstance(overlay, dict):
        return {}
    if problem := protected_overlay_error(overlay):
        raise OperatorError(problem)
    return overlay


def _effective_with_overlay(config_path: Path | None, machine: Path | None) -> EffectiveConfig:
    """Return the effective config, with a machine file's `[config]` overlay on top when named."""
    overlay = _read_machine_overlay(machine)
    if overlay is None:
        return load_effective(Path.cwd(), config_path)
    return load_effective_with_overlay(Path.cwd(), overlay, explicit_path=config_path)


def _cmd_config_show(
    config_path: Path | None,
    *,
    as_json: bool,
    keys: list[str] | None = None,
    descriptions: bool = False,
    machine: Path | None = None,
) -> int:
    """Print the effective config, every leaf with its origin, or the asked-for keys in full.

    Args:
        config_path: The invocation's `--config`.
        as_json: Print JSON.
        keys: Leaves or section prefixes to show untruncated, with their descriptions.
        descriptions: Print each leaf's description.
        machine: A machine file whose `[config]` overlay goes on top.

    Returns:
        The exit code.
    """
    eff = _effective_with_overlay(config_path, machine)
    resolved = resolved_config_values(eff.config)
    if keys:
        # An asked-for key is the one place a description can never bury the values.
        try:
            detail = render_key_detail(
                eff, keys, resolved=resolved, color=sys.stdout.isatty(), as_json=as_json
            )
        except KeyError as exc:
            error(_config_key_error(str(exc.args[0]), eff))
            return 2
        print(detail, end="")
        return 0
    text = render_show(
        eff,
        as_json=as_json,
        resolved=resolved,
        color=sys.stdout.isatty(),
        descriptions=descriptions,
    )
    print(text, end="")
    return 0


def _cmd_config_path() -> int:
    """Print every file and directory agent6 reads or writes, resolved.

    Returns:
        0.
    """
    user = effective_user()
    rows: list[tuple[str, Path, bool]] = [
        ("global config", global_config_path(user), True),
        ("repo config", repo_config_path(Path.cwd()), True),
        ("secrets", secrets_path(user), True),
        ("config dir", global_config_path(user).parent, False),
        ("state (all repos)", state_base(user), False),
        ("state (this repo)", state_dir(Path.cwd()), False),
        ("skills", data_dir(user) / "skills", False),
        ("cache", cache_dir(user), False),
    ]
    width = max(len(label) for label, _p, _f in rows)
    for label, p, is_file in rows:
        present = p.is_file() if is_file else p.is_dir()
        print(f"{label:<{width}}: {p}{'' if present else '  (not present)'}")
    return 0


def _cmd_config_presets(config_path: Path | None = None) -> int:
    """Print every known preset with the overrides it applies, the selected one marked.

    Returns:
        The exit code.
    """
    cat = preset_catalog(Path.cwd(), config_path)
    if cat.selected:
        print(f"preset = {cat.selected}  [{cat.source}]")
    else:
        print("no preset selected (plain defaults)")
    for info in cat.presets:
        tag = "built-in" if info.origin == "built-in" else f"user, {info.origin} config"
        if info.replaces_builtin:
            tag += ", replaces the built-in"
        sel = "  (selected)" if info.name == cat.selected else ""
        print(f"\n{info.name}  [{tag}]{sel}")
        leaves = flatten_leaves(info.overrides)
        if not leaves:
            print("  (plain defaults, no overrides)")
        for key, val in leaves.items():
            print(f"  {key} = {format_value(val)}")
    print(
        "\nSelect per run with --preset <name>;"
        " persist with `agent6 config set preset <name>` (--repo for this repo)."
    )
    return 0


def _open_target(target: Path) -> None:
    """Create the config dir and hand it back to the real operator before any write under sudo."""
    mkdir_for_real_user(target.parent)


def _cmd_config_fill(*, force: bool) -> int:
    """Write the defaults plus the global layer into the global config file.

    Never the repo layer and never a preset's effects.

    Args:
        force: Overwrite an existing file.

    Returns:
        The exit code.
    """
    target = resolved_write_path(global_config_path())
    _open_target(target)
    # Read and publish under the lock: a read before it would lose a concurrent `config set`.
    with writing_config(target):
        eff = load_global_only()
        if target.is_file() and not force:
            error(f"{target} already exists. Re-run with --force to overwrite.")
            return 2
        atomic_write(
            target,
            materialize(eff.config, keep_presets_from=target, keep_preset_selector=True),
        )
    print(f"Wrote fully-resolved config to {target}")
    return 0


def _config_write_target(*, repo: bool, machine: Path | None) -> tuple[Path, str]:
    """Return the file and the dotted-key prefix a config write targets.

    Args:
        repo: Write the in-repo config.
        machine: Write this machine file's `[config]` overlay, under the `config.` prefix.

    Returns:
        The file and the prefix.

    Raises:
        OperatorError: Both targets were named.
    """
    if machine is not None:
        if repo:
            raise OperatorError("use either --repo or --machine-file, not both")
        if machine.is_file():
            _read_machine_overlay(machine)
        return resolved_write_path(machine), "config."
    if repo:
        return resolved_write_path(repo_config_path(Path.cwd())), ""
    return resolved_write_path(global_config_path()), ""


def _reject_machine_protected(key: str, machine: Path | None) -> str | None:
    """Return the error for an operator-only key in a machine overlay, by the loader's own rule."""
    if machine is None:
        return None
    return protected_overlay_key_error(key)


def _machine_is_valid(text: str | None) -> bool:
    """Return whether the text parses as a complete, valid machine spec."""
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


def _leaf_problems(text: str) -> str:
    """Return a config validation error's `leaf: message` lines, the shape every writer prints."""
    leaves = [
        ln.strip()[2:].split(" (type=", 1)[0]
        for ln in text.splitlines()
        if ln.strip().startswith("- ")
    ]
    return "\n".join(leaves) if leaves else text


def _revalidate_machine(target: Path, prior_text: str | None, *, held: bool = True) -> str | None:
    """Re-validate a machine file after an overlay write, restoring the prior text on failure.

    Blocks only when the edit made a valid machine invalid; one already invalid is
    left for the author to finish.

    Args:
        target: The machine file.
        prior_text: The file before the write; None when it did not exist.
        held: The lock was held; when not, a broken edit is kept, saying so.

    Returns:
        The error to print, or None.
    """
    err: str | None = None
    try:
        data = read_toml_file(target)
        overlay = data.get("config", {})
        load_effective_with_overlay(Path.cwd(), overlay if isinstance(overlay, dict) else {})
        if "states" in data and _machine_is_valid(prior_text):
            load_machine(target)
    except ConfigError as exc:
        err = _leaf_problems(str(exc))
    except MachineError as exc:
        err = "; ".join(exc.problems)
    if err is None:
        return None
    return keep_or_rollback(target, prior_text, err, held=held)


def _warn_if_still_broken() -> None:
    """Warn when the config still fails to load after a kept write, naming the layer."""
    if (after := merged_config_error(Path.cwd())) is not None:
        warn(
            "the config is still invalid because of a value this edit did not"
            f" write; fix that one on its own:\n{after}"
        )


def _cmd_config_get(config_path: Path | None, key: str, *, machine: Path | None) -> int:
    """Print a leaf's effective value and the layer that set it.

    Args:
        config_path: The invocation's `--config`.
        key: The dotted leaf.
        machine: A machine file whose overlay goes on top.

    Returns:
        The exit code.
    """
    eff = _effective_with_overlay(config_path, machine)
    found = effective_leaf(eff, key)
    if found is None:
        error(_config_key_error(key, eff))
        return 2
    value, source = found
    print(f"{key} = {format_value(value)}  [{source}]")
    return 0


def _flag_shadow_note(key: str, config_path: Path | None) -> str | None:
    """Return a note when the `--config FILE` in force sets the key the write just landed under."""
    if config_path is None:
        return None
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return None
    if eff.sources.get(key) != "flag":
        return None
    return f"note: --config {config_path} overrides {key} while that flag is used."


def _cmd_config_set(
    key: str, value: str, *, repo: bool, machine: Path | None, config_path: Path | None = None
) -> int:
    """Set a scalar leaf in the target file.

    Args:
        key: The dotted leaf.
        value: The value as typed.
        repo: Write the in-repo config.
        machine: Write this machine file's overlay.
        config_path: The invocation's `--config`, for the shadow note.

    Returns:
        The exit code.
    """
    if err := _reject_machine_protected(key, machine):
        error(f"{err}")
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    parsed = parse_cli_value(value)
    if machine is None:
        err = set_config_value(Path.cwd(), key, value, to_repo=repo)
    else:
        _open_target(target)
        with writing_config(target) as held:
            prior = read_operator_file(target) if target.is_file() else None
            read_toml_file(target)  # no line surgery on a file that does not parse
            upsert_toml_leaf(target, prefix + key, parsed)
            err = _revalidate_machine(target, prior, held=held)
    if err:
        error(f"{err}")
        return 2
    if machine is None:
        _warn_if_still_broken()
    print(f"Set {key} = {format_value(parsed)} in {target}")
    if machine is None and (note := _flag_shadow_note(key, config_path)):
        print(note)
    return 0


def _config_key_error(key: str, eff: EffectiveConfig) -> str:
    """Return the error for a key that is a section or unknown, with its did-you-mean."""
    if any(candidate.startswith(key + ".") for candidate in resolved_config_values(eff.config)):
        return f"{key!r} is not a config leaf (see `agent6 config show`)."
    return unknown_key_error(key, Path.cwd(), eff=eff)


def _not_a_leaf(key: str, config_path: Path | None, machine: Path | None) -> str:
    """Return why the key cannot be unset; an MCP server table names the verb that removes it."""
    try:
        eff = _effective_with_overlay(config_path, machine)
    except ConfigError:
        return f"{key!r} is not a config leaf (see `agent6 config show`)."
    name = key.removeprefix("mcp.servers.")
    if name != key and "." not in name and name in eff.config.mcp.servers:
        return f"{key!r} is an MCP server entry; remove it with `agent6 mcp remove {name}`."
    return _config_key_error(key, eff)


def _cmd_config_unset(
    key: str, *, repo: bool, machine: Path | None, config_path: Path | None = None
) -> int:
    """Remove a leaf, so it reverts to the next layer or the built-in default.

    Args:
        key: The dotted leaf.
        repo: Edit the in-repo config.
        machine: Edit this machine file's overlay.
        config_path: The invocation's `--config`.

    Returns:
        The exit code.
    """
    if err := _reject_machine_protected(key, machine):
        error(f"{err}")
        return 2
    # An unset repairs a config that no longer loads, so a failing validation must not block it.
    try:
        known_leaf = effective_leaf(load_effective(Path.cwd(), config_path), key) is not None
    except ConfigError:
        known_leaf = True
    if not known_leaf:
        error(f"{_not_a_leaf(key, config_path, machine)}")
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    if not target.is_file():
        error(f"{target} does not exist; nothing to unset.")
        return 2
    if machine is None:
        res = unset_config_value(Path.cwd(), key, to_repo=repo)
        removed, err = res.removed, res.error
    else:
        with writing_config(target) as held:
            prior = read_operator_file(target)
            read_toml_file(target)  # no line surgery on a file that does not parse
            removed = remove_toml_leaf(target, prefix + key)
            err = _revalidate_machine(target, prior, held=held) if removed else None
    if err:
        error(f"unsetting {key} left an invalid config:\n{err}")
        return 2
    if not removed:
        print(f"{key} is not set in {target}; nothing to unset.")
        return 0
    if machine is None:
        _warn_if_still_broken()
    print(f"Unset {key} in {target}")
    if machine is None and (note := _flag_shadow_note(key, config_path)):
        print(note)
    return 0


def _config_list_edit(key: str, value: str, *, repo: bool, machine: Path | None, add: bool) -> int:
    """Add to or remove from a list leaf in the target file.

    Args:
        key: The dotted leaf.
        value: The entry.
        repo: Edit the in-repo config.
        machine: Edit this machine file's overlay.
        add: Add the entry; False removes it.

    Returns:
        The exit code.
    """
    if err := _reject_machine_protected(key, machine):
        error(f"{err}")
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    _open_target(target)
    # The lock spans from the read: two concurrent adds would otherwise drop one element.
    with writing_config(target) as held:
        current = read_toml_leaf(read_toml_file(target), prefix + key)
        if current is None:
            # A new override starts from the value it overrides, not [].
            try:
                base = (
                    load_effective(Path.cwd(), None)
                    if repo or machine is not None
                    else load_global_only()
                )
                inherited = effective_leaf(base, key)
            except ConfigError:
                inherited = None
            current = inherited[0] if inherited is not None and inherited[0] is not None else []
        if not isinstance(current, (list, tuple)):
            error(f"{key} is not a list field in {target}.")
            return 2
        parsed = parse_cli_value(value)
        items = list(current)
        if (parsed in items) == add:
            print(f"{format_value(parsed)} {'already' if add else 'not'} in {key}.")
            return 0
        items = [*items, parsed] if add else [x for x in items if x != parsed]
        prior = read_operator_file(target) if target.is_file() else None
        was_valid = machine is None and merged_config_error(Path.cwd()) is None
        upsert_toml_leaf(target, prefix + key, items)
        if machine is None:
            err = revalidate_write(
                Path.cwd(),
                target,
                prior,
                was_valid=was_valid,
                held=held,
                written=[(key, items)],
            )
        else:
            err = _revalidate_machine(target, prior, held=held)
        if err:
            error(f"{value!r} is not valid for {key}:\n{err}")
            return 2
    if machine is None:
        _warn_if_still_broken()
    verb, prep = ("Added", "to") if add else ("Removed", "from")
    print(f"{verb} {format_value(parsed)} {prep} {key} in {target}")
    return 0


def _entry_is_stale(entry: InvalidEntry) -> bool:
    """Return whether the entry's key no longer holds the value the unlocked diagnosis read."""
    try:
        data = read_toml_file(entry.path)
    except ConfigError:
        return True  # unreadable now: left to the loud paths
    # nan != nan, so a still-present nan would read as replaced on every pass and never be removed.
    return not _equal_tolerating_nan(read_toml_leaf(data, entry.file_key), entry.value)


def _equal_tolerating_nan(a: object, b: object) -> bool:
    """Return structural equality with NaN equal to NaN, recursing through dicts and lists."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal_tolerating_nan(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _equal_tolerating_nan(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def _cmd_config_fix(*, machine: Path | None) -> int:
    """Drop every invalid entry from the config, printing what it was and where it lived.

    Removing one entry can reveal another it shadowed, so the diagnosis repeats until
    the config is clean or nothing droppable remains. An entry the line surgery
    cannot reach is reported, never counted as removed.

    Args:
        machine: Fix this machine file's overlay instead.

    Returns:
        The exit code.
    """
    repo_root = Path.cwd()
    _read_machine_overlay(machine)
    removed: list[InvalidEntry] = []
    stuck: list[InvalidEntry] = []
    touched: set[Path] = set()
    diag = find_invalid_entries(repo_root, machine=machine)
    while diag.removable:
        progressed = False
        for entry in diag.removable:
            # The surgery publishes by rename, so it edits the file a link resolves to.
            target = resolved_write_path(entry.path)
            # Diagnosis ran unlocked, so a concurrent writer may have fixed this key since.
            with locked_file(target):
                if _entry_is_stale(entry):
                    continue
                try:
                    ok = (
                        remove_toml_table(target, entry.file_key)
                        if entry.is_table
                        else remove_toml_leaf(target, entry.file_key)
                    )
                except ConfigError:
                    # A leaf inside an inline table or dotted key: the surgery cannot carve it out.
                    ok = False
            if not ok:
                stuck.append(entry)
                continue
            progressed = True
            touched.add(target)
            removed.append(entry)
        if not progressed:
            # Nothing this pass could delete; re-diagnosing the same set would loop forever.
            break
        stuck = []
        diag = find_invalid_entries(repo_root, machine=machine)
    for path in touched:
        chown_to_real_user(path)
    for entry in removed:
        what = (
            f"[{entry.leaf}] (whole table)"
            if entry.is_table
            else f"{entry.leaf} = {format_value(entry.value)}"
        )
        print(f"Removed {what}  [{entry.layer}: {entry.path}]")
    if diag.blocked:
        error(
            f"config still invalid (not an auto-removable entry); fix it by hand:\n{diag.blocked}"
        )
        return 2
    if stuck:
        names = "\n".join(
            f"  {e.leaf} = {format_value(e.value)}  [{e.layer}: {e.path}]" for e in stuck
        )
        error(
            "config still invalid (flagged entries could not be auto-removed);"
            f" fix by hand:\n{names}"
        )
        return 2
    # Measured before claimed: a no-progress break lands here with the config still refused.
    final = find_invalid_entries(repo_root, machine=machine)
    if final.removable or final.blocked:
        error(
            "config still invalid (flagged entries changed under the lock"
            " this pass); re-run `agent6 config fix` or fix by hand."
        )
        return 2
    if not removed:
        print("Config is valid; nothing to fix.")
        return 0
    n = len(removed)
    print(f"Fixed the config: dropped {n} invalid entr{'y' if n == 1 else 'ies'}.")
    return 0
