# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 config` subcommands: show, fill, fix, path, presets, get, set, unset, add, remove."""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile

from agent6 import errors, paths, portable
from agent6.app import confine
from agent6.config import (
    ConfigError,
    io,
    layer,
    write,
)
from agent6.machine import (
    MachineError,
    load_machine,
    protected_overlay_error,
    protected_overlay_key_error,
)
from agent6.ui.cli import _common
from agent6.viewmodel import config_view


def _require_machine_file(machine: pathlib.Path | None) -> None:
    """Refuse a missing machine file instead of treating it as empty.

    Raises:
        OperatorError: The file does not exist.
    """
    if machine is not None and not machine.is_file():
        raise errors.OperatorError(f"no such machine file: {machine}")


def _read_machine_overlay(machine: pathlib.Path | None) -> dict[str, object] | None:
    """Read a machine file's `[config]` overlay.

    Returns:
        The overlay; None when no file is named.

    Raises:
        OperatorError: The file is missing or sets an operator-only key.
    """
    if machine is None:
        return None
    _require_machine_file(machine)
    overlay = io.read_toml_file(machine).get("config", {})
    if not isinstance(overlay, dict):
        return {}
    if problem := protected_overlay_error(overlay):
        raise errors.OperatorError(problem)
    return overlay


def _effective_with_overlay(
    config_path: pathlib.Path | None, machine: pathlib.Path | None
) -> layer.EffectiveConfig:
    """Return the effective config, with a machine file's `[config]` overlay on top when named."""
    overlay = _read_machine_overlay(machine)
    if overlay is None:
        return layer.load_effective(pathlib.Path.cwd(), config_path)
    return layer.load_effective_with_overlay(pathlib.Path.cwd(), overlay, explicit_path=config_path)


def _cmd_config_show(
    config_path: pathlib.Path | None,
    *,
    as_json: bool,
    keys: list[str] | None = None,
    descriptions: bool = False,
    machine: pathlib.Path | None = None,
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
    resolved = confine.resolved_config_values(eff.config)
    if keys:
        # An asked-for key is the one place a description can never bury the values.
        try:
            detail = config_view.render_key_detail(
                eff, keys, resolved=resolved, color=sys.stdout.isatty(), as_json=as_json
            )
        except KeyError as exc:
            _common.error(_config_key_error(str(exc.args[0]), eff))
            return 2
        print(detail, end="")
        return 0
    text = config_view.render_show(
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
    user = paths.effective_user()
    rows: list[tuple[str, pathlib.Path, bool]] = [
        ("global config", paths.global_config_path(user), True),
        ("repo config", paths.repo_config_path(pathlib.Path.cwd()), True),
        ("secrets", paths.secrets_path(user), True),
        ("config dir", paths.global_config_path(user).parent, False),
        ("state (all repos)", paths.state_base(user), False),
        ("state (this repo)", paths.state_dir(pathlib.Path.cwd()), False),
        ("skills", paths.data_dir(user) / "skills", False),
        ("cache", paths.cache_dir(user), False),
    ]
    width = max(len(label) for label, _p, _f in rows)
    for label, p, is_file in rows:
        present = p.is_file() if is_file else p.is_dir()
        print(f"{label:<{width}}: {p}{'' if present else '  (not present)'}")
    return 0


def _cmd_config_presets(config_path: pathlib.Path | None = None) -> int:
    """Print every known preset with the overrides it applies, the selected one marked.

    Returns:
        The exit code.
    """
    cat = layer.preset_catalog(pathlib.Path.cwd(), config_path)
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
        leaves = layer.flatten_leaves(info.overrides)
        if not leaves:
            print("  (plain defaults, no overrides)")
        for key, val in leaves.items():
            print(f"  {key} = {config_view.format_value(val)}")
    print(
        "\nSelect per run with --preset <name>;"
        " persist with `agent6 config set preset <name>` (--repo for this repo)."
    )
    return 0


def _open_target(target: pathlib.Path) -> None:
    """Create the config dir and hand it back to the real operator before any write under sudo."""
    paths.mkdir_for_real_user(target.parent)


def _cmd_config_fill(*, force: bool) -> int:
    """Write the defaults plus the global layer into the global config file.

    Never the repo layer and never a preset's effects.

    Args:
        force: Overwrite an existing file.

    Returns:
        The exit code.
    """
    target = write.resolved_write_path(paths.global_config_path())
    _open_target(target)
    # Read and publish under the lock: a read before it would lose a concurrent `config set`.
    with write.writing_config(target):
        eff = layer.load_global_only()
        if target.is_file() and not force:
            _common.error(f"{target} already exists. Re-run with --force to overwrite.")
            return 2
        portable.atomic_write(
            target,
            layer.materialize(eff.config, keep_presets_from=target, keep_preset_selector=True),
        )
    print(f"Wrote fully-resolved config to {target}")
    return 0


def _config_write_target(*, repo: bool, machine: pathlib.Path | None) -> tuple[pathlib.Path, str]:
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
            raise errors.OperatorError("use either --repo or --machine-file, not both")
        if machine.is_file():
            _read_machine_overlay(machine)
        return write.resolved_write_path(machine), "config."
    if repo:
        return write.resolved_write_path(paths.repo_config_path(pathlib.Path.cwd())), ""
    return write.resolved_write_path(paths.global_config_path()), ""


def _reject_machine_protected(key: str, machine: pathlib.Path | None) -> str | None:
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
            load_machine(pathlib.Path(tf.name))
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


def _revalidate_machine(
    target: pathlib.Path, prior_text: str | None, *, held: bool = True
) -> str | None:
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
        data = io.read_toml_file(target)
        overlay = data.get("config", {})
        layer.load_effective_with_overlay(
            pathlib.Path.cwd(), overlay if isinstance(overlay, dict) else {}
        )
        if "states" in data and _machine_is_valid(prior_text):
            load_machine(target)
    except ConfigError as exc:
        err = _leaf_problems(str(exc))
    except MachineError as exc:
        err = "; ".join(exc.problems)
    if err is None:
        return None
    return write.keep_or_rollback(target, prior_text, err, held=held)


def _warn_if_still_broken() -> None:
    """Warn when the config still fails to load after a kept write, naming the layer."""
    if (after := write.merged_config_error(pathlib.Path.cwd())) is not None:
        _common.warn(
            "the config is still invalid because of a value this edit did not"
            f" write; fix that one on its own:\n{after}"
        )


def _cmd_config_get(
    config_path: pathlib.Path | None, key: str, *, machine: pathlib.Path | None
) -> int:
    """Print a leaf's effective value and the layer that set it.

    Args:
        config_path: The invocation's `--config`.
        key: The dotted leaf.
        machine: A machine file whose overlay goes on top.

    Returns:
        The exit code.
    """
    eff = _effective_with_overlay(config_path, machine)
    found = layer.effective_leaf(eff, key)
    if found is None:
        _common.error(_config_key_error(key, eff))
        return 2
    value, source = found
    print(f"{key} = {config_view.format_value(value)}  [{source}]")
    return 0


def _flag_shadow_note(key: str, config_path: pathlib.Path | None) -> str | None:
    """Return a note when the `--config FILE` in force sets the key the write just landed under."""
    if config_path is None:
        return None
    try:
        eff = layer.load_effective(pathlib.Path.cwd(), config_path)
    except ConfigError:
        return None
    if eff.sources.get(key) != "flag":
        return None
    return f"note: --config {config_path} overrides {key} while that flag is used."


def _cmd_config_set(
    key: str,
    value: str,
    *,
    repo: bool,
    machine: pathlib.Path | None,
    config_path: pathlib.Path | None = None,
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
        _common.error(f"{err}")
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    parsed = io.parse_cli_value(value)
    if machine is None:
        err = write.set_config_value(pathlib.Path.cwd(), key, value, to_repo=repo)
    else:
        _open_target(target)
        with write.writing_config(target) as held:
            prior = errors.read_operator_file(target) if target.is_file() else None
            io.read_toml_file(target)  # no line surgery on a file that does not parse
            io.upsert_toml_leaf(target, prefix + key, parsed)
            err = _revalidate_machine(target, prior, held=held)
    if err:
        _common.error(f"{err}")
        return 2
    if machine is None:
        _warn_if_still_broken()
    print(f"Set {key} = {config_view.format_value(parsed)} in {target}")
    if machine is None and (note := _flag_shadow_note(key, config_path)):
        print(note)
    return 0


def _config_key_error(key: str, eff: layer.EffectiveConfig) -> str:
    """Return the error for a key that is a section or unknown, with its did-you-mean."""
    if any(
        candidate.startswith(key + ".") for candidate in confine.resolved_config_values(eff.config)
    ):
        return f"{key!r} is not a config leaf (see `agent6 config show`)."
    return write.unknown_key_error(key, pathlib.Path.cwd(), eff=eff)


def _not_a_leaf(key: str, config_path: pathlib.Path | None, machine: pathlib.Path | None) -> str:
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
    key: str, *, repo: bool, machine: pathlib.Path | None, config_path: pathlib.Path | None = None
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
        _common.error(f"{err}")
        return 2
    # An unset repairs a config that no longer loads, so a failing validation must not block it.
    try:
        known_leaf = (
            layer.effective_leaf(layer.load_effective(pathlib.Path.cwd(), config_path), key)
            is not None
        )
    except ConfigError:
        known_leaf = True
    if not known_leaf:
        _common.error(f"{_not_a_leaf(key, config_path, machine)}")
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    if not target.is_file():
        _common.error(f"{target} does not exist; nothing to unset.")
        return 2
    if machine is None:
        res = write.unset_config_value(pathlib.Path.cwd(), key, to_repo=repo)
        removed, err = res.removed, res.error
    else:
        with write.writing_config(target) as held:
            prior = errors.read_operator_file(target)
            io.read_toml_file(target)  # no line surgery on a file that does not parse
            removed = io.remove_toml_leaf(target, prefix + key)
            err = _revalidate_machine(target, prior, held=held) if removed else None
    if err:
        _common.error(f"unsetting {key} left an invalid config:\n{err}")
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


def _config_list_edit(
    key: str, value: str, *, repo: bool, machine: pathlib.Path | None, add: bool
) -> int:
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
        _common.error(f"{err}")
        return 2
    target, prefix = _config_write_target(repo=repo, machine=machine)
    _open_target(target)
    # The lock spans from the read: two concurrent adds would otherwise drop one element.
    with write.writing_config(target) as held:
        current = io.read_toml_leaf(io.read_toml_file(target), prefix + key)
        if current is None:
            # A new override starts from the value it overrides, not [].
            try:
                base = (
                    layer.load_effective(pathlib.Path.cwd(), None)
                    if repo or machine is not None
                    else layer.load_global_only()
                )
                inherited = layer.effective_leaf(base, key)
            except ConfigError:
                inherited = None
            current = inherited[0] if inherited is not None and inherited[0] is not None else []
        if not isinstance(current, (list, tuple)):
            _common.error(f"{key} is not a list field in {target}.")
            return 2
        parsed = io.parse_cli_value(value)
        items = list(current)
        if (parsed in items) == add:
            print(f"{config_view.format_value(parsed)} {'already' if add else 'not'} in {key}.")
            return 0
        items = [*items, parsed] if add else [x for x in items if x != parsed]
        prior = errors.read_operator_file(target) if target.is_file() else None
        was_valid = machine is None and write.merged_config_error(pathlib.Path.cwd()) is None
        io.upsert_toml_leaf(target, prefix + key, items)
        if machine is None:
            err = write.revalidate_write(
                pathlib.Path.cwd(),
                target,
                prior,
                was_valid=was_valid,
                held=held,
                written=[(key, items)],
            )
        else:
            err = _revalidate_machine(target, prior, held=held)
        if err:
            _common.error(f"{value!r} is not valid for {key}:\n{err}")
            return 2
    if machine is None:
        _warn_if_still_broken()
    verb, prep = ("Added", "to") if add else ("Removed", "from")
    print(f"{verb} {config_view.format_value(parsed)} {prep} {key} in {target}")
    return 0


def _entry_is_stale(entry: layer.InvalidEntry) -> bool:
    """Return whether the entry's key no longer holds the value the unlocked diagnosis read."""
    try:
        data = io.read_toml_file(entry.path)
    except ConfigError:
        return True  # unreadable now: left to the loud paths
    # nan != nan, so a still-present nan would read as replaced on every pass and never be removed.
    return not _equal_tolerating_nan(io.read_toml_leaf(data, entry.file_key), entry.value)


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


def _cmd_config_fix(*, machine: pathlib.Path | None) -> int:
    """Drop every invalid entry from the config, printing what it was and where it lived.

    Removing one entry can reveal another it shadowed, so the diagnosis repeats until
    the config is clean or nothing droppable remains. An entry the line surgery
    cannot reach is reported, never counted as removed.

    Args:
        machine: Fix this machine file's overlay instead.

    Returns:
        The exit code.
    """
    repo_root = pathlib.Path.cwd()
    _read_machine_overlay(machine)
    removed: list[layer.InvalidEntry] = []
    stuck: list[layer.InvalidEntry] = []
    touched: set[pathlib.Path] = set()
    diag = layer.find_invalid_entries(repo_root, machine=machine)
    while diag.removable:
        progressed = False
        for entry in diag.removable:
            # The surgery publishes by rename, so it edits the file a link resolves to.
            target = write.resolved_write_path(entry.path)
            # Diagnosis ran unlocked, so a concurrent writer may have fixed this key since.
            with portable.locked_file(target):
                if _entry_is_stale(entry):
                    continue
                try:
                    ok = (
                        io.remove_toml_table(target, entry.file_key)
                        if entry.is_table
                        else io.remove_toml_leaf(target, entry.file_key)
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
        diag = layer.find_invalid_entries(repo_root, machine=machine)
    for path in touched:
        paths.chown_to_real_user(path)
    for entry in removed:
        what = (
            f"[{entry.leaf}] (whole table)"
            if entry.is_table
            else f"{entry.leaf} = {config_view.format_value(entry.value)}"
        )
        print(f"Removed {what}  [{entry.layer}: {entry.path}]")
    if diag.blocked:
        _common.error(
            f"config still invalid (not an auto-removable entry); fix it by hand:\n{diag.blocked}"
        )
        return 2
    if stuck:
        names = "\n".join(
            f"  {e.leaf} = {config_view.format_value(e.value)}  [{e.layer}: {e.path}]"
            for e in stuck
        )
        _common.error(
            "config still invalid (flagged entries could not be auto-removed);"
            f" fix by hand:\n{names}"
        )
        return 2
    # Measured before claimed: a no-progress break lands here with the config still refused.
    final = layer.find_invalid_entries(repo_root, machine=machine)
    if final.removable or final.blocked:
        _common.error(
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
