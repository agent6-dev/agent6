# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config` subcommands (show/fill/path/get/set/unset/add/remove)."""

from __future__ import annotations

import difflib
import sys
import tempfile
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from agent6.config import (
    AnthropicProviderEntry,
    Config,
    ConfigError,
    OpenAIProviderEntry,
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
    InvalidEntry,
    effective_leaf,
    find_invalid_entries,
    flatten_leaves,
    leaf_keys,
    load_effective,
    load_effective_with_overlay,
    materialize,
    repo_config_path_for,
)
from agent6.machine import MachineError, load_machine
from agent6.models import registry as models_registry
from agent6.paths import (
    chown_to_real_user,
    effective_user,
    global_config_path,
    secrets_path,
)
from agent6.ui.cli._common import load_config_or_exit
from agent6.viewmodel.config_view import format_value, render_key_detail, render_show


def _cmd_config_show(config_path: Path | None, *, as_json: bool, key: str = "") -> int:
    eff = load_config_or_exit(Path.cwd(), config_path)
    if isinstance(eff, int):
        return eff
    resolved = models_registry.resolved_adaptive_values(eff.config)
    if key:
        # `config show <key>`: one leaf (or a whole section prefix), untruncated
        # (JSON mode filters to the same match set).
        detail = render_key_detail(
            eff, key, resolved=resolved, color=sys.stdout.isatty(), as_json=as_json
        )
        if detail is None:
            print(
                f"ERROR: no config key matches {key!r} (see `agent6 config show`).",
                file=sys.stderr,
            )
            return 2
        print(detail, end="")
        return 0
    text = render_show(eff, as_json=as_json, resolved=resolved, color=sys.stdout.isatty())
    print(text, end="")
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
    eff = load_config_or_exit(Path.cwd(), config_path)
    if isinstance(eff, int):
        return eff
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


def _restore_file(target: Path, text: str | None) -> None:
    """Put *target* back to *text*, or delete it when *text* is None."""
    if text is None:
        target.unlink(missing_ok=True)
    else:
        target.write_text(text, encoding="utf-8")


def _merged_config_error() -> str | None:
    """Validate the merged config as it sits on disk; the ConfigError message, or
    None when it is valid."""
    try:
        load_effective(Path.cwd(), None)
        return None
    except ConfigError as exc:
        return str(exc)


_PROVIDER_MEMBERS = (AnthropicProviderEntry, OpenAIProviderEntry)


def _provider_field_error(key: str, leaf: str, value: object) -> str | None:
    """Validate a ``providers.<name>.<leaf>`` write against the union members
    directly. The standalone minimal dict cannot carry the entry's
    discriminator, so union validation lands every error at the PARENT with
    pydantic-speak; the member models themselves give exact answers: a leaf on
    no member is an unknown key (with the members' own field pool as the
    did-you-mean universe), and a value every owning member rejects is invalid.
    None when some member accepts it (partial entries stay writable)."""
    fields = sorted({f for m in _PROVIDER_MEMBERS for f in m.model_fields})
    if leaf not in fields:
        close = difflib.get_close_matches(leaf, fields, n=2)
        hint = f". Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        return f"unknown provider key {key!r}{hint} (see `agent6 config show`)"
    errors: list[str] = []
    for member in _PROVIDER_MEMBERS:
        info = member.model_fields.get(leaf)
        if info is None:
            continue
        try:
            # rebuild_annotation carries the Field constraints (gt, pattern,
            # ...); the model's @field_validators do not travel with it, so
            # validator-only rejections still fall to the merged-dump path.
            TypeAdapter(info.rebuild_annotation()).validate_python(value)
            return None  # a member accepts it: the write stands
        except ValidationError as exc:
            errors.append(exc.errors()[0]["msg"])
    return f"{key}: {errors[0]}" if errors else None


def _unknown_key_error(key: str) -> str:
    """A human message for a key the schema forbids, with a did-you-mean.

    The pool is usually the SCHEMA defaults: this runs after the unknown key
    was already written, so the merged config no longer loads and the live
    branch (which would add real provider tables) only survives when a higher
    layer masks the write."""
    try:
        pool = leaf_keys(load_effective(Path.cwd(), None))
    except ConfigError:
        pool = sorted(flatten_leaves(Config().model_dump(mode="python")))
    close = difflib.get_close_matches(key, pool, n=2)
    hint = f". Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    return f"unknown config key {key!r}{hint} (see `agent6 config show`)"


def _written_value_error(key: str, value: object) -> str | None:
    """Validate the just-written ``key = value`` against the Config model on its
    own (a minimal dict, defaults for the rest), independent of the layer merge.
    A plain `config set` of an invalid value into a layer that a HIGHER layer
    masks (e.g. a global set the repo overlay shadows) would otherwise validate
    the merged config -- where the value is hidden -- and land the bad value in
    the file, only to explode later where the mask is absent. Rejects only when
    the error sits exactly at *key*, or at a parent of it for the schema's
    extra_forbidden (an unknown key or section), so a partial dynamic entry
    (a provider being filled in over several sets) is not falsely reverted."""
    if key == "profiles" or key.startswith("profiles."):
        # [profiles.*] is meta-config the loader strips BEFORE validation
        # (_apply_profile), so the Config schema forbids it by design; the
        # standalone check would falsely reject every legitimate profile write.
        # The merged re-validation still catches a profile body that breaks.
        return None
    parts = key.split(".")
    if parts[0] == "providers" and len(parts) == 3:
        return _provider_field_error(key, parts[2], value)
    nested: dict[str, object] = {}
    cur = nested
    for part in parts[:-1]:
        child: dict[str, object] = {}
        cur[part] = child
        cur = child
    cur[parts[-1]] = value
    try:
        Config.model_validate(nested)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            if err["type"] == "extra_forbidden" and (loc == key or key.startswith(loc + ".")):
                # An unknown top-level section errors at the SECTION (a parent
                # loc), not the leaf; both deserve the same friendly message,
                # not pydantic-speak or the merged-layer dump.
                return _unknown_key_error(key)
            if loc == key:
                msg = err["msg"]
                if err["type"] == "bool_parsing":
                    msg = f"expected true or false, got {value!r}"
                return f"{key}: {msg}"
    except ConfigError as exc:
        return str(exc)
    return None


def _revalidate_layered(
    target: Path,
    prior_text: str | None,
    *,
    was_valid: bool,
    written: tuple[str, object] | None = None,
) -> str | None:
    """Re-validate after a plain (global/repo) config write: revert only if this write
    BROKE a previously-valid config. If the config was already invalid -- a value left
    stale in a different, unedited layer -- keep the write and warn (a global set that
    the repo layer shadows hits this), so an already-broken config stays fixable through
    `config set` instead of the old catch-22 where one stale value blocked every write.

    *written* is the ``(key, value)`` a `config set` just wrote; its value is
    validated against the model even when a higher layer masks it in the merge.
    """
    if written is not None:
        value_err = _written_value_error(*written)
        if value_err is not None:
            _restore_file(target, prior_text)
            return value_err
    after = _merged_config_error()
    if after is None:
        return None  # valid -> success
    if was_valid:
        _restore_file(target, prior_text)  # the write broke a valid config -> fail loud
        return after
    print(
        "WARNING: the config is still invalid because of a value in another layer;"
        f" fix that one on its own:\n{after}",
        file=sys.stderr,
    )
    return None


def _revalidate_config(
    target: Path,
    prior_text: str | None,
    *,
    machine: Path | None,
    was_valid: bool = False,
    written: tuple[str, object] | None = None,
) -> str | None:
    """Re-validate the config after a write; restore *prior_text* on real failure.

    Returns a ready-to-print error message when THIS edit broke a previously-valid
    config (so the caller fails loud and the file is reverted), else None. A value
    left stale in a different, unedited layer never blocks an otherwise-valid write;
    *was_valid* is whether the merged config loaded before this write (see
    :func:`_revalidate_layered`). *written* is the ``(key, value)`` a `config set`
    wrote, checked standalone so a masked bad value is caught. The machine path
    keeps its own spec guard.
    """
    if machine is None:
        return _revalidate_layered(target, prior_text, was_valid=was_valid, written=written)
    err: str | None = None
    try:
        data = read_toml_file(target)
        overlay = data.get("config", {})
        load_effective_with_overlay(Path.cwd(), overlay if isinstance(overlay, dict) else {})
        # Validate the WHOLE machine spec too (not just the [config] overlay) so
        # `config set --machine-file` can't BREAK a runnable machine. Block only
        # when the edit made a previously-VALID machine invalid; a machine already
        # invalid (or a brand-new stub) is left for the author to finish.
        if "states" in data and _machine_is_valid(prior_text):
            load_machine(target)
    except ConfigError as exc:
        err = str(exc)
    except MachineError as exc:
        err = "; ".join(exc.problems)
    if err is not None:
        _restore_file(target, prior_text)
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
    was_valid = machine is None and _merged_config_error() is None  # loads BEFORE this write?
    try:
        upsert_toml_leaf(target, prefix + key, parsed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if err := _revalidate_config(
        target, prior, machine=machine, was_valid=was_valid, written=(key, parsed)
    ):
        print(f"ERROR: {err}", file=sys.stderr)
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
    was_valid = machine is None and _merged_config_error() is None  # loads BEFORE this unset?
    try:
        removed = remove_toml_leaf(target, prefix + key)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not removed:
        print(f"{key} is not set in {target}; nothing to unset.")
        return 0
    if err := _revalidate_config(target, prior, machine=machine, was_valid=was_valid):
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
    was_valid = machine is None and _merged_config_error() is None  # loads BEFORE this edit?
    try:
        upsert_toml_leaf(target, prefix + key, items)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if err := _revalidate_config(target, prior, machine=machine, was_valid=was_valid):
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


_MAX_FIX_PASSES = 25  # backstop; each pass drops >=1 leaf, so this is never reached


def _cmd_config_fix(*, machine: Path | None) -> int:
    """Drop every invalid entry from the config, printing what it was and where it
    lived (global / repo, or a machine's [config] overlay with --machine-file).

    Removing one entry can reveal another it shadowed, so it re-diagnoses until the
    config is clean or nothing droppable remains. An entry it cannot drop as a plain
    leaf (a non-absolute state_dir, a bad built-in default) is reported, not hidden.
    """
    repo_root = Path.cwd()
    removed: list[InvalidEntry] = []
    touched: set[Path] = set()
    diag = find_invalid_entries(repo_root, machine=machine)
    passes = 0
    while diag.removable and passes < _MAX_FIX_PASSES:
        for entry in diag.removable:
            if entry.is_table:
                remove_toml_table(entry.path, entry.file_key)
            else:
                remove_toml_leaf(entry.path, entry.file_key)
            touched.add(entry.path)
            removed.append(entry)
        passes += 1
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
        print(
            "ERROR: config still invalid (not an auto-removable entry); fix it by hand:\n"
            f"{diag.blocked}",
            file=sys.stderr,
        )
        return 2
    if not removed:
        print("Config is valid; nothing to fix.")
        return 0
    n = len(removed)
    print(f"Fixed the config: dropped {n} invalid entr{'y' if n == 1 else 'ies'}.")
    return 0
