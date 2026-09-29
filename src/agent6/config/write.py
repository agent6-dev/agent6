# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one config write cycle every editor uses.

A writer holds the config lock, refuses line surgery on a file that does not parse, writes
through the surgery in `io`, validates the written value standalone, then revalidates the
merged config and rolls back when this edit broke it. It raises `OperatorError` when the
edit cannot be attempted and returns an error string only when a landed edit failed
revalidation (rolled back, or kept when the lock failed open).
"""

from __future__ import annotations

import contextlib
import difflib
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from agent6.config._providers import Deployment, ProviderEntry
from agent6.config.io import (
    ConfigLeafValue,
    parse_cli_value,
    read_toml_file,
    remove_toml_leaf,
    remove_toml_table,
    upsert_toml_leaf,
    upsert_toml_table,
)
from agent6.config.layer import (
    EffectiveConfig,
    flatten_leaves,
    leaf_keys,
    load_effective,
)
from agent6.config.model import Config, ConfigError
from agent6.errors import OperatorError, read_operator_file
from agent6.paths import (
    chown_to_real_user,
    effective_user,
    global_config_path,
    mkdir_for_real_user,
    repo_config_path,
)
from agent6.portable import atomic_write, locked_file


def resolved_write_path(target: Path) -> Path:
    """Resolve a symlinked config path to the file a write must open.

    `atomic_write` publishes by rename, which would replace a dotfiles-managed symlink with
    a plain file. The link is followed only to a target the real operator owns, so a `sudo`
    write cannot be redirected into a root-owned file; every writer resolves here.

    Args:
        target: The config path.

    Returns:
        The path itself, or the link's target.

    Raises:
        OperatorError: The target, or the nearest existing directory of a target yet to be
            created, is unreadable or owned by another user.
    """
    if not target.is_symlink():
        return target
    resolved = target.resolve()
    owner = effective_user().uid
    # A dotfiles link often precedes its file; the ownership check moves to the nearest dir.
    checked = resolved
    while not checked.exists() and checked != checked.parent:
        checked = checked.parent
    try:
        checked_uid = checked.stat().st_uid
    except OSError as exc:
        raise OperatorError(f"config symlink {target} -> {resolved} is unreadable: {exc}") from exc
    if checked_uid != owner:
        whose = "" if checked == resolved else f" (its directory {checked})"
        raise OperatorError(
            f"config {target} is a symlink to {resolved}{whose}, owned by uid {checked_uid},"
            f" not you (uid {owner}); agent6 will not write through it"
        )
    return resolved


def _write_target(repo_root: Path, *, to_repo: bool) -> Path:
    """Return the layer's config file, resolved.

    Args:
        repo_root: The repo.
        to_repo: The repo layer instead of the global one.

    Returns:
        The file to write.
    """
    return resolved_write_path(repo_config_path(repo_root) if to_repo else global_config_path())


def _prepare_write_target(repo_root: Path, *, to_repo: bool) -> Path:
    """Return the layer's config file with its directory created for the real operator.

    Under `sudo` the handover is at creation, so a killed write never strands a root-owned
    dir a later non-root write cannot create its temp file in.

    Args:
        repo_root: The repo.
        to_repo: The repo layer instead of the global one.

    Returns:
        The file to write.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    mkdir_for_real_user(target.parent)
    return target


@contextlib.contextmanager
def writing_config(target: Path) -> Generator[bool]:
    """Hold the config write lock, handing the file back to the real operator on every exit.

    Under `sudo` every publish creates the file as root, so the handover is unconditional.

    Args:
        target: The config file.

    Yields:
        Whether the lock is held, for `keep_or_rollback`.
    """
    with locked_file(target) as held:
        try:
            yield held
        finally:
            chown_to_real_user(target)


def target_unparseable(target: Path) -> bool:
    """Return whether the file itself is no longer valid TOML.

    Args:
        target: The config file.

    Returns:
        True when it exists and does not parse.
    """
    try:
        read_toml_file(target)
    except ConfigError:
        return True
    return False


def merged_config_error(repo_root: Path) -> str | None:
    """Return the merged config's load error as it sits on disk, or None.

    Measured before a write, so `revalidate_write` tells this edit's breakage from an older
    one elsewhere.

    Args:
        repo_root: The repo.

    Returns:
        The error message, or None when the config loads.
    """
    try:
        load_effective(repo_root, None)
    except ConfigError as exc:
        return str(exc)
    return None


# Without the lock a snapshot restore could erase a concurrent writer's update.
_KEPT_NO_LOCK = (
    "(kept as written: the config lock could not be taken, so an automatic"
    " rollback might erase a concurrent edit; undo by hand or run `agent6 config fix`)"
)


def keep_or_rollback(target: Path, prior: str | None, err: str, *, held: bool) -> str:
    """Roll the file back to its prior text and hand the error back.

    Args:
        target: The config file.
        prior: The text before the edit; None deletes the file.
        err: The revalidation error.
        held: Whether the lock is held; without it the write is kept, since a restore could
            erase a concurrent writer's update.

    Returns:
        The error, with the kept-as-written note when the lock failed open.
    """
    if not held:
        return f"{err}\n{_KEPT_NO_LOCK}"
    if prior is None:
        target.unlink(missing_ok=True)
    else:
        atomic_write(target, prior)
    return err


# Derived: a hand-listed copy would leave a new entry type validated by nothing.
PROVIDER_MEMBERS: tuple[type[BaseModel], ...] = get_args(get_args(ProviderEntry)[0])


def provider_field_error(key: str, leaf: str, value: object) -> str | None:
    """Validate a `providers.<name>.<leaf>` write against the union members directly.

    A minimal standalone dict lacks the entry's discriminator, so each member is tried with
    its own `api_format` seeded; a partial entry stays writable.

    Args:
        key: The dotted key, for the message.
        leaf: The field name.
        value: The value.

    Returns:
        An error for a leaf no member has (with a did-you-mean) or a value every owning
        member rejects, every member's complaint de-duplicated; None when a member accepts it.
    """
    fields = sorted({f for m in PROVIDER_MEMBERS for f in m.model_fields})
    if leaf not in fields:
        close = difflib.get_close_matches(leaf, fields, n=2)
        hint = f". Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        return f"unknown provider key {key!r}{hint} (see `agent6 config show`)"
    errors: list[str] = []
    for member in PROVIDER_MEMBERS:
        if leaf not in member.model_fields:
            continue
        fmt = get_args(member.model_fields["api_format"].annotation)[0]
        try:
            member.model_validate({"api_format": fmt, leaf: value})
            return None
        except ValidationError as exc:
            # Only an error at the leaf or inside its value counts; a missing sibling does not.
            leaf_errs = [e["msg"] for e in exc.errors() if e["loc"] and e["loc"][0] == leaf]
            if not leaf_errs:
                return None
            errors.append(leaf_errs[0])
    if not errors:
        return None
    seen = list(dict.fromkeys(errors))
    return f"{key}: {' / '.join(seen)}"


def unknown_key_error(key: str, repo_root: Path, *, eff: EffectiveConfig | None = None) -> str:
    """Return the message for a key the schema forbids, with a did-you-mean.

    Args:
        key: The dotted key.
        repo_root: The repo whose effective leaves are the did-you-mean pool.
        eff: An effective config to take the pool from instead, when the caller holds one
            whose `--config` or `--machine-file` layers add keys.

    Returns:
        The message; the pool falls back to the schema defaults when the merged config no
        longer loads.
    """
    try:
        pool = leaf_keys(eff if eff is not None else load_effective(repo_root, None))
    except ConfigError:
        pool = sorted(flatten_leaves(Config().model_dump(mode="python")))
    close = difflib.get_close_matches(key, pool, n=2)
    hint = f". Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    return f"unknown config key {key!r}{hint} (see `agent6 config show`)"


def _section_leaves(doc: dict[str, Any], key: str) -> dict[str, Any]:
    """Return the scalar leaves of a key's own section as the document holds them.

    A cross-leaf rule is a `model_validator` on the section, so the standalone check needs
    the siblings; a nested table validates on its own and is dropped.

    Args:
        doc: The parsed file.
        key: The dotted key.

    Returns:
        The siblings by name; {} when the section is absent.
    """
    parts = key.split(".")[:-1]
    cur: Any = doc
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return {}
        cur = cur[part]
    if not isinstance(cur, dict):
        return {}
    return {k: v for k, v in cur.items() if not isinstance(v, dict)}


def written_value_error(
    key: str, value: object, *, repo_root: Path, section: dict[str, Any] | None = None
) -> str | None:
    """Validate a written `key = value` on its own, independent of the layer merge.

    A higher layer can mask an invalid value in the merge, so it would land in the file and
    explode where the mask is absent. The one owner every writer uses.

    Args:
        key: The dotted key.
        value: The value written.
        repo_root: The repo, for the did-you-mean pool.
        section: The section's sibling leaves, so a rule spanning two keys sees them.

    Returns:
        An error at the key, under it, or at a parent of it; None when the value is
        acceptable or the only complaint is a missing child, which means the written
        container is partial and the merged revalidation still catches a genuine absence.
    """
    parts = key.split(".")
    if parts[0] == "presets":
        # The schema forbids the table itself; a preset's leaf validates as the Config leaf.
        if len(parts) > 2:
            return written_value_error(
                ".".join(parts[2:]), value, repo_root=repo_root, section=section
            )
        return None
    if parts[0] == "providers" and len(parts) == 3:
        return provider_field_error(key, parts[2], value)
    nested: dict[str, object] = {}
    cur = nested
    for part in parts[:-1]:
        child: dict[str, object] = dict(section or {}) if part == parts[-2] else {}
        cur[part] = child
        cur = child
    cur[parts[-1]] = value
    try:
        Config.model_validate(nested)
    except ValidationError as exc:
        for err in exc.errors():
            message = _error_about(err, key, value, repo_root)
            if message is not None:
                return message
    except ConfigError as exc:
        return str(exc)
    return None


def _error_about(err: ErrorDetails, key: str, value: object, repo_root: Path) -> str | None:
    """Render one validation error as a message about the key.

    Args:
        err: The error.
        key: The dotted key written.
        value: The value written.
        repo_root: The repo, for the did-you-mean pool.

    Returns:
        The message, or None.
    """
    loc = ".".join(str(x) for x in err["loc"])
    if err["type"] == "extra_forbidden" and (loc == key or key.startswith(loc + ".")):
        # An unknown top-level section errors at the section, not the leaf.
        return unknown_key_error(key, repo_root)
    if err["type"] == "value_error" and "." not in loc and key.startswith(loc + "."):
        # A section rule is reported at the section; only a top-level one, since a name-keyed
        # entry is written a leaf at a time and its whole-entry rules would reject each write.
        return f"{key}: {err['msg']}"
    if loc != key and not loc.startswith(key + "."):
        return None
    if err["type"] == "missing":
        # The written container is partial; the merged revalidation catches a genuine absence.
        return None
    if err["type"] in ("bool_parsing", "bool_type"):
        detail = f"expected true or false, got {value!r}"
    elif err["type"] in ("tuple_type", "list_type"):
        detail = f'expected a list: config set {key} \'["a", "b"]\', or config add {key} a'
    else:
        detail = err["msg"]
    return f"{key}: {detail}"


def revalidate_write(
    repo_root: Path,
    target: Path,
    prior: str | None,
    *,
    was_valid: bool,
    held: bool = True,
    written: Sequence[tuple[str, object]] = (),
) -> str | None:
    """Reload the merged config after an edit and roll the file back when this edit broke it.

    A config that was already invalid keeps the edit, or a stale value in an unedited layer
    would refuse every write; `agent6 config fix` removes it. The caller holds
    `writing_config` across the whole cycle.

    Args:
        repo_root: The repo.
        target: The file edited.
        prior: Its text before the edit; None for a file this edit created.
        was_valid: Whether the merged config loaded before the edit.
        held: Whether the lock is held; without it the edit is kept and the error says so.
        written: The `(key, value)` pairs this edit wrote, each validated against the
            section as the file now holds it, so a value a higher layer masks in the merge
            is caught here.

    Returns:
        The error when this edit broke the config, else None.
    """
    try:
        doc = read_toml_file(target)
    except ConfigError as exc:
        # Unparseable TOML is always this edit's doing; a raise would escape the rollback.
        return keep_or_rollback(target, prior, str(exc), held=held)
    for wkey, wvalue in written:
        section = _section_leaves(doc, wkey)
        value_err = written_value_error(wkey, wvalue, repo_root=repo_root, section=section)
        if value_err is None:
            continue
        # A section rule can fire over a sibling that was already wrong; the value alone
        # settles whose fault it is, and the merged check below keeps a pre-broken config.
        if written_value_error(wkey, wvalue, repo_root=repo_root) is not None:
            return keep_or_rollback(target, prior, value_err, held=held)
    err = merged_config_error(repo_root)
    if err is None:
        return None
    # A target that no longer parses is always this write's doing, never another layer's.
    if not was_valid and not target_unparseable(target):
        return None  # broken before this edit; not ours to refuse
    return keep_or_rollback(target, prior, err, held=held)


def _models_at(model: type[BaseModel], part: str) -> tuple[type[BaseModel], ...] | None:
    """Return the models a field resolves to under a model.

    A name-keyed table (`providers`, `mcp.servers`) resolves through its value type; an
    optional section (`models.worker`) is a section, since written as a leaf it would
    collide with its own `[table]`.

    Args:
        model: The parent model.
        part: The field name.

    Returns:
        The member models, or None when the field is a leaf or unknown.
    """
    field = model.model_fields.get(part)
    if field is None:
        return None
    annotation = field.annotation
    if get_origin(annotation) is dict:
        return _model_members(get_args(annotation)[1]) or None
    return _model_members(annotation) or None


def _model_members(annotation: object) -> tuple[type[BaseModel], ...]:
    """Return every BaseModel in an annotation, unwrapping `Annotated` and unions.

    Args:
        annotation: The field annotation.

    Returns:
        The models, in declaration order.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)
    args = get_args(annotation)
    return tuple(m for arg in args for m in _model_members(arg))


def names_a_section(dotted_key: str) -> bool:
    """Return whether a dotted key names a `[table]` in the schema rather than a leaf.

    A section is written leaf by leaf so its siblings survive; a dict-typed leaf
    (`providers.<name>.extra_body`) is one value. A name-keyed table is a section at both
    levels: written whole, `providers` would replace every provider the operator has.

    Args:
        dotted_key: The key.

    Returns:
        True for a section.
    """
    models: tuple[type[BaseModel], ...] = (Config,)
    keyed = False  # the previous part was a name-keyed table, so this part is a name
    for part in dotted_key.split("."):
        if keyed:
            keyed = False
            continue
        resolved = [m for model in models for m in (_models_at(model, part) or ())]
        if not resolved:
            return False
        keyed = any(
            get_origin(model.model_fields[part].annotation) is dict
            for model in models
            if part in model.model_fields
        )
        models = tuple(resolved)
    return True


def set_config_value(
    repo_root: Path, dotted_key: str, raw_value: str, *, to_repo: bool = False
) -> str | None:
    """Set one leaf in the global or repo config.

    A section's own value (`config set context '{ a = 1, b = 2 }'`) is written per leaf so
    its siblings and comments survive; a dict-typed leaf is one value and is replaced.

    Args:
        repo_root: The repo.
        dotted_key: The key.
        raw_value: The value as a CLI would pass it, parsed like `config set`.
        to_repo: The repo layer instead of the global one.

    Returns:
        The error when the edit produced an invalid config (rolled back), else None.

    Raises:
        ConfigError: The target does not parse, the surgery refuses the key, or a section
            value is empty.
        OperatorError: The target cannot be written.
    """
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target) as held:
        prior = read_operator_file(target) if target.is_file() else None
        read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        parsed = parse_cli_value(raw_value)
        if isinstance(parsed, dict) and names_a_section(dotted_key):
            if not parsed:
                raise ConfigError(f"{dotted_key} = {{}} sets nothing; name the leaves to set")
            try:
                for leaf, val in parsed.items():
                    upsert_toml_leaf(target, f"{dotted_key}.{leaf}", val)
            except ConfigError as exc:
                # A refusal mid-way leaves earlier leaves on disk.
                return keep_or_rollback(target, prior, str(exc), held=held)
            written = [(f"{dotted_key}.{leaf}", v) for leaf, v in parsed.items()]
        else:
            upsert_toml_leaf(target, dotted_key, parsed)
            written = [(dotted_key, parsed)]
        return revalidate_write(
            repo_root, target, prior, was_valid=was_valid, held=held, written=written
        )


def set_config_table(
    repo_root: Path,
    table: str,
    fields: dict[str, ConfigLeafValue],
    *,
    to_repo: bool = False,
) -> str | None:
    """Insert or replace a whole `[table]` block, as `agent6 model` writes each role.

    Args:
        repo_root: The repo.
        table: The table name.
        fields: The leaves; a None value is omitted.
        to_repo: The repo layer instead of the global one.

    Returns:
        The error when the edit produced an invalid config (rolled back), else None.

    Raises:
        ConfigError: The target does not parse, or a value has no TOML form.
        OperatorError: The target cannot be written.
    """
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target) as held:
        prior = read_operator_file(target) if target.is_file() else None
        read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        upsert_toml_table(target, table, fields)
        return revalidate_write(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            # Per leaf: a whole-table pair would hide every leaf-level error.
            written=[(f"{table}.{k}", v) for k, v in fields.items() if v is not None],
        )


def provider_choices() -> dict[str, list[str]]:
    """Return the add-provider form's fixed choices, read from the schema so they never drift.

    Returns:
        The `api_format` and `deployment` values.
    """
    formats: list[str] = []
    for model in PROVIDER_MEMBERS:
        formats.extend(get_args(model.model_fields["api_format"].annotation))
    return {"api_format": formats, "deployment": list(get_args(Deployment))}


# The well-known names `agent6 connect` and the add-provider form land on the right host;
# `_default_base_url` knows only api.openai.com for the `openai` format.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {"api_format": "anthropic"},
    "chatgpt": {"api_format": "chatgpt"},
    "claude": {"api_format": "claude_code"},
    "openai": {"api_format": "openai", "base_url": "https://api.openai.com/v1"},
    "openrouter": {"api_format": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "ollama": {"api_format": "openai", "base_url": "http://localhost:11434/v1"},
}


def set_config_leaves(
    repo_root: Path,
    table: str,
    fields: dict[str, ConfigLeafValue],
    *,
    to_repo: bool = False,
) -> str | None:
    """Upsert individual `[table]` leaves, preserving sibling keys and comments.

    The update counterpart of `set_config_table`; one revalidation wraps every leaf write.

    Args:
        repo_root: The repo.
        table: The table name.
        fields: The leaves; a None value is omitted.
        to_repo: The repo layer instead of the global one.

    Returns:
        The error when the edit produced an invalid config (rolled back), else None.

    Raises:
        ConfigError: The target does not parse, or the surgery refuses a leaf (the file is
            rolled back first).
        OperatorError: The target cannot be written.
    """
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target) as held:
        prior = read_operator_file(target) if target.is_file() else None
        read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        try:
            for key, val in fields.items():
                if val is not None:
                    upsert_toml_leaf(target, f"{table}.{key}", val)
        except ConfigError as exc:
            # Earlier leaves may already have landed.
            raise ConfigError(keep_or_rollback(target, prior, str(exc), held=held)) from exc
        return revalidate_write(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            written=[(f"{table}.{k}", v) for k, v in fields.items() if v is not None],
        )


@dataclass(frozen=True, slots=True)
class UnsetResult:
    """How an unset ended.

    Attributes:
        removed: Whether anything was removed.
        error: The revalidation error when the removal broke the config (rolled back, or
            kept without the lock), else None.
    """

    removed: bool
    error: str | None = None


def unset_config_table(repo_root: Path, table: str, *, to_repo: bool = False) -> UnsetResult:
    """Remove a whole `[table]` with its subtables.

    For a name-keyed entry that is only valid whole: dropping one key of a
    `[mcp.servers.<name>]` leaves an invalid config, so the entry goes as a unit.

    Args:
        repo_root: The repo.
        table: The table name.
        to_repo: The repo layer instead of the global one.

    Returns:
        Whether the table was present, and the revalidation error if removing it broke the
        config.

    Raises:
        ConfigError: The target does not parse.
        OperatorError: The target cannot be written.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return UnsetResult(removed=False)
    with writing_config(target) as held:
        prior = read_operator_file(target)
        read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        if not remove_toml_table(target, table):
            return UnsetResult(removed=False)
        return UnsetResult(
            removed=True,
            error=revalidate_write(repo_root, target, prior, was_valid=was_valid, held=held),
        )


def unset_config_value(repo_root: Path, dotted_key: str, *, to_repo: bool = False) -> UnsetResult:
    """Remove one leaf, so it reverts to the next layer or the built-in default.

    Args:
        repo_root: The repo.
        dotted_key: The key.
        to_repo: The repo layer instead of the global one.

    Returns:
        Whether the leaf was set in the file, and the revalidation error if removing it
        broke the config.

    Raises:
        ConfigError: The target does not parse, or the key's ancestor is not a plain table.
        OperatorError: The target cannot be written.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return UnsetResult(removed=False)
    with writing_config(target) as held:
        prior = read_operator_file(target)
        read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        if not remove_toml_leaf(target, dotted_key):
            return UnsetResult(removed=False)
        return UnsetResult(
            removed=True,
            error=revalidate_write(repo_root, target, prior, was_valid=was_valid, held=held),
        )
