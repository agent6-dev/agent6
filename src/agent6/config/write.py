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
import dataclasses
import difflib
import pathlib
import shlex
from collections.abc import Generator, Sequence
from typing import Any, Literal, cast, get_args, get_origin

import pydantic
import pydantic.fields
import pydantic_core

from agent6 import errors as agent6_errors
from agent6 import paths, portable
from agent6.config import _base, _harness, _providers, io, layer
from agent6.config import model as config_model


def resolved_write_path(target: pathlib.Path) -> pathlib.Path:
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
    owner = paths.effective_user().uid
    # A dotfiles link often precedes its file; the ownership check moves to the nearest dir.
    checked = resolved
    while not checked.exists() and checked != checked.parent:
        checked = checked.parent
    try:
        checked_uid = checked.stat().st_uid
    except OSError as exc:
        raise agent6_errors.OperatorError(
            f"config symlink {target} -> {resolved} is unreadable: {exc}"
        ) from exc
    if checked_uid != owner:
        whose = "" if checked == resolved else f" (its directory {checked})"
        raise agent6_errors.OperatorError(
            f"config {target} is a symlink to {resolved}{whose}, owned by uid {checked_uid},"
            f" not you (uid {owner}); agent6 will not write through it"
        )
    return resolved


def _write_target(repo_root: pathlib.Path, *, to_repo: bool) -> pathlib.Path:
    """Return the layer's config file, resolved.

    Args:
        repo_root: The repo.
        to_repo: The repo layer instead of the global one.

    Returns:
        The file to write.
    """
    return resolved_write_path(
        paths.repo_config_path(repo_root) if to_repo else paths.global_config_path()
    )


def _prepare_write_target(repo_root: pathlib.Path, *, to_repo: bool) -> pathlib.Path:
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
    paths.mkdir_for_real_user(target.parent)
    return target


@contextlib.contextmanager
def writing_config(target: pathlib.Path) -> Generator[bool]:
    """Hold the config write lock, handing the file back to the real operator on every exit.

    Under `sudo` every publish creates the file as root, so the handover is unconditional.

    Args:
        target: The config file.

    Yields:
        Whether the lock is held, for `keep_or_rollback`.
    """
    with portable.locked_file(target) as held:
        try:
            yield held
        finally:
            paths.chown_to_real_user(target)


def target_unparseable(target: pathlib.Path) -> bool:
    """Return whether the file itself is no longer valid TOML.

    Args:
        target: The config file.

    Returns:
        True when it exists and does not parse.
    """
    try:
        io.read_toml_file(target)
    except config_model.ConfigError:
        return True
    return False


def merged_config_error(repo_root: pathlib.Path) -> str | None:
    """Return the merged config's load error as it sits on disk, or None.

    Measured before a write, so `revalidate_write` tells this edit's breakage from an older
    one elsewhere.

    Args:
        repo_root: The repo.

    Returns:
        The error message, or None when the config loads.
    """
    try:
        layer.load_effective(repo_root, ())
    except config_model.ConfigError as exc:
        return str(exc)
    return None


# Without the lock a snapshot restore could erase a concurrent writer's update.
_KEPT_NO_LOCK = (
    "(kept as written: the config lock could not be taken, so an automatic"
    " rollback might erase a concurrent edit; undo by hand or run `agent6 config fix`)"
)


def keep_or_rollback(target: pathlib.Path, prior: str | None, err: str, *, held: bool) -> str:
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
        portable.atomic_write(target, prior)
    return err


# Derived: a hand-listed copy would leave a new entry type validated by nothing.
PROVIDER_MEMBERS: tuple[type[pydantic.BaseModel], ...] = get_args(
    get_args(_providers.ProviderEntry)[0]
)


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
        except pydantic.ValidationError as exc:
            # Only an error at the leaf or inside its value counts; a missing sibling does not.
            leaf_errs = [e["msg"] for e in exc.errors() if e["loc"] and e["loc"][0] == leaf]
            if not leaf_errs:
                return None
            errors.append(leaf_errs[0])
    if not errors:
        return None
    seen = list(dict.fromkeys(errors))
    return f"{key}: {' / '.join(seen)}"


def unknown_key_error(
    key: str, repo_root: pathlib.Path, *, eff: layer.EffectiveConfig | None = None
) -> str:
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
        pool = layer.leaf_keys(eff if eff is not None else layer.load_effective(repo_root, ()))
    except config_model.ConfigError:
        pool = sorted(layer.flatten_leaves(config_model.Config().model_dump(mode="python")))
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


def _provider_key_refusal(key: str, value: object = None) -> str | None:
    """Return a provider-name refusal before a dotted-key write loses name boundaries."""
    parts = key.split(".")
    if parts[0] == "presets":
        parts = parts[2:]
    if not parts or parts[0] != "providers":
        return None
    if len(parts) >= 2:
        return _providers.provider_name_refusal(parts[1]) or None
    if isinstance(value, dict):
        for name in value:
            if refusal := _providers.provider_name_refusal(name):
                return refusal
    return None


def written_value_error(
    key: str,
    value: object,
    *,
    repo_root: pathlib.Path,
    section: dict[str, Any] | None = None,
    to_repo: bool = False,
) -> str | None:
    """Validate a written `key = value` on its own, independent of the layer merge.

    A higher layer can mask an invalid value in the merge, so it would land in the file and
    explode where the mask is absent. The one owner every writer uses.

    Args:
        key: The dotted key.
        value: The value written.
        repo_root: The repo, for the did-you-mean pool.
        section: The section's sibling leaves, so a rule spanning two keys sees them.
        to_repo: The write targets the repo layer, so a command in the error keeps `--repo`.

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
                ".".join(parts[2:]), value, repo_root=repo_root, section=section, to_repo=to_repo
            )
        return None
    if parts[0] == "providers" and len(parts) == 3:
        return _provider_key_refusal(key) or provider_field_error(key, parts[2], value)
    nested: dict[str, object] = {}
    cur = nested
    for part in parts[:-1]:
        child: dict[str, object] = dict(section or {}) if part == parts[-2] else {}
        cur[part] = child
        cur = child
    cur[parts[-1]] = value
    try:
        config_model.Config.model_validate(nested)
    except pydantic.ValidationError as exc:
        for err in exc.errors():
            message = _error_about(err, key, value, repo_root, to_repo=to_repo)
            if message is not None:
                return message
    except config_model.ConfigError as exc:
        return str(exc)
    return None


def _error_about(
    err: pydantic_core.ErrorDetails,
    key: str,
    value: object,
    repo_root: pathlib.Path,
    *,
    to_repo: bool,
) -> str | None:
    """Render one validation error as a message about the key.

    Args:
        err: The error.
        key: The dotted key written.
        value: The value written.
        repo_root: The repo, for the did-you-mean pool.
        to_repo: The write targets the repo layer, so a command in the message keeps `--repo`.

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
        target = f"--repo {key}" if to_repo else key
        detail = f'expected a list: config set {target} \'["a", "b"]\', or config add {target} a'
    else:
        detail = err["msg"]
    return f"{key}: {detail}"


def revalidate_write(
    repo_root: pathlib.Path,
    target: pathlib.Path,
    prior: str | None,
    *,
    was_valid: bool,
    held: bool = True,
    written: Sequence[tuple[str, object]] = (),
    to_repo: bool = False,
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
        to_repo: The target is the repo layer, so a command in the error keeps `--repo`.

    Returns:
        The error when this edit broke the config, else None.
    """
    try:
        doc = io.read_toml_file(target)
    except config_model.ConfigError as exc:
        # Unparseable TOML is always this edit's doing; a raise would escape the rollback.
        return keep_or_rollback(target, prior, str(exc), held=held)
    for wkey, wvalue in written:
        section = _section_leaves(doc, wkey)
        value_err = written_value_error(
            wkey, wvalue, repo_root=repo_root, section=section, to_repo=to_repo
        )
        if value_err is None:
            continue
        # A section rule can fire over a sibling that was already wrong; the value alone
        # settles whose fault it is, and the merged check below keeps a pre-broken config.
        if written_value_error(wkey, wvalue, repo_root=repo_root) is not None:
            return keep_or_rollback(target, prior, value_err, held=held)
    try:
        layer.load_effective(repo_root, ())
    except config_model.ConfigError as exc:
        err = str(exc) + _whole_table_hint(exc, written, to_repo=to_repo)
    else:
        return None
    # A target that no longer parses is always this write's doing, never another layer's.
    if not was_valid and not target_unparseable(target):
        return None  # broken before this edit; not ours to refuse
    return keep_or_rollback(target, prior, err, held=held)


def _whole_table_hint(
    exc: config_model.ConfigError, written: Sequence[tuple[str, object]], *, to_repo: bool
) -> str:
    """Return the line naming the one write that sets a table's required fields together.

    A table with required fields (`harness.metric`) cannot be built one leaf at a time: each
    write alone fails validation and rolls back.

    Args:
        exc: The merged config's load error.
        written: The `(key, value)` pairs the edit wrote.
        to_repo: The target is the repo layer, so the command keeps `--repo`.

    Returns:
        The hint, starting with a newline, or "" when no written table misses a field.
    """
    cause = exc.__cause__
    if not isinstance(cause, pydantic.ValidationError):
        return ""
    missing: dict[str, list[str]] = {}
    for issue in cause.errors():
        if issue["type"] == "missing":
            *table, field = (str(part) for part in issue["loc"])
            missing.setdefault(".".join(table), []).append(field)
    for key, _ in written:
        table, _, _leaf = key.rpartition(".")
        if table in missing:
            wrote = [k.rpartition(".")[2] for k, _ in written if k.rpartition(".")[0] == table]
            fields = ", ".join(f"{name} = ..." for name in (*wrote, *missing[table]))
            flag = " --repo" if to_repo else ""
            return (
                "\nset the table's fields in one write: "
                f"`agent6 config set{flag} {table} '{{ {fields} }}'`"
            )
    return ""


def _models_at(
    model: type[pydantic.BaseModel], part: str
) -> tuple[type[pydantic.BaseModel], ...] | None:
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


def _model_members(annotation: object) -> tuple[type[pydantic.BaseModel], ...]:
    """Return every BaseModel in an annotation, unwrapping `Annotated` and unions.

    Args:
        annotation: The field annotation.

    Returns:
        The models, in declaration order.
    """
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
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
    models: tuple[type[pydantic.BaseModel], ...] = (config_model.Config,)
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
    repo_root: pathlib.Path, dotted_key: str, raw_value: str, *, to_repo: bool = False
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
    parsed = io.parse_cli_value(raw_value)
    if refusal := _provider_key_refusal(dotted_key, parsed):
        return refusal
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target) as held:
        prior = agent6_errors.read_operator_file(target) if target.is_file() else None
        io.read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        if isinstance(parsed, dict) and names_a_section(dotted_key):
            if not parsed:
                raise config_model.ConfigError(
                    f"{dotted_key} = {{}} sets nothing; name the leaves to set"
                )
            try:
                for leaf, val in parsed.items():
                    io.upsert_toml_leaf(target, f"{dotted_key}.{leaf}", val)
            except config_model.ConfigError as exc:
                # A refusal mid-way leaves earlier leaves on disk.
                return keep_or_rollback(target, prior, str(exc), held=held)
            written = [(f"{dotted_key}.{leaf}", v) for leaf, v in parsed.items()]
        else:
            io.upsert_toml_leaf(target, dotted_key, parsed)
            written = [(dotted_key, parsed)]
        return revalidate_write(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            written=written,
            to_repo=to_repo,
        )


def set_config_table(
    repo_root: pathlib.Path,
    table: str,
    fields: dict[str, io.ConfigLeafValue],
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
    if refusal := _provider_key_refusal(table, fields):
        return refusal
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target) as held:
        prior = agent6_errors.read_operator_file(target) if target.is_file() else None
        io.read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        io.upsert_toml_table(target, table, fields)
        return revalidate_write(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            # Per leaf: a whole-table pair would hide every leaf-level error.
            written=[(f"{table}.{k}", v) for k, v in fields.items() if v is not None],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EntryTable:
    """A config table of named entries a form adds whole and removes whole.

    Attributes:
        path: The table's dotted key.
        label: The entry's name in a form or menu.
        fields: The leaves the add form offers, in order.
        leaf: The entry is one leaf whose value is the single form field, not a subtable.
    """

    path: str
    label: str
    fields: tuple[str, ...]
    leaf: bool = False


ENTRY_TABLES: tuple[EntryTable, ...] = (
    EntryTable("providers", "provider", ("api_format", "deployment", "base_url", "api_key_env")),
    EntryTable("mcp.servers", "MCP server", ("command", "url", "token_env", "enabled", "approve")),
    EntryTable("skills.state", "skill", ("state",), leaf=True),
)


def leaf_entries(config: config_model.Config, table: EntryTable) -> dict[str, str]:
    """Return a leaf table's entries by name, as the effective config holds them.

    Args:
        config: The effective config.
        table: A table with `leaf` set.

    Returns:
        The entry values by name.
    """
    node: object = config
    for part in table.path.split("."):
        node = getattr(node, part)
    return {name: str(value) for name, value in cast(dict[str, object], node).items()}


def _is_argv(table: EntryTable, field: str) -> bool:
    """Return whether a form field is an argv, so its line is split with shell quoting."""
    return any(_base.is_argv_field(info) for info in _entry_fields(table, field))


def _entry_fields(table: EntryTable, field: str) -> tuple[pydantic.fields.FieldInfo, ...]:
    """Return a form field's schema across the entry's model members.

    Args:
        table: The entry table.
        field: The form field.

    Returns:
        The member fields, or the mapping field for a scalar entry such as a skill state.
    """
    models: tuple[type[pydantic.BaseModel], ...] = (config_model.Config,)
    for part in table.path.split("."):
        resolved = tuple(m for model in models for m in (_models_at(model, part) or ()))
        if not resolved:
            return tuple(model.model_fields[part] for model in models)
        models = resolved
    return tuple(model.model_fields[field] for model in models if field in model.model_fields)


def _field_choices(annotation: Any) -> list[str]:
    """Return literal and boolean choices, including inside optional fields."""
    if annotation is bool:
        return ["true", "false"]
    if get_origin(annotation) is Literal:
        return [str(value) for value in get_args(annotation)]
    return [value for arg in get_args(annotation) for value in _field_choices(arg)]


def entry_choices(table: EntryTable) -> dict[str, list[str]]:
    """Return each fixed-choice form field's values from the entry schema.

    Args:
        table: The entry table.

    Returns:
        Literal choices and boolean choices (`true` and `false`), in declaration order.
    """
    choices: dict[str, list[str]] = {}
    for field in table.fields:
        values: list[str] = []
        for info in _entry_fields(table, field):
            values.extend(_field_choices(info.annotation))
        if values:
            choices[field] = list(dict.fromkeys(values))
    return choices


def entry_field_description(table: EntryTable, field: str) -> str:
    """Return a form field's description from the entry schema.

    Args:
        table: The entry table.
        field: The form field.

    Returns:
        The first member's description, or an empty string when absent.
    """
    return next((info.description or "" for info in _entry_fields(table, field)), "")


REVIEWER_FIELDS = ("focus", "provider", "model", "effort", "temperature")


def reviewer_choices() -> dict[str, list[str]]:
    """Return the reviewer form's fixed choices from the schema."""
    return {
        name: choices
        for name, field in _harness.ReviewerConfig.model_fields.items()
        if (choices := _field_choices(field.annotation))
    }


def add_reviewer(
    repo_root: pathlib.Path, fields: dict[str, str], *, to_repo: bool = False
) -> str | None:
    """Append a reviewer to the chosen layer's effective roster.

    Args:
        repo_root: The repo.
        fields: Form values; focus is required and blank optional fields are omitted.
        to_repo: The repo resolution instead of defaults plus the global config.

    Returns:
        The input or validation error, else None.

    Raises:
        ConfigError: The config cannot be loaded or the writer refuses the key.
        OperatorError: The target cannot be written.
    """
    if not fields.get("focus", "").strip():
        return "reviewer focus is required"
    values = {
        field: io.parse_cli_value(raw.strip()) if field == "temperature" else raw.strip()
        for field, raw in fields.items()
        if raw.strip()
    }
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target):
        eff = layer.load_effective(repo_root) if to_repo else layer.load_global_only()
        roster = [entry.model_dump(exclude_none=True) for entry in eff.config.review.reviewers]
        roster.append(values)
        return set_config_value(
            repo_root, "review.reviewers", io.format_toml_value(roster), to_repo=to_repo
        )


def remove_reviewer(repo_root: pathlib.Path, index: int, *, to_repo: bool = False) -> str | None:
    """Remove a zero-based position, writing an explicit empty list after the last seat.

    Args:
        repo_root: The repo.
        index: The reviewer's zero-based position, never its focus.
        to_repo: The repo resolution instead of defaults plus the global config.

    Returns:
        The input or validation error, else None.

    Raises:
        ConfigError: The config cannot be loaded or the writer refuses the key.
        OperatorError: The target cannot be written.
    """
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target):
        eff = layer.load_effective(repo_root) if to_repo else layer.load_global_only()
        roster = [entry.model_dump(exclude_none=True) for entry in eff.config.review.reviewers]
        if not 0 <= index < len(roster):
            return f"reviewer index {index} is out of range (roster has {len(roster)} reviewers)"
        del roster[index]
        return set_config_value(
            repo_root, "review.reviewers", io.format_toml_value(roster), to_repo=to_repo
        )


def add_entry(
    repo_root: pathlib.Path,
    table: EntryTable,
    name: str,
    fields: dict[str, str],
    *,
    to_repo: bool = False,
) -> str | None:
    """Add or update a named entry, preserving fields left blank.

    Args:
        repo_root: The repo.
        table: The entry table.
        name: The entry name, stripped; blank is refused.
        fields: Form values; a command line is split into argv with shell quoting.
        to_repo: The repo layer instead of the global one.

    Returns:
        The input or write error, else None.

    Raises:
        ConfigError: The target does not parse or the surgery refuses a leaf.
        OperatorError: The target cannot be written.
    """
    name = name.strip()
    if table.path == "providers" and (refusal := _providers.provider_name_refusal(name)):
        return refusal
    if not name:
        return f"{table.label} name is required"
    choices = entry_choices(table)
    values: dict[str, io.ConfigLeafValue] = {}
    for field, raw in fields.items():
        value = raw.strip()
        if not value:
            continue
        if _is_argv(table, field):
            try:
                values[field] = shlex.split(value)
            except ValueError as exc:
                return f"{field}: {exc}"
        elif choices.get(field) == ["true", "false"] and value in ("true", "false"):
            values[field] = value == "true"
        else:
            values[field] = value
    if table.leaf:
        return set_config_leaves(
            repo_root, table.path, {name: values.get(table.fields[0])}, to_repo=to_repo
        )
    return set_config_leaves(repo_root, f"{table.path}.{name}", values, to_repo=to_repo)


def remove_entry(
    repo_root: pathlib.Path, table: EntryTable, name: str, *, to_repo: bool = False
) -> UnsetResult:
    """Remove a whole named entry from the chosen layer.

    Args:
        repo_root: The repo.
        table: The entry table.
        name: The entry name.
        to_repo: The repo layer instead of the global one.

    Returns:
        Whether the entry was present and any revalidation error.
    """
    if table.path == "providers" and (refusal := _providers.provider_name_refusal(name)):
        return UnsetResult(removed=False, error=refusal)
    key = f"{table.path}.{name}"
    unset = unset_config_value if table.leaf else unset_config_table
    return unset(repo_root, key, to_repo=to_repo)


def entry_of(key: str) -> tuple[EntryTable, str] | None:
    """Return the named entry a config leaf belongs to, or None.

    Args:
        key: The dotted leaf key.

    Returns:
        The entry table and name, including for a leaf in an entry's subtable.
    """
    for table in ENTRY_TABLES:
        if key.startswith(table.path + "."):
            name, separator, _leaf = key[len(table.path) + 1 :].partition(".")
            if name and bool(separator) == (not table.leaf):
                return table, name
    return None


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
    repo_root: pathlib.Path,
    table: str,
    fields: dict[str, io.ConfigLeafValue],
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
    if refusal := _provider_key_refusal(table, fields):
        return refusal
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with writing_config(target) as held:
        prior = agent6_errors.read_operator_file(target) if target.is_file() else None
        io.read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        try:
            for key, val in fields.items():
                if val is not None:
                    io.upsert_toml_leaf(target, f"{table}.{key}", val)
        except config_model.ConfigError as exc:
            # Earlier leaves may already have landed.
            raise config_model.ConfigError(
                keep_or_rollback(target, prior, str(exc), held=held)
            ) from exc
        return revalidate_write(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            written=[(f"{table}.{k}", v) for k, v in fields.items() if v is not None],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class UnsetResult:
    """How an unset ended.

    Attributes:
        removed: Whether anything was removed.
        error: The revalidation error when the removal broke the config (rolled back, or
            kept without the lock), else None.
    """

    removed: bool
    error: str | None = None


def unset_config_table(
    repo_root: pathlib.Path, table: str, *, to_repo: bool = False
) -> UnsetResult:
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
    if refusal := _provider_key_refusal(table):
        return UnsetResult(removed=False, error=refusal)
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return UnsetResult(removed=False)
    with writing_config(target) as held:
        prior = agent6_errors.read_operator_file(target)
        io.read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        if not io.remove_toml_table(target, table):
            return UnsetResult(removed=False)
        return UnsetResult(
            removed=True,
            error=revalidate_write(repo_root, target, prior, was_valid=was_valid, held=held),
        )


def unset_config_value(
    repo_root: pathlib.Path, dotted_key: str, *, to_repo: bool = False
) -> UnsetResult:
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
    if refusal := _provider_key_refusal(dotted_key):
        return UnsetResult(removed=False, error=refusal)
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return UnsetResult(removed=False)
    with writing_config(target) as held:
        prior = agent6_errors.read_operator_file(target)
        io.read_toml_file(target)  # refuse line surgery on a file that does not parse
        was_valid = merged_config_error(repo_root) is None
        if not io.remove_toml_leaf(target, dotted_key):
            return UnsetResult(removed=False)
        return UnsetResult(
            removed=True,
            error=revalidate_write(repo_root, target, prior, was_valid=was_valid, held=held),
        )
