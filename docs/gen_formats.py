"""Generate the public format reference from serialized fields and producer docstrings."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pathlib
import re
import typing
from collections.abc import Iterable
from typing import Any

import pydantic

from agent6.providers import provider_types
from agent6.sessions import journal, retention
from agent6.viewmodel import state_view

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src/agent6"
_OUT = _ROOT / "docs/formats.md"
_BLOB = "https://github.com/agent6-dev/agent6/blob/master/"


def module_path(module: str) -> pathlib.Path:
    """Return the tree's file for a dotted module name, wherever agent6 is imported from."""
    return _SRC / (module.removeprefix("agent6.").replace(".", "/") + ".py")


class FormatError(Exception):
    """A serialized field has no documented meaning or a producer cannot be read."""


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """One serialized key, its declared type when available, and its source meaning."""

    name: str
    kind: str
    meaning: str


def descriptions(doc: str) -> dict[str, str]:
    """Return Google-style Attributes from a producer's docstring."""
    result: dict[str, str] = {}
    active = False
    key = ""
    for line in inspect.cleandoc(doc).splitlines():
        if line == "Attributes:":
            active = True
        elif active and line and not line.startswith(" "):
            active = False
        elif active and (match := re.match(r"    ([\w.\[\]*]+): (.*)", line)):
            key, text = match.groups()
            result[key] = text
        elif active and key and line.strip():
            result[key] += " " + line.strip()
    return result


def _meaning(docs: dict[str, str], name: str, source: str) -> str:
    if meaning := docs.get(name):
        return meaning.replace("|", "\\|")
    raise FormatError(f"{source}: undocumented serialized field {name}")


def _node(path: pathlib.Path, symbol: str) -> ast.FunctionDef | ast.ClassDef:
    tree: ast.AST = ast.parse(path.read_text(encoding="utf-8"))
    for part in symbol.split("."):
        found = next(
            (
                n
                for n in ast.iter_child_nodes(tree)
                if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == part
            ),
            None,
        )
        if found is None:
            raise FormatError(f"{path}: no producer {symbol}")
        tree = found
    if not isinstance(tree, ast.FunctionDef | ast.ClassDef):
        raise FormatError(f"{path}: not a producer {symbol}")
    return tree


def _pairs(node: ast.AST) -> Iterable[tuple[str, ast.expr]]:
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                yield key.value, value
            elif key is None:
                yield from _pairs(value)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        for kw in node.keywords:
            if kw.arg:
                yield kw.arg, kw.value
    elif isinstance(node, ast.IfExp):
        yield from _pairs(node.body)
        yield from _pairs(node.orelse)


def _bindings(node: ast.AST) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = child.value
        elif (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and child.value is not None
        ):
            result[child.target.id] = child.value
    return result


def _literal(
    node: ast.AST,
    prefix: str = "",
    bindings: dict[str, ast.expr] | None = None,
    seen: frozenset[str] = frozenset(),
) -> Iterable[str]:
    if isinstance(node, ast.Name) and bindings and node.id in bindings and node.id not in seen:
        yield from _literal(bindings[node.id], prefix, bindings, seen | {node.id})
    elif isinstance(node, ast.Tuple):
        for part in node.elts:
            yield from _literal(part, prefix, bindings, seen)
    elif isinstance(node, ast.ListComp):
        yield from _literal(node.elt, prefix + "[]", bindings, seen)
    elif isinstance(node, ast.IfExp):
        yield from _literal(node.body, prefix, bindings, seen)
        yield from _literal(node.orelse, prefix, bindings, seen)
    else:
        for key, value in _pairs(node):
            name = f"{prefix}.{key}" if prefix else key
            yield name
            yield from _literal(value, name, bindings, seen)


def _target(node: ast.AST) -> tuple[str, str]:
    if isinstance(node, ast.Name):
        return node.id, ""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        root, prefix = _target(node.value)
        if isinstance(node.slice.value, str):
            return root, f"{prefix}.{node.slice.value}".lstrip(".")
    return "", ""


def _responses(node: ast.AST, variable: str) -> Iterable[str]:
    """Yield keys from returned or sent dictionary literals."""
    bindings = _bindings(node)
    for child in ast.walk(node):
        if not variable and isinstance(child, ast.Return) and child.value is not None:
            value = child.value
            if isinstance(value, ast.Tuple):
                for part in value.elts:
                    yield from _literal(part, bindings=bindings)
            else:
                yield from _literal(value, bindings=bindings)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            method = child.func.attr
            if method in (
                {"append"} if variable else {"dumps", "_send_json", "_ok_or_err", "send"}
            ):
                index = 1 if method == "_ok_or_err" else 0
                if len(child.args) > index:
                    yield from _literal(child.args[index], bindings=bindings)


def _keys(node: ast.AST, variable: str) -> Iterable[str]:
    """Yield keys written to one output variable."""
    if not variable or variable == "@append":
        yield from _responses(node, variable)
        return
    for child in ast.walk(node):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(child, ast.Assign):
            targets, value = child.targets, child.value
        elif isinstance(child, ast.AnnAssign):
            targets, value = [child.target], child.value
        for target in targets:
            for part in target.elts if isinstance(target, ast.Tuple) else [target]:
                root, name = _target(part)
                if root != variable:
                    continue
                if name:
                    yield name
                if value is not None:
                    yield from _literal(value, name, _bindings(node))


def dictionary_fields(
    path: pathlib.Path, symbol: str, variable: str = "", branch: str = ""
) -> tuple[Field, ...]:
    """Return dictionary keys from a producer, with meanings from its Attributes section."""
    node = _node(path, symbol)
    docs = descriptions(ast.get_docstring(node) or "")
    body: ast.AST = node
    if branch:
        body = next(
            n
            for n in ast.walk(node)
            if isinstance(n, ast.If)
            and any(isinstance(v, ast.Constant) and v.value == branch for v in ast.walk(n.test))
        )
        body = ast.Module(body=body.body, type_ignores=[])
    if path.is_relative_to(_SRC):
        module = importlib.import_module(
            "agent6." + str(path.relative_to(_SRC)).removesuffix(".py").replace("/", ".")
        )
        for candidate in ast.walk(node):
            if isinstance(candidate, ast.FunctionDef) and isinstance(candidate.returns, ast.Name):
                annotation = vars(module).get(candidate.returns.id)
                if typing.is_typeddict(annotation):
                    hints = typing.get_type_hints(annotation)
                    names = dict.fromkeys(_keys(body, variable))
                    return tuple(
                        Field(n, _type_text(hints[n]), _meaning(docs, n, str(path))) for n in names
                    )
    names = dict.fromkeys(_keys(body, variable))
    return tuple(Field(n, "JSON", _meaning(docs, n, f"{path}:{symbol}")) for n in names)


def composed_fields(cls: type[Any], producers: tuple[Producer, ...]) -> tuple[Field, ...]:
    """Return a serialized model with the builder's additions and removals applied."""
    fields = {f.name: f for f in model_fields(cls)}
    if cls.__name__ == "SessionState":
        fields = {name: fields[name] for name in state_view.PUBLIC_STATE_FIELDS}
    for producer in producers:
        fields.update(
            (f.name, f)
            for f in dictionary_fields(
                producer.path, producer.symbol, producer.variable, producer.branch
            )
        )
        for node in ast.walk(_node(producer.path, producer.symbol)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func.value) == producer.variable
                and node.func.attr == "pop"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                fields.pop(str(node.args[0].value), None)
    return tuple(fields.values())


def _types(annotation: Any) -> Iterable[type[Any]]:
    if typing.get_origin(annotation):
        for arg in typing.get_args(annotation):
            yield from _types(arg)
    elif isinstance(annotation, type) and (
        dataclasses.is_dataclass(annotation) or issubclass(annotation, pydantic.BaseModel)
    ):
        yield annotation


def model_fields(cls: type[Any]) -> tuple[Field, ...]:
    """Return fields from the dataclass or model actually used by serialization."""
    docs = descriptions(inspect.getdoc(cls) or "")
    hints = typing.get_type_hints(cls)
    if issubclass(cls, pydantic.BaseModel):
        fields = cls.model_fields
        for name, info in fields.items():
            if info.description:
                docs[name] = info.description
        names = list(fields)
    elif typing.is_typeddict(cls):
        names = list(hints)
    else:
        names = [f.name for f in dataclasses.fields(cls)]
    return tuple(
        Field(name, _type_text(hints[name]), _meaning(docs, name, cls.__qualname__))
        for name in names
    )


def _type_text(value: Any) -> str:
    text = str(value).replace("typing.", "")
    text = re.sub(r"<class '([^']+)'>", r"\1", text)
    # Python 3.13 prints `pathlib._local.Path`; 3.12 and 3.14 print `pathlib.Path`.
    text = re.sub(r"\b(\w+)\._local\.", r"\1.", text)
    text = re.sub(r"agent6\.[\w.]+\.([A-Z]\w*)", r"\1", text)
    return _optional_as_union(text).replace("|", "\\|")


def _optional_as_union(text: str) -> str:
    """Spell every `Optional[X]` as `X | None`, nested ones included, as Python 3.14 prints it.

    Args:
        text: A type's text.

    Returns:
        The text with no `Optional[` left.
    """
    while (start := text.find("Optional[")) >= 0:
        depth, end = 0, start + len("Optional[")
        for end in range(start + len("Optional["), len(text)):
            depth += {"[": 1, "]": -1}.get(text[end], 0)
            if depth < 0:
                break
        inner = text[start + len("Optional[") : end]
        text = f"{text[:start]}{inner} | None{text[end + 1 :]}"
    return text


def _table(fields: Iterable[Field]) -> str:
    rows = ["| Field | Type | Meaning |", "| --- | --- | --- |"]
    rows.extend(f"| `{f.name}` | `{f.kind}` | {f.meaning} |" for f in fields)
    return "\n".join(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class Producer:
    """Identify an output builder, not a duplicate field schema."""

    module: str
    symbol: str
    surface: str
    variable: str = ""
    branch: str = ""

    @property
    def path(self) -> pathlib.Path:
        """Return the producer module path."""
        return module_path(self.module)

    def render(self) -> str:
        """Return a table of the keys this builder writes."""
        node = _node(self.path, self.symbol)
        meaning = (ast.get_docstring(node) or "").split("\n\n")[0].replace("\n", " ")
        fields = dictionary_fields(self.path, self.symbol, self.variable, self.branch)
        source = self.path.relative_to(_ROOT)
        return (
            f"### {self.surface}\n\n**Support: public.** {meaning}\n\n"
            f"Source: [`{self.module}.{self.symbol}`]({_BLOB}{source}).\n\n" + _table(fields)
        )


# These entries select producers; field names, types and meanings come from their code.
CLI = (
    Producer("viewmodel.listing", "summary_row", "`sessions list --json` row"),
    Producer("viewmodel.listing", "row_json", "`sessions list --json` group fields", "out"),
    Producer("ui.cli.sessions_show", "_cmd_status", "`sessions show --json`"),
)
SESSION = (
    Producer("viewmodel.state_view", "session_state_as_dict", "`attach --json` snapshot", "d"),
    Producer(
        "viewmodel.state_view",
        "session_state_as_dict",
        "`attach --json` approval and commit fields",
        "row",
    ),
    Producer("viewmodel.wire", "branch_facts", "`attach --json` branch fields", "out"),
    Producer("viewmodel.wire", "manifest_header", "`attach --json` header fields", "header"),
    Producer("viewmodel.wire", "session_snapshot", "`/api/session/<id>` snapshot", "snap"),
)
BUDGET = Producer("viewmodel.state_view", "budget_dict", "`budget` rendered fields", "d")
MACHINE = (
    Producer(
        "viewmodel.machine.state", "machine_state_as_dict", "`attach --json` machine snapshot", "d"
    ),
    Producer("viewmodel.wire", "machine_snapshot", "`/api/machine/<name>` snapshot", "d"),
)
WEB = (
    Producer(
        "ui.web.sse", "stream_session.send_frame", "`/api/session/<id>/events` frame fields", "d"
    ),
    Producer(
        "ui.web.sse", "stream_machine", "`/api/machine/<name>/events` snapshot frame", "payload"
    ),
    Producer("ui.web.sse", "stream_machine", "`/api/machine/<name>/events` frames"),
    Producer("ui.web.sse", "_with_idle_age", "`/api/machine/<name>/events` reasoning age", "fresh"),
    Producer("ui.web.server", "_Handler._route", "`/api/meta` payload", branch="/api/meta"),
    Producer(
        "ui.web.server",
        "_Handler._route",
        "`/api/config/entry_tables` payload",
        branch="/api/config/entry_tables",
    ),
    Producer(
        "ui.web.server",
        "_Handler._route",
        "`/api/config/suggest/<key>` payload",
        branch="/api/config/suggest/",
    ),
    Producer("ui.web.server", "_Handler._login", "`/api/login` result"),
    Producer("ui.web.server", "_Handler._send_sign_in_required", "Web API sign-in refusal"),
    Producer("ui.web.server", "_Handler._ok_or_err", "Web POST result envelope"),
    Producer("ui.web.server", "_Handler._route_post", "Web POST result fields"),
    Producer("ui.web.actions", "fork_run", "`/api/session/<id>/fork` result"),
    Producer("ui.web.actions", "undo_session", "`/api/session/<id>/undo` result"),
    Producer("ui.web.actions", "review_run", "`/api/session/<id>/review` result"),
    Producer("ui.web.model", "hub_payload", "`/api/hub` payload"),
    Producer("ui.web.model", "_machine_row", "`/api/hub` machine row", "entry"),
    Producer("ui.web.model", "list_machine_files", "`/api/hub` machine-file rows"),
    Producer("ui.web.model", "routes_payload", "`/api/routes` payload"),
    Producer("ui.web.model", "merge_preview_payload", "`GET /api/session/<id>/merge` payload"),
    Producer(
        "ui.web.model", "resume_defaults_payload", "`/api/session/<id>/resume_defaults` payload"
    ),
    Producer("ui.web.model", "conversation_payload", "`/api/session/<id>/conversation` payload"),
    Producer(
        "ui.web.model", "conversation_items", "`/api/session/<id>/conversation` item", "entry"
    ),
    Producer("ui.web.model", "restate_payload", "`/api/session/<id>/restate` payload"),
    Producer(
        "ui.web.model", "machine_conversation_payload", "`/api/machine/<name>/conversation` payload"
    ),
    Producer(
        "ui.web.model",
        "machine_reasoning_snapshot",
        "`/api/machine/<name>/reasoning` fields",
        "snap",
    ),
    Producer("ui.web.model", "_diff_payload", "`/api/session/<id>/diff` payload"),
    Producer("viewmodel.config_view", "_leaf_json", "`/api/config` setting"),
    Producer("ui.web.model", "config_payload", "`/api/config` editing fields", "setting"),
    Producer("ui.web.git_model", "_ref_row", "`/api/git/refs` ref row"),
    Producer("ui.web.git_model", "_file_row", "Git changed-file row"),
    Producer("ui.web.git_model", "_log_row", "`/api/git/graph` commit row"),
    Producer("ui.web.git_model", "_decorations", "`/api/git/graph` ref decoration", "@append"),
    Producer("ui.web.git_model", "_endpoint", "`/api/git/compare` endpoint"),
    Producer("ui.web.git_model", "refs_payload", "`/api/git/refs` payload"),
    Producer("ui.web.git_model", "graph_payload", "`/api/git/graph` payload"),
    Producer("ui.web.git_model", "commit_payload", "`/api/git/commit/<rev>` payload"),
    Producer("ui.web.git_model", "compare_payload", "`/api/git/compare` payload"),
    Producer("ui.web.git_model", "worktree_payload", "`/api/git/worktree` payload"),
    Producer("ui.web.git_model", "diff_payload", "`/api/git/diff` payload"),
    Producer("ui.web.git_model", "search_payload", "`/api/git/search` payload"),
    Producer("ui.web.git_model", "tree_payload", "`/api/git/tree` payload"),
    Producer("ui.web.git_model", "text_payload", "`/api/git/text` payload"),
    Producer("ui.web.git_model", "blame_payload", "`/api/git/blame` payload"),
    Producer("ui.web.review_routes", "comparison", "`/api/git/findings` payload"),
    Producer("ui.web.review_routes", "comparison", "`/api/git/findings` finding", "@append"),
    Producer("ui.web.review_routes", "panel", "`/api/reviews/<panel_id>` payload"),
    Producer("ui.web.review_routes", "_diff", "`/api/reviews/<panel_id>/diff` payload"),
    Producer("ui.web.review_routes", "_file", "`/api/reviews/<panel_id>/file` payload"),
    Producer("viewmodel.review", "as_dict", "`/api/session/<id>/findings` review fields", "result"),
    Producer("viewmodel.review", "as_dict", "`/api/session/<id>/findings` finding fields", "row"),
    Producer("viewmodel.receipt", "as_dict", "`/api/session/<id>/receipt` fields", "payload"),
)
MODELS = (
    ("app.lane_launch", "LaneLaunch"),
    ("sessions.completion", "CompletionRecord"),
    ("sessions.lane_dispatch", "DispatchRecord"),
    ("sessions.manifest", "SessionManifest"),
    ("ui.cli.ps_cmd", "_Row"),
    ("viewmodel.state_types", "SessionState"),
    ("viewmodel.machine.state", "MachineState"),
    ("viewmodel.review", "MergeReviewGuard"),
    ("viewmodel.receipt", "ReceiptView"),
    ("viewmodel.lanes", "ChildView"),
    ("viewmodel.lanes", "DispatchView"),
    ("review_records", "FindingAction"),
    ("app.review.actions", "ActionResult"),
    ("vcs.git.files", "BlameLine"),
    ("vcs.git.files", "TreeFile"),
)


def _models() -> list[type[Any]]:
    queue = [getattr(importlib.import_module(f"agent6.{mod}"), name) for mod, name in MODELS]
    server = importlib.import_module("agent6.ui.web.server")
    queue.extend(
        cls for name, cls in vars(server).items() if name.endswith("Body") and name != "_Body"
    )
    result: list[type[Any]] = []
    for cls in queue:
        if cls in result:
            continue
        result.append(cls)
        for annotation in typing.get_type_hints(cls).values():
            queue.extend(t for t in _types(annotation) if t not in result)
    return result


def _model_section(cls: type[Any]) -> str:
    source = module_path(cls.__module__).relative_to(_ROOT)
    meaning = (inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " ")
    surface = cls.__name__
    if cls.__module__ == "agent6.ui.cli.ps_cmd" and surface == "_Row":
        surface = "`agent6 ps` JSON row"
    return (
        f"### {surface}\n\n**Support: public.** {meaning}\n\n"
        f"Source: [`{cls.__module__}.{cls.__name__}`]({_BLOB}{source}).\n\n"
        + _table(model_fields(cls))
    )


def _journal() -> str:
    parts = [
        Producer(
            "sessions.event_log", "EventSink.emit", "`logs.jsonl` event envelope", "payload"
        ).render()
    ]
    for event_type, record in sorted(journal.EVENT_TYPES.items()):
        fields = _record_fields(record, "")
        meaning = (inspect.getdoc(record) or "").split("\n\n")[0].replace("\n", " ")
        parts.append(f"### {event_type}\n\n**Support: public.** {meaning}\n\n{_table(fields)}")
    return "\n\n".join(parts)


# The JSON kind a record field's annotation declares, as the page names it.
_RECORD_KINDS = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "collections.abc.Sequence[Any]": "list",
    "dict[str, Any]": "object",
    "tuple[str, ...]": "list",
}


def _record_fields(record: type[Any], prefix: str) -> list[Field]:
    """Return a record's rows, a nested record's fields under dotted names."""
    rows: list[Field] = []
    for name, info in record.model_fields.items():
        optional = "" if info.is_required() else ", optional"
        nested, kind = _nested_record(info.annotation)
        rows.append(
            Field(prefix + name, kind + optional, (info.description or "").replace("|", "\\|"))
        )
        if nested is not None:
            rows.extend(_record_fields(nested, f"{prefix}{name}{'[].' if kind == 'list' else '.'}"))
    return rows


def _nested_record(annotation: Any) -> tuple[type[Any] | None, str]:
    """Return the record an annotation nests (bare, optional or listed) and its JSON kind."""
    for t in _types(annotation):
        if issubclass(t, pydantic.BaseModel):
            text = _type_text(annotation)
            return t, "list" if text.startswith("collections.abc.Sequence") else "object"
    return None, _record_kind(annotation)


def _record_kind(annotation: Any) -> str:
    text = _type_text(annotation).replace("\\|", "|")
    base = text.replace(" | None", "")
    return _RECORD_KINDS[base]


_HEADER = """<!-- Generated by docs/gen_formats.py; do not edit by hand.
Edit the producers' docstrings or descriptions, then run:
    uv run python docs/gen_formats.py
Pinned by tests/unit/test_formats_doc.py. -->

# Formats

Use this reference to read agent6's journal, CLI JSON, manifest and HTTP payloads.
Field names come from the serializers, dataclasses and models.
Meanings come from their docstrings and descriptions.
`JSON` means a dictionary builder has no narrower declared field type.
Python tuples serialize as JSON arrays, `None` as `null`, and paths as strings.
Nested dictionary keys use dots and array items use `[]`.

## Support levels

[Data contracts](data-contracts.md#support-levels) defines the support levels.

Formats are not stable until 1.0.

- **Durable but private:** snapshots preserve run history; transcripts retain diagnostic history.
- **Disposable:** caches and live [coordination files](architecture.md#session-state-on-disk).
- Config TOML is public; [Configuration](config.md) owns its generated setting reference.
- Machine definitions are public; see [Agent state machines](state-machines.md).

## Journal

`logs.jsonl` contains one JSON object per line.
A torn last line is dropped the next time the journal opens for appending.
The envelope below surrounds each event's payload.
The journal records describe current writers; historical reads tolerate unknown events.
Optional event fields may be absent or null.
Integer values also satisfy a record's `float` field, but booleans do not.

Review targets retain `snapshot_commit` as the cloneable commit for the exact `head_tree`.
`snapshot_ref` names its session-owned hidden ref; `hidden_paths` and `paths` keep their coverage.
It defaults empty on historical records, whose unanchored trees may be unavailable.
The ref also retains the base tree and is removed with its session, not by chain pruning.

"""


def build_markdown() -> str:
    """Return the page without starting a server or reading a user's state."""
    sections = [
        _HEADER,
        _journal(),
        "## Provider transcripts\n\n**Support: durable but private.**\n\n"
        "`transcripts/<timestamp>-<seq>.json` holds one record per model attempt.\n"
        "The sequence is padded to six digits in the filename.\n"
        "The record is written with the redacted request before the send, with "
        '`outcome: "incomplete"`, and rewritten in place when the attempt ends: `completed`, '
        "`failed`, `interrupted` (a steer) or `aborted` (a stop or a cancellation).\n"
        "A record stays `incomplete` while the call runs, and after a process ended before the "
        "attempt did; it does not say whether the provider received or billed the request.\n"
        "`sessions transcript` marks every attempt that did not complete.\n"
        "`model_call_id` joins a transcript to its logical call; `request_id` joins it to "
        "its `provider.request` event.\n"
        "Retries share the call id and have separate request ids.\n"
        "Older transcripts may omit these ids and `outcome`.\n"
        "Secret-bearing request and response headers are redacted, and their credential values "
        "of at least eight characters are scrubbed wherever they appear in the record.\n"
        "[Attachment payloads are omitted](security.md#media-retention) and replaced with "
        "metadata, so these files cannot replay the attachment bytes.\n"
        "Each session keeps an `objects/` store: one file per distinct JSON value, named by the "
        "SHA-256 of its canonical JSON (`sort_keys=True`, compact separators) after redaction "
        "and media omission.\n"
        'A transcript\'s request holds a reference, `{"$object": "<sha256 hex>"}`, in place of '
        "each message (a Responses `input` item) and of its `system`, `instructions` and "
        "`tools` values; a numbered checkpoint references its messages the same way.\n"
        "An object is written before any record that references it, the pre-send record "
        "included, so an unchanged retry writes nothing new.\n"
        "`loop_state.json` and `turn_in_flight.json` stay full: they are recovery state.\n"
        "Readers materialize references, accept older full records, and raise "
        "`ObjectStoreError` naming a missing object and its record; `sessions transcript "
        "--json` prints materialized records.\n\n"
        "### Artifact pruning\n\n"
        "`sessions prune --older-than DAYS` removes `transcripts/`, `checkpoints/` and "
        "`objects/` together from finished sessions whose last execution ended more than DAYS "
        "ago, writing `pruned.json` first so an interrupted prune cannot reuse a sequence.\n"
        "The manifest, journal, receipt, recovery state and graph store stay.\n"
        "`sessions transcript` reports the prune, a fork at a removed checkpoint refuses with "
        "the same notice, and a resumed session continues its transcript sequence above the "
        "saved one.\n"
        "Nothing is deleted automatically and nothing is compressed.\n\n"
        + _table(model_fields(retention.Pruned))
        + "\n\n### Attempt record\n\n"
        + f"Source: [`providers.provider_types.TranscriptAttempt.write`]"
        f"({_BLOB}src/agent6/providers/provider_types.py).\n\n"
        + _table(
            dictionary_fields(
                module_path("providers.provider_types"), "TranscriptAttempt.write", "payload"
            )
            + tuple(
                Field(f"usage.{field.name}", field.kind, field.meaning)
                for field in model_fields(provider_types.ProviderUsage)
            )
        ),
        "## CLI JSON",
    ]
    sections.append(
        "`sessions list --json` emits `summary_row` objects with `row_json` additions.\n"
        "`sessions show --json` emits one `_cmd_status` object.\n"
        "`ps --json` emits an array of `_Row` records, including nested lanes."
    )
    sections.extend(p.render() for p in CLI)
    summary = _node(module_path("budget"), "BudgetTracker.format_summary")
    sections.append(
        "## CLI text\n\n### Token + cost summary\n\n"
        "The end-of-run summary groups usage by provider/model route.\n"
        "The journal's `budget.update` event carries the cumulative totals above.\n\n"
        f"Source: [`budget.BudgetTracker.format_summary`]({_BLOB}src/agent6/budget.py).\n\n"
        + _table(
            Field(name, "text", meaning)
            for name, meaning in descriptions(ast.get_docstring(summary) or "").items()
        )
    )
    sections.append("## Session snapshots\n\n**Support: public.**")
    sections.append(
        "`attach --json`, `/api/session/<id>` and `/api/draft/<name>` read `SessionState` "
        "with the serializer's replacements and header fields below.\n"
        "The serializer removes `task_label` and renders `log_tail` as strings.\n"
        "The `row` additions belong to approvals and commits, not the snapshot root."
    )
    session_cls = importlib.import_module("agent6.viewmodel.state_types").SessionState
    sections.append(_table(composed_fields(session_cls, SESSION[:1] + SESSION[2:])))
    sections.append(
        "Sources: "
        + ", ".join(f"[`{p.symbol}`]({_BLOB}{p.path.relative_to(_ROOT)})" for p in SESSION)
        + "."
    )
    sections.append(SESSION[1].render())
    sections.append(BUDGET.render())
    sections.append("## Machine snapshots\n\n**Support: public.**")
    sections.append(
        "A machine's `attach --json` and `/api/machine/<name>` read `MachineState` "
        "with the computed fields below."
    )
    machine_cls = importlib.import_module("agent6.viewmodel.machine.state").MachineState
    sections.append(_table(composed_fields(machine_cls, MACHINE)))
    sections.append(
        "Sources: "
        + ", ".join(f"[`{p.symbol}`]({_BLOB}{p.path.relative_to(_ROOT)})" for p in MACHINE)
        + "."
    )
    sections.append("## Web API")
    sections.append(
        "[Web UI](web.md#the-http-api) describes authentication, routes, limits and status codes.\n"
        "The response builders below name keys read directly from their dictionary literals "
        "and assignments; conditional keys appear only on the paths that write them.\n"
        "Request bodies use the typed `Body` records below.\n"
        "Git blob responses carry bytes rather than JSON."
    )
    sections.extend(p.render() for p in WEB)
    write = importlib.import_module("agent6.config.write")
    reviewer = importlib.import_module("agent6.config.harness_config").ReviewerConfig
    sections.append(
        "### Reviewer choices\n\n**Support: public.** "
        "`/api/config/reviewer_choices` returns the enum choices from "
        "`config.write.reviewer_choices`, keyed by reviewer field.\n\n"
        + _table(
            Field(name, "list[str]", reviewer.model_fields[name].description or "")
            for name in write.reviewer_choices()
        )
    )
    sections.append("## Typed records")
    sections.append(
        "`SessionManifest` is the private, durable `manifest.json` record.\n"
        "`HarnessStamp.gate` separates the pinned command and source from execution availability.\n"
        "`ComparisonStamp` holds group facts on the coordinator.\n"
        "`PlacementStamp` links each lane to it.\n"
        "Other records are nested objects used by the CLI and web serializers above.\n"
        "Properties are absent unless a serializer adds them explicitly."
    )
    sections.extend(
        _model_section(cls)
        for cls in _models()
        if cls.__name__ not in {"SessionState", "MachineState", "TaskLabel", "LogLine"}
    )
    return "\n\n".join(sections).rstrip() + "\n"


if __name__ == "__main__":
    _OUT.write_text(build_markdown(), encoding="utf-8")
    print(f"gen_formats: wrote {_OUT}")
