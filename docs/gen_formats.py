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

from agent6.sessions import event_schema

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
    names = dict.fromkeys(_keys(body, variable))
    return tuple(Field(n, "JSON", _meaning(docs, n, f"{path}:{symbol}")) for n in names)


def composed_fields(cls: type[Any], producers: tuple[Producer, ...]) -> tuple[Field, ...]:
    """Return a serialized model with the builder's additions and removals applied."""
    fields = {f.name: f for f in model_fields(cls)}
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
            f"### {self.module}.{self.symbol}"
            + (f" ({self.variable})" if self.variable else "")
            + (f" ({self.branch})" if self.branch else "")
            + f"\n\n**Support: public.** {meaning}\n\n"
            + f"Source: [`{self.module}.{self.symbol}`]({_BLOB}{source}).\n\n"
            + _table(fields)
        )


# These entries select producers; field names, types and meanings come from their code.
CLI = (
    Producer("viewmodel.listing", "summary_row"),
    Producer("viewmodel.listing", "row_json", "out"),
    Producer("ui.cli.sessions_show", "_cmd_status"),
)
SESSION = (
    Producer("viewmodel.state_view", "session_state_as_dict", "d"),
    Producer("viewmodel.state_view", "session_state_as_dict", "row"),
    Producer("viewmodel.wire", "branch_facts", "out"),
    Producer("viewmodel.wire", "manifest_header", "header"),
    Producer("viewmodel.wire", "session_snapshot", "snap"),
)
MACHINE = (
    Producer("viewmodel.machine_state", "machine_state_as_dict", "d"),
    Producer("viewmodel.wire", "machine_snapshot", "d"),
)
WEB = (
    Producer("ui.web.sse", "stream_session.send_frame", "d"),
    Producer("ui.web.sse", "stream_machine", "payload"),
    Producer("ui.web.sse", "stream_machine"),
    Producer("ui.web.sse", "_with_idle_age", "fresh"),
    Producer("ui.web.server", "_Handler._route", branch="/api/meta"),
    Producer("ui.web.server", "_Handler._route", branch="/api/config/entry_tables"),
    Producer("ui.web.server", "_Handler._route", branch="/api/config/suggest/"),
    Producer("ui.web.server", "_Handler._login"),
    Producer("ui.web.server", "_Handler._send_sign_in_required"),
    Producer("ui.web.server", "_Handler._ok_or_err"),
    Producer("ui.web.server", "_Handler._route_post"),
    Producer("ui.web.actions", "fork_run"),
    Producer("ui.web.actions", "undo_session"),
    Producer("ui.web.actions", "run_plan"),
    Producer("ui.web.actions", "review_run"),
    Producer("ui.web.model", "hub_payload"),
    Producer("ui.web.model", "_machine_row", "entry"),
    Producer("ui.web.model", "list_machine_files"),
    Producer("ui.web.model", "routes_payload"),
    Producer("ui.web.model", "resume_defaults_payload"),
    Producer("ui.web.model", "conversation_payload"),
    Producer("ui.web.model", "conversation_items", "entry"),
    Producer("ui.web.model", "restate_payload"),
    Producer("ui.web.model", "machine_conversation_payload"),
    Producer("ui.web.model", "machine_reasoning_snapshot", "snap"),
    Producer("ui.web.model", "_diff_payload"),
    Producer("viewmodel.config_view", "_leaf_json"),
    Producer("ui.web.model", "config_payload", "setting"),
    Producer("ui.web.git_model", "_ref_row"),
    Producer("ui.web.git_model", "_file_row"),
    Producer("ui.web.git_model", "_log_row"),
    Producer("ui.web.git_model", "_decorations", "@append"),
    Producer("ui.web.git_model", "_endpoint"),
    Producer("ui.web.git_model", "refs_payload"),
    Producer("ui.web.git_model", "graph_payload"),
    Producer("ui.web.git_model", "commit_payload"),
    Producer("ui.web.git_model", "compare_payload"),
    Producer("ui.web.git_model", "worktree_payload"),
    Producer("ui.web.git_model", "diff_payload"),
    Producer("ui.web.git_model", "search_payload"),
    Producer("ui.web.git_model", "tree_payload"),
    Producer("ui.web.git_model", "text_payload"),
    Producer("ui.web.git_model", "blame_payload"),
    Producer("ui.web.review_routes", "comparison"),
    Producer("ui.web.review_routes", "comparison", "@append"),
    Producer("ui.web.review_routes", "panel"),
    Producer("ui.web.review_routes", "_diff"),
    Producer("ui.web.review_routes", "_file"),
    Producer("viewmodel.review", "as_dict", "result"),
    Producer("viewmodel.review", "as_dict", "row"),
    Producer("viewmodel.receipt", "as_dict", "payload"),
)
MODELS = (
    ("sessions.manifest", "SessionManifest"),
    ("ui.cli.ps_cmd", "_Row"),
    ("viewmodel.state_types", "SessionState"),
    ("viewmodel.machine_state", "MachineState"),
    ("viewmodel.review_queue", "ReviewQueueEntry"),
    ("viewmodel.review", "MergeReviewGuard"),
    ("viewmodel.receipt", "ReceiptView"),
    ("review_records", "FindingAction"),
    ("app.review_actions", "ActionResult"),
    ("vcs.git_files", "BlameLine"),
    ("vcs.git_files", "TreeFile"),
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
    return (
        f"### {cls.__name__}\n\n**Support: public.** {meaning}\n\n"
        f"Source: [`{cls.__module__}.{cls.__name__}`]({_BLOB}{source}).\n\n"
        + _table(model_fields(cls))
    )


def _journal() -> str:
    parts = [Producer("sessions.event_log", "EventSink.emit", "payload").render()]
    for event in event_schema.EVENTS:
        fields = [
            Field(f.name, f.kind + ("" if f.required else ", optional"), f.meaning)
            for f in event.fields
        ]
        parts.append(
            f"### {event.type}\n\n**Support: public.** {event.meaning}\n\n{_table(fields)}"
        )
    return "\n\n".join(parts)


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

- **Public:** an integration surface documented here, not a promise of compatibility before 1.0.
- **Durable but private:** saved state agent6 needs to resume or replay, not an integration API.
- **Disposable:** derived caches and live coordination files that do not define saved history.

All public shapes may change before 1.0 without a compatibility shim.

- **Durable but private:** [snapshots and transcripts](data-contracts.md) preserve run history.
- **Disposable:** [coordination files](architecture.md#session-state-on-disk) serve live workers.
- Config TOML is public; [Configuration](config.md) owns its generated setting reference.
- Machine definitions are public; see [Agent state machines](state-machines.md).

## Journal

`logs.jsonl` contains one JSON object per line.
The envelope below surrounds each event's payload.
The event registry describes current writers; historical reads tolerate unknown events.
Optional event fields may be absent or null.
Integer values also satisfy the registry's `float` kind, but booleans do not.

"""


def build_markdown() -> str:
    """Return the page without starting a server or reading a user's state."""
    sections = [_HEADER, _journal(), "## CLI JSON"]
    sections.append(
        "`sessions list --json` emits `summary_row` objects with `row_json` additions.\n"
        "`sessions show --json` emits one `_cmd_status` object.\n"
        "`ps --json` emits an array of `_Row` records, including nested lanes."
    )
    sections.extend(p.render() for p in CLI)
    sections.append("## Session snapshots\n\n**Support: public.**")
    sections.append(
        "`attach --json`, `/api/session/<id>` and `/api/draft/<name>` read `SessionState` "
        "with the serializer's replacements and header fields below.\n"
        "The serializer removes `task_label` and renders `log_tail` as strings.\n"
        "The `row` additions belong to approvals and commit steps, not the snapshot root."
    )
    session_cls = importlib.import_module("agent6.viewmodel.state_types").SessionState
    sections.append(_table(composed_fields(session_cls, SESSION[:1] + SESSION[2:])))
    sections.append(
        "Sources: "
        + ", ".join(f"[`{p.symbol}`]({_BLOB}{p.path.relative_to(_ROOT)})" for p in SESSION)
        + "."
    )
    sections.append(SESSION[1].render())
    sections.append("## Machine snapshots\n\n**Support: public.**")
    sections.append(
        "A machine's `attach --json` and `/api/machine/<name>` read `MachineState` "
        "with the computed fields below."
    )
    machine_cls = importlib.import_module("agent6.viewmodel.machine_state").MachineState
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
        "`SessionManifest` is the public `manifest.json` record, with nested stamps below.\n"
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
