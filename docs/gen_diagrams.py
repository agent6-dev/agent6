# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Generate docs/architecture.md from docs/architecture_template.md.

Two markers expand from the current source: `<!-- diagram: NAME -->` becomes a
mermaid block and `<!-- generated: NAME -->` becomes generated text or a list. A diagram
carries a shape (the layer chain, the stage order, the gate chain); names that
only need listing are listed, and the package list is each package's docstring.
Regenerate with `uv run python docs/gen_diagrams.py`; pinned by
tests/unit/test_gen_diagrams.py.
"""

from __future__ import annotations

import ast
import functools
import itertools
import pathlib
import re
import subprocess
import sys
import tempfile

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "agent6"
_TREE_URL = "https://github.com/agent6-dev/agent6/tree/master/src/agent6"
_TEMPLATE = _ROOT / "docs" / "architecture_template.md"
_OUT = _ROOT / "docs" / "architecture.md"
# A diagram owns its line; a generated list substitutes in place, so it can sit in a sentence.
_DIAGRAM = re.compile(r"^<!-- diagram: ([a-z-]+) -->$")
_GENERATED = re.compile(r"<!-- generated: ([a-z-]+) -->")

# The documented layering, top to bottom; every other top-level package is substrate.
_CORE_LAYERS = ("ui", "app", "harness", "tools", "sandbox")
# A package docstring ends with this line; the names on it are checked against the directory.
_MAIN_MODULES = "Main modules: "


def _nid(name: str) -> str:
    """Return a mermaid-safe node id; the prefix keeps a name such as `end` off mermaid's words."""
    return "n_" + re.sub(r"\W", "_", name)


@functools.cache
def _tach_graph() -> str:
    """Return tach's module graph as mermaid text."""
    # From this interpreter, never `uv run`: a uv spawn re-syncs, uninstalling a tested wheel.
    with tempfile.TemporaryDirectory(dir=_ROOT) as directory:
        output = pathlib.Path(directory) / "modules.mmd"
        subprocess.run(
            [sys.executable, "-m", "tach", "show", "--mermaid", "-o", str(output)],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
        )
        return output.read_text(encoding="utf-8")


def _package_edges(mermaid_graph: str) -> tuple[set[tuple[str, str]], set[str]]:
    """Return the graph's top-level package edges and the substrate packages."""
    edges: set[tuple[str, str]] = set()
    substrate: set[str] = set()
    for line in mermaid_graph.splitlines():
        m = re.match(r"\s*(\S+) --> (\S+)", line)
        if m is None:
            continue
        a, b = (part.split(".")[1] if "." in part else part for part in m.groups())
        if a == b or "agent6" in (a, b):
            continue
        edges.add((a, b))
        substrate.update(x for x in (a, b) if x not in _CORE_LAYERS)
    return edges, substrate


def _layering_mermaid() -> str:
    """Return the layer chain, with any import that climbs it drawn dashed.

    Every layer may import every layer below it, so the real edges would draw a mesh
    the chain already says; an upward edge means the map is stale.
    """
    edges, _ = _package_edges(_tach_graph())
    rank = {name: i for i, name in enumerate(_CORE_LAYERS)}
    lines = ["graph TD"]
    lines += [f'    {_nid(n)}["{n}"]' for n in _CORE_LAYERS]
    lines += [f"    {_nid(a)} --> {_nid(b)}" for a, b in itertools.pairwise(_CORE_LAYERS)]
    lines += [
        f'    {_nid(a)} -. "climbs the stack" .-> {_nid(b)}'
        for a, b in sorted(edges)
        if a in rank and b in rank and rank[a] > rank[b]
    ]
    return "\n".join(lines)


def _substrate_names() -> str:
    """Return the substrate packages as a sorted inline list."""
    _, substrate = _package_edges(_tach_graph())
    return ", ".join(f"`{name}`" for name in sorted(substrate))


def _package_sentences(package: pathlib.Path) -> list[str]:
    """Return a package docstring's sentences, a wrapped one joined.

    Args:
        package: The package's directory.

    Returns:
        The summary, then each later sentence, the main modules last.

    Raises:
        SystemExit: The package has no docstring, or its docstring names no main modules or
            one the directory does not hold.
    """
    init = package / "__init__.py"
    where = init.relative_to(_SRC.parent).as_posix()
    doc = ast.get_docstring(ast.parse(init.read_text(encoding="utf-8")))
    if not doc:
        raise SystemExit(f"{where} has no docstring")
    sentences: list[str] = []
    for line in filter(None, doc.splitlines()):
        if sentences and not sentences[-1].endswith("."):
            sentences[-1] += f" {line}"
        else:
            sentences.append(line)
    if not sentences[-1].startswith(_MAIN_MODULES):
        raise SystemExit(f"{where}: the docstring does not end with `{_MAIN_MODULES.strip()}`")
    missing = [
        name
        for name in re.findall(r"`(\w+)`", sentences[-1])
        if not ((package / f"{name}.py").is_file() or (package / name / "__init__.py").is_file())
    ]
    if missing:
        raise SystemExit(f"{where} names no such module: {', '.join(missing)}")
    return sentences


def _layers_first(package: pathlib.PurePath) -> tuple[int, tuple[str, ...]]:
    """Return a sort key: the root, the layers top to bottom, then the other packages by name."""
    top = package.parts[:1]
    if not top:
        return -1, ()
    layer = _CORE_LAYERS.index(top[0]) if top[0] in _CORE_LAYERS else len(_CORE_LAYERS)
    return layer, package.parts


def _packages() -> str:
    """Return every package's docstring as a list: its summary, then a bullet per sentence."""
    lines: list[str] = []
    packages = (init.parent.relative_to(_SRC) for init in _SRC.rglob("__init__.py"))
    for package in sorted(packages, key=_layers_first):
        name = package.as_posix() if package.parts else "agent6"
        url = f"{_TREE_URL}/{package.as_posix()}" if package.parts else _TREE_URL
        summary, *rest = _package_sentences(_SRC / package)
        lines.append(f"- **[`{name}`]({url})**: {summary}")
        lines += [f"    - {sentence}" for sentence in rest]
    return "\n".join(lines)


def _tier_callgraph(rel_path: str, tier: tuple[str, ...]) -> str:
    """Return the mermaid call graph of one file's tier: direct calls between its members."""
    tree = ast.parse((_ROOT / rel_path).read_text(encoding="utf-8"))
    members = set(tier)
    edges: set[tuple[str, str]] = set()

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.cur: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            prev, self.cur = self.cur, node.name
            self.generic_visit(node)
            self.cur = prev

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]  # noqa: N815  # ast.NodeVisitor dispatch name

        def visit_Call(self, node: ast.Call) -> None:
            name = None
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in members and self.cur in members and name != self.cur:
                edges.add((self.cur, name))
            self.generic_visit(node)

    V().visit(tree)
    connected = {n for e in edges for n in e}
    lines = ["graph TD"]
    for name in tier:
        if name in connected:
            label = name.lstrip("_")
            lines.append(f'    {_nid(label)}["{label}"]')
    for a, b in sorted(edges):
        lines.append(f"    {_nid(a.lstrip('_'))} --> {_nid(b.lstrip('_'))}")
    return "\n".join(lines)


# The modules that hold the lifecycle's stages, by the name their callers import them under.
_STAGE_MODULES = {
    "execution": "src/agent6/app/execution.py",
    "run": "src/agent6/app/run.py",
    "session_prepare": "src/agent6/app/session/prepare.py",
}
# The stage behind a call through a callable the lifecycle hands in, by the attribute called:
# the execution calls a run's gate step as `inputs.gate`.
_HANDED_STAGES = {"gate": ("src/agent6/app/run.py", "infer_gate")}


@functools.cache
def _functions(rel_path: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return a file's functions by name, nested ones included."""
    tree = ast.parse((_ROOT / rel_path).read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _stage_called(func: ast.expr, rel_path: str) -> tuple[str, str] | None:
    """Return the file and name of the stage a call's target names, or None for any other call."""
    if isinstance(func, ast.Name):
        return (rel_path, func.id) if func.id in _functions(rel_path) else None
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
        return None
    if func.attr in _HANDED_STAGES:
        return _HANDED_STAGES[func.attr]
    path = _STAGE_MODULES.get(func.value.id)
    return (path, func.attr) if path is not None and func.attr in _functions(path) else None


def _calls_in_order(
    rel_path: str, func: str, tier: tuple[str, ...], *, following: tuple[str, ...] = ()
) -> list[str]:
    """Return the tier functions a function calls, in source order, each once.

    A composition function's information is its order. A call to a stage, a function of the
    lifecycle's own modules, stands for the stage's calls. A context manager's calls come
    where its block ends, since that is where it releases. A nested function's body is left
    out: a closure's calls happen when it is called.

    Args:
        rel_path: The file.
        func: The calling function.
        tier: The functions that count, qualified when their receiver matters.
        following: The stages being read, so a stage that calls back into one is read once.

    Returns:
        The names.
    """
    target = _functions(rel_path)[func]
    members = set(tier)
    seen: list[str] = []

    def see(names: list[str]) -> None:
        seen.extend(name for name in names if name not in seen)

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name = ast.unparse(node.func)
            if name not in members and isinstance(node.func, ast.Attribute):
                name = node.func.attr
            stage = None if name in members else _stage_called(node.func, rel_path)
            if name in members:
                see([name])
            self.generic_visit(node)
            if stage is not None and stage[1] not in (func, *following):
                see(_calls_in_order(*stage, tier, following=(func, *following)))

        def visit_With(self, node: ast.With) -> None:
            for statement in node.body:
                self.visit(statement)
            for item in node.items:
                self.visit(item)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is target:
                self.generic_visit(node)

    V().visit(target)
    return seen


# The run lifecycle's stages worth drawing; the extractor reads their order from the source.
_RUN_LIFECYCLE_TIER = (
    "session_config",
    "headless_check",
    "route_preflight",
    "hold_workspace_and_select_isolation",
    "git_preflight",
    "infer_verify_if_unset",
    "settle_gate",
    "write_session_manifest",
    "build_session_providers",
    "build_session_tools",
    "wf.resume",
    "wf.run",
    "end_auto_stash",
    "finalize_auto_merge",
    "print_session_end",
    "session_end",
    "session_exit_code",
)
_DISPATCH_TIER = (
    "dispatch",
    "_dispatch_inner",
    "_run_handler",
    "_approve_mcp_call",
)


def _run_lifecycle_mermaid() -> str:
    """Return the calls `run_task` makes through its stages, the execution body among them.

    Raises:
        SystemExit: A lifecycle tier member matched no calls.
    """
    stages = _calls_in_order("src/agent6/app/run.py", "run_task", _RUN_LIFECYCLE_TIER)
    if missing := set(_RUN_LIFECYCLE_TIER) - set(stages):
        raise SystemExit(f"run-lifecycle tier matched no calls: {', '.join(sorted(missing))}")
    lines = ["graph TD", '    n_run_task["run_task"]']
    lines += [f'    {_nid(name)}["{name}"]' for name in stages]
    chain = ["run_task", *stages]
    lines += [f"    {_nid(a)} --> {_nid(b)}" for a, b in itertools.pairwise(chain)]
    return "\n".join(lines)


def _tool_name_constants() -> dict[str, str]:
    """Return each input class's `TOOL_NAME` from tools/schema.py."""
    tree = ast.parse((_ROOT / "src/agent6/tools/schema.py").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "TOOL_NAME"
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
                and stmt.value.value
            ):
                out[node.name] = stmt.value.value
    return out


def _handler_names() -> list[str]:
    """Return the tool names the handler table routes, in table order.

    The schema constants name the tools on the wire.

    Raises:
        SystemExit: A table key is not a known `TOOL_NAME`.
    """
    constants = _tool_name_constants()
    tree = ast.parse((_ROOT / "src/agent6/tools/dispatch.py").read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_HANDLERS"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        for key in node.value.keys:
            # `schema.<Input>.TOOL_NAME`: the class sits behind the module the table imports
            cls = key.value if isinstance(key, ast.Attribute) and key.attr == "TOOL_NAME" else None
            cls_name = cls.id if isinstance(cls, ast.Name) else getattr(cls, "attr", None)
            if cls_name in constants:
                names.append(constants[cls_name])
            else:
                shown = ast.dump(key) if key is not None else "**"
                raise SystemExit(f"handler table key is not a known <Input>.TOOL_NAME: {shown}")
    return names


def _dispatch_mermaid() -> str:
    """Return the gate chain every tool call passes, ending at the handler table as one node."""
    graph = _tier_callgraph("src/agent6/tools/dispatch.py", _DISPATCH_TIER)
    table = f'    n_table["handler table: {len(_handler_names())} tools"]'
    edge = f"    {_nid('run_handler')} -.->|by name| n_table"
    return "\n".join([graph, table, edge])


def _tool_names() -> str:
    """Return the dispatch table's tools as an inline list."""
    return ", ".join(f"`{name}`" for name in _handler_names())


_BLOCKS = {
    "layering": _layering_mermaid,
    "packages": _packages,
    "run-lifecycle": _run_lifecycle_mermaid,
    "substrate-names": _substrate_names,
    "tool-dispatch": _dispatch_mermaid,
    "tool-names": _tool_names,
}


def render(template: str) -> str:
    """Return the page rendered from the template."""
    out: list[str] = [
        "<!-- Generated from docs/architecture_template.md by docs/gen_diagrams.py;"
        " edit that, then regenerate. -->",
    ]
    for line in template.splitlines():
        diagram = _DIAGRAM.match(line)
        if diagram is not None:
            out.extend(["```mermaid", _BLOCKS[diagram.group(1)](), "```"])
            continue
        out.append(_GENERATED.sub(lambda m: _BLOCKS[m.group(1)](), line))
    return "\n".join(out) + "\n"


def main() -> None:
    """Write the page."""
    page = render(_TEMPLATE.read_text(encoding="utf-8"))
    _OUT.write_text(page, encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_ROOT)} ({len(page.splitlines())} lines)")


if __name__ == "__main__":
    sys.exit(main())
