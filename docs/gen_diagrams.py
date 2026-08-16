# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Generate docs/architecture.md from docs/architecture_template.md.

Two marker kinds expand from the current source, so the page cannot drift
from the code:

- ``<!-- diagram: NAME -->`` becomes a mermaid block.
- ``<!-- generated: NAME -->`` becomes a line of text (the package and tool
  name lists, which read better as prose than as boxes).

A diagram carries the SHAPE: the layer chain, the stage order, the gate
chain. Names that only need listing are listed. Drawing every module and
every tool as a node produced a page-wide hairball at a tenth the legible
type size.

Run by the pages workflow before ``mkdocs build``; ``docs/architecture.md`` is
committed generated output, pinned by tests/unit/test_gen_diagrams.py.
Regenerate with ``uv run python docs/gen_diagrams.py``.
"""

from __future__ import annotations

import ast
import functools
import itertools
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE = _ROOT / "docs" / "architecture_template.md"
_OUT = _ROOT / "docs" / "architecture.md"
# A diagram owns its line and expands to a fenced block; a generated name
# list substitutes in place, so it can sit inside a sentence.
_DIAGRAM = re.compile(r"^<!-- diagram: ([a-z-]+) -->$")
_GENERATED = re.compile(r"<!-- generated: ([a-z-]+) -->")

# The documented layering, top to bottom. Every other top-level package is
# shared substrate.
_CORE_LAYERS = ("ui", "app", "workflows", "tools", "sandbox")


def _nid(name: str) -> str:
    """Mermaid-safe node id for *name*. Ids are DATA (package, method, and
    tool names), and mermaid claims bare words like ``graph``, ``call`` and
    ``end`` anywhere in a flowchart body; the prefix keeps every id off that
    list, and the raw name rides only inside a quoted label."""
    return "n_" + re.sub(r"\W", "_", name)


@functools.cache
def _tach_graph() -> str:
    # tach runs from the current interpreter's environment, never via
    # `uv run`: a uv spawn re-syncs the project, which would uninstall a
    # wheel-installed agent6 from the venv under a suite testing that wheel.
    proc = subprocess.run(
        [sys.executable, "-m", "tach", "show", "--mermaid", "-o", "/dev/stdout"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )
    return proc.stdout


def _package_edges(mermaid_graph: str) -> tuple[set[tuple[str, str]], set[str]]:
    """`tach show`'s module graph as (top-level edges, substrate packages)."""
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
    """The layer chain, plus any import that climbs it.

    Every layer may import every layer below it, so drawing all the real
    edges draws a fully-connected five-node mesh that says nothing the chain
    does not. What the chain cannot show is a violation, so an edge running
    upward is drawn dashed and labelled: tach checks the same rule, and
    seeing one here means the map is stale.
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
    """The substrate packages, as a sorted inline list."""
    _, substrate = _package_edges(_tach_graph())
    return ", ".join(f"`{name}`" for name in sorted(substrate))


def _tier_callgraph(rel_path: str, tier: tuple[str, ...]) -> str:
    """Mermaid callgraph of the named tier in one file: edges are direct
    calls (``self.X(...)`` or bare ``X(...)``) between tier members."""
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

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

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


def _calls_in_order(rel_path: str, func: str, tier: tuple[str, ...]) -> list[str]:
    """The *tier* functions *func* calls, in source order, first call only.

    A composition function's information is its ORDER; a star of edges from
    the caller carries none of it.
    """
    tree = ast.parse((_ROOT / rel_path).read_text(encoding="utf-8"))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func
    )
    members = set(tier)
    seen: list[str] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            if name in members and name not in seen:
                seen.append(name)
            self.generic_visit(node)

    V().visit(target)
    return seen


# The run lifecycle's stage functions, in the order run_task composes them
# (the extractor reads the order out of the source; this list decides which
# calls are stages worth drawing).
_RUN_LIFECYCLE_TIER = (
    "session_config",
    "headless_approval_refusal",
    "select_isolation",
    "git_preflight",
    "infer_verify_if_unset",
    "drop_gate_if_unrunnable",
    "pin_gate",
    "write_session_manifest",
    "build_session_providers",
    "build_session_tools",
    "finalize_auto_stash",
    "finalize_auto_merge",
    "print_session_end",
    "fire_notify_hook",
    "session_exit_code",
)
_DISPATCH_TIER = (
    "dispatch",
    "_dispatch_inner",
    "_run_handler",
    "_approve_mcp_call",
)


def _run_lifecycle_mermaid() -> str:
    """`run_task`'s stages as one chain, in the order it calls them."""
    stages = _calls_in_order("src/agent6/app/run.py", "run_task", _RUN_LIFECYCLE_TIER)
    lines = ["graph TD", '    n_run_task["run_task"]']
    lines += [f'    {_nid(name)}["{name}"]' for name in stages]
    chain = ["run_task", *stages]
    lines += [f"    {_nid(a)} --> {_nid(b)}" for a, b in itertools.pairwise(chain)]
    return "\n".join(lines)


def _tool_name_constants() -> dict[str, str]:
    """`{input class: TOOL_NAME}` from tools/schema.py."""
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
    """The tool names the dispatcher's handler table routes, in table order.

    Resolved through the schema's `TOOL_NAME` constants, never the handler
    METHOD names: `RunVerifyInput` routes `run_verify_command` while its
    method is `_run_verify`, so the method name advertises a tool the model
    cannot call. An unresolvable key is a loud failure, not a guess.
    """
    constants = _tool_name_constants()
    tree = ast.parse((_ROOT / "src/agent6/tools/dispatch.py").read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "_handlers"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        for key in node.value.keys:
            if (
                isinstance(key, ast.Attribute)
                and key.attr == "TOOL_NAME"
                and isinstance(key.value, ast.Name)
                and key.value.id in constants
            ):
                names.append(constants[key.value.id])
            else:
                raise SystemExit(
                    f"handler table key is not a known <Input>.TOOL_NAME: {ast.dump(key)}"
                )
    return names


def _dispatch_mermaid() -> str:
    """The gate chain every tool call passes, ending at the handler table.

    The table's entries are a SET reached by name, not a call sequence:
    drawing one node per tool fanned twenty-odd dead-end boxes across the
    page. The count rides on the node and the names are listed below it.
    """
    graph = _tier_callgraph("src/agent6/tools/dispatch.py", _DISPATCH_TIER)
    table = f'    n_table["handler table: {len(_handler_names())} tools"]'
    edge = f"    {_nid('run_handler')} -.->|by name| n_table"
    return "\n".join([graph, table, edge])


def _tool_names() -> str:
    """The dispatch table's tools, as an inline list in table order."""
    return ", ".join(f"`{name}`" for name in _handler_names())


_BLOCKS = {
    "layering": _layering_mermaid,
    "run-lifecycle": _run_lifecycle_mermaid,
    "tool-dispatch": _dispatch_mermaid,
    "substrate-names": _substrate_names,
    "tool-names": _tool_names,
}


def render(template: str) -> str:
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
    page = render(_TEMPLATE.read_text(encoding="utf-8"))
    _OUT.write_text(page, encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_ROOT)} ({len(page.splitlines())} lines)")


if __name__ == "__main__":
    sys.exit(main())
