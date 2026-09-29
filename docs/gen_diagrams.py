# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Generate docs/architecture.md from docs/architecture_template.md.

Two markers expand from the current source: `<!-- diagram: NAME -->` becomes a
mermaid block and `<!-- generated: NAME -->` becomes a line of text. A diagram
carries a shape (the layer chain, the stage order, the gate chain); names that
only need listing are listed. Regenerate with `uv run python
docs/gen_diagrams.py`; pinned by tests/unit/test_gen_diagrams.py.
"""

from __future__ import annotations

import ast
import functools
import itertools
import pathlib
import re
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = _ROOT / "docs" / "architecture_template.md"
_OUT = _ROOT / "docs" / "architecture.md"
# A diagram owns its line; a generated list substitutes in place, so it can sit in a sentence.
_DIAGRAM = re.compile(r"^<!-- diagram: ([a-z-]+) -->$")
_GENERATED = re.compile(r"<!-- generated: ([a-z-]+) -->")

# The documented layering, top to bottom; every other top-level package is substrate.
_CORE_LAYERS = ("ui", "app", "harness", "tools", "sandbox")


def _nid(name: str) -> str:
    """Return a mermaid-safe node id; the prefix keeps a name such as `end` off mermaid's words."""
    return "n_" + re.sub(r"\W", "_", name)


@functools.cache
def _tach_graph() -> str:
    """Return tach's module graph as mermaid text."""
    # From this interpreter, never `uv run`: a uv spawn re-syncs, uninstalling a tested wheel.
    proc = subprocess.run(
        [sys.executable, "-m", "tach", "show", "--mermaid", "-o", "/dev/stdout"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )
    return proc.stdout


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


def _calls_in_order(
    rel_path: str, func: str, tier: tuple[str, ...], *, own_body_only: bool = False
) -> list[str]:
    """Return the tier functions a function calls, in source order, each once.

    A composition function's information is its order.

    Args:
        rel_path: The file.
        func: The calling function.
        tier: The functions that count.
        own_body_only: Leave nested functions' bodies out; a closure's calls happen
            when it is called.

    Returns:
        The names.
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

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is target or not own_body_only:
                self.generic_visit(node)

    V().visit(target)
    return seen


# The run lifecycle's stages worth drawing; the extractor reads their order from the source.
_RUN_LIFECYCLE_TIER = (
    "session_config",
    "headless_approval_refusal",
    "route_preflight",
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
    """Return `run_task`'s stages as one chain, the execution body spliced in at the hand-off."""
    outer = _calls_in_order(
        "src/agent6/app/run.py",
        "run_task",
        (*_RUN_LIFECYCLE_TIER, "run_execution"),
        own_body_only=True,
    )
    # The execution calls the lifecycle's gate step (`inputs.gate`) between the
    # providers and the tools; run's is the `_gate` closure in run_task.
    gate = _calls_in_order("src/agent6/app/run.py", "_gate", _RUN_LIFECYCLE_TIER)
    execution = _calls_in_order(
        "src/agent6/app/_execution.py", "run_execution", (*_RUN_LIFECYCLE_TIER, "gate")
    )
    body = [stage for name in execution for stage in (gate if name == "gate" else [name])]
    stages = [stage for name in outer for stage in (body if name == "run_execution" else [name])]
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

    Resolved through the schema's `TOOL_NAME` constants, never the method names,
    which advertise tools the model cannot call.

    Raises:
        SystemExit: A table key is not a known `TOOL_NAME`.
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
    "run-lifecycle": _run_lifecycle_mermaid,
    "tool-dispatch": _dispatch_mermaid,
    "substrate-names": _substrate_names,
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
