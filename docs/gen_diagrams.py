# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Generate docs/internals.md from docs/internals_template.md.

Each ``<!-- diagram: NAME -->`` marker becomes a mermaid block built from the
current source, so the diagrams cannot drift from the code:

- ``layering``: the top-level package graph, collapsed from ``tach show``'s
  module graph (the core layers drawn edge-by-edge, the shared substrate
  grouped as one cluster).
- ``turn-pipeline``: the agent loop's drive tier, AST-extracted from
  ``workflows/loop.py`` (direct ``self.X()`` calls between the named phase
  methods).

Run by the pages workflow before ``mkdocs build``; ``docs/internals.md`` is
generated output and never committed. Regenerate locally with
``uv run python docs/gen_diagrams.py``.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE = _ROOT / "docs" / "internals_template.md"
_OUT = _ROOT / "docs" / "internals.md"
_MARKER = re.compile(r"^<!-- diagram: ([a-z-]+) -->$")

# The documented layering, drawn edge-by-edge; everything else collapses into
# the shared-substrate cluster.
_CORE_LAYERS = ("ui", "app", "workflows", "tools", "sandbox")

# The drive tier of workflows/loop.py: the per-turn phases run/resume fan
# into. Deeper tiers regenerate from source (the extractor takes any name
# list); keeping this curated is what keeps the diagram readable.
_PIPELINE_TIER = (
    "run",
    "resume",
    "_drive_loop",
    "_turn_pre_call",
    "_turn_provider_call",
    "_turn_dispatch_tools",
    "_turn_auto_commit_and_metric",
    "_turn_review_triggers",
    "_turn_finish_gates",
    "_turn_notices",
    "_turn_stop_checks",
    "_maybe_compact",
    "_summarise_and_restart",
    "_maybe_handle_steer",
    "_save_resume_snapshot",
)


def _nid(name: str) -> str:
    """Mermaid-safe node id for *name*. Ids are DATA (package, method, and
    tool names), and mermaid claims bare words like ``graph``, ``call`` and
    ``end`` anywhere in a flowchart body; the prefix keeps every id off that
    list, and the raw name rides only inside a quoted label."""
    return "n_" + re.sub(r"\W", "_", name)


def _layering_mermaid() -> str:
    proc = subprocess.run(
        ["uv", "run", "tach", "show", "--mermaid", "-o", "/dev/stdout"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )
    return _layering_from_tach(proc.stdout)


def _layering_from_tach(mermaid_graph: str) -> str:
    edges: set[tuple[str, str]] = set()
    substrate: set[str] = set()
    for line in mermaid_graph.splitlines():
        m = re.match(r"\s*(\S+) --> (\S+)", line)
        if m is None:
            continue
        a, b = (part.split(".")[1] if "." in part else part for part in m.groups())
        if a == b or "agent6" in (a, b):
            continue
        if a in _CORE_LAYERS and b in _CORE_LAYERS:
            edges.add((a, b))
        else:
            substrate.update(x for x in (a, b) if x not in _CORE_LAYERS)
    lines = ["graph TD"]
    lines += [f'    {_nid(n)}["{n}"]' for n in _CORE_LAYERS if any(n in e for e in edges)]
    lines += [f"    {_nid(a)} --> {_nid(b)}" for a, b in sorted(edges)]
    lines.append('    subgraph substrate["shared substrate (every layer may use)"]')
    row = sorted(substrate)
    # Rows of six: one long invisible-link chain renders wider than a phone.
    for i in range(0, len(row), 6):
        lines.append("        " + " ~~~ ".join(f'{_nid(x)}["{x}"]' for x in row[i : i + 6]))
    lines.append("    end")
    return "\n".join(lines)


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


# The run lifecycle's stage functions (app/run.py composes them) and the
# dispatcher's gate chain; curated like the pipeline tier.
_RUN_LIFECYCLE_TIER = (
    "run_task",
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


def _dispatch_mermaid() -> str:
    """The dispatch gate chain plus the handler TABLE: handlers are reached
    through ``self._handlers[name]``, not direct calls, so those edges are
    read out of the table literal and drawn dashed (via table)."""
    graph = _tier_callgraph("src/agent6/tools/dispatch.py", _DISPATCH_TIER)
    tree = ast.parse((_ROOT / "src/agent6/tools/dispatch.py").read_text(encoding="utf-8"))
    handlers: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "_handlers"
            and isinstance(node.value, ast.Dict)
        ):
            for v in node.value.values:
                if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
                    handlers.append(v.attr)
    lines = [graph]
    for h in handlers:
        label = h.lstrip("_")
        lines.append(f'    {_nid(label)}["{label}"]')
        lines.append(f"    {_nid('run_handler')} -.->|table| {_nid(label)}")
    return "\n".join(lines)


_DIAGRAMS = {
    "layering": _layering_mermaid,
    "turn-pipeline": lambda: _tier_callgraph("src/agent6/workflows/loop.py", _PIPELINE_TIER),
    "run-lifecycle": lambda: _tier_callgraph("src/agent6/app/run.py", _RUN_LIFECYCLE_TIER),
    "tool-dispatch": _dispatch_mermaid,
}


def render(template: str) -> str:
    out: list[str] = [
        "<!-- Generated from docs/internals_template.md by docs/gen_diagrams.py;"
        " edit those, then regenerate. -->",
    ]
    for line in template.splitlines():
        marker = _MARKER.match(line)
        if marker is None:
            out.append(line)
            continue
        body = _DIAGRAMS[marker.group(1)]()
        out.extend(["```mermaid", body, "```"])
    return "\n".join(out) + "\n"


def main() -> None:
    page = render(_TEMPLATE.read_text(encoding="utf-8"))
    _OUT.write_text(page, encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_ROOT)} ({len(page.splitlines())} lines)")


if __name__ == "__main__":
    sys.exit(main())
