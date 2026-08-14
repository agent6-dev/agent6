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
    "_turn_critic_triggers",
    "_turn_finish_gates",
    "_turn_notices",
    "_turn_stop_checks",
    "_maybe_compact",
    "_summarise_and_restart",
    "_maybe_handle_steer",
    "_save_resume_snapshot",
)


def _layering_mermaid() -> str:
    proc = subprocess.run(
        ["uv", "run", "tach", "show", "--mermaid", "-o", "/dev/stdout"],
        capture_output=True,
        text=True,
        check=True,
        cwd=_ROOT,
    )
    edges: set[tuple[str, str]] = set()
    substrate: set[str] = set()
    for line in proc.stdout.splitlines():
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
    lines += [f"    {a} --> {b}" for a, b in sorted(edges)]
    lines.append('    subgraph substrate["shared substrate (every layer may use)"]')
    row = sorted(substrate)
    # Rows of six: one long invisible-link chain renders wider than a phone.
    for i in range(0, len(row), 6):
        lines.append("        " + " ~~~ ".join(row[i : i + 6]))
    lines.append("    end")
    return "\n".join(lines)


def _pipeline_mermaid() -> str:
    tree = ast.parse((_ROOT / "src/agent6/workflows/loop.py").read_text(encoding="utf-8"))
    tier = set(_PIPELINE_TIER)
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
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in tier
                and self.cur in tier
                and node.func.attr != self.cur
            ):
                edges.add((self.cur, node.func.attr))
            self.generic_visit(node)

    V().visit(tree)
    connected = {n for e in edges for n in e}
    lines = ["graph TD"]
    for name in _PIPELINE_TIER:
        if name in connected:
            label = name.lstrip("_")
            lines.append(f'    {name.lstrip("_")}["{label}"]')
    for a, b in sorted(edges):
        lines.append(f"    {a.lstrip('_')} --> {b.lstrip('_')}")
    return "\n".join(lines)


_DIAGRAMS = {
    "layering": _layering_mermaid,
    "turn-pipeline": _pipeline_mermaid,
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
