# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The generated internals diagrams stay parseable by mermaid.

Node ids are DATA (package, method, and tool names lifted from source), and
mermaid reserves bare words like ``graph``, ``call``, and ``end`` anywhere in
a flowchart body: the package named ``graph`` rendered the layering diagram
as a syntax-error box on the published site. Every emitted id must therefore
be prefixed, with the raw name carried only inside a quoted label."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]

# Words mermaid's flowchart lexer claims anywhere in the body; a bare node id
# equal to one parses as the keyword and kills the whole diagram.
_MERMAID_KEYWORDS = {
    "call",
    "class",
    "classDef",
    "click",
    "direction",
    "end",
    "flowchart",
    "graph",
    "linkStyle",
    "style",
    "subgraph",
}


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_diagrams", _ROOT / "docs" / "gen_diagrams.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ids_in(diagram: str) -> set[str]:
    """Every token in node-id position: quoted labels and |edge labels| are
    stripped first, so what remains on a body line is ids, arrows, and link
    operators."""
    ids: set[str] = set()
    for line in diagram.splitlines()[1:]:
        stripped = line.strip()
        if stripped in ("end",) or stripped.startswith("subgraph "):
            continue
        bare = re.sub(r'"[^"]*"', "", stripped)
        bare = re.sub(r"\|[^|]*\|", "", bare)
        for token in re.findall(r"[A-Za-z_]\w*", bare):
            ids.add(token)
    return ids


def test_no_emitted_node_id_is_a_mermaid_keyword() -> None:
    """Rendered over the real tree: every diagram's ids stay off the keyword
    list (the layering diagram used to emit the ``graph`` package bare)."""
    gen = _load_generator()
    page = gen.render((_ROOT / "docs" / "internals_template.md").read_text(encoding="utf-8"))
    blocks = re.findall(r"```mermaid\n(.*?)```", page, re.S)
    assert blocks, "no diagrams rendered"
    for block in blocks:
        clashes = _ids_in(block) & _MERMAID_KEYWORDS
        assert not clashes, f"bare mermaid keyword(s) emitted as node ids: {sorted(clashes)}"


def test_a_tier_member_named_like_a_keyword_is_safe(tmp_path: Path) -> None:
    """The callgraph extractor must survive a source function named ``end`` or
    ``call``: the name rides into the diagram only as a quoted label, never as
    the id."""
    gen = _load_generator()
    src = tmp_path / "mod.py"
    src.write_text(
        "def call() -> None:\n    end()\n\ndef end() -> None:\n    pass\n",
        encoding="utf-8",
    )
    # An absolute path: pathlib's `_ROOT / abs` resolves to abs, so the
    # extractor reads the fixture instead of a repo file.
    diagram = gen._tier_callgraph(str(src), ("call", "end"))
    clashes = _ids_in(diagram) & _MERMAID_KEYWORDS
    assert not clashes, f"bare mermaid keyword(s) emitted as node ids: {sorted(clashes)}"
    assert '["call"]' in diagram and '["end"]' in diagram
