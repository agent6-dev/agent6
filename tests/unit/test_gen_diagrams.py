# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The generated architecture diagrams stay parseable by mermaid.

Node ids are data (package, method and tool names), and mermaid reserves bare words like `graph`,
`call` and `end` in a flowchart body, so every emitted id is prefixed and the raw name rides only
inside a quoted label.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import types

_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Words mermaid's flowchart lexer claims anywhere; a node id equal to one kills the diagram.
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


def _load_generator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_diagrams", _ROOT / "docs" / "gen_diagrams.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ids_in(diagram: str) -> set[str]:
    """Every token in node-id position, with labels stripped first in every shape mermaid has."""
    ids: set[str] = set()
    for line in diagram.splitlines()[1:]:
        stripped = line.strip()
        if stripped in ("end",) or stripped.startswith("subgraph "):
            continue
        bare = re.sub(r'"[^"]*"', "", stripped)
        bare = re.sub(r"\|[^|]*\|", "", bare)
        bare = re.sub(r"\[[^\]]*\]", "", bare)
        bare = re.sub(r"\{[^}]*\}", "", bare)
        for token in re.findall(r"[A-Za-z_]\w*", bare):
            ids.add(token)
    return ids


def test_no_emitted_node_id_is_a_mermaid_keyword() -> None:
    """Rendered over the real tree, every diagram's ids stay off the keyword list."""
    gen = _load_generator()
    page = gen.render((_ROOT / "docs" / "architecture_template.md").read_text(encoding="utf-8"))
    blocks = re.findall(r"```mermaid\n(.*?)```", page, re.S)
    assert blocks, "no diagrams rendered"
    for block in blocks:
        clashes = _ids_in(block) & _MERMAID_KEYWORDS
        assert not clashes, f"bare mermaid keyword(s) emitted as node ids: {sorted(clashes)}"


def test_a_tier_member_named_like_a_keyword_is_safe(tmp_path: pathlib.Path) -> None:
    """The callgraph extractor survives a source function named `end` or `call`."""
    gen = _load_generator()
    src = tmp_path / "mod.py"
    src.write_text(
        "def call() -> None:\n    end()\n\ndef end() -> None:\n    pass\n",
        encoding="utf-8",
    )
    # An absolute path: `_ROOT / abs` resolves to abs, so the extractor reads the fixture.
    diagram = gen._tier_callgraph(str(src), ("call", "end"))
    clashes = _ids_in(diagram) & _MERMAID_KEYWORDS
    assert not clashes, f"bare mermaid keyword(s) emitted as node ids: {sorted(clashes)}"
    assert '["call"]' in diagram and '["end"]' in diagram


def test_the_generator_never_spawns_uv() -> None:
    """The layering diagram invokes tach from the running interpreter's environment, never `uv`.

    `uv run` re-syncs the project, which swaps a wheel-installed agent6 for the checkout mid-suite.
    """
    src = (_ROOT / "docs" / "gen_diagrams.py").read_text(encoding="utf-8")
    assert '"uv"' not in src and "'uv'" not in src


def test_the_generated_tool_list_names_tools_a_model_can_call() -> None:
    """Every tool name the page lists resolves through the schema's TOOL_NAME constants."""
    from agent6.tools import schema

    gen = _load_generator()
    listed = re.findall(r"`([a-z0-9_]+)`", gen._tool_names())
    real = {
        obj.TOOL_NAME
        for obj in vars(schema).values()
        if isinstance(obj, type) and getattr(obj, "TOOL_NAME", "")
    }
    assert listed, "no tools listed"
    assert not set(listed) - real, sorted(set(listed) - real)
    assert "run_verify_command" in listed


def test_every_asset_the_site_config_references_exists() -> None:
    """Deleting a docs asset leaves no dangling `extra_css` or `extra_javascript` entry."""
    config = (_ROOT / "docs" / "mkdocs.yml").read_text(encoding="utf-8")
    refs = re.findall(r"^\s*-\s+(assets/\S+)$", config, re.M)
    assert refs, "no asset references found in the site config"
    assert not [r for r in refs if not (_ROOT / "docs" / r).is_file()]


def test_architecture_page_is_not_stale() -> None:
    """`docs/architecture.md` is rendered from the template plus the current source, committed.

    Regenerate with `uv run python docs/gen_diagrams.py`.
    """
    gen = _load_generator()
    template = (_ROOT / "docs" / "architecture_template.md").read_text(encoding="utf-8")
    committed = (_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert gen.render(template) == committed
