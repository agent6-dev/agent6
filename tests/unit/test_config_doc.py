# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Staleness pin for the generated config page.

``docs/config.md`` is rendered from ``docs/config_template.md`` plus the config model by
``docs/gen_config.py``; this re-renders it in memory and asserts the committed file matches. The fix
is never to edit the page: run ``uv run python docs/gen_config.py``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

from agent6.config import layer

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_generator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("gen_config", _ROOT / "docs" / "gen_config.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_page_is_not_stale() -> None:
    gen = _load_generator()
    template = (_ROOT / "docs" / "config_template.md").read_text(encoding="utf-8")
    generated: str = gen.render(template)
    committed = (_ROOT / "docs" / "config.md").read_text(encoding="utf-8")
    assert generated == committed, (
        "docs/config.md is stale; regenerate it with: uv run python docs/gen_config.py"
    )


def test_every_leaf_has_a_description() -> None:
    """Every leaf has a description; an empty Meaning cell looks complete and says nothing."""
    gen = _load_generator()
    undescribed = sorted(path for path, (_, desc) in gen.leaves().items() if not desc.strip())
    assert not undescribed, f"config leaves with no Field(description=...): {undescribed}"


def test_every_leaf_reaches_the_page() -> None:
    """Every config leaf appears as a row in the rendered page.

    The staleness pin re-renders the same template, so a `<!-- config-table: ... -->` marker that
    stops matching loses the same rows on both sides.
    """
    gen = _load_generator()
    page = (_ROOT / "docs" / "config.md").read_text(encoding="utf-8")

    def documented(path: str) -> bool:
        parts = path.split(".")
        return any(f"`{'.'.join(parts[i:])}`" in page for i in range(len(parts)))

    missing = sorted(path for path in gen.leaves() if not documented(path))
    assert not missing, f"config leaves with no row on the page: {missing}"


def test_layer_table_names_the_config_file_layer_as_config_show_does(
    tmp_path: pathlib.Path,
) -> None:
    """The layer table names the `--config FILE` layer as `config show` and `config get` print it.

    The page called it `explicit`, a name no surface prints.
    """
    explicit = tmp_path / "x.toml"
    explicit.write_text('[sandbox]\nnetwork = "session"\n', encoding="utf-8")
    printed = layer.load_effective(tmp_path, explicit).sources["sandbox.network"]
    page = (_ROOT / "docs" / "config.md").read_text(encoding="utf-8")
    rows = [line.split("|") for line in page.splitlines() if line.startswith("|")]
    assert [row[1].strip() for row in rows if row[2].strip() == "`--config FILE`"] == [printed]
