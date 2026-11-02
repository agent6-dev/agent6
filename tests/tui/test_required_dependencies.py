from __future__ import annotations

import ast
import inspect
import types

import pytest

from agent6.ui.tui import config_forms, config_page, copy_method, forms


@pytest.mark.parametrize("module", [config_forms, config_page, copy_method, forms])
def test_textual_is_imported_as_a_required_dependency(module: types.ModuleType) -> None:
    """Required Textual imports cannot offer installation of a nonexistent extra."""
    source = inspect.getsource(module)
    tree = ast.parse(source)
    assert any(isinstance(node, ast.ImportFrom) and node.module == "textual" for node in tree.body)
    assert "agent6[tui]" not in source
