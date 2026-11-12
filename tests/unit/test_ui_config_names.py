"""UI literals name real config fields, not retired spellings.

This is a static literal check, not data-flow analysis of dynamically assembled labels.
The generated config-page tests separately pin schema fields to their documentation.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections.abc import Iterator

import pytest
import tree_sitter
import tree_sitter_language_pack

from agent6 import config
from agent6.viewmodel import config_view

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_UI = _ROOT / "src/agent6/ui"
_ROOTS = "|".join(sorted({*config.Config.model_fields, "oauth"}))
_REFERENCE = re.compile(
    rf"(?<![\w.])\[?(?P<root>{_ROOTS})\]?"
    r"(?P<tail>(?:\.(?:[a-z_][a-z_0-9]*|<name>|\*))+)(?![\w])"
)
# Exact non-config literals: events, checks, filenames, modules, telemetry and a suggestion key.
_NON_CONFIG = {
    ("acp/updates.py", "budget.update"),
    ("cli/check_sandbox.py", "container.discovery"),
    ("cli/check_sandbox.py", "container.exposure"),
    ("cli/check_sandbox.py", "sandbox.grants"),
    ("cli/check_sandbox.py", "sandbox.refusal"),
    ("cli/completions_cmd.py", "agent6.fish"),
    ("cli/completions_cmd.py", "agent6.xsh"),
    ("cli/live.py", "agent6.ui.tui"),
    ("cli/machine_cmds.py", "machine.asm.toml"),
    ("cli/watch.py", "machine.asm.toml"),
    ("telemetry/export.py", "agent6.export.scope"),
    ("telemetry/export.py", "agent6.repo"),
    ("telemetry/otlp.py", "agent6.telemetry"),
    ("tui/__main__.py", "python -m agent6.ui.tui"),
    ("tui/machines.py", "machine.asm.toml"),
    ("web/client.js", "/api/config/suggest/parallel.models"),
}


def _python_strings(source: str) -> Iterator[tuple[int, str]]:
    tree = ast.parse(source)
    docstrings = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def _javascript_strings(source: str) -> Iterator[tuple[int, str]]:
    data = source.encode()
    parser = tree_sitter_language_pack.get_parser("javascript")
    tree = parser.parse(data)
    assert not tree.root_node.has_error
    query = tree_sitter.Query(
        tree_sitter_language_pack.get_language("javascript"),
        "(string) @literal (template_string) @literal",
    )
    for node in tree_sitter.QueryCursor(query).captures(tree.root_node).get("literal", []):
        line = data[: node.start_byte].count(b"\n") + 1
        yield line, data[node.start_byte + 1 : node.end_byte - 1].decode()


def _unknown_names(module: str, text: str) -> list[str]:
    if (module, text) in _NON_CONFIG:
        return []
    unknown: list[str] = []
    for match in _REFERENCE.finditer(text):
        name = match["root"] + match["tail"]
        # --secret writes a header value in secrets.toml, not the public config's list.
        if name == "telemetry.headers.<name>" and module in {
            "cli/config_args.py",
            "cli/config_secrets.py",
        }:
            continue
        if name.startswith("oauth."):
            name = "providers.<name>." + name
        names = (
            [name.replace("*", role, 1) for role in config.ModelsConfig.model_fields]
            if name.startswith("models.*")
            else [name]
        )
        for leaf in names:
            if config_view.field_schema(config.Config, leaf.split(".")) is None:
                unknown.append(leaf)
    return unknown


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            path,
            id=str(path.relative_to(_UI)),
            marks=pytest.mark.needs_grammar(language="javascript") if path.suffix == ".js" else (),
        )
        for path in sorted(_UI.rglob("*"))
        if path.suffix in {".js", ".py"}
    ],
)
def test_ui_config_literals_exist(path: pathlib.Path) -> None:
    """A retired leaf in help, an error, a label or a web string fails with its location."""
    source = path.read_text(encoding="utf-8")
    strings = _python_strings(source) if path.suffix == ".py" else _javascript_strings(source)
    module = str(path.relative_to(_UI))
    unknown = [
        f"{module}:{line}: {name}"
        for line, text in strings
        for name in _unknown_names(module, text)
    ]
    assert not unknown, "Unknown config references:\n" + "\n".join(unknown)


@pytest.mark.parametrize(
    "text",
    [
        "Set sandbox.retired true",
        "Default: [git].retired",
        "models.default.retired",
        "mcp.servers.*.sandbox.retired",
        "oauth.retired in the provider",
    ],
)
def test_unknown_config_names_are_rejected(text: str) -> None:
    assert _unknown_names("cli/common.py", text)
