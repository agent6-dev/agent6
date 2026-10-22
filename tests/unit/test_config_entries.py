from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.config import io, layer, write


@pytest.mark.parametrize("to_repo", [False, True])
@pytest.mark.parametrize(
    ("table_path", "fields", "expected"),
    [
        (
            "providers",
            {"api_format": "openai", "base_url": " https://example.com/v1 "},
            {"api_format": "openai", "base_url": "https://example.com/v1"},
        ),
        (
            "mcp.servers",
            {"command": 'python -m server "two words"', "enabled": "false", "approve": "ask"},
            {
                "command": ["python", "-m", "server", "two words"],
                "enabled": False,
                "approve": "ask",
            },
        ),
        ("skills.state", {"state": "always"}, "always"),
    ],
)
def test_add_and_remove_entry(
    tmp_path: pathlib.Path,
    table_path: str,
    fields: dict[str, str],
    expected: object,
    *,
    to_repo: bool,
) -> None:
    """Entries land in the chosen layer and disappear whole, leaving siblings intact."""
    table = next(t for t in write.ENTRY_TABLES if t.path == table_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    target = paths.repo_config_path(repo) if to_repo else paths.global_config_path()
    assert write.add_entry(repo, table, " sibling ", fields, to_repo=to_repo) is None
    assert write.add_entry(repo, table, " example ", fields, to_repo=to_repo) is None
    doc = io.read_toml_file(target)
    for part in table_path.split("."):
        doc = doc[part]
    assert doc == {"sibling": expected, "example": expected}

    assert write.remove_entry(repo, table, "example", to_repo=to_repo) == write.UnsetResult(True)
    doc = io.read_toml_file(target)
    for part in table_path.split("."):
        doc = doc[part]
    assert doc == {"sibling": expected}
    assert write.remove_entry(repo, table, "example", to_repo=to_repo) == write.UnsetResult(False)
    layer.load_effective(repo)


def test_mcp_entry_keeps_blank_fields_and_removes_subtables(tmp_path: pathlib.Path) -> None:
    """An MCP update preserves omitted leaves; removal includes its sandbox grants."""
    table = next(t for t in write.ENTRY_TABLES if t.path == "mcp.servers")
    assert write.add_entry(tmp_path, table, "local", {"command": "server --stdio"}) is None
    assert write.set_config_value(tmp_path, "mcp.servers.local.sandbox.network", "none") is None
    assert write.add_entry(tmp_path, table, "local", {"command": " ", "approve": "allow"}) is None
    server = layer.load_effective(tmp_path).config.mcp.servers["local"]
    assert server.command == ("server", "--stdio")
    assert server.approve == "allow"
    assert server.sandbox is not None
    assert write.remove_entry(tmp_path, table, "local") == write.UnsetResult(True)
    assert not layer.load_effective(tmp_path).config.mcp.servers
    assert "local" not in paths.global_config_path().read_text()


def test_bad_entry_input_leaves_config_untouched(tmp_path: pathlib.Path) -> None:
    """A missing name or malformed command is refused before any write."""
    table = next(t for t in write.ENTRY_TABLES if t.path == "mcp.servers")
    error = write.add_entry(tmp_path, table, " ", {"command": "server"})
    assert error and "name is required" in error
    error = write.add_entry(tmp_path, table, "local", {"command": 'server "unclosed'})
    assert error and "command" in error and "quotation" in error
    assert not paths.global_config_path().exists()


@pytest.mark.parametrize(
    ("table_path", "choices", "field", "description_part"),
    [
        (
            "providers",
            {
                "api_format": ["anthropic", "openai", "chatgpt", "claude_code"],
                "deployment": ["direct", "vertex", "azure"],
            },
            "base_url",
            "endpoint",
        ),
        (
            "mcp.servers",
            {"enabled": ["true", "false"], "approve": ["ask", "allow"]},
            "command",
            "argv",
        ),
        ("skills.state", {"state": ["enabled", "disabled", "always"]}, "state", "Per-skill state"),
    ],
)
def test_entry_form_schema(
    table_path: str, choices: dict[str, list[str]], field: str, description_part: str
) -> None:
    """Forms expose literal and boolean choices and the schema's field descriptions."""
    table = next(t for t in write.ENTRY_TABLES if t.path == table_path)
    assert write.entry_choices(table) == choices
    assert description_part in write.entry_field_description(table, field)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("providers.foo.base_url", ("providers", "foo")),
        ("mcp.servers.local.command", ("mcp.servers", "local")),
        ("mcp.servers.local.sandbox.network", ("mcp.servers", "local")),
        ("skills.state.test-skill", ("skills.state", "test-skill")),
        ("sandbox.network", None),
        ("providers", None),
        ("mcp.servers", None),
        ("skills.state", None),
        ("review.reviewers", None),
    ],
)
def test_entry_of(key: str, expected: tuple[str, str] | None) -> None:
    """Only entry leaves resolve to a named entry, including nested MCP grants."""
    found = write.entry_of(key)
    assert (None if found is None else (found[0].path, found[1])) == expected
