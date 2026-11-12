from __future__ import annotations

import concurrent.futures
import pathlib
import threading
from collections.abc import Callable

import pytest

from agent6 import paths, portable
from agent6.config import io, layer, write


def _interleave_edits(
    target: pathlib.Path,
    waiting_edit: Callable[[], object],
    held_edit: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object]:
    """Queue one editor behind a real lock while its holder completes another edit."""
    attempted = threading.Event()
    real_lock = portable.lock_exclusive

    def observe_lock(fd: int, *, blocking: bool) -> None:
        attempted.set()
        real_lock(fd, blocking=blocking)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with portable.locked_file(target) as held:
            assert held
            monkeypatch.setattr(portable, "lock_exclusive", observe_lock)
            waiting = executor.submit(waiting_edit)
            assert attempted.wait(timeout=5), "the queued editor did not reach the lock"
            result = held_edit()
        return waiting.result(timeout=5), result


@pytest.mark.parametrize("to_repo", [False, True])
@pytest.mark.parametrize("remove", [False, True])
def test_reviewer_edits_read_the_roster_under_the_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, to_repo: bool, remove: bool
) -> None:
    """A queued append or removal keeps the reviewer's edit made by the lock holder."""
    assert (
        write.set_config_value(
            tmp_path, "review.reviewers", '[{ focus = "first" }]', to_repo=to_repo
        )
        is None
    )
    target = paths.repo_config_path(tmp_path) if to_repo else paths.global_config_path()

    def waiting_edit() -> str | None:
        if remove:
            return write.remove_reviewer(tmp_path, 0, to_repo=to_repo)
        return write.add_reviewer(tmp_path, {"focus": "queued"}, to_repo=to_repo)

    assert _interleave_edits(
        target,
        waiting_edit,
        lambda: write.add_reviewer(tmp_path, {"focus": "holder"}, to_repo=to_repo),
        monkeypatch,
    ) == (None, None)
    expected = ["holder"] if remove else ["first", "holder", "queued"]
    assert [r.focus for r in layer.load_effective(tmp_path).config.review.reviewers] == expected


@pytest.mark.parametrize("to_repo", [False, True])
@pytest.mark.parametrize("remove", [False, True])
def test_entry_edits_preserve_a_concurrent_sibling(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, to_repo: bool, remove: bool
) -> None:
    """An entry update or removal cannot erase a sibling added while it waits for the lock."""
    table = next(t for t in write.ENTRY_TABLES if t.path == "mcp.servers")
    assert write.add_entry(tmp_path, table, "first", {"command": "server"}, to_repo=to_repo) is None
    target = paths.repo_config_path(tmp_path) if to_repo else paths.global_config_path()

    def waiting_edit() -> str | write.UnsetResult | None:
        if remove:
            return write.remove_entry(tmp_path, table, "first", to_repo=to_repo)
        return write.add_entry(tmp_path, table, "first", {"approve": "allow"}, to_repo=to_repo)

    assert _interleave_edits(
        target,
        waiting_edit,
        lambda: write.add_entry(tmp_path, table, "sibling", {"command": "other"}, to_repo=to_repo),
        monkeypatch,
    ) == (write.UnsetResult(True) if remove else None, None)
    servers = layer.load_effective(tmp_path).config.mcp.servers
    assert servers["sibling"].command == ("other",)
    if remove:
        assert "first" not in servers
    else:
        assert servers["first"].command == ("server",)
        assert servers["first"].approve == "allow"


@pytest.mark.parametrize("to_repo", [False, True])
@pytest.mark.parametrize(
    ("table_path", "fields", "expected"),
    [
        (
            "providers",
            {"backend": "openai", "base_url": " https://example.com/v1 "},
            {"backend": "openai", "base_url": "https://example.com/v1"},
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


@pytest.mark.parametrize("to_repo", [False, True])
def test_add_reviewer_uses_target_resolution(tmp_path: pathlib.Path, *, to_repo: bool) -> None:
    """An append keeps the chosen layer's effective roster, not a higher layer's roster."""
    assert (
        write.set_config_value(
            tmp_path, "review.reviewers", '[{ focus = "global", effort = "off" }]'
        )
        is None
    )
    assert (
        write.set_config_value(tmp_path, "review.reviewers", '[{ focus = "repo" }]', to_repo=True)
        is None
    )
    assert (
        write.add_reviewer(
            tmp_path,
            {
                "focus": " tests ",
                "model": "",
                "effort": "high",
                "temperature": "0",
            },
            to_repo=to_repo,
        )
        is None
    )
    target = paths.repo_config_path(tmp_path) if to_repo else paths.global_config_path()
    first = {"focus": "repo"} if to_repo else {"focus": "global", "effort": "off"}
    assert io.read_toml_file(target)["review"]["reviewers"] == [
        first,
        {"focus": "tests", "effort": "high", "temperature": 0.0},
    ]


def test_a_roster_written_as_an_array_of_tables_takes_an_edit(tmp_path: pathlib.Path) -> None:
    """`[[review.reviewers]]` blocks, the natural TOML for a roster, take an add and a remove.

    The writer refused the edit: the blocks declared `review` without a `[review]` header,
    so the roster editor failed on a hand-written config.
    """
    cfg = paths.global_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('[[review.reviewers]]\nfocus = "security"\n', encoding="utf-8")

    assert write.add_reviewer(tmp_path, {"focus": "tests"}) is None
    roster = layer.load_global_only().config.review.reviewers
    assert [r.focus for r in roster] == ["security", "tests"]
    assert write.remove_reviewer(tmp_path, 0) is None
    assert [r.focus for r in layer.load_global_only().config.review.reviewers] == ["tests"]


@pytest.mark.parametrize("to_repo", [False, True])
def test_remove_reviewer_by_position_and_keep_empty_override(
    tmp_path: pathlib.Path, *, to_repo: bool
) -> None:
    """Shared focuses stay distinct; removing the last inherited reviewer writes an empty list."""
    assert (
        write.set_config_value(
            tmp_path,
            "review.reviewers",
            '[{ focus = "tests", effort = "off" }, { focus = "tests", temperature = 0.5 }]',
        )
        is None
    )
    assert write.remove_reviewer(tmp_path, 0, to_repo=to_repo) is None
    target = paths.repo_config_path(tmp_path) if to_repo else paths.global_config_path()
    assert io.read_toml_file(target)["review"]["reviewers"] == [
        {"focus": "tests", "temperature": 0.5}
    ]
    assert write.remove_reviewer(tmp_path, 0, to_repo=to_repo) is None
    assert io.read_toml_file(target)["review"]["reviewers"] == []
    assert layer.load_effective(tmp_path).config.review.reviewers == ()
    if to_repo:
        assert len(layer.load_global_only().config.review.reviewers) == 2


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"focus": " "},
        {"focus": "tests", "provider": "p"},
        {"focus": "tests", "temperature": "3"},
        {"focus": "tests", "temperature": "warm"},
        {"focus": "tests", "effort": "unknown"},
        {"focus": "tests", "unknown": "value"},
    ],
)
def test_invalid_reviewer_keeps_roster(tmp_path: pathlib.Path, fields: dict[str, str]) -> None:
    """Invalid form values cannot replace the existing roster."""
    assert write.set_config_value(tmp_path, "review.reviewers", '[{ focus = "security" }]') is None
    target = paths.global_config_path()
    before = target.read_text()
    assert write.add_reviewer(tmp_path, fields)
    assert target.read_text() == before


@pytest.mark.parametrize("index", [-1, 1])
def test_remove_reviewer_refuses_out_of_range(tmp_path: pathlib.Path, index: int) -> None:
    """An invalid position cannot remove a different reviewer or rewrite the config."""
    assert write.set_config_value(tmp_path, "review.reviewers", '[{ focus = "security" }]') is None
    target = paths.global_config_path()
    before = target.read_text()
    assert write.remove_reviewer(tmp_path, index)
    assert target.read_text() == before


def test_reviewer_effort_choices() -> None:
    """The optional effort field offers its schema choices to both forms."""
    assert write.reviewer_choices() == {"effort": ["off", "low", "medium", "high", "xhigh", "max"]}


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
                "backend": ["anthropic", "openai", "chatgpt", "responses", "claude_code"],
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
        ("skills.state", {"state": ["enabled", "disabled", "always"]}, "state", "per-skill state"),
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


def test_preset_key_prefixes_the_table_only_under_a_preset() -> None:
    """An edit under the selected preset writes `presets.<name>.<key>`; none writes the key.

    The `[agent6]` table and the preset tables never nest under the selected preset.
    """
    assert write.preset_key("fast", "sandbox.run_commands") == "presets.fast.sandbox.run_commands"
    assert write.preset_key("", "sandbox.run_commands") == "sandbox.run_commands"
    assert write.preset_key("fast", "agent6.preset") == "agent6.preset"
    assert (
        write.preset_key("fast", "presets.slow.sandbox.network") == "presets.slow.sandbox.network"
    )


@pytest.mark.parametrize("to_repo", [False, True])
def test_add_preset_creates_a_table_and_selects_it(
    tmp_path: pathlib.Path, *, to_repo: bool
) -> None:
    """Add preset writes `[presets.<name>]` once and the selector; Remove preset drops both."""
    assert write.add_preset(tmp_path, "fast", to_repo=to_repo) is None
    target = paths.repo_config_path(tmp_path) if to_repo else paths.global_config_path()
    assert "[presets.fast]" in target.read_text(encoding="utf-8")
    assert io.read_toml_file(target)["agent6"]["preset"] == "fast"
    assert write.preset_layer(tmp_path, "fast") is to_repo
    assert (
        write.set_config_value(tmp_path, "presets.fast.sandbox.network", "host", to_repo=to_repo)
        is None
    )
    assert write.add_preset(tmp_path, "fast", to_repo=to_repo) is None
    assert io.read_toml_file(target)["presets"]["fast"] == {"sandbox": {"network": "host"}}
    assert write.add_preset(tmp_path, " ", to_repo=to_repo) == "preset name is required"
    assert (
        write.add_preset(tmp_path, "none", to_repo=to_repo)
        == '"none" is reserved for selecting no preset'
    )
    assert write.remove_preset(tmp_path, "fast").removed is True
    saved = io.read_toml_file(target)
    assert "presets" not in saved and "agent6" not in saved
    assert write.preset_layer(tmp_path, "fast") is None
    assert write.remove_preset(tmp_path, "fast").removed is False
    # A selected built-in has no table: Remove preset still resets the selector.
    assert write.set_config_value(tmp_path, "agent6.preset", "review", to_repo=to_repo) is None
    assert write.remove_preset(tmp_path, "review").removed is True
    assert "agent6" not in io.read_toml_file(target)


def test_entries_and_reviewers_land_under_the_selected_preset(tmp_path: pathlib.Path) -> None:
    """An entry or a reviewer added under a preset writes its table, and is removed from it."""
    assert write.add_preset(tmp_path, "fast") is None
    provider = next(t for t in write.ENTRY_TABLES if t.path == "providers")
    fields = {"backend": "openai", "base_url": "https://acme.example/v1", "api_key_env": "ACME"}
    assert write.add_entry(tmp_path, provider, "acme", fields, preset="fast") is None
    target = paths.global_config_path()
    assert io.read_toml_file(target)["presets"]["fast"]["providers"]["acme"]["backend"] == "openai"
    assert "providers" not in io.read_toml_file(target)
    assert write.add_reviewer(tmp_path, {"focus": "tests"}, preset="fast") is None
    assert io.read_toml_file(target)["presets"]["fast"]["review"]["reviewers"] == [
        {"focus": "tests"}
    ]
    assert write.remove_reviewer(tmp_path, 0, preset="fast") is None
    assert io.read_toml_file(target)["presets"]["fast"]["review"]["reviewers"] == []
    assert write.remove_entry(tmp_path, provider, "acme", preset="fast").removed is True
    assert "providers" not in io.read_toml_file(target)["presets"]["fast"]


def test_a_write_under_a_built_in_preset_keeps_the_built_ins_values(tmp_path: pathlib.Path) -> None:
    """The first `presets.review.*` write copies the built-in in, so `review` keeps its reviewers.

    A `[presets.<name>]` table replaces the built-in of that name wholesale; one page edit
    under a selected built-in dropped every other value it set.
    """
    assert write.set_config_value(tmp_path, "presets.review.sandbox.network", "host") is None
    saved = io.read_toml_file(paths.global_config_path())["presets"]["review"]
    assert saved["sandbox"] == {"network": "host"}
    assert saved["review"]["when"] == "finish"
    assert saved["review"]["reviewers"] == [
        {"focus": "security"},
        {"focus": "correctness"},
        {"focus": "tests"},
    ]
    # A second write under the same name seeds nothing more.
    assert write.set_config_value(tmp_path, "presets.review.review.concurrency", "2") is None
    saved = io.read_toml_file(paths.global_config_path())["presets"]["review"]
    assert saved["review"]["concurrency"] == 2 and saved["sandbox"] == {"network": "host"}
    assert write.set_config_value(tmp_path, "agent6.preset", "review") is None
    eff = layer.load_effective(tmp_path)
    assert eff.config.sandbox.network == "host" and len(eff.config.review.reviewers) == 3
