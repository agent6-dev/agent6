# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The in-process file boundary (`Workspace`).

`sandbox.hide_paths` was wired only into the jail policy, so the tools -- which
run IN-PROCESS, outside the jail, and ask no approval -- bypassed it: `read_file`
returned a hidden secret, `list_dir` showed it, and `apply_edit` WROTE into one.
With a workspace root containing agent6's own config dir (root=$HOME) that
extended to `secrets.toml` and to `config.toml`, whose next load sets isolation
and run_commands -- cross-run loosening by persistence.

The tools are the front door of the file axis and the jail is the fence, so the
boundary is derived from config VALUES and holds at EVERY isolation level: a
degradation (auto falling back, macOS having no jail) must never widen it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.errors import ToolError

_SECRET = "AWS_SECRET_ACCESS_KEY=leaked-xyz"


def _dispatcher(root: Path, cfg: Config, isolation: str = "none") -> ToolDispatcher:
    return ToolDispatcher(root=root, config=cfg, isolation=isolation)  # pyright: ignore[reportArgumentType]


def _hiding(root: Path, *rel: str) -> Config:
    return Config.model_validate({"sandbox": {"hide_paths": [str(root / r) for r in rel]}})


def _dispatch(root: Path, cfg: Config, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    d = _dispatcher(root, cfg)
    try:
        return d.dispatch(tool, args).to_wire()
    finally:
        d.close()


def test_read_file_refuses_a_hidden_path(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(_SECRET, encoding="utf-8")
    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(tmp_path, _hiding(tmp_path, ".env"), "read_file", {"path": ".env"})


def test_list_dir_hides_the_entry_but_says_how_many(tmp_path: Path) -> None:
    """Filtered, not named: the listing stays true ("something is hidden")
    without disclosing what, and the model stops probing."""
    (tmp_path / ".env").write_text(_SECRET, encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = _dispatch(tmp_path, _hiding(tmp_path, ".env"), "list_dir", {"path": "."})
    assert out["entries"] == ["main.py"]
    assert out["hidden"] == 1


def test_list_dir_omits_the_count_when_nothing_is_hidden(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = _dispatch(tmp_path, Config(), "list_dir", {"path": "."})
    assert out["entries"] == ["main.py"]
    assert "hidden" not in out


def test_apply_edit_refuses_to_write_a_hidden_path(tmp_path: Path) -> None:
    """The write half: refusing the read while allowing the write would leave
    the model able to plant content in a path the operator hid."""
    secret = tmp_path / ".env"
    secret.write_text(_SECRET, encoding="utf-8")
    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(
            tmp_path,
            _hiding(tmp_path, ".env"),
            "apply_edit",
            {"path": ".env", "edits": [{"old_string": "leaked-xyz", "new_string": "PWNED"}]},
        )
    assert secret.read_text(encoding="utf-8") == _SECRET


def test_a_normal_path_is_untouched(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = _dispatch(tmp_path, _hiding(tmp_path, ".env"), "read_file", {"path": "main.py"})
    assert out["content"] == "x = 1\n"


def test_a_hidden_file_never_reaches_the_symbol_index(tmp_path: Path) -> None:
    """find_definition would otherwise leak the symbol NAMES and line numbers
    of a file nothing is allowed to read."""
    (tmp_path / "secrets.py").write_text("def leaked_symbol():\n    pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def public_symbol():\n    pass\n", encoding="utf-8")
    cfg = _hiding(tmp_path, "secrets.py")
    defs = _dispatch(tmp_path, cfg, "find_definition", {"symbol": "leaked_symbol"})
    assert defs["definitions"] == []
    ok = _dispatch(tmp_path, cfg, "find_definition", {"symbol": "public_symbol"})
    assert [d["path"] for d in ok["definitions"]] == ["main.py"]


@pytest.mark.parametrize("isolation", ["strict", "hardened", "none"])
def test_the_boundary_holds_at_every_isolation_level(tmp_path: Path, isolation: str) -> None:
    """The rule: config VALUES define the boundary, never the isolation level.
    `none` has no jail at all (and is what macOS resolves to), so a boundary
    that tracked the level would vanish exactly where it is the only one left.
    """
    (tmp_path / ".env").write_text(_SECRET, encoding="utf-8")
    d = _dispatcher(tmp_path, _hiding(tmp_path, ".env"), isolation)
    try:
        with pytest.raises(ToolError, match="hidden from this run"):
            d.dispatch("read_file", {"path": ".env"})
    finally:
        d.close()


def test_agent6s_own_secrets_are_denied_when_the_workspace_contains_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workspace root that CONTAINS the config dir (root=$HOME) put
    `secrets.toml` -- provider keys -- inside the tree the tools may reach, with
    no hide_paths entry naming it. The builtin private dirs are denied too."""
    root = tmp_path / "home"
    root.mkdir()
    # The same override the suite's own isolation uses; it outranks XDG.
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(root / ".config" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(root / ".local" / "state" / "agent6"))
    cfg_dir = root / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "secrets.toml").write_text('[fake]\nKEY="fake-not-real"\n', encoding="utf-8")

    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(root, Config(), "read_file", {"path": ".config/agent6/secrets.toml"})


def test_the_config_a_later_run_loads_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistence, not just disclosure: editing `~/.config/agent6/config.toml`
    sets `isolation` / `run_commands` for the NEXT run."""
    root = tmp_path / "home"
    root.mkdir()
    # The same override the suite's own isolation uses; it outranks XDG.
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(root / ".config" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(root / ".local" / "state" / "agent6"))
    cfg_dir = root / ".config" / "agent6"
    cfg_dir.mkdir(parents=True)
    conf = cfg_dir / "config.toml"
    conf.write_text('[sandbox]\nisolation = "strict"\n', encoding="utf-8")

    with pytest.raises(ToolError, match="hidden from this run"):
        _dispatch(
            root,
            Config(),
            "apply_edit",
            {
                "path": ".config/agent6/config.toml",
                "edits": [{"old_string": '"strict"', "new_string": '"none"'}],
            },
        )
    assert 'isolation = "strict"' in conf.read_text(encoding="utf-8")


def test_a_workspace_inside_a_private_dir_refuses_at_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tool call would refuse, so the run is told why up front instead of
    failing on every path. One exactly-known case, not an enumeration."""
    from agent6.app.confine import check_workspace_outside_private_dirs

    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state" / "agent6"))
    inside = tmp_path / "state" / "agent6" / "somerepo"
    inside.mkdir(parents=True)
    refusal = check_workspace_outside_private_dirs(inside)
    assert refusal is not None and "private" in refusal

    ordinary = tmp_path / "project"
    ordinary.mkdir()
    assert check_workspace_outside_private_dirs(ordinary) is None
