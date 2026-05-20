# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.tools.dispatch — path safety, edit semantics, no-net I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher, ToolError

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.planner]
provider = "anthropic"
model = "x"
[models.worker]
provider = "anthropic"
model = "x"
[models.critic]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[models.summarizer]
provider = "anthropic"
model = "x"
[sandbox]
profile = "auto"
network = "no"
run_commands = "no"
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
default = "implement"
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _config(tmp_path: Path) -> Config:
    from agent6.config import load_config

    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def test_read_file_ok(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("read_file", {"path": "hello.txt"})
    assert out["content"] == "hi"


def test_absolute_path_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch("read_file", {"path": "/etc/passwd"})


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.\."):
        d.dispatch("read_file", {"path": "../outside.txt"})


def test_apply_edit_create_and_replace(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "create", "old_string": "", "new_string": "x = 1\n"}],
        },
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"
    d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 2"}],
        },
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 2\n"


def test_apply_edit_non_unique_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\na\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not unique"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "a", "new_string": "b"}],
            },
        )


def test_apply_edit_missing_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("hello\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not found"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "bye", "new_string": "x"}],
            },
        )


def test_run_command_disabled_when_no(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="disabled"):
        d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert "run_command" not in d.available_tool_names()


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Unknown"):
        d.dispatch("nope", {})


def test_grep_finds_match(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("hello world\nfoo\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("grep", {"pattern": "hello", "path": "."})
    assert len(out["hits"]) == 1
    assert out["hits"][0]["text"] == "hello world"


def test_list_dir(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "y.txt").write_text("y", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("list_dir", {"path": "."})
    assert "x/" in out["entries"]
    assert "y.txt" in out["entries"]
