# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-run memory wiring: the add_memory / invalidate_memory tools, the
<memories> system-prompt block, and the Workflow's state_dir loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config, load_config
from agent6.memory import MemoryEntry, add, invalidate, list_entries
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.schema import (
    LOOP_EXTRA_TOOLS,
    AddMemoryInput,
    InvalidateMemoryInput,
)
from agent6.types import RepoSummary
from agent6.workflows import loop as loopmod
from agent6.workflows._prompt_blocks import memories_block
from agent6.workflows.loop import Workflow

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
isolation = "auto"
run_commands = "no"
protect_git = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
[workflow]
verify_command = ["true"]
[budget]
max_tokens_fallback = 2000000
"""


def _silent(_msg: str) -> None:
    return None


def _config(tmp_path: Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _repo(tmp_path: Path) -> RepoSummary:
    return RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def _entry(scope: Any = "facts", body: str = "x", **kw: Any) -> MemoryEntry:
    defaults: dict[str, Any] = {
        "id": "01HXXXXXXXXXXXXXXXXXXXXXX0",
        "scope": scope,
        "created_at": "2026-01-01T00:00:00Z",
        "body": body,
    }
    defaults.update(kw)
    return MemoryEntry(**defaults)


# --- schema / tool lists -------------------------------------------------


def test_loop_extra_tools_include_memory_tools() -> None:
    names = {t.TOOL_NAME for t in LOOP_EXTRA_TOOLS}
    assert AddMemoryInput.TOOL_NAME in names
    assert InvalidateMemoryInput.TOOL_NAME in names


def test_tool_definitions_expose_memory_tools_in_run_mode_only(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, state_dir=tmp_path / "state")
    run_names = {t.name for t in loopmod.tool_definitions(d, mode="run")}  # pyright: ignore[reportPrivateUsage]
    assert "add_memory" in run_names
    assert "invalidate_memory" in run_names
    for mode in ("plan", "ask", "machine", "agent"):
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_memory" not in names, mode
        assert "invalidate_memory" not in names, mode


def test_add_memory_input_bounds() -> None:
    with pytest.raises(ValueError):
        AddMemoryInput(scope="facts", body="")
    with pytest.raises(ValueError):
        AddMemoryInput(scope="facts", body="x" * 2001)
    with pytest.raises(ValueError):
        AddMemoryInput(scope="notes", body="x")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError):
        InvalidateMemoryInput(id="short", reason="r")


# --- dispatcher ----------------------------------------------------------


def test_dispatch_add_memory_persists(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    state = tmp_path / "state"
    d = ToolDispatcher(root=tmp_path, config=cfg, state_dir=state)
    out = d.dispatch(
        "add_memory", {"scope": "facts", "body": "the suite takes 4 minutes"}
    ).to_wire()
    assert len(out["id"]) == 26
    entries = list_entries(state, "facts")
    assert [e.body for e in entries] == ["the suite takes 4 minutes"]
    assert entries[0].id == out["id"]


def test_dispatch_invalidate_memory_roundtrip(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    state = tmp_path / "state"
    e = add(state, "decisions", "use tabs")
    d = ToolDispatcher(root=tmp_path, config=cfg, state_dir=state)
    out = d.dispatch("invalidate_memory", {"id": e.id, "reason": "operator uses spaces"}).to_wire()
    assert out["id"] == e.id
    assert out["invalidated_at"]
    assert not list_entries(state, "decisions")[0].is_active


def test_dispatch_memory_tools_unwired_raise(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="no memory store"):
        d.dispatch("add_memory", {"scope": "facts", "body": "x"})
    with pytest.raises(ToolError, match="no memory store"):
        d.dispatch("invalidate_memory", {"id": "0" * 26, "reason": "r"})


def test_dispatch_invalidate_memory_unknown_id(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, state_dir=tmp_path / "state")
    with pytest.raises(ToolError, match="no memory with id"):
        d.dispatch("invalidate_memory", {"id": "0" * 26, "reason": "r"})


def test_dispatch_memory_tools_blocked_outside_run_mode(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    for mode in ("plan", "ask", "machine"):
        d = ToolDispatcher(root=tmp_path, config=cfg, state_dir=tmp_path / "state", mode=mode)
        with pytest.raises(ToolError, match="not available"):
            d.dispatch("add_memory", {"scope": "facts", "body": "x"})
        with pytest.raises(ToolError, match="not available"):
            d.dispatch("invalidate_memory", {"id": "0" * 26, "reason": "r"})


# --- <memories> block ------------------------------------------------------


def test_memories_block_run_mode_renders_guidance_when_empty() -> None:
    block = memories_block((), mode="run")
    assert block.startswith("<memories>")
    assert block.rstrip().endswith("</memories>")
    assert "(none recorded yet)" in block
    assert "add_memory" in block


def test_memories_block_absent_for_readonly_modes_when_empty() -> None:
    assert memories_block((), mode="plan") == ""
    assert memories_block((), mode="ask") == ""


def test_memories_block_groups_scopes_and_shows_ids() -> None:
    a = _entry("facts", "fact body", id="01AAAAAAAAAAAAAAAAAAAAAAAA")
    b = _entry("preferences", "pref body", id="01BBBBBBBBBBBBBBBBBBBBBBBB")
    block = memories_block((a, b), mode="plan")
    assert "[facts]" in block and "[preferences]" in block
    assert "[decisions]" not in block
    assert f"- {a.id} (2026-01-01): fact body" in block
    # Read-only modes get no write-tool guidance.
    assert "add_memory" not in block


def test_memories_block_clips_long_entries() -> None:
    e = _entry("facts", "y" * 5000)
    block = memories_block((e,), mode="run")
    assert "[clipped]" in block
    assert "y" * 1201 not in block


def test_memories_block_elides_oldest_beyond_cap() -> None:
    entries = tuple(
        _entry(
            "facts",
            f"note {i} " + "z" * 1100,
            id=f"01{i:024d}",
            created_at=f"2026-01-{i + 1:02d}T00:00:00Z",
        )
        for i in range(20)
    )
    block = memories_block(entries, mode="run")
    assert "older memories elided" in block
    # Newest survives, oldest goes.
    assert "note 19 " in block
    assert "note 0 " not in block


def test_memories_block_multiline_bodies_indent() -> None:
    e = _entry("decisions", "first line\nsecond line")
    block = memories_block((e,), mode="run")
    assert "): first line\n  second line" in block


# --- system prompt assembly ------------------------------------------------


def test_build_system_prompt_injects_memories(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    mem = (_entry("facts", "verify needs the venv"),)
    run = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=_repo(tmp_path), mode="run", memories=mem, notes="", skills=None
    )
    assert "<memories>" in run
    assert "verify needs the venv" in run
    plan = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=_repo(tmp_path), mode="plan", memories=mem, notes="", skills=None
    )
    assert "verify needs the venv" in plan


def test_build_system_prompt_run_mode_always_has_memories_block(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config=cfg, repo=_repo(tmp_path), mode="run", memories=(), notes="", skills=None
    )
    assert "<memories>" in run


def test_build_system_prompt_machine_modes_never_see_memories(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    mem = (_entry("facts", "verify needs the venv"),)
    for mode in ("machine", "agent"):
        text = loopmod.build_system_prompt(  # pyright: ignore[reportPrivateUsage]
            config=cfg, repo=_repo(tmp_path), mode=mode, memories=mem, notes="", skills=None
        )
        assert "<memories>" not in text, mode


# --- Workflow._load_memories -----------------------------------------------


def _wf(tmp_path: Path, **kw: Any) -> Workflow:
    defaults: dict[str, Any] = {
        "root": tmp_path,
        "config": MagicMock(),
        "provider": MagicMock(),
        "dispatcher": MagicMock(),
        "logger": _silent,
    }
    defaults.update(kw)
    return Workflow(**defaults)


def test_workflow_load_memories_filters_invalidated(tmp_path: Path) -> None:
    state = tmp_path / "state"
    keep = add(state, "facts", "keep me")
    drop = add(state, "facts", "drop me")
    invalidate(state, drop.id, "stale")
    wf = _wf(tmp_path, state_dir=state)
    loaded = wf._load_memories()  # pyright: ignore[reportPrivateUsage]
    assert [e.id for e in loaded] == [keep.id]


def test_workflow_load_memories_unreadable_reaches_a_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degrading to an empty <memories> block is right, but it was announced
    only through the loop logger, which the console view filters out unless
    AGENT6_DEBUG=1 and a detached run sends to DEVNULL. No event meant the TUI,
    web, and `runs log` never saw it: every recorded memory silently vanished
    from the prompt. It must be an event like any other run error."""
    from agent6.memory import MemoryStoreError
    from agent6.workflows import loop as loop_mod

    class _Capture:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def emit(self, event_type: str, /, **fields: Any) -> None:
            self.events.append({"type": event_type, **fields})

    def _boom(_state_dir: Path) -> list[Any]:
        raise MemoryStoreError("facts.md is not valid UTF-8")

    monkeypatch.setattr(loop_mod, "memory_list_entries", _boom)
    ev = _Capture()
    wf = _wf(tmp_path, state_dir=tmp_path / "state", events=ev)
    assert wf._load_memories() == ()  # pyright: ignore[reportPrivateUsage]
    unavailable = [e for e in ev.events if e["type"] == "loop.memories.unavailable"]
    assert unavailable, "a degraded memory load must reach the surfaces"
    assert "UTF-8" in str(unavailable[0].get("error", ""))


def test_workflow_load_memories_unset_or_machine_mode_empty(tmp_path: Path) -> None:
    state = tmp_path / "state"
    add(state, "facts", "present")
    assert _wf(tmp_path)._load_memories() == ()  # pyright: ignore[reportPrivateUsage]
    wf = _wf(tmp_path, state_dir=state, mode="agent")
    assert wf._load_memories() == ()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_workflow_load_memories_unreadable_store_warns_not_raises(tmp_path: Path) -> None:
    state = tmp_path / "state"
    add(state, "facts", "present")
    (state / "memories" / "facts.md").chmod(0o000)
    logs: list[str] = []
    wf = _wf(tmp_path, state_dir=state, logger=logs.append)
    try:
        assert wf._load_memories() == ()  # pyright: ignore[reportPrivateUsage]
    finally:
        (state / "memories" / "facts.md").chmod(0o600)
    assert any("memories unavailable" in m for m in logs)


def test_memories_block_pinned_survives_the_trim() -> None:
    """An old pinned entry outlives 20 newer bulky entries; unpinned old ones
    elide as before, and the elided count stays truthful."""
    old_pinned = _entry(
        "decisions",
        "PINNED-DECISION squash merges only",
        id="01" + "0" * 24,
        created_at="2025-06-01T00:00:00Z",
        pinned=True,
    )
    noise = tuple(
        _entry(
            "facts",
            f"note {i} " + "z" * 1100,
            id=f"01{i:024d}",
            created_at=f"2026-01-{i + 1:02d}T00:00:00Z",
        )
        for i in range(1, 20)
    )
    block = memories_block((old_pinned, *noise), mode="run")
    assert "PINNED-DECISION squash merges only" in block
    assert "[pinned]" in block
    assert "older memories elided" in block
    assert "note 1 " not in block  # oldest unpinned still elides


def test_memories_block_a_small_pin_outlives_a_larger_one_that_does_not_fit() -> None:
    """Pins are costed first but are not a contiguous newest window: stopping the
    pinned pass at the first pin that does not fit skipped every smaller pin
    behind it, and the unpinned pass then spent the same leftover budget on a
    LARGER unpinned entry. A pin lost its slot to a non-pin. The note must also
    say a pinned entry was among the elided, since pins are promised to survive
    the newest-win trim."""
    big = tuple(
        _entry(
            "facts",
            f"bigpin {i} " + "y" * 1200,
            id=f"01{i:024d}",
            created_at=f"2026-03-{i + 2:02d}T00:00:00Z",
            pinned=True,
        )
        for i in range(10)
    )
    tiny = _entry(
        "facts",
        "TINYPIN keep me",
        id="01" + "9" * 24,
        created_at="2026-01-01T00:00:00Z",
        pinned=True,
    )
    unpinned = _entry(
        "facts",
        "UNPINNED and bigger than the tiny pin " + "z" * 100,
        id="01" + "8" * 24,
        created_at="2026-01-02T00:00:00Z",
    )
    block = memories_block((*big, tiny, unpinned), mode="run")
    assert "TINYPIN keep me" in block
    assert "of them pinned" in block


def test_memories_block_pins_over_cap_elide_oldest_pins() -> None:
    """Pins never break the block byte bound: when pins alone exceed the cap,
    the oldest pinned elide (counted in the note) and the newest pinned render."""
    pins = tuple(
        _entry(
            "facts",
            f"pin {i} " + "y" * 1150,
            id=f"01{i:024d}",
            created_at=f"2026-02-{i + 1:02d}T00:00:00Z",
            pinned=True,
        )
        for i in range(12)  # 12 x ~1248 chars > 12000 cap
    )
    block = memories_block(pins, mode="run")
    assert "older memories elided" in block
    assert "pin 11 " in block  # newest pinned kept
    assert "pin 0 " not in block  # oldest pinned elided
    assert len(block) < 14_000  # bound holds (cap + headers)


def test_content_blocks_are_required_at_assembly() -> None:
    """memories/notes/skills carry no defaults: an assembly site cannot
    silently omit a block, so `prompt show` and the loop cannot drift apart
    (one under-reporting what a session receives, the other sending nothing)."""
    import inspect

    from agent6.workflows._prompt_blocks import build_system_prompt

    sig = inspect.signature(build_system_prompt)
    for name in ("memories", "notes", "skills"):
        assert sig.parameters[name].default is inspect.Parameter.empty, name
