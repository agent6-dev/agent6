# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""FROZEN model-facing wire: the bytes each tool handler serializes to the LLM.

The loop JSON-dumps a dispatched tool's result verbatim into the tool_result
the model reads (workflows/loop.py). That JSON -- keys, key ORDER (dicts
preserve insertion order), and value formats -- is frozen LLM I/O: a drift
silently changes every model's tool feedback. This pins a representative
handler from each family, including the optional-field, score-append, preview,
and error shapes, so the 8b typed-result reshape must reproduce the bytes.

``_wire`` bridges the reshape: ``dispatch`` returns a bare dict today and a
typed result carrying ``to_wire()`` after. Either way this compares the exact
model-facing bytes, so the file stays green across the change unedited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from agent6.config import Config, load_config
from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.runs.layout import RunLayout
from agent6.tools.dispatch import ToolDispatcher, ToolError

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
profile = "auto"
agent_network = "open"
run_commands = "yes"
protect_git = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _config(tmp_path: Path, *, extra: str = "") -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML + extra, encoding="utf-8")
    return load_config(p)


def _wire(result: object) -> dict[str, Any]:
    """Model-facing bytes of a dispatch result, before or after the typed
    reshape. Post-reshape ``dispatch`` returns a result with ``to_wire()``;
    today it returns the dict itself."""
    to_wire = getattr(result, "to_wire", None)
    return to_wire() if callable(to_wire) else result  # type: ignore[return-value]


def _dumps(result: object) -> str:
    return json.dumps(_wire(result), ensure_ascii=False)


# --- content access family ---------------------------------------------------


def test_wire_read_file_full(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("read_file", {"path": "hello.txt"})) == (
        '{"content": "hi", "size": 2, "lines_total": 1}'
    )


def test_wire_read_file_slice(tmp_path: Path) -> None:
    (tmp_path / "abc.txt").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("read_file", {"path": "abc.txt", "offset": 1, "limit": 1})
    assert _dumps(out) == (
        '{"content": "b\\n", "size": 2, "lines_total": 3, "offset": 1, "lines_returned": 1}'
    )


def test_wire_read_file_full_agrees_with_slice_on_lines_total(tmp_path: Path) -> None:
    """Full and partial reads of one unchanged file must report the same
    lines_total: the count the paging args index into (splitlines), not the
    newline-count+1 heuristic that overshot every newline-terminated file."""
    (tmp_path / "abc.txt").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("read_file", {"path": "abc.txt"})) == (
        '{"content": "a\\nb\\nc\\n", "size": 6, "lines_total": 3}'
    )


def test_wire_read_file_offset_past_eof(tmp_path: Path) -> None:
    """A paging overshoot returns an empty slice with lines_returned=0, not the
    negative end-minus-offset arithmetic."""
    (tmp_path / "abc.txt").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("read_file", {"path": "abc.txt", "offset": 10, "limit": 5})
    assert _dumps(out) == (
        '{"content": "", "size": 0, "lines_total": 3, "offset": 10, "lines_returned": 0}'
    )


def test_wire_list_dir(tmp_path: Path) -> None:
    sub = tmp_path / "d"
    sub.mkdir()
    (sub / "a.txt").write_text("", encoding="utf-8")
    (sub / "b.txt").write_text("", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("list_dir", {"path": "d"})) == '{"entries": ["a.txt", "b.txt"]}'


# --- search family -----------------------------------------------------------


def test_wire_grep_hit(tmp_path: Path) -> None:
    (tmp_path / "h.txt").write_text("hello world\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("grep", {"path": "h.txt", "pattern": "hello"})) == (
        '{"hits": [{"path": "h.txt", "line": 1, "text": "hello world"}], "truncated": false}'
    )


def test_wire_grep_empty(tmp_path: Path) -> None:
    (tmp_path / "h.txt").write_text("hello\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    assert _dumps(d.dispatch("grep", {"path": "h.txt", "pattern": "zzz"})) == (
        '{"hits": [], "truncated": false}'
    )


# --- filesystem-write family (applied + preview) -----------------------------


def test_wire_apply_edit_applied(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch(
        "apply_edit",
        {"path": "new.txt", "edits": [{"kind": "create", "new_string": "x\n"}]},
    )
    assert _dumps(out) == '{"applied": ["create"], "path": "new.txt"}'


def test_wire_apply_edit_preview_carries_would_apply(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "preview": True,
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 99"}],
        },
    )
    w = _wire(out)
    # Preview shape: fixed key order, and would_apply present ONLY for apply_edit.
    assert list(w) == [
        "preview",
        "path",
        "diff",
        "hunks",
        "bytes_before",
        "bytes_after",
        "truncated",
        "would_apply",
    ]
    assert w["preview"] is True
    assert w["would_apply"] == ["replace"]
    assert w["hunks"] == 1


def test_wire_apply_patch_preview_omits_would_apply(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("apply_patch", {"path": "f.py", "patch": patch, "preview": True})
    w = _wire(out)
    assert list(w) == [
        "preview",
        "path",
        "diff",
        "hunks",
        "bytes_before",
        "bytes_after",
        "truncated",
    ]


# --- run-control family ------------------------------------------------------


def test_wire_finish_run(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("finish_run", {"summary": "done", "result": {"k": 1}})
    assert _dumps(out) == '{"acknowledged": true, "summary": "done", "result": {"k": 1}}'


def test_wire_finish_run_null_result(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    out = d.dispatch("finish_run", {"summary": "done"})
    assert _dumps(out) == '{"acknowledged": true, "summary": "done", "result": null}'


def test_wire_finish_planning(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), mode="plan")
    out = d.dispatch("finish_planning", {"summary": "s", "plan_markdown": "# Plan\n"})
    assert _dumps(out) == '{"acknowledged": true, "summary": "s", "plan_bytes": 7}'


def test_wire_ask_user(tmp_path: Path) -> None:
    d = ToolDispatcher(
        root=tmp_path,
        config=_config(tmp_path),
        questioner=lambda qs: tuple("ans" for _ in qs),
    )
    out = d.dispatch("ask_user", {"questions": [{"question": "q?", "options": ["a", "b"]}]})
    assert _dumps(out) == '{"answers": ["ans"]}'


# --- DAG family (dynamic ULID ids -> pin order + shape) ----------------------


def test_wire_add_task_order(tmp_path: Path) -> None:
    cur = GraphCurator(RunLayout(state_dir=tmp_path / ".agent6", run_id="r"))
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    d = ToolDispatcher(
        root=tmp_path, config=_config(tmp_path), curator=cur, run_root_node_id=root.id
    )
    w = _wire(d.dispatch("add_task", {"title": "sub"}))
    assert list(w) == ["id", "parent_id", "title", "status"]
    assert w == {"id": w["id"], "parent_id": root.id, "title": "sub", "status": "pending"}


# --- execution family (jail-backed; mock run_in_jail) ------------------------


def _cmd_result(**kw: Any):
    from agent6.types import CommandResult

    base = dict(argv=("x",), returncode=0, stdout="", stderr="", duration_s=0.5, exec_failed=False)
    base.update(kw)
    return CommandResult(**base)  # type: ignore[arg-type]


def test_wire_run_verify(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with mock.patch("agent6.tools.dispatch.run_in_jail", return_value=_cmd_result(stdout="ok")):
        out = d.dispatch("run_verify_command", {})
    assert _dumps(out) == (
        '{"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.5, "exec_failed": false}'
    )


def test_wire_run_command(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with mock.patch(
        "agent6.tools.dispatch.run_in_jail",
        return_value=_cmd_result(returncode=3, stdout="o", stderr="e"),
    ):
        out = d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert _dumps(out) == (
        '{"returncode": 3, "stdout": "o", "stderr": "e", "duration_s": 0.5, "exec_failed": false}'
    )


def test_wire_run_metric_appends_score(tmp_path: Path) -> None:
    extra = (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/true"]\n'
        'pattern = "CYCLES: (\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path, extra=extra))
    with mock.patch(
        "agent6.tools.dispatch.run_in_jail", return_value=_cmd_result(stdout="CYCLES: 42")
    ):
        out = d.dispatch("run_metric_command", {})
    # score is APPENDED after the exec fields, in that order.
    assert _dumps(out) == (
        '{"returncode": 0, "stdout": "CYCLES: 42", "stderr": "", "duration_s": 0.5,'
        ' "exec_failed": false, "score": 42.0}'
    )


# --- error shape (loop wraps a raised ToolError) -----------------------------


def test_wire_tool_error_shape(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with pytest.raises(ToolError) as exc:
        d.dispatch("no_such_tool", {})
    # The loop serializes a raised ToolError as {"error": str(exc)} (loop.py).
    assert json.dumps({"error": str(exc.value)}) == '{"error": "Unknown tool: no_such_tool"}'
