# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""PINNED operator-facing surface: every ToolResult.summary() string.

The `tool.result` event's `summary` is the one-line log/TUI/web line the
operator reads for every dispatched tool. The 8b reshape rewrote it from the
old 12-branch key-sniffer (`summarize_result`, base tree) into per-type
``summary()`` methods on inspection-only equivalence; this file makes each
string a tested contract, so future drift is a deliberate edit here.

Expected strings are derived from the base-tree sniffer branch each wire shape
used to hit, with ONE deliberate change the reshape report itemized: an MCP
``RawResult`` whose opaque payload happens to carry sniffer-matching keys now
summarizes as the generic "ok" (the sniffer used to guess from the keys).

The completeness test walks the concrete subclasses in agent6.tools.results so
a NEW result type cannot ship without pinning its summary here.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent6.tools import results as results_mod
from agent6.tools.results import (
    AddDependencyResult,
    AddMemoryResult,
    AddTaskResult,
    AnswersResult,
    DefinitionsResult,
    DocsContentResult,
    DocsIndexResult,
    EditResult,
    ExecResult,
    FinishPlanningResult,
    FinishRunResult,
    GrepResult,
    InvalidateMemoryResult,
    ListDirResult,
    ListTasksResult,
    MetricResult,
    OutlineResult,
    PatchResult,
    PreviewResult,
    RawResult,
    ReadFileResult,
    ReferencesResult,
    SetCursorResult,
    SkillResult,
    ToolResult,
    UpdateTaskResult,
)

_HIT: dict[str, Any] = {"path": "a.py", "line": 1, "text": "x"}
_SYM: dict[str, Any] = {"name": "f", "kind": "function", "line": 1, "col": 0}
_LOC: dict[str, Any] = {"path": "a.py", "line": 1, "col": 0}
_TASK: dict[str, Any] = {"id": "01A", "title": "t", "status": "pending"}
_LONG_TITLE = "audit the provider transport layer for retry storms and dedupe them all"

# (case id, result, exact expected summary). One row per concrete type, plus a
# second row wherever summary() has a conditional branch (truncated suffix,
# blank-answer counting, title clipping).
CASES: list[tuple[str, ToolResult, str]] = [
    # content access
    ("docs_index", DocsIndexResult(available=("architecture", "security")), "ok"),
    ("docs_content", DocsContentResult(name="security", content="body", truncated=False), "ok"),
    ("read_file", ReadFileResult(content="hi", size=2, lines_total=1), "2 bytes"),
    (
        "read_file_slice",
        ReadFileResult(content="b\n", size=2, lines_total=3, offset=1, lines_returned=1),
        "2 bytes",
    ),
    ("list_dir", ListDirResult(entries=("a.txt", "b/")), "2 entries"),
    # search / navigation
    ("grep", GrepResult(hits=(_HIT,), truncated=False), "1 matches"),
    ("grep_truncated", GrepResult(hits=(_HIT, _HIT), truncated=True), "2 matches (truncated)"),
    ("grep_timeout", GrepResult(hits=(), truncated=True, timeout=True), "0 matches (truncated)"),
    ("outline", OutlineResult(symbols=(_SYM, _SYM, _SYM), truncated=False), "3 symbols"),
    ("outline_truncated", OutlineResult(symbols=(_SYM,), truncated=True), "1 symbols (truncated)"),
    ("definitions", DefinitionsResult(definitions=(_LOC,), truncated=False), "1 definitions"),
    (
        "definitions_truncated",
        DefinitionsResult(definitions=(_LOC,), truncated=True),
        "1 definitions (truncated)",
    ),
    ("references", ReferencesResult(references=(), truncated=False), "0 references"),
    (
        "references_truncated",
        ReferencesResult(references=(_LOC,), truncated=True),
        "1 references (truncated)",
    ),
    # filesystem writes
    (
        "apply_edit",
        EditResult(applied=("create",), path="new.txt"),
        "applied=['create'] path=new.txt",
    ),
    (
        "apply_edit_multi",
        EditResult(applied=("replace", "replace~indent"), path="src/m.py"),
        "applied=['replace', 'replace~indent'] path=src/m.py",
    ),
    ("apply_patch", PatchResult(path="f.py", bytes_written=5), "patched path=f.py bytes=5"),
    (
        "preview",
        PreviewResult(
            path="f.py",
            diff="-x\n+y\n",
            hunks=1,
            bytes_before=2,
            bytes_after=2,
            truncated=False,
            would_apply=("replace",),
        ),
        "ok",
    ),
    # execution
    (
        "exec",
        ExecResult(returncode=1, stdout="", stderr="boom", duration_s=0.5, exec_failed=False),
        "exit=1 in 0.5s",
    ),
    (
        "exec_duration_fmt",
        ExecResult(returncode=0, stdout="", stderr="", duration_s=12.34, exec_failed=False),
        "exit=0 in 12.3s",
    ),
    (
        "metric",
        MetricResult(
            returncode=0,
            stdout="CYCLES: 42",
            stderr="",
            duration_s=0.5,
            exec_failed=False,
            score=42.0,
        ),
        "exit=0 in 0.5s",
    ),
    # run control
    ("finish_run", FinishRunResult(summary_text="done", result=None), "ok"),
    ("finish_planning", FinishPlanningResult(summary_text="s", plan_bytes=7), "ok"),
    ("ask_user", AnswersResult(answers=("yes", "", " ", "no")), "2/4 answered"),
    # DAG
    (
        "add_task",
        AddTaskResult(id="01A", parent_id=None, title="t", status="pending"),
        "pending: t",
    ),
    (
        "add_task_title_clipped",
        AddTaskResult(id="01A", parent_id=None, title=_LONG_TITLE, status="pending"),
        f"pending: {_LONG_TITLE[:60]}",
    ),
    ("update_task", UpdateTaskResult(id="01A", status="done", title="t"), "done: t"),
    ("set_cursor", SetCursorResult(cursor="01A"), "ok"),
    ("add_dependency", AddDependencyResult(id="01A", title="t", depends_on=("01B",)), "ok"),
    ("list_tasks", ListTasksResult(tasks=(_TASK, _TASK), count=2), "2 tasks"),
    # operator knowledge
    ("add_memory", AddMemoryResult(id="01M", scope="facts", created_at="2026"), "ok"),
    ("invalidate_memory", InvalidateMemoryResult(id="01M", invalidated_at="2026"), "ok"),
    (
        "use_skill",
        SkillResult(skill="deploy", file="SKILL.md", content="12345"),
        "skill deploy/SKILL.md (5 chars)",
    ),
    # MCP passthrough: DELIBERATE change from the base-tree sniffer, which would
    # have guessed "0 matches" from this payload's keys; the opaque server dict
    # now summarizes as the generic "ok" (reshape report, security-adjacent §).
    ("mcp_raw", RawResult({"hits": [], "truncated": False}), "ok"),
]


@pytest.mark.parametrize(
    ("result", "expected"), [(r, e) for _, r, e in CASES], ids=[i for i, _, _ in CASES]
)
def test_summary_string_is_pinned(result: ToolResult, expected: str) -> None:
    assert result.summary() == expected


def test_base_summary_fallback_is_ok() -> None:
    """A result type that does not override summary() reports the generic "ok"
    (the sniffer's fallback: echoing the tool name doubled the name column)."""

    class _Minimal(ToolResult):
        def to_wire(self) -> dict[str, Any]:
            return {}

    assert _Minimal().summary() == "ok"


def test_every_concrete_result_type_is_pinned() -> None:
    """A new ToolResult subclass cannot ship without pinning its summary here.

    Compares by class NAME over the module's own namespace (not
    ``__subclasses__``, which under the test import layout lists each class
    twice with distinct identities)."""
    concrete = {
        name
        for name, obj in vars(results_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, results_mod.ToolResult)
        and obj is not results_mod.ToolResult
    }
    covered = {type(r).__name__ for _, r, _ in CASES}
    assert concrete == covered
