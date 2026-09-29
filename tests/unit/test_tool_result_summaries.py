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

_HIT: dict[str, Any] = {"path": "a.py", "line": 1, "text": "x"}
_SYM: dict[str, Any] = {"name": "f", "kind": "function", "line": 1, "col": 0}
_LOC: dict[str, Any] = {"path": "a.py", "line": 1, "col": 0}
_TASK: dict[str, Any] = {"id": "01A", "title": "t", "status": "pending"}
_LONG_TITLE = "audit the provider transport layer for retry storms and dedupe them all"

# (case id, result, exact expected summary). One row per concrete type, plus a
# second row wherever summary() has a conditional branch (truncated suffix,
# blank-answer counting, title clipping).
CASES: list[tuple[str, results_mod.ToolResult, str]] = [
    # content access
    ("docs_index", results_mod.DocsIndexResult(available=("architecture", "security")), "ok"),
    (
        "docs_content",
        results_mod.DocsContentResult(name="security", content="body", size=4, truncated=False),
        "ok",
    ),
    ("read_file", results_mod.ReadFileResult(content="hi", size=2, lines_total=1), "2 bytes"),
    (
        "read_file_slice",
        results_mod.ReadFileResult(
            content="b\n", size=2, lines_total=3, start_line=2, lines_returned=1
        ),
        "2 bytes",
    ),
    ("list_dir", results_mod.ListDirResult(entries=("a.txt", "b/")), "2 entries"),
    # search / navigation
    (
        "outline",
        results_mod.OutlineResult(symbols=(_SYM, _SYM, _SYM), truncated=False),
        "3 symbols",
    ),
    (
        "outline_truncated",
        results_mod.OutlineResult(symbols=(_SYM,), truncated=True),
        "1 symbols (truncated)",
    ),
    (
        "definitions",
        results_mod.DefinitionsResult(definitions=(_LOC,), truncated=False),
        "1 definitions",
    ),
    (
        "definitions_truncated",
        results_mod.DefinitionsResult(definitions=(_LOC,), truncated=True),
        "1 definitions (truncated)",
    ),
    ("references", results_mod.ReferencesResult(references=(), truncated=False), "0 references"),
    (
        "references_truncated",
        results_mod.ReferencesResult(references=(_LOC,), truncated=True),
        "1 references (truncated)",
    ),
    # filesystem writes
    (
        "apply_edit",
        results_mod.EditResult(applied=("create",), path="new.txt"),
        "applied=['create'] path=new.txt",
    ),
    (
        "apply_edit_multi",
        results_mod.EditResult(applied=("replace", "replace~indent"), path="src/m.py"),
        "applied=['replace', 'replace~indent'] path=src/m.py",
    ),
    (
        "apply_patch",
        results_mod.PatchResult(path="f.py", bytes_written=5),
        "patched path=f.py bytes=5",
    ),
    (
        "apply_patch_delete",
        results_mod.PatchResult(path="f.py", bytes_written=0, deleted=("f.py",)),
        "deleted path=f.py",
    ),
    (
        "apply_patch_multi_mixed",
        results_mod.PatchResult(
            path="a.py", bytes_written=2, files=(("a.py", 2),), deleted=("old.py",)
        ),
        "patched 1 files, deleted 1 bytes=2",
    ),
    (
        "preview",
        results_mod.PreviewResult(
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
    # the web
    (
        "fetch",
        results_mod.FetchResult(
            url="https://x/y", status=200, content_type="text/plain", body="hello"
        ),
        "200 · 5 bytes",
    ),
    # other sessions
    (
        "sessions_roster",
        results_mod.SessionsResult(sessions=("[a-b-AAAAAA] ask · 2026-07-31T01:00: q",)),
        "1 session",
    ),
    (
        "sessions_read",
        results_mod.SessionsResult(sessions=("[a] run: x", "[b] ask: y"), conversation="user: hi"),
        "2 sessions",
    ),
    # background commands
    (
        "background_one",
        results_mod.BackgroundResult(shells=("[bg1] running: sleep 300",)),
        "[bg1] running: sleep 300",
    ),
    (
        "background_many",
        results_mod.BackgroundResult(
            shells=("[bg1] running: a", "[bg2] exited (exit 1): b"), output="x"
        ),
        "2 background",
    ),
    # execution
    (
        "exec",
        results_mod.ExecResult(
            returncode=1, stdout="", stderr="boom", duration_s=0.5, exec_failed=False
        ),
        "exit=1 in 0.5s",
    ),
    (
        "exec_duration_fmt",
        results_mod.ExecResult(
            returncode=0, stdout="", stderr="", duration_s=12.34, exec_failed=False
        ),
        "exit=0 in 12.3s",
    ),
    (
        "exec_timed_out",
        results_mod.ExecResult(
            returncode=124, stdout="", stderr="", duration_s=240.1, exec_failed=False, timeout_s=240
        ),
        "exit=124 (timed out at 240s) in 240.1s",
    ),
    (
        "metric",
        results_mod.MetricResult(
            returncode=0,
            stdout="CYCLES: 42",
            stderr="",
            duration_s=0.5,
            exec_failed=False,
            score=42.0,
        ),
        "exit=0 in 0.5s",
    ),
    (
        "metric_timed_out",
        results_mod.MetricResult(
            returncode=124,
            stdout="",
            stderr="",
            duration_s=240.1,
            exec_failed=False,
            score=None,
            timeout_s=240,
        ),
        "exit=124 (timed out at 240s) in 240.1s",
    ),
    # run control
    ("finish_session", results_mod.FinishSessionResult(summary_text="done", result=None), "ok"),
    ("finish_planning", results_mod.FinishPlanningResult(summary_text="s", plan_bytes=7), "ok"),
    ("ask_user", results_mod.AnswersResult(answers=("yes", "", " ", "no")), "2/4 answered"),
    # DAG
    (
        "add_task",
        results_mod.AddTaskResult(id="01A", parent_id=None, title="t", status="pending"),
        "pending: t",
    ),
    (
        "add_task_title_clipped",
        results_mod.AddTaskResult(id="01A", parent_id=None, title=_LONG_TITLE, status="pending"),
        f"pending: {_LONG_TITLE[:60]}",
    ),
    ("update_task", results_mod.UpdateTaskResult(id="01A", status="done", title="t"), "done: t"),
    ("list_tasks", results_mod.ListTasksResult(tasks=(_TASK, _TASK), count=2), "2 tasks"),
    # operator knowledge
    (
        "use_skill",
        results_mod.SkillResult(skill="deploy", file="SKILL.md", content="12345"),
        "skill deploy/SKILL.md (5 chars)",
    ),
    # MCP passthrough: DELIBERATE change from the base-tree sniffer, which would
    # have guessed "0 matches" from this payload's keys; the opaque server dict
    # now summarizes as the generic "ok" (reshape report, security-adjacent §).
    ("mcp_raw", results_mod.RawResult({"hits": [], "truncated": False}), "ok"),
]


@pytest.mark.parametrize(
    ("result", "expected"), [(r, e) for _, r, e in CASES], ids=[i for i, _, _ in CASES]
)
def test_summary_string_is_pinned(result: results_mod.ToolResult, expected: str) -> None:
    assert result.summary() == expected


def test_base_summary_fallback_is_ok() -> None:
    """A result type without its own summary() reports "ok", not the tool name doubled."""

    class _Minimal(results_mod.ToolResult):
        def to_wire(self) -> dict[str, Any]:
            return {}

    assert _Minimal().summary() == "ok"


def test_every_concrete_result_type_is_pinned() -> None:
    """A new ToolResult subclass cannot ship without pinning its summary here.

    Compares by class NAME over the module's own namespace (not
    ``__subclasses__``, which under the test import layout lists each class
    twice with distinct identities).
    """
    concrete = {
        name
        for name, obj in vars(results_mod).items()
        if isinstance(obj, type)
        and issubclass(obj, results_mod.ToolResult)
        and obj is not results_mod.ToolResult
    }
    covered = {type(r).__name__ for _, r, _ in CASES}
    assert concrete == covered
