# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Typed tool-handler results: every handler returns one of these frozen values
instead of a bare dict, each owning its two representations -- the exact
model-facing ``to_wire()`` dict and the one-line human ``summary()``.

- ``to_wire()`` -- the exact dict the loop JSON-dumps into the model's
  tool_result. This is frozen LLM I/O: keys, key ORDER (dicts preserve
  insertion order, so field/emit order here is load-bearing), and value
  formats must match what the handler used to build inline. Pinned by
  ``tests/unit/test_tool_result_wire.py``.
- ``summary()`` -- the one-line human string for the log tail / TUI. Replaces
  the 12-branch key-sniffer that used to guess the tool from the dict's keys;
  each result states its own summary.

Internal values, so frozen dataclasses (not pydantic): the wire dict is
produced at the boundary, never validated back in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


class ToolResult(abc.ABC):
    """One tool handler's typed result: it owns the model-facing ``to_wire()``
    dict and its one-line ``summary()``."""

    __slots__ = ()

    @abc.abstractmethod
    def to_wire(self) -> dict[str, Any]:
        """The model-facing dict, JSON-serialized verbatim by the loop."""

    def summary(self) -> str:
        """One-line log/TUI summary. Defaults to the old sniffer's fallback."""
        return "ok"


def _trunc(truncated: bool) -> str:
    return " (truncated)" if truncated else ""


# --- content access ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocsIndexResult(ToolResult):
    """agent6_docs with no name: the list of available docs."""

    available: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"available": list(self.available)}


@dataclass(frozen=True, slots=True)
class DocsContentResult(ToolResult):
    """agent6_docs for a named doc."""

    name: str
    content: str
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "content": self.content, "truncated": self.truncated}


@dataclass(frozen=True, slots=True)
class ReadFileResult(ToolResult):
    content: str
    size: int
    lines_total: int
    # Present together only for a partial read (offset/limit given); a full
    # read omits both. offset can legitimately be 0 for a slice, so None is
    # the "full read" sentinel.
    offset: int | None = None
    lines_returned: int | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "content": self.content,
            "size": self.size,
            "lines_total": self.lines_total,
        }
        if self.offset is not None:
            out["offset"] = self.offset
            out["lines_returned"] = self.lines_returned
        return out

    def summary(self) -> str:
        return f"{self.size} bytes"


@dataclass(frozen=True, slots=True)
class ListDirResult(ToolResult):
    entries: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"entries": list(self.entries)}

    def summary(self) -> str:
        return f"{len(self.entries)} entries"


# --- search / navigation -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrepResult(ToolResult):
    hits: tuple[dict[str, Any], ...]
    truncated: bool
    # Only the wall-clock-timeout return carries this key.
    timeout: bool = False

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"hits": list(self.hits), "truncated": self.truncated}
        if self.timeout:
            out["timeout"] = True
        return out

    def summary(self) -> str:
        return f"{len(self.hits)} matches{_trunc(self.truncated)}"


@dataclass(frozen=True, slots=True)
class OutlineResult(ToolResult):
    symbols: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"symbols": list(self.symbols), "truncated": self.truncated}

    def summary(self) -> str:
        return f"{len(self.symbols)} symbols{_trunc(self.truncated)}"


@dataclass(frozen=True, slots=True)
class DefinitionsResult(ToolResult):
    """find_definition and find_definition_lsp: same envelope, different rows."""

    definitions: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"definitions": list(self.definitions), "truncated": self.truncated}

    def summary(self) -> str:
        return f"{len(self.definitions)} definitions{_trunc(self.truncated)}"


@dataclass(frozen=True, slots=True)
class ReferencesResult(ToolResult):
    """find_references and find_references_lsp."""

    references: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        return {"references": list(self.references), "truncated": self.truncated}

    def summary(self) -> str:
        return f"{len(self.references)} references{_trunc(self.truncated)}"


# --- filesystem writes -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditResult(ToolResult):
    """apply_edit that wrote (not preview)."""

    applied: tuple[str, ...]
    path: str

    def to_wire(self) -> dict[str, Any]:
        return {"applied": list(self.applied), "path": self.path}

    def summary(self) -> str:
        return f"applied={list(self.applied)} path={self.path}"


@dataclass(frozen=True, slots=True)
class PatchResult(ToolResult):
    """apply_patch that wrote (not preview)."""

    path: str
    bytes_written: int

    def to_wire(self) -> dict[str, Any]:
        return {"path": self.path, "bytes_written": self.bytes_written}

    def summary(self) -> str:
        return f"patched path={self.path} bytes={self.bytes_written}"


@dataclass(frozen=True, slots=True)
class PreviewResult(ToolResult):
    """apply_edit/apply_patch with preview=true: the dry-run diff. apply_edit
    carries would_apply (the per-edit kinds); apply_patch does not."""

    path: str
    diff: str
    hunks: int
    bytes_before: int
    bytes_after: int
    truncated: bool
    would_apply: tuple[str, ...] | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "preview": True,
            "path": self.path,
            "diff": self.diff,
            "hunks": self.hunks,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "truncated": self.truncated,
        }
        if self.would_apply is not None:
            out["would_apply"] = list(self.would_apply)
        return out


# --- execution (jail-backed) -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecResult(ToolResult):
    """run_command and run_verify_command: the jailed command's outcome."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool

    def to_wire(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "exec_failed": self.exec_failed,
        }

    def summary(self) -> str:
        return f"exit={self.returncode} in {self.duration_s:.1f}s"


@dataclass(frozen=True, slots=True)
class MetricResult(ToolResult):
    """run_metric_command: the jail outcome plus the parsed score, appended
    after the exec fields."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool
    score: float | None

    @classmethod
    def from_exec(cls, res: ExecResult, score: float | None) -> MetricResult:
        return cls(
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration_s=res.duration_s,
            exec_failed=res.exec_failed,
            score=score,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "exec_failed": self.exec_failed,
            "score": self.score,
        }

    def summary(self) -> str:
        return f"exit={self.returncode} in {self.duration_s:.1f}s"


# --- run control -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinishRunResult(ToolResult):
    summary_text: str
    result: dict[str, Any] | None

    def to_wire(self) -> dict[str, Any]:
        return {"acknowledged": True, "summary": self.summary_text, "result": self.result}


@dataclass(frozen=True, slots=True)
class FinishPlanningResult(ToolResult):
    summary_text: str
    plan_bytes: int

    def to_wire(self) -> dict[str, Any]:
        return {"acknowledged": True, "summary": self.summary_text, "plan_bytes": self.plan_bytes}


@dataclass(frozen=True, slots=True)
class AnswersResult(ToolResult):
    answers: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"answers": list(self.answers)}

    def summary(self) -> str:
        answered = sum(1 for a in self.answers if str(a).strip())
        return f"{answered}/{len(self.answers)} answered"


# --- DAG (task graph) --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AddTaskResult(ToolResult):
    id: str
    parent_id: str | None
    title: str
    status: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "title": self.title,
            "status": self.status,
        }

    def summary(self) -> str:
        return f"{self.status}: {str(self.title)[:60]}"


@dataclass(frozen=True, slots=True)
class UpdateTaskResult(ToolResult):
    id: str
    status: str
    title: str

    def to_wire(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "title": self.title}

    def summary(self) -> str:
        return f"{self.status}: {str(self.title)[:60]}"


@dataclass(frozen=True, slots=True)
class SetCursorResult(ToolResult):
    cursor: str | None

    def to_wire(self) -> dict[str, Any]:
        return {"acknowledged": True, "cursor": self.cursor}


@dataclass(frozen=True, slots=True)
class AddDependencyResult(ToolResult):
    id: str
    title: str
    depends_on: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "depends_on": list(self.depends_on)}


@dataclass(frozen=True, slots=True)
class ListTasksResult(ToolResult):
    tasks: tuple[dict[str, Any], ...]
    count: int

    def to_wire(self) -> dict[str, Any]:
        return {"tasks": list(self.tasks), "count": self.count}

    def summary(self) -> str:
        return f"{self.count} tasks"


# --- operator knowledge ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AddMemoryResult(ToolResult):
    id: str
    scope: str
    created_at: str

    def to_wire(self) -> dict[str, Any]:
        return {"id": self.id, "scope": self.scope, "created_at": self.created_at}


@dataclass(frozen=True, slots=True)
class InvalidateMemoryResult(ToolResult):
    id: str
    invalidated_at: str

    def to_wire(self) -> dict[str, Any]:
        return {"id": self.id, "invalidated_at": self.invalidated_at}


@dataclass(frozen=True, slots=True)
class SkillResult(ToolResult):
    skill: str
    file: str
    content: str

    def to_wire(self) -> dict[str, Any]:
        return {"skill": self.skill, "file": self.file, "content": self.content}

    def summary(self) -> str:
        return f"skill {self.skill}/{self.file} ({len(self.content)} chars)"


# --- MCP passthrough ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawResult(ToolResult):
    """An operator-configured MCP server's result: an opaque dict forwarded to
    the model unchanged. agent6 does not know its shape, so the summary is the
    generic 'ok'."""

    payload: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return self.payload
