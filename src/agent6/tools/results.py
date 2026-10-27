# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Typed tool-handler results.

Every handler returns one of these frozen values. `to_wire()` is the dict the loop JSON-dumps
into the model's tool result: keys, key order and value formats are the model-facing contract,
pinned by `tests/unit/test_tool_result_wire.py`. `summary()` is the one-line human string for
the log tail and the TUI. Frozen dataclasses, not pydantic: the wire dict is produced at the
boundary, never validated back in.
"""

from __future__ import annotations

import abc
import dataclasses
import shlex
from typing import Any

from agent6 import evidence
from agent6.tools import operator_prompts


class ToolResult(abc.ABC):
    """One tool handler's typed result."""

    __slots__ = ()

    @abc.abstractmethod
    def to_wire(self) -> dict[str, Any]:
        """Return the model-facing dict, JSON-serialized verbatim by the loop."""

    def summary(self) -> str:
        """Return the one-line log and TUI summary, "ok" unless the result says more."""
        return "ok"


def _trunc(truncated: bool) -> str:
    return " (truncated)" if truncated else ""


@dataclasses.dataclass(frozen=True, slots=True)
class DocsIndexResult(ToolResult):
    """agent6_docs with no name: the list of available docs."""

    available: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"available": list(self.available)}


@dataclasses.dataclass(frozen=True, slots=True)
class DocsContentResult(ToolResult):
    """agent6_docs for a named doc.

    Attributes:
        name: The doc's name.
        content: The doc's text, capped.
        size: The doc's full length in characters, so a cut content still names its size.
        truncated: Whether the content was cut.
    """

    name: str
    content: str
    size: int
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {
            "name": self.name,
            "content": self.content,
            "size": self.size,
            "truncated": self.truncated,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ReadFileResult(ToolResult):
    """read_file's content and line counts.

    Attributes:
        content: The text returned.
        size: Its size in bytes.
        lines_total: The line count of the file, or of the capped prefix when truncated.
        start_line: The first line returned; None for a full read, with `lines_returned`.
        lines_returned: The number of lines returned; None for a full read.
        truncated: The file was larger than the read cap, so the content and the line counts
            are of the capped prefix only.
    """

    content: str
    size: int
    lines_total: int
    start_line: int | None = None
    lines_returned: int | None = None
    truncated: bool = False

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        out: dict[str, Any] = {
            "content": self.content,
            "size": self.size,
            "lines_total": self.lines_total,
        }
        if self.start_line is not None:
            out["start_line"] = self.start_line
            out["lines_returned"] = self.lines_returned
        if self.truncated:
            out["truncated"] = True
        return out

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{self.size} bytes{' (truncated)' if self.truncated else ''}"


@dataclasses.dataclass(frozen=True, slots=True)
class ListDirResult(ToolResult):
    """list_dir's entries.

    Attributes:
        entries: The visible names, directories with a trailing slash.
        hidden: How many entries the workspace boundary hides, counted rather than named.
        truncated: Whether the listing stopped at the cap.
    """

    entries: tuple[str, ...]
    hidden: int = 0
    truncated: bool = False

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        out: dict[str, Any] = {"entries": list(self.entries)}
        if self.hidden:
            out["hidden"] = self.hidden
        if self.truncated:
            out["truncated"] = True
        return out

    def summary(self) -> str:
        """Return the one-line summary."""
        extra = f", {self.hidden} hidden" if self.hidden else ""
        cut = " (truncated)" if self.truncated else ""
        return f"{len(self.entries)} entries{extra}{cut}"


@dataclasses.dataclass(frozen=True, slots=True)
class OutlineResult(ToolResult):
    """outline's symbol rows, each {name, kind, line, col}."""

    symbols: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"symbols": list(self.symbols), "truncated": self.truncated}

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{len(self.symbols)} symbols{_trunc(self.truncated)}"


@dataclasses.dataclass(frozen=True, slots=True)
class DefinitionsResult(ToolResult):
    """find_definition's rows, each {name, kind, path, line, col}."""

    definitions: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"definitions": list(self.definitions), "truncated": self.truncated}

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{len(self.definitions)} definitions{_trunc(self.truncated)}"


@dataclasses.dataclass(frozen=True, slots=True)
class ReferencesResult(ToolResult):
    """find_references's rows, each {name, path, line, col}."""

    references: tuple[dict[str, Any], ...]
    truncated: bool

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"references": list(self.references), "truncated": self.truncated}

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{len(self.references)} references{_trunc(self.truncated)}"


@dataclasses.dataclass(frozen=True, slots=True)
class EditResult(ToolResult):
    """apply_edit that wrote.

    Attributes:
        applied: The kind of each edit applied, in order.
        path: The workspace-relative path.
        created: The write made the file; off the wire, read by the memory use record.
    """

    applied: tuple[str, ...]
    path: str
    created: bool = False

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"applied": list(self.applied), "path": self.path}

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"applied={list(self.applied)} path={self.path}"


@dataclasses.dataclass(frozen=True, slots=True)
class PatchResult(ToolResult):
    """apply_patch that wrote.

    Attributes:
        path: The first file's path.
        bytes_written: The bytes written in all.
        files: One (path, bytes_written) row per file for a multi-file patch; empty for one file.
        deleted: The paths the patch deleted, disjoint from `files`.
        healed: The hunks the matcher healed rather than matched exactly (`~rstrip`, `~indent`,
            `~moved`), so the model knows its context was off.
    """

    path: str
    bytes_written: int
    files: tuple[tuple[str, int], ...] = ()
    deleted: tuple[str, ...] = ()
    healed: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {"path": self.path, "bytes_written": self.bytes_written}
        if self.files:
            wire["files"] = [{"path": p, "bytes_written": b} for p, b in self.files]
        if self.deleted:
            wire["deleted"] = list(self.deleted)
        if self.healed:
            wire["healed"] = list(self.healed)
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        if not self.deleted:
            if self.files:
                return f"patched {len(self.files)} files bytes={self.bytes_written}"
            return f"patched path={self.path} bytes={self.bytes_written}"
        if not self.files:
            if len(self.deleted) == 1:
                return f"deleted path={self.deleted[0]}"
            return f"deleted {len(self.deleted)} files"
        return (
            f"patched {len(self.files)} files, deleted {len(self.deleted)}"
            f" bytes={self.bytes_written}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PreviewResult(ToolResult):
    """An edit tool's dry run.

    Attributes:
        path: The first file's path.
        diff: The unified diff, capped.
        hunks: The hunk count.
        bytes_before: The size on disk before.
        bytes_after: The size on disk after.
        truncated: Whether the diff was cut.
        would_apply: The kind of each edit, for apply_edit; None for apply_patch.
        files: Every previewed path in patch order for a multi-file patch; empty for one file.
        healed: The hunks the matcher would heal rather than match exactly.
    """

    path: str
    diff: str
    hunks: int
    bytes_before: int
    bytes_after: int
    truncated: bool
    would_apply: tuple[str, ...] | None = None
    files: tuple[str, ...] = ()
    healed: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
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
        if self.files:
            out["files"] = list(self.files)
        if self.healed:
            out["healed"] = list(self.healed)
        return out


@dataclasses.dataclass(frozen=True, slots=True)
class FetchResult(ToolResult):
    """fetch's response: one URL's text, with a 30x's Location since redirects are not followed."""

    url: str
    status: int
    content_type: str
    body: str
    location: str = ""

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {
            "url": self.url,
            "status": self.status,
            "content_type": self.content_type,
            "body": self.body,
        }
        if self.location and 300 <= self.status < 400:
            wire["location"] = self.location
            wire["note"] = "redirects are not followed; fetch this URL if you still want it"
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{self.status} · {len(self.body)} bytes"


@dataclasses.dataclass(frozen=True, slots=True)
class ExecResult(ToolResult):
    """The jailed command's outcome, for run_command and run_verify_command.

    A `background_id` means the command was handed back and still runs.
    `stopped_by_operator` means an immediate stop ended it before it finished, with no live handle.

    Attributes:
        returncode: The exit code, or None while running or when the code is unknown.
        stdout: The output so far.
        stderr: The errors so far.
        duration_s: The wall-clock time until the result.
        exec_failed: The command could not be started.
        command: What ran, for a command the model did not choose (a verify gate).
        background_id: The background job the command continues as, when handed back.
        timeout_s: The wall-clock cap the runner enforced, 0 when none; with rc 124 it tells a
            timeout from a failure.
        stopped_by_operator: The operator's immediate stop ended the command before it finished.
    """

    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool
    command: tuple[str, ...] = ()
    background_id: str = ""
    timeout_s: float = 0.0
    stopped_by_operator: bool = False
    stdout_truncated: bool | None = None
    stderr_truncated: bool | None = None
    verification: evidence.VerificationEvidence | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "exec_failed": self.exec_failed,
        }
        if self.stdout_truncated is not None:
            wire["stdout_truncated"] = self.stdout_truncated
        if self.stderr_truncated is not None:
            wire["stderr_truncated"] = self.stderr_truncated
        if self.verification is not None:
            wire["verification"] = dataclasses.asdict(self.verification)
        if self.command:
            wire["command"] = shlex.join(self.command)
        if self.returncode == 124 and self.timeout_s > 0:
            wire["timed_out"] = True
            wire["timeout_s"] = self.timeout_s
        if self.stopped_by_operator:
            wire["stopped_by_operator"] = True
        elif self.background_id:
            wire["still_running"] = True
            wire["background_id"] = self.background_id
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        if self.stopped_by_operator:
            return f"operator's stop ended the command before it finished ({self.duration_s:.1f}s)"
        if self.background_id:
            return f"still running as {self.background_id} after {self.duration_s:.1f}s"
        if self.returncode == 124 and self.timeout_s > 0:
            return f"exit=124 (timed out at {self.timeout_s:.0f}s) in {self.duration_s:.1f}s"
        return f"exit={self.returncode} in {self.duration_s:.1f}s"


@dataclasses.dataclass(frozen=True, slots=True)
class MetricResult(ToolResult):
    """run_metric_command's outcome: the exec fields plus the parsed score."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    exec_failed: bool
    score: float | None
    timeout_s: float = 0.0

    @classmethod
    def from_exec(cls, res: ExecResult, score: float | None) -> MetricResult:
        """Pair an exec result with its parsed score.

        Args:
            res: The metric command's exec result.
            score: The parsed score, or None when the metric pattern found none.

        Returns:
            The metric result.

        Raises:
            ValueError: The command was handed back; a score needs a verdict.
        """
        if res.returncode is None:  # pragma: no cover - the gate sets no check-in
            raise ValueError("a metric command cannot be handed back: a score needs a verdict")
        return cls(
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration_s=res.duration_s,
            exec_failed=res.exec_failed,
            score=score,
            timeout_s=res.timeout_s,
        )

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "exec_failed": self.exec_failed,
            "score": self.score,
        }
        if self.returncode == 124 and self.timeout_s > 0:
            wire["timed_out"] = True
            wire["timeout_s"] = self.timeout_s
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        if self.returncode == 124 and self.timeout_s > 0:
            return f"exit=124 (timed out at {self.timeout_s:.0f}s) in {self.duration_s:.1f}s"
        return f"exit={self.returncode} in {self.duration_s:.1f}s"


@dataclasses.dataclass(frozen=True, slots=True)
class FinishSessionResult(ToolResult):
    """finish_session's acknowledgement.

    Attributes:
        summary_text: The model's summary for the operator.
        result: The structured payload, when the task named a result schema.
        stale_gate: The model's proposed verify command, recorded for the operator only.
    """

    summary_text: str
    result: dict[str, Any] | None
    stale_gate: str = ""

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {
            "acknowledged": True,
            "summary": self.summary_text,
            "result": self.result,
        }
        if self.stale_gate:
            # Said plainly, so the model does not finish believing it swapped the gate.
            wire["stale_gate"] = (
                f"recorded for the operator: {self.stale_gate}."
                " This run's gate is unchanged and this run does not pass."
            )
        return wire


@dataclasses.dataclass(frozen=True, slots=True)
class FinishPlanningResult(ToolResult):
    """finish_planning's acknowledgement."""

    summary_text: str
    plan_bytes: int

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"acknowledged": True, "summary": self.summary_text, "plan_bytes": self.plan_bytes}


@dataclasses.dataclass(frozen=True, slots=True)
class AnswersResult(ToolResult):
    """ask_user's outcome.

    Attributes:
        id: The prompt's id, the key a late answer arrives under.
        answers: Aligned to the questions by index; None while pending.
        status: `answered`, `pending` (queued, the answers arrive later) or `unanswered`.
        note: The state the call met, when it was not answered.
        asked: The questions' texts, index-aligned; off the wire, the loop's decision record.
    """

    id: str
    answers: tuple[str, ...] | None
    status: operator_prompts.QuestionStatus = "answered"
    note: str = ""
    asked: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "answers": None if self.answers is None else list(self.answers),
        }
        if self.note:
            wire["note"] = self.note
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        if self.answers is None:
            return "queued, no answer yet"
        if self.status == "unanswered":
            return "unanswered"
        answered = sum(1 for a in self.answers if str(a).strip())
        return f"{answered}/{len(self.answers)} answered"


@dataclasses.dataclass(frozen=True, slots=True)
class AddTaskResult(ToolResult):
    """add_task's new node."""

    id: str
    parent_id: str | None
    title: str
    status: str

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "title": self.title,
            "status": self.status,
        }

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{self.status}: {str(self.title)[:60]}"


@dataclasses.dataclass(frozen=True, slots=True)
class UpdateTaskResult(ToolResult):
    """update_task's node after the change.

    Attributes:
        id: The task id.
        status: The status after the change.
        title: The task's title.
        depends_on: The dependencies after the change.
        note: What the graph did beyond the status: marking a task in_progress claims the focus.
    """

    id: str
    status: str
    title: str
    depends_on: tuple[str, ...] = ()
    note: str = ""

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "title": self.title,
            "depends_on": list(self.depends_on),
        }
        if self.note:
            wire["note"] = self.note
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{self.status}: {str(self.title)[:60]}"


@dataclasses.dataclass(frozen=True, slots=True)
class ListTasksResult(ToolResult):
    """list_tasks's rows, each {id, parent_id, title, status, acceptance, relevant_paths, ...}."""

    tasks: tuple[dict[str, Any], ...]
    count: int

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"tasks": list(self.tasks), "count": self.count}

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{self.count} tasks"


@dataclasses.dataclass(frozen=True, slots=True)
class SkillResult(ToolResult):
    """use_skill's content: one file of one skill."""

    skill: str
    file: str
    content: str

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return {"skill": self.skill, "file": self.file, "content": self.content}

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"skill {self.skill}/{self.file} ({len(self.content)} chars)"


@dataclasses.dataclass(frozen=True, slots=True)
class RawResult(ToolResult):
    """An MCP server's result: an opaque dict forwarded to the model unchanged."""

    payload: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        return self.payload


@dataclasses.dataclass(frozen=True, slots=True)
class BackgroundResult(ToolResult):
    """A background command tool's result.

    The roster rides on every one, so the model also learns that a command it started has died.
    """

    shells: tuple[str, ...]
    output: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {"shells": list(self.shells)}
        if self.output is not None:
            wire["output"] = self.output
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        return self.shells[0] if len(self.shells) == 1 else f"{len(self.shells)} background"


@dataclasses.dataclass(frozen=True, slots=True)
class SessionsResult(ToolResult):
    """The project's sessions, and one session's conversation when asked for."""

    sessions: tuple[str, ...]
    conversation: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {"sessions": list(self.sessions)}
        if self.conversation is not None:
            wire["conversation"] = self.conversation
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        return f"{len(self.sessions)} session{'' if len(self.sessions) == 1 else 's'}"


@dataclasses.dataclass(frozen=True, slots=True)
class FindingBrief:
    """One finding as `list_findings` reports it.

    Attributes:
        id: The id the operator's commands take.
        status: open or resolved.
        severity: block, warn or nit.
        category: The reviewer's category.
        location: The cited path and lines, with the side when it is the base.
        title: The finding's title.
        detail: The finding's detail.
        pre_existing: The base location and reason when the cited line bytes predate the run.
        resolution: A resolved finding's disposition and reason.
        fix_state: A requested fix's delivery state; "" when none was requested.
    """

    id: str
    status: str
    severity: str
    category: str
    location: str
    title: str
    detail: str
    pre_existing: str = ""
    resolution: str = ""
    fix_state: str = ""

    def to_wire(self) -> dict[str, str]:
        """Return the wire dict, the empty fields left out."""
        return {key: value for key, value in dataclasses.asdict(self).items() if value}


@dataclasses.dataclass(frozen=True, slots=True)
class FindingsResult(ToolResult):
    """list_findings: the session's findings.

    Attributes:
        findings: The findings listed, in journal order.
        note: The journal's legacy notice, when its review counts predate structured findings.
    """

    findings: tuple[FindingBrief, ...]
    note: str = ""

    def to_wire(self) -> dict[str, Any]:
        """Return the wire dict."""
        wire: dict[str, Any] = {"findings": [f.to_wire() for f in self.findings]}
        if self.note:
            wire["note"] = self.note
        return wire

    def summary(self) -> str:
        """Return the one-line summary."""
        open_count = sum(1 for f in self.findings if f.status == "open")
        resolved = len(self.findings) - open_count
        return f"{open_count} open" + (f", {resolved} resolved" if resolved else "")
