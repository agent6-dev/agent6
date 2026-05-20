# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validates incoming LLM tool calls and executes them.

All filesystem reads/writes are clamped to *root* (the repo cwd). All command
execution goes through agent6.sandbox.jail.run_in_jail. Capability gating
(`run_commands = "no" | "ask" | "yes"`) is enforced here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent6.config import Config
from agent6.events import EventSink
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.tools.schema import (
    ALL_TOOLS,
    ApplyEditInput,
    GrepInput,
    ListDirInput,
    ReadFileInput,
    RunCommandInput,
    RunVerifyInput,
)
from agent6.types import CommandResult, JailPolicy, SandboxProfile


class ToolError(Exception):
    """The LLM tried something the tool layer refused."""


class _Approver(Protocol):
    def __call__(self, prompt: str, /) -> bool: ...


def _default_approver(prompt: str) -> bool:  # pragma: no cover — interactive
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


@dataclass(frozen=True, slots=True)
class _SafePath:
    abs_path: Path
    rel_path: Path


def _resolve_in_root(root: Path, candidate: str) -> _SafePath:
    """Resolve *candidate* relative to *root* and ensure it stays inside *root*."""
    if candidate.startswith("/"):
        raise ToolError(f"Absolute paths not allowed: {candidate!r}")
    if ".." in Path(candidate).parts:
        raise ToolError(f"Path contains '..': {candidate!r}")
    abs_path = (root / candidate).resolve()
    try:
        rel = abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ToolError(f"Path escapes repo root: {candidate!r}") from exc
    return _SafePath(abs_path=abs_path, rel_path=rel)


class ToolDispatcher:
    """Runtime tool dispatcher. Constructed once per workflow run."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        sandbox_profile: SandboxProfile = "strict",
        approver: _Approver | None = None,
        events: EventSink | None = None,
    ) -> None:
        self._root = root.resolve()
        self._config = config
        self._sandbox_profile: SandboxProfile = sandbox_profile
        self._approver: _Approver = approver or _default_approver
        self._events = events
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            ReadFileInput.TOOL_NAME: self._read_file,
            ListDirInput.TOOL_NAME: self._list_dir,
            GrepInput.TOOL_NAME: self._grep,
            ApplyEditInput.TOOL_NAME: self._apply_edit,
            RunVerifyInput.TOOL_NAME: self._run_verify,
            RunCommandInput.TOOL_NAME: self._run_command,
        }
        self._available = {cls.TOOL_NAME for cls in ALL_TOOLS}

    @property
    def root(self) -> Path:
        return self._root

    def available_tool_names(self) -> tuple[str, ...]:
        # run_command is filtered out if disabled.
        names = list(self._available)
        if self._config.sandbox.run_commands == "no":
            names = [n for n in names if n != RunCommandInput.TOOL_NAME]
        return tuple(sorted(names))

    def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        if name not in self._handlers:
            raise ToolError(f"Unknown tool: {name}")
        if name == RunCommandInput.TOOL_NAME and self._config.sandbox.run_commands == "no":
            raise ToolError("run_command is disabled by config (run_commands = 'no')")
        self._emit("tool.call", name=name, args=_truncate_args(raw_input))
        try:
            result = self._handlers[name](raw_input)
        except ToolError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise
        except Exception as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise ToolError(f"{name} failed: {exc}") from exc
        self._emit("tool.result", name=name, ok=True, summary=_summarize_result(name, result))
        return result

    def _emit(self, event_type: str, /, **fields: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, **fields)

    # ----- handlers -----

    def _read_file(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ReadFileInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_file():
            raise ToolError(f"Not a file: {args.path}")
        try:
            content = sp.abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"File is not UTF-8 text: {args.path}") from exc
        return {"content": content, "size": len(content)}

    def _list_dir(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ListDirInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_dir():
            raise ToolError(f"Not a directory: {args.path}")
        entries: list[str] = []
        for entry in sorted(sp.abs_path.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.name + suffix)
        return {"entries": entries}

    def _grep(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = GrepInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        try:
            pat = re.compile(args.pattern, re.IGNORECASE if args.case_insensitive else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc
        hits: list[dict[str, Any]] = []
        targets: list[Path]
        if sp.abs_path.is_file():
            targets = [sp.abs_path]
        else:
            targets = [p for p in sp.abs_path.rglob("*") if p.is_file()]
        for path in targets:
            if any(part.startswith(".") for part in path.relative_to(self._root).parts):
                continue
            try:
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    start=1,
                ):
                    if pat.search(line):
                        hits.append(
                            {
                                "path": str(path.relative_to(self._root)),
                                "line": lineno,
                                "text": line[:500],
                            }
                        )
                        if len(hits) >= 500:
                            return {"hits": hits, "truncated": True}
            except OSError:
                continue
        return {"hits": hits, "truncated": False}

    def _apply_edit(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ApplyEditInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        # Write-outside-cwd is enforced by _resolve_in_root already (root == cwd).
        applied: list[str] = []
        existing = sp.abs_path.read_text(encoding="utf-8") if sp.abs_path.exists() else None
        new_content = existing
        for i, edit in enumerate(args.edits):
            if edit.kind == "create":
                if existing is not None and i == 0:
                    raise ToolError(f"create requested but file already exists: {args.path}")
                new_content = edit.new_string
                applied.append("create")
            else:
                if new_content is None:
                    raise ToolError(f"replace requested but file does not exist: {args.path}")
                occurrences = new_content.count(edit.old_string)
                if occurrences == 0:
                    # Include the current file content (truncated) so the worker's
                    # retry within the step sees the actual on-disk state rather
                    # than its (possibly stale) prior view.
                    snippet = (
                        new_content
                        if len(new_content) <= 4000
                        else (
                            new_content[:2000]
                            + f"\n... <truncated {len(new_content) - 4000} chars> ...\n"
                            + new_content[-2000:]
                        )
                    )
                    raise ToolError(
                        f"old_string not found in {args.path} (edit #{i}). "
                        f"Current file content:\n---BEGIN {args.path}---\n"
                        f"{snippet}\n---END {args.path}---"
                    )
                if occurrences > 1:
                    raise ToolError(
                        f"old_string is not unique in {args.path} "
                        f"(edit #{i}, {occurrences} matches); add more surrounding "
                        f"context to make it unique"
                    )
                new_content = new_content.replace(edit.old_string, edit.new_string, 1)
                applied.append("replace")
        if new_content is None:
            raise ToolError("No content to write")
        sp.abs_path.parent.mkdir(parents=True, exist_ok=True)
        sp.abs_path.write_text(new_content, encoding="utf-8")
        return {"applied": applied, "path": str(sp.rel_path)}

    def _run_verify(self, _raw: dict[str, Any]) -> dict[str, Any]:
        argv = tuple(self._config.workflow.verify_command)
        self._emit("verify.start", cmd=list(argv))
        res = self._run_argv_in_jail(argv, label="verify_command")
        self._emit(
            "verify.end",
            cmd=list(argv),
            exit_code=res["returncode"],
            duration_s=res["duration_s"],
            stdout_tail=str(res["stdout"])[-2000:],
            stderr_tail=str(res["stderr"])[-2000:],
        )
        return res

    def _run_command(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = RunCommandInput.model_validate(raw)
        if self._config.sandbox.run_commands == "ask":
            ok = self._approver(f"Allow run_command {args.argv!r}?")
            if not ok:
                raise ToolError("run_command denied by user")
        return self._run_argv_in_jail(args.argv, label="run_command")

    def _run_argv_in_jail(self, argv: tuple[str, ...], *, label: str) -> dict[str, Any]:
        allow_network = self._config.sandbox.network == "allow"
        policy = JailPolicy(
            cwd=self._root,
            argv=argv,
            profile=self._sandbox_profile,
            env=tuple(sorted(_passthrough_env().items())),
            allow_network=allow_network,
        )
        try:
            res: CommandResult = run_in_jail(policy)
        except JailUnavailableError as exc:
            raise ToolError(f"{label}: jail unavailable: {exc}") from exc
        return {
            "returncode": res.returncode,
            "stdout": res.stdout[-20_000:],
            "stderr": res.stderr[-20_000:],
            "duration_s": res.duration_s,
        }


_PASSTHROUGH_ENV_KEYS = ("LANG", "LC_ALL", "TERM", "CI")


def _passthrough_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _PASSTHROUGH_ENV_KEYS if k in os.environ}


def _truncate_args(raw: dict[str, Any], *, max_value_chars: int = 200) -> dict[str, Any]:
    """Cheap argument preview for telemetry; truncates strings longer than
    *max_value_chars* and lists longer than 10 items."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str) and len(v) > max_value_chars:
            out[k] = v[:max_value_chars] + f"… ({len(v)} chars)"
        elif isinstance(v, list | tuple) and len(v) > 10:
            out[k] = [*list(v[:10]), f"… ({len(v)} items)"]
        else:
            out[k] = v
    return out


def _summarize_result(name: str, result: dict[str, Any]) -> str:
    """One-line human-readable summary for the TUI / log tail."""
    if "size" in result:
        return f"{result['size']} bytes"
    if "entries" in result and isinstance(result["entries"], list):
        return f"{len(result['entries'])} entries"
    if "hits" in result and isinstance(result["hits"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['hits'])} matches{more}"
    if "applied" in result:
        return f"applied={result['applied']} path={result.get('path')}"
    if "returncode" in result:
        return f"exit={result['returncode']} in {result.get('duration_s', 0):.1f}s"
    return name
