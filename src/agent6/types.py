# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Internal value types, frozen dataclasses, constructed by us only.

Compare with `agent6.models` which holds pydantic models at LLM/config boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TernaryMode = Literal["no", "ask", "yes"]
# `none` is the unsandboxed profile selected on hosts without the Linux
# sandbox (macOS, or any non-Linux platform). It runs child commands as
# plain subprocesses with no kernel-enforced confinement; it is only ever
# reached via `select_profile` on a non-Linux host and never from config.
SandboxProfile = Literal["strict", "hardened", "none"]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of running a command (in or out of the jail)."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class JailPolicy:
    """What the jail is allowed to do for a single child invocation."""

    cwd: Path
    argv: tuple[str, ...]
    profile: SandboxProfile = "strict"
    env: tuple[tuple[str, str], ...] = ()
    allow_network: bool = False
    extra_ro_paths: tuple[Path, ...] = ()
    extra_rw_paths: tuple[Path, ...] = ()
    # Paths inside ``cwd`` that the launcher must make read-only from the
    # child's view. Strict re-binds them RO on top of the workspace mount;
    # hardened switches its Landlock rules from "RW on cwd" to "R on cwd
    # + RW on each top-level entry except these". Used to keep an
    # LLM-driven ``run_command`` from rewriting ``.git`` even though it
    # lives inside the project root.
    extra_protect_paths: tuple[Path, ...] = ()
    timeout_s: float = 600.0


@dataclass(frozen=True, slots=True)
class RepoSummary:
    """Compact view of a repository handed to the planner."""

    root: Path
    branch: str
    head_sha: str
    file_count: int
    top_level: tuple[str, ...]
    agents_md: str
    recent_log: str
    # Top co-change pairs mined from `git log --name-only`. Tuple
    # of (file_a, file_b, count) sorted by count desc. Empty when the
    # repo has insufficient history (e.g. fresh --depth=1 clone in the
    # realworld bench) or when no pair co-changed at least 2 commits.
    co_change_pairs: tuple[tuple[str, str, int], ...] = ()
    # Top "hot" symbols mined from the tree-sitter index. Tuple
    # of (name, kind, def_path, def_line, files_referenced) sorted by
    # cross-file reference count desc. Complements co_change_pairs:
    # works on fresh repos (no history needed). Empty when the index
    # is disabled or no symbol crosses the min_files_referenced
    # threshold.
    hot_symbols: tuple[tuple[str, str, str, int, int], ...] = ()
    # Compact directory map built from `git ls-files`. Multi-line
    # string of `path/  (N files: a, b, ...)` rows, capped so it stays
    # within a few KB. Empty outside a git repo or when ls-files fails.
    repo_map: str = ""
    # per-file symbol outline mined from the tree-sitter index.
    # Multi-line string of `PATH:` headers followed by `  KIND NAME:LINE`
    # rows, ordered by source position. Capped so the block never exceeds
    # a few KB of system-prompt space; oversized files are truncated with
    # a `... (+N more)` row, and overflow at the file level is summarised
    # as `... (N more files)`. Empty when no parser is available or the
    # index is disabled.
    symbol_outline: str = ""


@dataclass(frozen=True, slots=True)
class FileContext:
    """Files relevant to a single workflow step, gathered deterministically by the workflow."""

    files: tuple[tuple[Path, str], ...] = field(default_factory=tuple)

    def as_text(self, max_chars_per_file: int = 200_000) -> str:
        chunks: list[str] = []
        for path, content in self.files:
            body = (
                content
                if len(content) <= max_chars_per_file
                else (content[:max_chars_per_file] + "\n…[truncated]…\n")
            )
            chunks.append(f"### {path}\n```\n{body}\n```")
        return "\n\n".join(chunks)


@dataclass(frozen=True, slots=True)
class SandboxReport:
    """Result of one sandbox self-test."""

    name: str
    ok: bool
    detail: str
