# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Internal value types, frozen dataclasses, constructed by us only.

Compare with the pydantic models at the trust boundaries: `agent6.config.model`
(config), `agent6.providers.types` (LLM I/O), `agent6.tools.schema` (tool
inputs), `agent6.machine.model` (machine files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TernaryMode = Literal["no", "ask", "yes"]
# `none` is the UNSANDBOXED profile: child commands run as plain subprocesses
# with no kernel-enforced confinement. Reached when the host has no confinement
# mechanism at all (non-Linux, or a Linux kernel offering neither userns nor
# Landlock), or as a deliberate operator opt-out on any host via
# `sandbox.profile = "none"`, `--dangerously-disable-sandbox`, or
# `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` (self-authorizing, with a loud warning;
# see `detect.select_profile`).
SandboxProfile = Literal["strict", "hardened", "none"]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of running a command (in or out of the jail)."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    # True when the launcher could not execute the binary at all (bad path, not
    # on the jail PATH, missing interpreter, or a symlink that escapes the
    # sandbox roots). Distinct from "ran and exited non-zero": a
    # model can fix its own argv, but an operator verify/metric command that
    # cannot execute is a config/sandbox problem the run must surface loudly.
    exec_failed: bool = False

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
    # Real-location RO+exec bind mounts for operator-installed tools that live
    # outside the system dirs (uv in ~/.local/bin or the /opt target a
    # /usr/local/bin symlink resolves to), so a verify/run command finds them.
    # Distinct from ``extra_ro_paths`` (remapped under /ro, which breaks symlinks);
    # these keep their real paths. Read+execute only, never writable.
    tool_paths: tuple[Path, ...] = ()
    timeout_s: float = 600.0
    # Per-process memory cap in MiB (RLIMIT_DATA, set by the launcher in the
    # child before exec and inherited by every descendant); 0 disables. The
    # dataclass default matches ``[sandbox].memory_limit_mb`` so call sites
    # that do not carry config (probes, offline script tests) stay bounded.
    memory_limit_mb: int = 4096


@dataclass(frozen=True, slots=True)
class CoChangePair:
    """Two files that changed together, and how many commits they co-changed in."""

    file_a: str
    file_b: str
    count: int


@dataclass(frozen=True, slots=True)
class HotSymbol:
    """A symbol whose rename/signature change would ripple across files."""

    name: str
    kind: str
    def_path: str
    def_line: int
    files_referenced: int


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
    co_change_pairs: tuple[CoChangePair, ...] = ()
    # Top "hot" symbols mined from the tree-sitter index. Tuple
    # of (name, kind, def_path, def_line, files_referenced) sorted by
    # cross-file reference count desc. Complements co_change_pairs:
    # works on fresh repos (no history needed). Empty when the index
    # is disabled or no symbol crosses the min_files_referenced
    # threshold.
    hot_symbols: tuple[HotSymbol, ...] = ()
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
    # False when root is not a git repository (`agent6 ask` runs anywhere;
    # run/plan require git up front). branch/head_sha/recent_log/repo_map
    # are then empty and the prompt names the situation instead of
    # rendering a fake repo header.
    is_git: bool = True


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
