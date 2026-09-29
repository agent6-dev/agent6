# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Infer a verify command for a run when none is configured.

Cheapest source first: the `## Verify command` or `## Test` section of
AGENTS.md or an inline `Verify:` line, then the repo's own files (a root
`verify.sh`, package.json, a Makefile target, a Python manifest, Cargo, go.mod,
loose tests), then an injected LLM call over the manifests and AGENTS.md. The
result lives in memory for one run and is never written to config. A shell
pipeline is wrapped as `sh -c`; `sh` and operator tools such as `uv` resolve on
the jail PATH.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shlex
from collections.abc import Callable

# A command line with any of these is a shell construct, wrapped in `sh -c`.
_SHELL_META = re.compile(r"(\|\||&&|[|&;<>`]|\$\()")
_VERIFY_HEADING = re.compile(r"^#{1,6}\s*(verify|test)\b", re.IGNORECASE)
_INLINE_VERIFY = re.compile(r"^\s*(?:verify|test)\s*:\s*(.+)$", re.IGNORECASE)
_MAKE_TARGET = re.compile(r"^([A-Za-z0-9_-]+)\s*:", re.MULTILINE)


@dataclasses.dataclass(frozen=True, slots=True)
class InferredVerify:
    """A verify command inferred for one run, never persisted.

    Attributes:
        argv: The command.
        source: Where it came from: "agents_md", "package.json", "Makefile:test", "llm", ...
    """

    argv: tuple[str, ...]
    source: str


def line_to_argv(cmd: str) -> tuple[str, ...] | None:
    """Return one command line as argv, a shell pipeline wrapped in `sh -c`; None when empty."""
    cmd = cmd.strip()
    if not cmd:
        return None
    if _SHELL_META.search(cmd):
        return ("sh", "-c", cmd)
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    return tuple(parts) or None


def _block_to_argv(block: list[str]) -> tuple[str, ...] | None:
    """Return a fenced block as one argv: comments dropped, continued lines joined."""
    logical: list[str] = []
    for raw in block:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        logical.append(line)
    if not logical:
        return None
    joined = " ".join(part.rstrip("\\").strip() for part in logical).strip()
    return line_to_argv(joined)


def _first_fenced_block(lines: list[str], start: int) -> list[str] | None:
    """Return the first fenced block opening within 12 lines of the start, else None."""
    i = start
    while i < len(lines) and i < start + 12:
        if lines[i].lstrip().startswith("```"):
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            return body
        i += 1
    return None


def read_agents_md(repo_root: pathlib.Path) -> str:
    """Return the repo's AGENTS.md text; "" when absent or unreadable."""
    path = repo_root / "AGENTS.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def verify_from_agents_md(agents_md: str) -> tuple[str, ...] | None:
    """Return the verify command AGENTS.md states, or None.

    A `## Verify command` or `## Test` heading followed by a fenced block, or an
    inline `Verify:` or `Test:` line.
    """
    if not agents_md:
        return None
    lines = agents_md.splitlines()
    for i, line in enumerate(lines):
        if _VERIFY_HEADING.match(line.strip()):
            block = _first_fenced_block(lines, i + 1)
            if block is not None and (argv := _block_to_argv(block)) is not None:
                return argv
    for line in lines:
        m = _INLINE_VERIFY.match(line)
        if m and (argv := line_to_argv(_unquote_code(m.group(1)))) is not None:
            return argv
    return None


def _unquote_code(text: str) -> str:
    """Return an inline command minus its backticks, which would read as a shell substitution."""
    text = text.strip().removesuffix(".") if text.rstrip().endswith("`.") else text.strip()
    if len(text) > 1 and text[0] == "`" and text[-1] == "`":
        return text[1:-1]
    return text


def _has_make_target(text: str, target: str) -> bool:
    """Return whether a Makefile defines the target."""
    return any(m.group(1) == target for m in _MAKE_TARGET.finditer(text))


def _python(repo_root: pathlib.Path) -> str:
    """Return the interpreter a pytest gate runs with: the project's `.venv`, else `python3`."""
    return ".venv/bin/python" if (repo_root / ".venv" / "bin" / "python").exists() else "python3"


Signal = tuple[tuple[str, ...], str]


def _verify_sh(repo_root: pathlib.Path) -> Signal | None:
    """Return the root `verify.sh` as the command, or None."""
    script = repo_root / "verify.sh"
    if not script.is_file():
        return None
    argv = ("./verify.sh",) if os.access(script, os.X_OK) else ("sh", "verify.sh")
    return (argv, "verify.sh")


def _package_json(repo_root: pathlib.Path) -> Signal | None:
    """Return `npm test` when package.json has a test script, or None."""
    pkg = repo_root / "package.json"
    if not pkg.is_file():
        return None
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if isinstance(scripts, dict) and isinstance(scripts.get("test"), str):
        return (("npm", "test", "--silent"), "package.json")
    return None


def _makefile(repo_root: pathlib.Path) -> Signal | None:
    """Return `make test` or `make check` when a Makefile defines it, or None."""
    for mk in ("Makefile", "makefile", "GNUmakefile"):
        p = repo_root / mk
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            txt = ""
        for target in ("test", "check"):
            if _has_make_target(txt, target):
                return (("make", target), f"Makefile:{target}")
    return None


def _python_manifest(repo_root: pathlib.Path) -> Signal | None:
    """Return a pytest run when a Python manifest exists, or None."""
    manifests = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "setup.py")
    if any((repo_root / f).is_file() for f in manifests):
        return ((_python(repo_root), "-m", "pytest", "-q"), "pyproject")
    return None


def _cargo(repo_root: pathlib.Path) -> Signal | None:
    """Return `cargo test` when Cargo.toml exists, or None."""
    return (
        (("cargo", "test", "--quiet"), "Cargo.toml")
        if (repo_root / "Cargo.toml").is_file()
        else None
    )


def _go(repo_root: pathlib.Path) -> Signal | None:
    """Return `go test` when go.mod exists, or None."""
    return (("go", "test", "./..."), "go.mod") if (repo_root / "go.mod").is_file() else None


def _loose_python_tests(repo_root: pathlib.Path) -> Signal | None:
    """Return a pytest run when loose `test_*.py` files exist, or None."""
    if any(repo_root.glob("test_*.py")) or any((repo_root / "tests").glob("test_*.py")):
        return ((_python(repo_root), "-m", "pytest", "-q"), "test_*.py")
    return None


# Loose Python tests last, so a Rust or Go repo's `tests/` dir never reads as pytest.
_REPO_SIGNALS = (
    _verify_sh,
    _package_json,
    _makefile,
    _python_manifest,
    _cargo,
    _go,
    _loose_python_tests,
)


def verify_from_repo_signals(repo_root: pathlib.Path) -> Signal | None:
    """Return the first repo signal that matches, or None."""
    return next((found for probe in _REPO_SIGNALS if (found := probe(repo_root))), None)


# The files fed to the LLM.
_MANIFEST_FILES = (
    "pyproject.toml",
    "package.json",
    "Makefile",
    "Cargo.toml",
    "go.mod",
    "tox.ini",
    "pytest.ini",
    "noxfile.py",
)

VERIFY_INFER_SYSTEM_PROMPT = (
    "You infer the single command a CI/verify step runs to decide whether a change to THIS"
    " repository passes (build + tests). You are given the repo's manifest files and AGENTS.md.\n\n"
    'Reply with ONLY a JSON array of argv strings and nothing else, e.g. ["pytest","-q"].\n'
    "Hard rules:\n"
    "- The command runs in a locked-down sandbox: PATH is /usr/bin:/bin, the standard bin"
    " dirs that exist (/usr/local/bin, ~/.local/bin, ~/.cargo/bin, ...), and the repo dir,"
    " with an ephemeral $HOME and NO network. Operator tools resolve, so `uv run pytest`"
    " works (it uses the already-synced venv; the sandbox cannot sync). Prefer the"
    " project's real runner: `uv run ...` for a uv project, else a stdlib .venv/bin/python"
    " or /usr/bin/python3, or system cargo/go/node/make.\n"
    "- Prefer the project's real fast test/build command, not a lint-only or syntax check.\n"
    '- If you need a shell pipeline, return ["sh","-c","<pipeline>"].\n'
    "- If you genuinely cannot determine one, return []."
)


def gather_repo_manifests(repo_root: pathlib.Path, agents_md: str, *, cap: int = 4000) -> str:
    """Return the clipped manifest files and AGENTS.md as the LLM call's context."""
    parts: list[str] = []
    try:
        top = sorted(p.name + ("/" if p.is_dir() else "") for p in repo_root.iterdir())
    except OSError:
        top = []
    parts.append("<top-level>\n" + " ".join(top[:80]) + "\n</top-level>")
    for name in _MANIFEST_FILES:
        p = repo_root / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f'<file path="{name}">\n{text[:cap]}\n</file>')
    if agents_md.strip():
        parts.append(f"<AGENTS.md>\n{agents_md[:cap]}\n</AGENTS.md>")
    return "\n\n".join(parts)


def parse_llm_verify(text: str) -> tuple[str, ...] | None:
    """Return the JSON argv array in the model's reply, or None."""
    if not text:
        return None
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(x, str) and x.strip() for x in data):
        return None
    return tuple(x for x in data)


def infer_verify_command(
    repo_root: pathlib.Path,
    agents_md: str,
    *,
    llm_call: Callable[[str], str] | None = None,
) -> InferredVerify | None:
    """Infer a verify command: AGENTS.md, then the repo's files, then the LLM.

    Args:
        repo_root: The repository.
        agents_md: The AGENTS.md text.
        llm_call: Takes the gathered context and returns the model's text; None skips
            the LLM tier, as does a repo with no manifest and no AGENTS.md prose.

    Returns:
        The command and its source, or None when unknown.
    """
    argv = verify_from_agents_md(agents_md)
    if argv is not None:
        return InferredVerify(argv=argv, source="agents_md")
    sig = verify_from_repo_signals(repo_root)
    if sig is not None:
        return InferredVerify(argv=sig[0], source=sig[1])
    readable = bool(agents_md.strip()) or any(
        (repo_root / name).is_file() for name in _MANIFEST_FILES
    )
    if llm_call is not None and readable:
        context = gather_repo_manifests(repo_root, agents_md)
        try:
            raw = llm_call(context)
        except Exception:  # never fails the run
            raw = ""
        argv = parse_llm_verify(raw)
        if argv is not None:
            return InferredVerify(argv=argv, source="llm")
    return None
