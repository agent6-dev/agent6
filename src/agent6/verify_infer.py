# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Infer a verify command for a run when none is configured.

Cheapest source first: the `## Verify command` or `## Test` section of
AGENTS.md or an inline `Verify:` line, then the repo's own files (a root
`verify.sh`, package.json, a Makefile target, a Python manifest, Cargo, go.mod,
loose tests), then an injected LLM call over the manifests and AGENTS.md. A
command whose first word is a bare name the jail PATH lacks is passed over for
the next source, and the reason is kept. The result lives in memory for one run
and is never written to config. A shell pipeline is wrapped as `sh -c`; `sh` and
operator tools such as `uv` resolve on the jail PATH. The `## Setup command` section
of AGENTS.md is read here too, by the same fence reader.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shlex
from collections.abc import Callable

from agent6 import git_ops
from agent6.sandbox import tool_paths
from agent6.tools import _path_safety
from agent6.tools import errors as tools_errors

# A command line with any of these, or a leading assignment, is shell, wrapped in `sh -c`.
_SHELL_META = re.compile(r"[|&;<>`$*]")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
# A shell operator glued to an unquoted first word (`ls|cat`, `(cd`): the program ends before it.
_OPERATOR = re.compile(r"[|&;<>]")
# A prompt is a whole first word, so `$HOME`, `$(...)` and `>>>` stay intact.
_PROMPT = re.compile(r"^[$>](?:\s+|$)")
# The POSIX builtins a shell line may open with before its program: the jail's own `sh` has
# them; `source` is bash's, absent from dash.
_SH_BUILTINS = frozenset({".", ":", "cd", "eval", "exec", "export", "set", "umask", "unset"})
_VERIFY_HEADING = re.compile(r"^#{1,6}\s*(verify|test)\b", re.IGNORECASE)
_SETUP_HEADING = re.compile(r"^#{1,6}\s*setup command\b", re.IGNORECASE)
_INLINE_VERIFY = re.compile(r"^\s*(?:verify|test)\s*:\s*(.+)$", re.IGNORECASE)
_MAKE_TARGET = re.compile(r"^([A-Za-z0-9_-]+)\s*:", re.MULTILINE)
# The test script `npm init` writes, which exists to fail.
_NPM_PLACEHOLDER = re.compile(r"""echo\s+(["']?)Error: no test specified\1\s*&&\s*exit\s+1""")


@dataclasses.dataclass(frozen=True, slots=True)
class InferredVerify:
    """A verify command inferred for one run, never persisted.

    Attributes:
        argv: The command; empty when nothing was inferred.
        source: Where it came from: "agents_md", "package.json", "Makefile:test", "llm", ...;
            "none" when nothing was inferred.
        passed_over: Each source inference did not use, with the reason, as (source, why).
    """

    argv: tuple[str, ...]
    source: str
    passed_over: tuple[tuple[str, str], ...] = ()

    @property
    def label(self) -> str:
        """The source name for notices; the journal keeps `source` as its token."""
        return "AGENTS.md" if self.source == "agents_md" else self.source


def passed_over_text(inferred: InferredVerify) -> str:
    """Return what inference passed over and why, as one clause; "" when it used everything."""
    return "; ".join(f"{source} {why}" for source, why in inferred.passed_over)


def _strip_comment(line: str) -> str:
    """Return a shell line minus its `#` comment; a `#` in quotes or inside a word stays."""
    quote = ""
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif ch == "\\" and quote != "'":
            escaped = True
        elif quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line


def command_head(argv: tuple[str, ...]) -> str:
    """Return the program a command runs first: past a `sh -c` wrap and `NAME=value` words.

    A pipeline or an `&&` chain is judged by its first program, the one that must exist for
    any of it to run, with or without spaces around the operator; a word the line holds only
    quoted or escaped, and a plain argv, are read as they are. A line whose first word is a
    shell builtin reads as that builtin.

    Args:
        argv: The command, plain or a shell line wrapped as `sh -c LINE`.

    Returns:
        The program's name or path, else the argv's first word (`sh` when the wrapped line
        names no program or does not parse); "" for an empty argv.
    """
    words, raw = argv, ""
    if argv[:2] == ("sh", "-c"):
        raw = argv[2] if len(argv) > 2 else ""
        try:
            words = tuple(shlex.split(raw))
        except ValueError:
            words = ()
    for word in words:
        if _ASSIGNMENT.match(word):
            continue
        # A word the line holds as is was unquoted: an operator in it ends the program's name.
        if word not in raw.split():
            return word
        if head := _OPERATOR.split(word.lstrip("("), maxsplit=1)[0]:
            return head
    return argv[0] if argv else ""


def line_to_argv(cmd: str) -> tuple[str, ...] | None:
    """Return one command line as argv, shell syntax wrapped in `sh -c`; None when empty.

    Args:
        cmd: The line; a `#` comment is dropped.

    Returns:
        The argv; None when the line holds no command, or has no shell syntax and does not
        parse.
    """
    cmd = _strip_comment(cmd.strip())
    if not cmd:
        return None
    if _SHELL_META.search(cmd) or _ASSIGNMENT.match(cmd):
        return ("sh", "-c", cmd)
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    return tuple(parts) or None


def _block_to_argv(block: list[str]) -> tuple[str, ...] | None:
    """Return a fenced block as one argv.

    A block with a `$` prompt, or opening with a `>` prompt, keeps only prompted lines and
    their continuations. Prompts and comments are dropped, a line ending in a shell operator
    or a backslash continues on the next, and the commands chain with `&&`. A `>` line after
    a plain line is a redirect, and a continuation prompt in a transcript.

    Args:
        block: The lines between the fences.

    Returns:
        The argv; None when the block holds no command, or a single command that has no shell
        syntax and does not parse.
    """
    transcript = _has_prompt(block)
    commands: list[str] = []
    continued = False
    for raw in block:
        line = raw.strip()
        if transcript and not continued and not _PROMPT.match(line):
            continue
        line = _strip_comment(_PROMPT.sub("", line) if transcript else line)
        if not line:
            continue
        # A trailing `\` or `;` behind an odd run of backslashes is literal, not an operator.
        continues = line.endswith(("&&", "||", "|")) or (
            line.endswith(("\\", ";")) and not _escaped_end(line)
        )
        part = (line.removesuffix("\\") if continues else line).rstrip()
        if continued:
            commands[-1] += " " + part
        else:
            commands.append(part)
        continued = continues
    return line_to_argv(" && ".join(commands))


def _escaped_end(line: str) -> bool:
    """Return whether the line's last character sits behind an odd run of backslashes."""
    body = line[:-1]
    return (len(body) - len(body.rstrip("\\"))) % 2 == 1


def _has_prompt(block: list[str]) -> bool:
    """Return whether the block is a transcript: a `$` prompt anywhere, or a `>` prompt first.

    A `>` line after a plain line is a redirect. Lines are read as `_block_to_argv` reads
    them: comments off, and a blank line skipped.

    Args:
        block: The lines between the fences.

    Returns:
        True for a transcript.
    """
    lines = [line for line in (_strip_comment(raw.strip()) for raw in block) if line]
    if any(line.startswith("$") and _PROMPT.match(line) for line in lines):
        return True
    return bool(lines) and lines[0].startswith(">") and _PROMPT.match(lines[0]) is not None


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


def read_agents_md(
    repo_root: pathlib.Path, hide: tuple[str, ...] = (), *, at: str | None = None
) -> str:
    """Return AGENTS.md from the directory or a commit, "" when missing or out of bounds.

    Args:
        repo_root: The directory whose AGENTS.md is read.
        hide: The hidden paths.
        at: Read this commit's copy instead of the checkout's; None reads the directory.

    Returns:
        The text, or "" when missing, hidden or outside the git root.

    Raises:
        GitError: A requested commit cannot be read or git fails.
    """
    base, here = git_ops.toplevel(repo_root), repo_root.resolve()
    if base is None or not here.is_relative_to(base.resolve()):
        return (
            "" if at is not None else _path_safety.read_workspace_text(repo_root, "AGENTS.md", hide)
        )
    root = base.resolve()
    name = (here.relative_to(root) / "AGENTS.md").as_posix()
    if at is None:
        return _path_safety.read_workspace_text(root, name, hide)
    try:
        _path_safety.host_workspace(root, hide).resolve_read(name)
    except (OSError, tools_errors.ToolError):
        return ""
    return git_ops.read_file_at(root, at, name)


def verify_from_agents_md(agents_md: str) -> tuple[str, ...] | None:
    """Return the verify command AGENTS.md states, or None.

    A `## Verify command` or `## Test` heading followed by a fenced block, or an
    inline `Verify:` or `Test:` line. Both readers strip whole-word `$` and `>` prompts.

    Args:
        agents_md: The AGENTS.md text.

    Returns:
        The command as argv, a heading's block taking precedence over an inline line; None
        when AGENTS.md states neither.
    """
    if (argv := _heading_block_argv(agents_md, _VERIFY_HEADING)) is not None:
        return argv
    for line in agents_md.splitlines():
        m = _INLINE_VERIFY.match(line)
        # The comment goes first: backticks closed after it would survive the unquote.
        if m:
            command = _unquote_code(_strip_comment(m.group(1))).strip()
            if (argv := line_to_argv(_PROMPT.sub("", command))) is not None:
                return argv
    return None


def setup_from_agents_md(agents_md: str) -> tuple[str, ...] | None:
    """Return the setup command AGENTS.md states, or None.

    A `## Setup command` heading followed by a fenced block, read as the verify block is.

    Args:
        agents_md: The AGENTS.md text.

    Returns:
        The command as argv; None when AGENTS.md states none.
    """
    return _heading_block_argv(agents_md, _SETUP_HEADING)


def _heading_block_argv(agents_md: str, heading: re.Pattern[str]) -> tuple[str, ...] | None:
    """Return the first fenced block under a matching heading as one argv, or None.

    Args:
        agents_md: The AGENTS.md text.
        heading: The pattern a heading line matches, `#` marks included.

    Returns:
        The argv; None when no matching heading's block reads as a command.
    """
    lines = agents_md.splitlines()
    for i, line in enumerate(lines):
        if heading.match(line.strip()):
            block = _first_fenced_block(lines, i + 1)
            if block is not None and (argv := _block_to_argv(block)) is not None:
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


def _hidden(repo_root: pathlib.Path, name: str, hide: tuple[str, ...]) -> bool:
    """Return whether the path, or the path it resolves to, is under a hidden path."""
    path = repo_root.resolve() / name
    workspace = _path_safety.host_workspace(repo_root, hide)
    # realpath, not resolve(): on Python 3.12 resolve() raises RuntimeError on a symlink loop.
    return workspace.is_denied(path) or workspace.is_denied(pathlib.Path(os.path.realpath(path)))


def _present(repo_root: pathlib.Path, name: str, hide: tuple[str, ...]) -> bool:
    """Return whether the file exists and is not under a hidden path."""
    return (repo_root / name).is_file() and not _hidden(repo_root, name, hide)


def _python(repo_root: pathlib.Path, hide: tuple[str, ...]) -> str:
    """Return the interpreter a pytest gate runs with: the project's `.venv`, else `python3`."""
    return ".venv/bin/python" if _present(repo_root, ".venv/bin/python", hide) else "python3"


Signal = tuple[tuple[str, ...], str]
# The sources that each name one ecosystem's own test runner.
_ECOSYSTEM_SOURCES = frozenset({"Cargo.toml", "go.mod", "package.json", "pyproject"})
# What sh would expand or parse in a word: operators, substitutions, globs, `~`, a comment.
_SHELL_SYNTAX = frozenset("&|;<>()$`\\*?[]~#{}")


def _verify_sh(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return the root `verify.sh` as the command, or None."""
    if not _present(repo_root, "verify.sh", hide):
        return None
    argv = ("./verify.sh",) if os.access(repo_root / "verify.sh", os.X_OK) else ("sh", "verify.sh")
    return (argv, "verify.sh")


def _package_json(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return `npm test` when package.json has a test script other than npm's placeholder.

    In a jail without npm the script itself is the gate when it is a plain command whose
    runner the jail has on its PATH; one with shell syntax (`&&`, a pipe, a glob, an env
    assignment, a second line) needs npm's shell and stays `npm test`, as does a runner given
    as a path, which can be untracked or outside the jail, and a script with a `pretest` or
    `posttest` hook, which only npm runs.

    Args:
        repo_root: The directory whose package.json is read.
        hide: `[sandbox].hide_paths`; a hidden package.json reads as absent.

    Returns:
        The command and its source, "package.json"; None when there is no test script, or
        only npm's placeholder.
    """
    try:
        data = json.loads(_path_safety.read_workspace_text(repo_root, "package.json", hide))
    except ValueError:
        data = {}
    scripts = data.get("scripts") if isinstance(data, dict) else None
    script = scripts.get("test") if isinstance(scripts, dict) else None
    if not isinstance(script, str) or _NPM_PLACEHOLDER.fullmatch(script.strip()):
        return None
    hooked = isinstance(scripts, dict) and any(k in scripts for k in ("pretest", "posttest"))
    if not hooked and not tool_paths.jail_resolves("npm") and (argv := _plain_argv(script)):
        return (argv, "package.json")
    return (("npm", "test", "--silent"), "package.json")


def _plain_argv(script: str) -> tuple[str, ...]:
    """Return a shell script as an argv the jail can run, or () when it needs a shell.

    Args:
        script: The package.json test script.

    Returns:
        The script's words; () when it needs a shell, names its runner by path, or the jail's
        PATH lacks the runner.
    """
    if "\n" in script.strip():
        return ()  # sh runs each line as a command of its own
    try:
        words = shlex.split(script)
    except ValueError:
        return ()
    if not words or "=" in words[0] or any(set(w) & _SHELL_SYNTAX for w in words):
        return ()
    # A path runner counts as found, and one missing where the gate runs ends the run unstarted.
    return tuple(words) if "/" not in words[0] and tool_paths.jail_resolves(words[0]) else ()


def _makefile(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return `make test` or `make check` when a Makefile defines it, or None."""
    for mk in ("Makefile", "makefile", "GNUmakefile"):
        txt = _path_safety.read_workspace_text(repo_root, mk, hide)
        for target in ("test", "check"):
            if _has_make_target(txt, target):
                return (("make", target), f"Makefile:{target}")
    return None


def _python_manifest(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return a pytest run when a Python manifest exists, or None."""
    manifests = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "setup.py")
    if any(_present(repo_root, f, hide) for f in manifests):
        return ((_python(repo_root, hide), "-m", "pytest", "-q"), "pyproject")
    return None


def _cargo(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return `cargo test` when Cargo.toml exists, or None."""
    found = _present(repo_root, "Cargo.toml", hide)
    return (("cargo", "test", "--quiet"), "Cargo.toml") if found else None


def _go(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return `go test` when go.mod exists, or None."""
    return (("go", "test", "./..."), "go.mod") if _present(repo_root, "go.mod", hide) else None


def _loose_python_tests(repo_root: pathlib.Path, hide: tuple[str, ...]) -> Signal | None:
    """Return a pytest run when loose `test_*.py` files exist, or None."""
    tests = (*repo_root.glob("test_*.py"), *(repo_root / "tests").glob("test_*.py"))
    if any(_present(repo_root, t.relative_to(repo_root).as_posix(), hide) for t in tests):
        return ((_python(repo_root, hide), "-m", "pytest", "-q"), "test_*.py")
    return None


def repo_signals(repo_root: pathlib.Path, hide: tuple[str, ...] = ()) -> list[Signal]:
    """Return every repo signal that matches, the preferred one first.

    Args:
        repo_root: The directory whose files are probed.
        hide: `[sandbox].hide_paths`; a hidden file reads as absent.

    Returns:
        The matches as (command, source) pairs.
    """
    # Loose Python tests last, so a Rust or Go repo's `tests/` dir never reads as pytest.
    probes = (
        _verify_sh,
        _package_json,
        _makefile,
        _python_manifest,
        _cargo,
        _go,
        _loose_python_tests,
    )
    return [found for probe in probes if (found := probe(repo_root, hide))]


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


def gather_repo_manifests(
    repo_root: pathlib.Path, agents_md: str, hide: tuple[str, ...] = (), *, cap: int = 4000
) -> str:
    """Return the clipped manifest files and AGENTS.md as the LLM call's context."""
    parts: list[str] = []
    try:
        top = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in repo_root.iterdir()
            if not _hidden(repo_root, p.name, hide)
        )
    except OSError:
        top = []
    parts.append("<top-level>\n" + " ".join(top[:80]) + "\n</top-level>")
    for name in _MANIFEST_FILES:
        text = _path_safety.read_workspace_text(repo_root, name, hide)
        if text:
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
    hide: tuple[str, ...] = (),
) -> InferredVerify:
    """Infer a verify command: AGENTS.md, then the repo's files, then the LLM.

    A candidate whose first word does not resolve in the jail is passed over for the next; a
    shell builtin heading a shell line is the jail's own `sh`.
    When one ecosystem's manifest gives the gate, the other ecosystems' manifests are recorded
    as passed over too.

    Args:
        repo_root: The repository.
        agents_md: The AGENTS.md text.
        llm_call: Takes the gathered context and returns the model's text; None skips
            the LLM tier, as does a repo with no manifest and no AGENTS.md prose.
        hide: `[sandbox].hide_paths`; a hidden file reads as absent.

    Returns:
        The command, its source and what was passed over; an empty command when unknown.
    """
    passed: list[tuple[str, str]] = []

    def runnable(argv: tuple[str, ...], named_by: str) -> bool:
        head = command_head(argv)
        if tool_paths.jail_resolves(head) or (argv[:2] == ("sh", "-c") and head in _SH_BUILTINS):
            return True
        why = f"names `{shlex.join(argv)}`, and the sandbox has no `{head}`"
        passed.append((named_by, why))
        return False

    argv = verify_from_agents_md(agents_md)
    if argv is not None and runnable(argv, "AGENTS.md"):
        return InferredVerify(argv, "agents_md", tuple(passed))
    signals = repo_signals(repo_root, hide)
    for i, (argv, source) in enumerate(signals):
        if runnable(argv, source):
            if source in _ECOSYSTEM_SOURCES:
                # One ecosystem's runner is the gate; the other ecosystems here are named.
                passed += [
                    (other, f"names `{shlex.join(cmd)}`, which this gate does not run")
                    for cmd, other in signals[i + 1 :]
                    if other in _ECOSYSTEM_SOURCES
                ]
            return InferredVerify(argv, source, tuple(passed))
    readable = bool(agents_md.strip()) or any(
        _present(repo_root, name, hide) for name in _MANIFEST_FILES
    )
    if llm_call is not None and readable:
        context = gather_repo_manifests(repo_root, agents_md, hide)
        try:
            raw = llm_call(context)
        except Exception:  # never fails the run
            raw = ""
        argv = parse_llm_verify(raw)
        if argv is not None and runnable(argv, "the reviewer model"):
            return InferredVerify(argv, "llm", tuple(passed))
    return InferredVerify((), "none", tuple(passed))
