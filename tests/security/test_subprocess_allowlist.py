# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The subprocess audit as a test: every module that spawns a child directly is on the allow-list.

Argv there is fixed or operator-chosen, except notify-send's documented data argument.
A new name here is a security review with a `Security review note:` commit paragraph, not
a test to update in passing. Broader than the AGENTS.md `rg` one-liner on purpose:
subprocess.call / check_call / check_output, a bare `from subprocess import Popen`, an
aliased `import subprocess as sp`, and the os.system / os.exec* / os.posix_spawn family
all spawn children too, and a regex for run|Popen alone would wave them through.
"""

from __future__ import annotations

import ast
import pathlib
import re
from unittest import mock

import pytest

import agent6

# Any subprocess-module call (dotted or aliased), plus the os-level spawn
# family. Kept as source text (not AST) so a match inside a generated-code
# string literal is still surfaced for review rather than silently skipped.
_PATTERN = re.compile(
    r"subprocess\.(run|Popen|call|check_call|check_output|getoutput|getstatusoutput)"
    r"|from subprocess import"
    r"|import subprocess as"
    r"|os\.(system|popen|posix_spawn|posix_spawnp"
    r"|execv|execve|execvp|execvpe|execl|execle|execlp|spawn\w+)"
    r"|pty\.spawn"
    r"|create_subprocess_(exec|shell)"
)

# Reviewed direct-subprocess modules; the rationale for each is recorded in the
# security invariants section of CLAUDE.md/AGENTS.md and docs/security.md.
# completions_cmd.py matches inside a string literal only (the generated xonsh
# completer, which runs in the operator's shell, not in agent6).
ALLOWED = {
    "app/notify_hook.py",
    "app/machine/scriptcheck.py",
    "app/machine/agent.py",
    "providers/claude_code_session.py",
    "providers/claude_code.py",
    "providers/token_command.py",
    "sandbox/container.py",  # discovery and fixed bootstrap; model argv uses JailSession
    "sandbox/detect.py",
    "sandbox/jail.py",
    "sandbox/jail_protocol.py",  # Popen annotations only, pinned below
    "sandbox/jail_session.py",
    "sessions/process.py",
    "ui/btw.py",
    "ui/cli/ask.py",
    "ui/cli/live.py",
    "ui/cli/connect_oauth.py",
    "ui/cli/completions_cmd.py",
    "ui/cli/history_cmds.py",
    "ui/cli/plan_watch.py",
    "vcs/git_command.py",
    "vcs/git_ops.py",
    "vcs/git_policy.py",
    "ui/cli/sessions_cmds.py",
    "ui/cli/skills_cmds.py",
    "ui/cli/system_cmds.py",
    "ui/notify.py",
    "ui/spawn.py",
    "ui/tui/clipboard.py",
    "ui/tui/conversation.py",
}


def test_protocol_framing_never_spawns_a_process() -> None:
    path = pathlib.Path(agent6.__file__).resolve().parent / "sandbox/jail_protocol.py"
    tree = ast.parse(path.read_text())
    imports: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                origin = alias.name if alias.asname else name
                imports.setdefault(name, set()).add(origin)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports.setdefault(alias.asname or alias.name, set()).add(
                    f"{node.module}.{alias.name}"
                )
    annotations = {
        part
        for node in ast.walk(tree)
        for field, value in ast.iter_fields(node)
        if field in {"annotation", "returns"} and isinstance(value, ast.AST)
        for part in ast.walk(value)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            reference = node.func
        elif isinstance(node, (ast.Attribute, ast.Name)) and node not in annotations:
            reference = node
        else:
            continue
        while isinstance(reference, ast.Subscript):
            reference = reference.value
        name, dot, rest = ast.unparse(reference).partition(".")
        for origin in imports.get(name, {name}):
            qualified = origin + dot + rest
            assert not _PATTERN.search(qualified), (
                f"protocol framing references a process-spawning API: {qualified}"
            )


def test_direct_subprocess_stays_on_the_allowlist() -> None:
    src = pathlib.Path(agent6.__file__).resolve().parent
    matches = {
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if _PATTERN.search(p.read_text(encoding="utf-8"))
    }
    unexpected = matches - ALLOWED
    stale = ALLOWED - matches
    assert not unexpected, f"unreviewed direct subprocess use: {sorted(unexpected)}"
    assert not stale, f"allow-list entries with no match left (prune them): {sorted(stale)}"


# Every stdlib call that RUNS something. The allow-list above is only as good as
# this list: a form missing here is a new execution site the suite waves through.
_EXECUTING_FORMS = (
    "subprocess.run(argv)",
    "subprocess.Popen(argv)",
    "subprocess.call(argv)",
    "subprocess.check_call(argv)",
    "subprocess.check_output(argv)",
    "subprocess.getoutput(cmd)",
    "subprocess.getstatusoutput(cmd)",
    "from subprocess import run",
    "import subprocess as sp",
    "asyncio.create_subprocess_exec(*argv)",
    "asyncio.create_subprocess_shell(cmd)",
    "os.system(cmd)",
    "os.popen(cmd)",
    "os.posix_spawn(exe, argv, env)",
    "os.execv(exe, argv)",
    "os.execvpe(exe, argv, env)",
    "os.spawnv(mode, exe, argv)",
    "pty.spawn(argv)",
)


@pytest.mark.parametrize("form", _EXECUTING_FORMS)
def test_the_scan_catches_every_way_to_run_something(form: str) -> None:
    """The pattern catches `getoutput`, `getstatusoutput`, `os.popen` and `pty.spawn`.

    Each runs a command through a shell; absent from the pattern, adding one would keep this
    file green, the one test standing between a new execution site and the release.
    """
    assert _PATTERN.search(form), f"a new execution site written as {form!r} would pass unnoticed"


@pytest.mark.parametrize(
    "form",
    (
        *(form for form in _EXECUTING_FORMS if not form.startswith(("from ", "import "))),
        "callbacks = [os.system]; callbacks[0](cmd)",
        "def launch(factory=subprocess.Popen):\n    return factory(argv)",
        "def transport(proc: subprocess.Popen(argv)):\n    pass",
        "from asyncio import create_subprocess_exec as spawn\nspawn(*argv)",
        "from os import posix_spawn as spawn\nspawn(exe, argv, env)",
        "from os import system\nsystem(cmd)",
        "from pty import spawn\nspawn(argv)",
        "from subprocess import Popen\nPopen(argv)",
        "from subprocess import Popen as spawn\nlaunch = spawn\nlaunch(argv)",
        "from subprocess import Popen as spawn\nspawn(argv)\nfrom os import read as spawn",
        "from subprocess import Popen as spawn\nspawn[bytes](argv)",
        "from subprocess import run as spawn\nspawn(argv)",
        "import asyncio as aio\naio.create_subprocess_shell(cmd)",
        "import os as operating_system\noperating_system.system(cmd)",
        "import pty as terminal\nterminal.spawn(argv)",
        "import subprocess as sp\nlaunch = sp.Popen[bytes]\nlaunch(argv)",
        "import subprocess as sp\nsp.Popen(argv)",
        "launch = subprocess.Popen; launch(argv)",
        "launch: object = subprocess.Popen; launch(argv)",
        "subprocess.Popen[bytes](argv)",
    ),
)
def test_protocol_audit_rejects_execution_forms(form: str) -> None:
    """The protocol's annotation exemption must not admit executable references."""
    source = "import asyncio\nimport os\nimport pty\nimport subprocess\n" + form
    with (
        mock.patch.object(pathlib.Path, "read_text", return_value=source),
        pytest.raises(AssertionError),
    ):
        test_protocol_framing_never_spawns_a_process()


@pytest.mark.parametrize(
    ("imports", "annotation"),
    [
        ("from subprocess import Popen", "Popen[bytes]"),
        ("import subprocess", "subprocess.Popen[bytes]"),
        ("import subprocess as sp", "sp.Popen[bytes]"),
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "async def transport(proc: {annotation}) -> {annotation}:\n    return proc\n",
        "def transport(proc: {annotation}) -> {annotation}:\n    return proc\n",
        "proc: {annotation}\nproc = None\n",
    ],
)
def test_protocol_audit_allows_process_annotations(
    imports: str, annotation: str, template: str
) -> None:
    source = imports + "\n" + template.format(annotation=annotation)
    with mock.patch.object(pathlib.Path, "read_text", return_value=source):
        test_protocol_framing_never_spawns_a_process()
