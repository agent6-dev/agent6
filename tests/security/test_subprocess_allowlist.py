# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The subprocess audit as a test: every module that spawns a child process
directly is on the reviewed allow-list below (argv fixed or operator-chosen,
never LLM output -- that goes through run_in_jail). A new name here is a
security review with a `Security review note:` commit paragraph, not a test to
update in passing.

Broader than the AGENTS.md `rg` one-liner on purpose: subprocess.call /
check_call / check_output, a bare `from subprocess import Popen`, an aliased
`import subprocess as sp`, and the os.system / os.exec* / os.posix_spawn family
all spawn children too, and a regex for run|Popen alone would wave them
through."""

from __future__ import annotations

import re
from pathlib import Path

import agent6

# Any subprocess-module call (dotted or aliased), plus the os-level spawn
# family. Kept as source text (not AST) so a match inside a generated-code
# string literal is still surfaced for review rather than silently skipped.
_PATTERN = re.compile(
    r"subprocess\.(run|Popen|call|check_call|check_output)"
    r"|from subprocess import"
    r"|import subprocess as"
    r"|os\.(system|posix_spawn|posix_spawnp|execv|execve|execvp|execvpe|execl|execle|execlp|spawn\w+)"
    r"|create_subprocess_(exec|shell)"
)

# Reviewed direct-subprocess modules; the rationale for each is recorded in the
# security invariants section of CLAUDE.md/AGENTS.md and docs/security.md.
# completions_cmd.py matches inside a string literal only (the generated xonsh
# completer, which runs in the operator's shell, not in agent6).
ALLOWED = {
    "app/finalize.py",
    "app/machine/_preflight.py",
    "app/machine_agent.py",
    "git_ops.py",
    "providers/token_command.py",
    "runs/ipc.py",
    "sandbox/detect.py",
    "sandbox/host_spawn.py",
    "sandbox/jail.py",
    "tools/lsp.py",
    "tools/mcp_client.py",
    "ui/cli/_ask.py",
    "ui/cli/_live.py",
    "ui/cli/_steer.py",
    "ui/cli/completions_cmd.py",
    "ui/cli/history_cmds.py",
    "ui/cli/plan_watch.py",
    "ui/cli/review_cmds.py",
    "ui/cli/runs_cmds.py",
    "app/machine/_scriptcheck.py",
    "ui/cli/skills_cmds.py",
    "ui/cli/system_cmds.py",
    "ui/notify.py",
    "ui/spawn.py",
    "ui/tui/clipboard.py",
    "ui/tui/conversation.py",
}


def test_direct_subprocess_stays_on_the_allowlist() -> None:
    src = Path(agent6.__file__).resolve().parent
    matches = {
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if _PATTERN.search(p.read_text(encoding="utf-8"))
    }
    unexpected = matches - ALLOWED
    stale = ALLOWED - matches
    assert not unexpected, f"unreviewed direct subprocess use: {sorted(unexpected)}"
    assert not stale, f"allow-list entries with no match left (prune them): {sorted(stale)}"
