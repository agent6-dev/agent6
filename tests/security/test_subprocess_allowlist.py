# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The CLAUDE.md subprocess audit (`rg 'subprocess\\.(run|Popen)' src/agent6/`)
as a test: every module that spawns a child process directly is on the reviewed
allow-list below (argv fixed or operator-chosen, never LLM output -- that goes
through run_in_jail). A new name here is a security review with a
`Security review note:` commit paragraph, not a test to update in passing."""

from __future__ import annotations

import re
from pathlib import Path

import agent6

_PATTERN = re.compile(r"subprocess\.(run|Popen)")

# Reviewed direct-subprocess modules; the rationale for each is recorded in the
# security invariants section of CLAUDE.md/AGENTS.md and docs/security.md.
# completions_cmd.py matches inside a string literal only (the generated xonsh
# completer, which runs in the operator's shell, not in agent6).
ALLOWED = {
    "app/finalize.py",
    "app/run.py",
    "git_ops.py",
    "graph/client.py",
    "providers/token_command.py",
    "sandbox/detect.py",
    "sandbox/host_spawn.py",
    "sandbox/jail.py",
    "tools/lsp.py",
    "tools/mcp_client.py",
    "ui/bridge/notify.py",
    "ui/bridge/spawn.py",
    "ui/cli/_ask.py",
    "ui/cli/_live.py",
    "ui/cli/_steer.py",
    "ui/cli/completions_cmd.py",
    "ui/cli/history_cmds.py",
    "ui/cli/machine_cmds.py",
    "ui/cli/plan_watch.py",
    "ui/cli/resume.py",
    "ui/cli/review_cmds.py",
    "ui/cli/runs_cmds.py",
    "ui/cli/scriptcheck.py",
    "ui/cli/skills_cmds.py",
    "ui/cli/system_cmds.py",
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
