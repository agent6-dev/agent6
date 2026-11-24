"""Apply a treatment in a fresh interpreter without importing from the trial checkout."""

from __future__ import annotations

import contextlib
import pathlib
import sys
from unittest import mock

from bench import compaction_arms

from agent6.config import model as config_model
from agent6.harness.compaction import retention
from agent6.sandbox import detect
from agent6.sessions import ipc
from agent6.ui.cli import entry, exec_cmds

SID = "compaction-bench"


def command(function: str, *args: str) -> list[str]:
    """Start trusted bench code with the checkout absent from the import path."""
    source = pathlib.Path(__file__).resolve().parents[1]
    code = (
        f"import sys; sys.path.insert(0, {str(source)!r}); from bench import compaction_worker; "
        f"sys.exit(compaction_worker.{function}(*sys.argv[1:]))"
    )
    return [sys.executable, "-I", "-c", code, *args]


def full_jail() -> None:
    """Refuse live inference unless the worker's default isolation resolves to full."""
    level = detect.resolve_isolation("auto", detect.probed_environment())
    if level != "full":
        raise SystemExit(f"live benchmark requires full isolation; default resolved to {level}")


def grade(repo: str) -> int:
    cfg = config_model.Config.model_validate_json(sys.stdin.buffer.read())
    if cfg.sandbox.run_commands == "deny":
        raise SystemExit("sandbox.run_commands denies the small scorer")
    scorer = pathlib.Path(__file__).with_name("six_modules") / "grade.py"
    return exec_cmds.exec_new(cfg, pathlib.Path(repo), ("python3", "-I", "-c", scorer.read_text()))


def resume(mode: str, config: str, model: str, steer: str, route: str) -> int:
    if route != "scripted":
        full_jail()
    request = iter([""])
    with contextlib.ExitStack() as stack:
        if mode in ("keep", "pointers"):
            stack.enter_context(
                mock.patch.object(ipc, "read_compact_request", lambda _: next(request, None))
            )
        if mode == "pointers":
            stack.enter_context(
                mock.patch.object(retention, "select", compaction_arms.pointer_tail)
            )
        return entry.cli_main(
            ["resume", "--config", config, "--model", model, "--steer", steer, SID]
        )
