"""Write a run's log under the state dir a test's `agent6 web` serves."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from agent6 import paths
from agent6.sessions import process


def write_run(
    tmp_path: pathlib.Path, name: str, rows: list[dict[str, Any]], *, live: bool = False
) -> pathlib.Path:
    """Write a run's log after its start record; a live run's worker is this process.

    Returns:
        The run's session dir.
    """
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / name
    session.mkdir(parents=True)
    start = {"type": "session.start", "mode": "run", "user_task": "Parse CSV rows"}
    (session / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [start, *rows]))
    if live:
        process.write_worker_pid(session, os.getpid())
    return session
