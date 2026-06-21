# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan`/`watch` and run-id resolution helpers."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from agent6.cli._common import _runs_dir
from agent6.run_id import RunIdError, resolve_run_id


def _event_epoch(value: object) -> float | None:
    """Parse an event ``ts`` to epoch seconds, or None if unparseable.

    EventSink writes ``ts`` as an ISO-8601 string (``datetime.isoformat``),
    so the elapsed-time anchor must parse that, not only bare numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _resolve_plan_run_id(run_id: str) -> str | None:
    """Resolve a (possibly prefix) run-id under the per-repo run-state dir.

    Prints an error and returns None on failure. Used by ``run --from-plan``,
    ``plan --show``, and ``plan --edit``.
    """
    runs_dir = _runs_dir(Path.cwd())
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None
    plan = runs_dir / resolved / "plan.md"
    if not plan.is_file():
        print(
            f"ERROR: {resolved} has no plan.md (was it created with `agent6 plan`?)",
            file=sys.stderr,
        )
        return None
    return resolved


def _cmd_plan_show(run_id: str) -> int:
    """Print a planning run's plan.md to stdout."""
    resolved = _resolve_plan_run_id(run_id)
    if resolved is None:
        return 2
    plan = _runs_dir(Path.cwd()) / resolved / "plan.md"
    sys.stdout.write(plan.read_text(encoding="utf-8"))
    return 0


def _cmd_plan_edit(run_id: str) -> int:
    """Open a planning run's plan.md in $EDITOR (default: vi).

    Operator-controlled argv (the editor name + the resolved plan path),
    not LLM-controlled, so direct subprocess.run is allowed.
    """
    resolved = _resolve_plan_run_id(run_id)
    if resolved is None:
        return 2
    plan = _runs_dir(Path.cwd()) / resolved / "plan.md"
    editor = os.environ.get("EDITOR", "vi")
    try:
        result = subprocess.run([editor, str(plan)], check=False)
    except OSError as exc:
        print(f"ERROR: failed to spawn editor {editor!r}: {exc}", file=sys.stderr)
        return 1
    return result.returncode


def _most_recent_run_id(runs_dir: Path) -> str | None:
    """Return the directory name (= run id) of the most recently mtime'd run.

    Used by `agent6 watch` (no arg), `agent6 run --continue`, and the
    history-graph subcommand. Returns None when there are no runs yet (the
    per-repo run-state dir is missing) or when the directory exists but is empty.
    """
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].name


def _most_recent_plan_run_id(runs_dir: Path) -> str | None:
    """Most recently mtime'd run dir that holds a ``plan.md`` (a plan run).

    Used by bare `agent6 run` (no task) to offer the latest plan for execution.
    """
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and (p / "plan.md").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else None


def _cmd_watch(run_id: str, *, plain: bool = False, since: int = 0) -> int:  # noqa: PLR0911
    """Read-only live view of a run directory.

    Default is the textual TUI viewer. ``--plain`` switches to a no-deps
    line tail of ``logs.jsonl``; useful in headless terminals
    or when ``textual`` isn't installed.
    """
    runs_dir = _runs_dir(Path.cwd())
    if run_id:
        try:
            resolved = resolve_run_id(runs_dir, run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        target = runs_dir / resolved
    else:
        if not runs_dir.is_dir():
            print(f"ERROR: no runs directory at {runs_dir}", file=sys.stderr)
            return 2
        candidates = sorted(
            (p for p in runs_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"ERROR: no runs found under {runs_dir}", file=sys.stderr)
            return 2
        target = candidates[0]
        print(f"[agent6] watching most recent run: {target.name}", file=sys.stderr)
    if not target.is_dir():
        print(f"ERROR: no such run dir: {target}", file=sys.stderr)
        return 2
    if plain:
        return _cmd_watch_plain(target, since=since)
    try:
        from agent6.ui.app import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "HINT: pass --plain for a no-deps text tail of logs.jsonl.",
            file=sys.stderr,
        )
        return 3
    return run_tui(target)


def _cmd_tui() -> int:
    """The TUI hub (`agent6 tui`): browse runs and start new work. Loops between
    the home screen and the dashboard, opening a run watches it, then returns
    here on close."""
    try:
        from agent6.ui.app import run_tui  # noqa: PLC0415 - lazy: textual is optional
        from agent6.ui.home import run_home  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("HINT: the TUI needs 'textual' (part of the base install).", file=sys.stderr)
        return 3
    cwd = Path.cwd()
    agent6_dir = _runs_dir(cwd).parent
    while True:
        run_dir = run_home(agent6_dir, cwd)
        if run_dir is None:
            return 0
        run_tui(run_dir)


def _format_plain_event(line: str, *, run_start_ts: float | None) -> str:
    """Pretty-print one logs.jsonl line as `<elapsed> <type> key=val ...`.

    Falls back to the raw line on parse error so a corrupt event doesn't
    abort the tail. ``run_start_ts`` is the wall-clock timestamp of the
    earliest event seen so far; used to render relative elapsed seconds.
    """
    raw = line.rstrip("\n")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(obj, dict):
        return raw
    ts = _event_epoch(obj.get("ts"))
    event = obj.get("event") or obj.get("type") or "?"
    if ts is not None and run_start_ts is not None:
        elapsed = max(0.0, ts - run_start_ts)
        ts_str = f"+{elapsed:7.1f}s"
    else:
        ts_str = "        "
    skip = {"ts", "event", "type", "run_id"}
    pairs: list[str] = []
    for k, v in obj.items():
        if k in skip:
            continue
        if isinstance(v, str):
            shown = v if len(v) <= 80 else v[:77] + "..."
            pairs.append(f"{k}={shown!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            pairs.append(f"{k}={v}")
        else:
            blob = json.dumps(v, default=str)
            shown = blob if len(blob) <= 80 else blob[:77] + "..."
            pairs.append(f"{k}={shown}")
    return f"{ts_str} {event:30s} {' '.join(pairs)}"


def _cmd_watch_plain(target: Path, *, since: int) -> int:  # noqa: PLR0912, PLR0915
    """Tail ``logs.jsonl`` line-by-line with no extra deps.

    Polls the file with 0.25s sleeps; rotates when the inode changes.
    Pretty-prints each event with the type and key fields. Returns 0 on
    EOF (run dir gone) or KeyboardInterrupt.
    """
    events_path = target / "logs.jsonl"
    if not events_path.is_file():
        print(f"ERROR: no logs.jsonl in {target}", file=sys.stderr)
        return 2

    # Read the first event for the elapsed-time anchor.
    run_start_ts: float | None = None
    try:
        with events_path.open(encoding="utf-8") as fh:
            first = fh.readline()
        if first:
            obj0 = json.loads(first)
            if isinstance(obj0, dict):
                run_start_ts = _event_epoch(obj0.get("ts"))
    except (OSError, json.JSONDecodeError):
        run_start_ts = None

    print(
        f"[agent6] tailing {events_path} (--plain). Ctrl-C to exit.",
        file=sys.stderr,
    )

    try:
        fh = events_path.open(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot open {events_path}: {exc}", file=sys.stderr)
        return 2

    try:
        if since > 0:
            # Replay the last `since` lines before following.
            try:
                lines = fh.readlines()
            except OSError as exc:
                print(f"ERROR: read failed: {exc}", file=sys.stderr)
                return 2
            for line in lines[-since:]:
                print(_format_plain_event(line, run_start_ts=run_start_ts))
        else:
            # Seek to end; only show new events going forward.
            fh.seek(0, 2)
        try:
            current_ino = events_path.stat().st_ino
        except OSError:
            current_ino = -1
        while True:
            line = fh.readline()
            if line:
                print(_format_plain_event(line, run_start_ts=run_start_ts), flush=True)
                continue
            # No new data: check for rotation and sleep briefly.
            try:
                new_ino = events_path.stat().st_ino
            except OSError:
                time.sleep(0.5)
                continue
            if new_ino != current_ino:
                with contextlib.suppress(OSError):
                    fh.close()
                try:
                    fh = events_path.open(encoding="utf-8")
                except OSError:
                    time.sleep(0.5)
                    continue
                current_ino = new_ino
                continue
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[agent6] watch --plain: stopped.", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            fh.close()
