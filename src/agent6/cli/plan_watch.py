# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan`/`agent6 watch` and run-id resolution helpers."""

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
from agent6.cli._console_view import ConsoleView
from agent6.frontend.approval import read_worker_pid, worker_is_alive
from agent6.run_id import RunIdError, resolve_run_id
from agent6.viewmodel import run_mtime, tail_events


def event_epoch(value: object) -> float | None:
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
    ``plan show``, and ``plan edit``.
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
    """Return the directory name (= run id) of the most recently active run.

    Used by `agent6 watch` (no arg), `agent6 run --continue`, and the
    history-graph subcommand. Returns None when there are no runs yet (the
    per-repo run-state dir is missing) or when the directory exists but is empty.
    """
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=run_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].name


def _most_recent_plan_run_id(runs_dir: Path) -> str | None:
    """Most recently active run dir that holds a ``plan.md`` (a plan run).

    Used by bare `agent6 run` (no task) to offer the latest plan for execution.
    """
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and (p / "plan.md").is_file()),
        key=run_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else None


def _cmd_watch(run_id: str, *, tui: bool = False, since: int = 0, raw: bool = False) -> int:
    """Read-only live view of a run directory.

    Default follows the run's conversation (the same render as ``agent6 run``).
    ``--raw`` is the no-deps event-line tail; ``--tui`` the full-screen dashboard.
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
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs found under {runs_dir}", file=sys.stderr)
            return 2
        target = runs_dir / latest
        print(f"[agent6] watching most recent run: {target.name}", file=sys.stderr)
    if not target.is_dir():
        print(f"ERROR: no such run dir: {target}", file=sys.stderr)
        return 2
    if not tui:
        return _cmd_watch_plain(target, since=since) if raw else _watch_transcript(target)
    try:
        from agent6.tui.app import run_tui  # noqa: PLC0415 - lazy: textual is optional
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "HINT: drop --tui for a no-deps text tail of logs.jsonl.",
            file=sys.stderr,
        )
        return 3
    return run_tui(target)


def _resolve_run_dir(runs_dir: Path, run_id: str) -> Path | None:
    """Resolve a run id (or the most-recent run when empty) to its run dir."""
    if run_id:
        try:
            return runs_dir / resolve_run_id(runs_dir, run_id)
        except RunIdError:
            return None
    if not runs_dir.is_dir():
        return None
    # Sort by logs.jsonl activity (run_mtime), not dir mtime: a viewer opening a
    # run writes frontend.pid into its dir and would otherwise float it to latest.
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=run_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _scan_run_events(events_path: Path) -> dict[str, object]:
    """Single pass over logs.jsonl. Returns start_ep, last_ep, last_type,
    iteration, end_reason, and the latest `loop.budget` token/cost totals.
    Tolerant of torn/short lines."""
    out: dict[str, object] = {
        "start_ep": None,
        "last_ep": None,
        "last_type": None,
        "iteration": None,
        "end_reason": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }
    if not events_path.is_file():
        return out
    with contextlib.suppress(OSError):
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            ep = event_epoch(e.get("ts"))
            etype = e.get("type")
            if etype == "run.start" and out["start_ep"] is None:
                out["start_ep"] = ep
            if isinstance(e.get("iteration"), int):
                out["iteration"] = e["iteration"]
            if etype == "run.end":
                out["end_reason"] = e.get("reason")
            if etype == "loop.budget":
                for k in ("input_tokens", "output_tokens", "cost_usd"):
                    if e.get(k) is not None:
                        out[k] = e[k]
            if ep is not None:
                out["last_ep"] = ep
            if isinstance(etype, str):
                out["last_type"] = etype
    return out


def _print_fork_lineage(manifest: dict[str, object]) -> None:
    """Print the fork-lineage line for a run created by `agent6 fork` (no-op
    otherwise)."""
    parent = manifest.get("parent_run_id")
    if not (isinstance(parent, str) and parent):
        return
    sha = manifest.get("forked_from_sha")
    sha_note = f" ({sha[:12]})" if isinstance(sha, str) and sha else ""
    print(f"forked from: {parent}@turn {manifest.get('forked_from_turn')}{sha_note}")


def _cmd_status(run_id: str, *, as_json: bool = False) -> int:
    """One-shot liveness + progress summary for a run, then exit (no follower).

    Answers "is this run still alive, and what is it doing?" from the run dir
    alone: the worker.pid (probed with signal 0, so liveness is known even while
    the worker is blocked in a long provider call that emits no events) plus the
    last event, current iteration, and elapsed time from logs.jsonl. For a quick
    or scripted check; `agent6 watch` is the live follower.
    """
    runs_dir = _runs_dir(Path.cwd())
    target = _resolve_run_dir(runs_dir, run_id)
    if target is None or not target.is_dir():
        print(f"ERROR: no run found ({run_id or 'latest'}) under {runs_dir}", file=sys.stderr)
        return 2

    manifest: dict[str, object] = {}
    with contextlib.suppress(OSError, ValueError):
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))

    ev = _scan_run_events(target / "logs.jsonl")
    start_ep = ev["start_ep"]
    last_ep = ev["last_ep"]
    last_type = ev["last_type"]
    iteration = ev["iteration"]
    end_reason = ev["end_reason"]

    pid = read_worker_pid(target)
    alive = worker_is_alive(target)
    now = time.time()
    last_age = (now - last_ep) if isinstance(last_ep, (int, float)) else None
    elapsed = (
        (last_ep - start_ep)
        if isinstance(last_ep, (int, float)) and isinstance(start_ep, (int, float))
        else None
    )

    if end_reason is not None:
        state = f"finished ({end_reason})"
    elif alive and last_age is not None and last_age > 120:
        state = "running (in a long step — provider call?)"
    elif alive:
        state = "running"
    elif last_ep is None:
        state = "unknown (no events yet)"
    else:
        state = "stopped (no live worker, no run.end — likely crashed or killed)"

    models = manifest.get("models")
    worker = models.get("worker") if isinstance(models, dict) else None
    model = (worker.get("model") if isinstance(worker, dict) else worker) or "?"

    if as_json:
        print(
            json.dumps(
                {
                    "run_id": target.name,
                    "mode": manifest.get("mode"),
                    "model": model,
                    "state": state,
                    "alive": alive,
                    "pid": pid,
                    "iteration": iteration,
                    "last_event": last_type,
                    "last_event_age_s": round(last_age, 1) if last_age is not None else None,
                    "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
                    "reason": end_reason,
                    "input_tokens": ev["input_tokens"],
                    "output_tokens": ev["output_tokens"],
                    "cost_usd": ev["cost_usd"],
                    "parent_run_id": manifest.get("parent_run_id"),
                    "forked_from_turn": manifest.get("forked_from_turn"),
                    "forked_from_sha": manifest.get("forked_from_sha"),
                }
            )
        )
        return 0

    pid_note = ""
    if alive:
        pid_note = f"  — worker pid {pid} alive"
    elif pid is not None and end_reason is None:
        pid_note = f"  — worker pid {pid} not running"
    print(f"run:        {target.name}  (mode={manifest.get('mode', '?')})")
    _print_fork_lineage(manifest)
    print(f"model:      {model}")
    print(f"state:      {state}{pid_note}")
    print(f"iteration:  {iteration if iteration is not None else '-'}")
    print(
        f"last event: {last_type or '-'}"
        f"{f'  ({_fmt_dur(last_age)} ago)' if last_age is not None else ''}"
    )
    print(f"elapsed:    {_fmt_dur(elapsed)}")
    if ev["input_tokens"] is not None or ev["cost_usd"] is not None:
        cost = ev["cost_usd"]
        cost_s = f"  cost ~${cost:.4f}" if isinstance(cost, (int, float)) else ""
        print(f"usage:      in={ev['input_tokens'] or 0} out={ev['output_tokens'] or 0}{cost_s}")
    return 0


def _cmd_tui() -> int:
    """The TUI hub (`agent6 tui`): browse runs and start new work. Loops between
    the home screen and the dashboard, opening a run watches it, then returns
    here on close."""
    try:
        from agent6.tui.app import QUIT_HUB_CODE, run_tui  # noqa: PLC0415 - lazy: textual optional
        from agent6.tui.home import run_home  # noqa: PLC0415
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
        # Esc in the dashboard returns here (reopen home); q quits the hub.
        if run_tui(run_dir, from_hub=True) == QUIT_HUB_CODE:
            return 0


def format_plain_event(line: str, *, run_start_ts: float | None) -> str:
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
    ts = event_epoch(obj.get("ts"))
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


def _watch_transcript(target: Path) -> int:
    """Follow a run's conversation live: fold ``logs.jsonl`` through the same
    ``ConsoleView`` as ``agent6 run``, so an attached viewer sees what the run
    prints. Renders from the start, tails until the run ends (a finished run just
    renders and exits), then returns; Ctrl-C exits early. A detach emits no
    run.end, so watching a detached run follows the background resume to its end."""
    events_path = target / "logs.jsonl"
    if not events_path.is_file():
        print(f"ERROR: no logs.jsonl in {target}", file=sys.stderr)
        return 2
    print(f"[agent6] following {target.name}. Ctrl-C to exit.", file=sys.stderr)
    view = ConsoleView(sys.stdout)
    try:
        for event in tail_events(events_path, follow=True, stop_when_finished=True):
            view.feed(event)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
    return 0


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

    # Read the first event for the elapsed-time anchor. Binary readline: a
    # torn-mid-UTF-8 first line must not crash the watch before it starts.
    run_start_ts: float | None = None
    try:
        with events_path.open("rb") as fh:
            first = fh.readline()
        if first:
            obj0 = json.loads(first.decode("utf-8"))
            if isinstance(obj0, dict):
                run_start_ts = event_epoch(obj0.get("ts"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        run_start_ts = None

    print(
        f"[agent6] tailing {events_path}. Ctrl-C to exit.",
        file=sys.stderr,
    )

    # Binary reads throughout: the writer flushes long lines in several
    # syscalls, so a read can hit EOF mid multibyte UTF-8 sequence and a
    # text-mode readline would raise UnicodeDecodeError. Complete lines are
    # decoded (errors="replace"); a partial tail stays buffered until its
    # newline arrives.
    try:
        fh = events_path.open("rb")
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
            for raw in lines[-since:]:
                line = raw.decode("utf-8", errors="replace")
                print(format_plain_event(line, run_start_ts=run_start_ts))
        else:
            # Seek to end; only show new events going forward.
            fh.seek(0, 2)
        try:
            current_ino = events_path.stat().st_ino
        except OSError:
            current_ino = -1
        pending = b""
        while True:
            chunk = fh.readline()
            if chunk:
                pending += chunk
                if not pending.endswith(b"\n"):
                    continue  # partial line at EOF; the rest arrives next read
                line = pending.decode("utf-8", errors="replace")
                pending = b""
                print(format_plain_event(line, run_start_ts=run_start_ts), flush=True)
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
                    fh = events_path.open("rb")
                except OSError:
                    time.sleep(0.5)
                    continue
                current_ino = new_ino
                pending = b""
                continue
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            fh.close()
