# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""File-based IPC between the workflow process and a front-end.

The workflow process and a front-end (the Textual TUI or the `agent6 web`
server) run as separate OS processes; the front-end just tails JSONL and
answers prompts by writing files. When an approval is needed:

1. The workflow process writes an `approval.prompt` event to logs.jsonl
   and then polls `<session_dir>/approvals/<id>.answer` for a result.
2. If a `<session_dir>/frontends/` claim points at a live process, the
   workflow process waits for the front-end to write the answer file.
   Otherwise it falls back to a plain stdin prompt.
3. The front-end (when present) presents a modal / control, then writes
   `<session_dir>/approvals/<id>.answer` containing the operator's literal
   choice: `yes`, `no`, `session` or `session-deny`. What a choice GRANTS is
   the asking side's to decide (see `Approver`), not the front-end's -- the
   front-end reports the click.

We use the filesystem rather than a socket because:
- the JSONL log is already the cross-process contract,
- the front-end may crash without taking the workflow down with it,
- any front-end can mirror it (the TUI, the web server, a VS Code extension).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent6.events import EventSink
from agent6.portable import atomic_write

APPROVAL_DIR_NAME = "approvals"
QUESTION_DIR_NAME = "questions"
FRONTENDS_DIR = "frontends"
WORKER_PID_FILE = "worker.pid"  # the run's worker process, for `agent6 sessions show` liveness
STEER_ANSWER_FILE = "steer.answer"

# How long the answer polls keep waiting after the front-end liveness gate goes
# dark before falling back headless (deny / ""). A transient drop (a phone
# locking its browser, a page reload, a web server restart) re-registers within
# seconds; without the grace, one 0.2s poll landing in that gap silently denied
# a pending approval. 30s outlasts a reload while a truly-gone front-end still
# fails over well before the answer timeout.
FRONTEND_DEAD_GRACE_S = 30.0


def approvals_dir(session_dir: Path) -> Path:
    p = session_dir / APPROVAL_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _answer_path(directory: Path, answer_id: str) -> Path:
    """Resolve ``<directory>/<answer_id>.answer``, refusing an id that escapes the
    directory (a path separator, ``..``, or an absolute path). A front-end always
    answers an id from a prompt it rendered, but that id crosses a trust boundary
    in the web server, so containment stays a hard check on the write primitive."""
    target = directory / f"{answer_id}.answer"
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"unsafe answer id: {answer_id!r}")
    return target


def clear_pending_answers(session_dir: Path) -> None:
    """Drop stale bridge state at run/resume START: leftover `*.answer` files
    from a prior session (the id counters reset, so an old answer would be read
    instead of prompting) and a leftover `steer.request` marker (which would
    otherwise trigger a phantom steer prompt that no live front-end answers).
    Best-effort. Stale front-end claims need no sweep here: `frontend_is_live`
    prunes dead claims on every probe, and live watchers' claims must survive
    so their modals stay wired up."""
    for sub in (APPROVAL_DIR_NAME, QUESTION_DIR_NAME):
        d = session_dir / sub
        if d.is_dir():
            for f in d.glob("*.answer"):
                with contextlib.suppress(OSError):
                    f.unlink()
    clear_steer_answer(session_dir)
    clear_steer_request(session_dir)
    # A leftover stop/compact marker from the prior session would instantly
    # re-stop (or re-compact) the fresh one.
    clear_stop_request(session_dir)
    clear_compact_request(session_dir)


def register_frontend(session_dir: Path, pid: int) -> None:
    """Register *pid* as a live answering front-end: one claim file per
    front-end (``frontends/<pid>``), so any number can watch concurrently
    (web + TUI + attach, or several of one kind) and none can deregister
    another. The name is the claim; the file is empty."""
    d = session_dir / FRONTENDS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / str(pid)).touch()


def unregister_frontend(session_dir: Path, pid: int) -> None:
    """Drop *pid*'s own claim; other front-ends' claims are untouched."""
    with contextlib.suppress(OSError):
        (session_dir / FRONTENDS_DIR / str(pid)).unlink()


def pid_alive(pid: int) -> bool:
    """True iff a live process WE OWN has *pid* (signal 0 probes without killing).

    PermissionError reads as DEAD: agent6's workers and front-ends are always
    spawned by the same user that later probes them, so a foreign-owned pid
    can only mean the original process died and the kernel reused the number
    for another user's process -- a run rendered "running" forever off such a
    pid hung the /parallel lane await permanently."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# /proc exists on Linux; on macOS `ps` answers the same question instead.
_HAS_PROC = Path("/proc").is_dir()


def _ps_start_time(pid: int) -> str:
    """Start-time identity via ``ps -o lstart=`` ("" for a dead pid or a host
    without ps). Fixed argv over a pid agent6 itself recorded, never LLM
    output; see the subprocess allowlist in docs/security.md."""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode(errors="replace").strip()


def _proc_start_time(pid: int) -> str:
    """Start-time identity for *pid*, or "" when it cannot be read (the
    process just exited): field 22 of /proc/<pid>/stat on Linux, ``ps`` where
    /proc is absent (macOS -- whose small pid_max recycles pids fast, so the
    plain kill-0 probe misread reuse as liveness there too). The comm field
    may contain spaces/parens, so split after the LAST ')'."""
    if not _HAS_PROC:
        return _ps_start_time(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return ""
    rest = stat.rpartition(")")[2].split()
    return rest[19] if len(rest) > 19 else ""


def write_worker_pid(session_dir: Path, pid: int) -> None:
    """Record the session's worker pid so `agent6 sessions show` can probe liveness even
    while the worker is blocked in a long provider call (no events emitted).
    The start-time identity rides along after the pid (/proc ticks on Linux,
    `ps` lstart text elsewhere) so a recycled pid -- same number, different
    process, after a SIGKILL'd worker left the file behind -- cannot make a
    dead run read running forever, which blocked resume and hung the
    /parallel lane await."""
    # Atomic like every sibling publish: a plain write truncates first, so a
    # reader in that window sees a PREFIX of the pid with the identity stripped
    # -- and a prefix naming a live process you own reads alive with nothing
    # left to refute it, the exact recycled-pid lie this record exists to kill.
    record = f"{pid} {_proc_start_time(pid)}".rstrip()
    atomic_write(session_dir / WORKER_PID_FILE, record)


def emit_session_start(
    events: EventSink, session_dir: Path, event_type: str, /, **fields: Any
) -> None:
    """Emit a start-family event (``session.start`` / ``loop.resume.start``)
    with the worker pid already on disk: the status fold reads a started
    session with no pid file as one whose worker exited."""
    write_worker_pid(session_dir, os.getpid())
    events.emit(event_type, **fields)


def clear_worker_pid(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / WORKER_PID_FILE).unlink()


def _read_pid_record(session_dir: Path) -> tuple[int, str] | None:
    """The recorded ``(pid, start_time)``; start_time is "" when none was
    recorded. Split once only: the `ps` lstart identity contains spaces."""
    try:
        tokens = (session_dir / WORKER_PID_FILE).read_text(encoding="utf-8").split(maxsplit=1)
        return int(tokens[0]), tokens[1].strip() if len(tokens) > 1 else ""
    except (OSError, ValueError, IndexError):
        return None


def read_worker_pid(session_dir: Path) -> int | None:
    rec = _read_pid_record(session_dir)
    return None if rec is None else rec[0]


def worker_is_alive(session_dir: Path) -> bool:
    """True iff worker.pid points at a live process that IS the recorded worker:
    the pid is alive AND, when a start time was recorded, today's start time
    matches. A recycled pid fails the match and reads dead."""
    rec = _read_pid_record(session_dir)
    if rec is None:
        return False
    pid, recorded_start = rec
    # `os.kill(0, 0)` probes the process GROUP and `os.kill(-1, 0)` every
    # process, so both answer alive: 0 and -1 are not pids, as frontend_is_live
    # already knows.
    if pid <= 0 or not pid_alive(pid):
        return False
    if not recorded_start:
        return True
    return _proc_start_time(pid) == recorded_start


def frontend_is_live(session_dir: Path) -> bool:
    """True when ANY registered front-end is a live process we own. Prunes
    dead claims (hard-killed front-ends) in passing so a stale claim can
    never block the answer poll and the dir stays tidy."""
    try:
        entries = list((session_dir / FRONTENDS_DIR).iterdir())
    except OSError:
        return False
    live = False
    for f in entries:
        try:
            pid = int(f.name)
        except ValueError:
            pid = -1
        if pid > 0 and pid_alive(pid):
            live = True
        else:
            with contextlib.suppress(OSError):
                f.unlink()
    return live


def _write_answer_atomic(target: Path, text: str) -> None:
    """Write an answer file via a UNIQUE temp + fsync + atomic replace.

    The reader polls on existence every 0.2s, so a plain write_text would
    expose an empty/partial file it consumes as deny / "". The temp must be
    unique per call (portable.atomic_write's mkstemp): two concurrently-live
    front-ends (attach + web on one run, or two web threads) answering the
    same prompt would race on a shared fixed sibling .tmp, the loser hitting
    FileNotFoundError after the winner's rename -- a 500 on an answer that
    actually landed."""
    atomic_write(target, text)


def _consume_answer(target: Path) -> str | None:
    """Read + delete *target* (consume, so it is never re-read on a later
    prompt/resume), or None when absent."""
    try:
        txt = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    with contextlib.suppress(FileNotFoundError):
        target.unlink()
    return txt


def _await_answer(
    target: Path, live: Path, *, timeout_s: float, poll_s: float, dead_grace_s: float
) -> str | None:
    """Poll for *target*, consume it, and return its text.

    Returns None when the front-end registered on *live* stays dead for
    *dead_grace_s* consecutive seconds (see FRONTEND_DEAD_GRACE_S) or when
    *timeout_s* elapses. A file that vanishes between polls is not-yet-answered,
    never an error. A final consume runs before either None verdict, so an answer landing
    between the round's read and the verdict is honoured rather than denied
    with its file left on disk."""
    deadline = time.monotonic() + timeout_s
    dead_since: float | None = None
    while time.monotonic() < deadline:
        if (txt := _consume_answer(target)) is not None:
            return txt
        if frontend_is_live(live):
            dead_since = None
        else:
            now = time.monotonic()
            if dead_since is None:
                dead_since = now
            if now - dead_since >= dead_grace_s:
                break
        time.sleep(poll_s)
    return _consume_answer(target)


def write_answer(session_dir: Path, prompt_id: str, answer: str) -> None:
    """Called by a front-end (TUI or web) with the operator's literal choice:
    "yes", "no", "session" or "session-deny"."""
    target = _answer_path(approvals_dir(session_dir), prompt_id)
    _write_answer_atomic(target, answer)


def clear_answer(session_dir: Path, prompt_id: str) -> None:
    """Drop any pre-existing answer for *prompt_id* so an answer written BEFORE
    the prompt was emitted is never consumed. Prompt ids are deterministic
    sequential counters (approval-1, ...), so a front-end (or a hostile POST)
    could pre-write approvals/approval-1.answer and the run would silently
    honor it the moment it reached that approval, auto-approving a command the
    operator never saw. The run process clears the slot immediately before
    emitting the prompt (it alone knows the exact emit moment); a legitimate
    answer is only ever written after the front-end renders the prompt, so
    none is lost. Mirrors clear_steer_answer for the steer bridge."""
    with contextlib.suppress(OSError):
        _answer_path(approvals_dir(session_dir), prompt_id).unlink(missing_ok=True)


def clear_question_answers(session_dir: Path, question_id: str) -> None:
    """The ask_user analogue of :func:`clear_answer`: drop a pre-written answer
    for *question_id* before its prompt is emitted."""
    with contextlib.suppress(OSError):
        _answer_path(questions_dir(session_dir), question_id).unlink(missing_ok=True)


# "Allow (or deny) for the rest of the session": one marker file per SCOPE,
# checked before every prompt in that scope. A scope is what the operator was
# answering about, so a standing answer grants what the prompt said and no more.
# The whole vocabulary: the three command tools share one, and each MCP server
# has its own (server names are [A-Za-z0-9_-]+, so a scope is always a safe file
# suffix and two servers never collide).
#
# Markers are NOT `*.answer`s, so clear_pending_answers leaves them in place:
# the choice persists across this run's resumes (a detached run then keeps going
# without a front-end to prompt). They live in the run's approvals dir, so other
# runs are unaffected and a fresh run prompts again.
COMMAND_SCOPE = "command"
MCP_SCOPE_PREFIX = "mcp."
SESSION_ALLOW_FILE = "session.allow"
SESSION_DENY_FILE = "session.deny"


def set_session_allow(session_dir: Path, scope: str) -> None:
    """Record the operator's 'allow all of *scope* for the session' choice."""
    d = approvals_dir(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    _write_answer_atomic(d / f"{SESSION_ALLOW_FILE}.{scope}", "1")


def session_allow_set(session_dir: Path, scope: str) -> bool:
    return (approvals_dir(session_dir) / f"{SESSION_ALLOW_FILE}.{scope}").exists()


def set_session_deny(session_dir: Path, scope: str) -> None:
    """Record the mirror choice: 'none of *scope* for the rest of the session'.

    A single "no" answers one call, exactly as a single "yes" approves one; only
    the session choices persist. Denying for the session WITHDRAWS the tools
    rather than refusing each call, so the model stops spending turns on a door
    that will not open.
    """
    d = approvals_dir(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    _write_answer_atomic(d / f"{SESSION_DENY_FILE}.{scope}", "1")


def session_deny_set(session_dir: Path, scope: str) -> bool:
    return (approvals_dir(session_dir) / f"{SESSION_DENY_FILE}.{scope}").exists()


def record_answer(session_dir: Path, answer: str, scope: str | None) -> bool:
    """Apply the operator's literal *answer* and return the verdict for THIS call.

    The one place an answer's meaning is decided. A session choice persists only
    when the prompt offered one: `scope=None` is a gate with no standing answer
    (`fetch`), and an "allow all" arriving on one anyway grants nothing beyond
    the call it was clicked on. Anything unrecognised is a deny, so a truncated
    or hand-written answer file cannot approve.
    """
    if scope:
        if answer == "session":
            set_session_allow(session_dir, scope)
        elif answer == "session-deny":
            set_session_deny(session_dir, scope)
    return answer in {"yes", "session"}


def effective_run_commands(configured: str, session_dir: Path) -> str:
    """What the command policy IS right now: "no" | "ask" | "yes".

    One answer from three inputs, so every consumer agrees: the configured
    knob, the operator's session choice, and the away-mode a detached run was
    left with. Only "ask" is movable -- a configured "yes" or "no" is the
    operator's standing policy and no in-run choice overrides it.

    "no" means the tools are WITHDRAWN, not refused per call: that is the same
    wiring for `run_commands = "no"`, `--no-commands`, deny-for-session and an
    away-mode of "deny", so the rules fall out consistently.
    """
    if configured != "ask":
        return configured
    if session_allow_set(session_dir, COMMAND_SCOPE):
        return "yes"
    if session_deny_set(session_dir, COMMAND_SCOPE) or away_mode(session_dir) == "deny":
        return "no"
    return "ask"


# How a DETACHED run (no terminal to prompt) handles run_command approvals and
# ask_user questions: "deny" auto-denies, "wait" blocks until a front-end
# reattaches and answers. "approve" is not stored here -- detach approve-all
# sets the command scope's allow marker. Persists like it (not an *.answer).
AWAY_MODE_FILE = "away.mode"


def set_away_mode(session_dir: Path, mode: str) -> None:
    """Record the detach 'while away' choice ("deny" | "wait")."""
    if mode not in ("deny", "wait"):
        raise ValueError(
            f"away.mode is 'deny' or 'wait', got {mode!r} (approve-all reuses session.allow)"
        )
    d = approvals_dir(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    _write_answer_atomic(d / AWAY_MODE_FILE, mode)


def away_mode(session_dir: Path) -> str:
    """ "deny", "wait", or "" (unset -- interactive/foreground default flow)."""
    try:
        return (approvals_dir(session_dir) / AWAY_MODE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def clear_away_mode(session_dir: Path) -> None:
    """Drop the detach 'while away' choice. Called when an INTERACTIVE (tty) run or
    resume starts: the operator is back at the terminal, so a stale away-mode from a
    prior detach must not keep auto-denying/waiting."""
    with contextlib.suppress(FileNotFoundError):
        (approvals_dir(session_dir) / AWAY_MODE_FILE).unlink()


def read_answer(
    session_dir: Path,
    prompt_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Called by the workflow. Returns the operator's literal choice ("yes",
    "no", "session", "session-deny"), or None on timeout or once the front-end
    has stayed dead past ``dead_grace_s`` (a shorter drop keeps waiting).

    ``live_dir`` overrides which dir the liveness gate probes for front-end claims
    (defaults to ``session_dir``). A machine agent state reads answers from its
    per-state dir but the front-end registers on the instance dir, so it passes
    the instance dir here."""
    target = approvals_dir(session_dir) / f"{prompt_id}.answer"
    txt = _await_answer(
        target,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
    return None if txt is None else txt.strip().lower()


# --- agent->user question bridge (the `ask_user` tool) -----------------------
# Same shape as approvals, but the answer is a free string (a selected option or
# typed text). The workflow emits `question.prompt`, polls for the answer file;
# the TUI shows a modal and writes it. Falls back to stdin (then a default) when
# no TUI is live, so headless runs never hang.


def questions_dir(session_dir: Path) -> Path:
    p = session_dir / QUESTION_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_question_answers(session_dir: Path, question_id: str, answers: Sequence[str]) -> None:
    """Called by a front-end when the user answers the question(s). Answers align to
    the prompt's `questions` by index and are stored as a JSON list."""
    _write_answer_atomic(
        _answer_path(questions_dir(session_dir), question_id), json.dumps(list(answers))
    )


def read_question_answers(
    session_dir: Path,
    question_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> tuple[str, ...] | None:
    """Called by the workflow. Returns the answers tuple (aligned to the prompt's
    questions), or None on timeout or once the front-end has stayed dead past
    ``dead_grace_s``. ``live_dir`` overrides the liveness-gate dir (see
    :func:`read_answer`)."""
    target = questions_dir(session_dir) / f"{question_id}.answer"
    raw = _await_answer(
        target,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return (raw,)  # a bare free-text answer (not JSON) -> single answer
    return tuple(str(x) for x in data) if isinstance(data, list) else (str(data),)


# --- mid-run steering bridge (Ctrl-C while the TUI owns the terminal) --------
# Single-slot: only one steer prompt is ever outstanding (the SIGINT handler
# sets a flag the loop drains at its next boundary). The run process triggers a
# steer by emitting `session.steer_requested`; the TUI shows a modal and writes the
# answer here; the run process reads it. The answer is a free string:
# "" = continue, "abort" = stop, anything else = a steering instruction.


def write_steer_answer(session_dir: Path, answer: str) -> None:
    """Called by a front-end when the user answers the steer prompt."""
    _write_answer_atomic(session_dir / STEER_ANSWER_FILE, answer)


def clear_steer_answer(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_ANSWER_FILE).unlink()


def steer_answer_is_abort(session_dir: Path) -> bool:
    """Non-blocking peek: True if a pending steer answer is a stop. Lets a long
    streaming model turn bail immediately instead of only at the between-step
    boundary. Does NOT consume the answer -- the boundary still handles it if the
    stream ends first."""
    try:
        answer = (session_dir / STEER_ANSWER_FILE).read_text(encoding="utf-8").strip().lower()
    except (OSError, ValueError):  # missing/unreadable, or non-UTF-8: not an abort
        return False
    # Exactly the Stop contract: every front-end's Stop writes "abort", and the
    # between-step boundary (_maybe_handle_steer) also stops only on "abort". A
    # typed steer instruction -- even the word "stop" -- is an instruction, not a
    # stop; interrupting mid-stream on it would diverge from the boundary.
    return answer == "abort"


# A steer can also be INITIATED from the TUI (the `s` key) without Ctrl-C: the
# dashboard drops this marker, the run notices it at its next safe boundary (same
# as the SIGINT flag), prompts via the modal, and clears it. Decoupled from
# signals so a watcher process can request a steer the run picks up.
STEER_REQUEST_FILE = "steer.request"


def request_steer(session_dir: Path) -> None:
    """TUI-initiated steer: drop a marker the session polls at its next boundary."""
    with contextlib.suppress(OSError):
        (session_dir / STEER_REQUEST_FILE).write_text("", encoding="utf-8")


def steer_request_pending(session_dir: Path) -> bool:
    return (session_dir / STEER_REQUEST_FILE).exists()


def clear_steer_request(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_REQUEST_FILE).unlink()


STOP_REQUEST_FILE = "stop.request"


def request_stop(session_dir: Path) -> None:
    """Front-end "stop after this step": drop a marker the session polls at each
    completed-iteration boundary and honors by ending the run cleanly there
    (the finished step's tool results and auto-commit land first). The
    immediate stop stays the steer "abort" answer, which interrupts mid-turn."""
    with contextlib.suppress(OSError):
        (session_dir / STOP_REQUEST_FILE).write_text("", encoding="utf-8")


def stop_request_pending(session_dir: Path) -> bool:
    return (session_dir / STOP_REQUEST_FILE).exists()


def clear_stop_request(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STOP_REQUEST_FILE).unlink()


COMPACT_REQUEST_FILE = "compact.request"


def request_compact(session_dir: Path, focus: str = "") -> bool:
    """Front-end-initiated manual compaction: drop a marker the session polls at its
    next safe boundary and honors by forcing a context compaction (mirrors
    steer). The marker body is the operator's optional summary *focus*
    (`/compact <focus>`); "" is a plain compact. Published atomically: the run
    polls `read_compact_request` every boundary, so a plain write exposed an
    empty/partial focus it consumed (and then cleared) as the real one.

    Returns whether the marker landed. A failed write must not raise into a TUI
    action or a web handler, but every front-end reported "compaction requested"
    unconditionally, so a read-only or full state dir looked like success and
    nothing ever compacted."""
    try:
        atomic_write(session_dir / COMPACT_REQUEST_FILE, focus)
    except OSError:
        return False
    return True


def read_compact_request(session_dir: Path) -> str | None:
    """The pending compact request's focus text, or None when no request is
    pending ("" = a plain compact with no focus)."""
    try:
        return (session_dir / COMPACT_REQUEST_FILE).read_text(encoding="utf-8")
    except OSError:
        return None


def clear_compact_request(session_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (session_dir / COMPACT_REQUEST_FILE).unlink()


def read_steer_answer(
    session_dir: Path,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Called by the workflow when the TUI is live. Returns the answer string
    (consuming the file), or None on timeout or once the front-end has stayed
    dead past ``dead_grace_s``. ``live_dir`` overrides the liveness-gate dir
    (see :func:`read_answer`)."""
    return _await_answer(
        session_dir / STEER_ANSWER_FILE,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
