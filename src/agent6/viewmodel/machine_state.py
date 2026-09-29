# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a machine instance's journal and spec into the watch view every front-end renders.

The machine analogue of state.py: which states exist, where the machine is, the path
taken and how it ended. The reasoning inside an agent state is itself a run log and
folds through `SessionState`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
from collections.abc import Sequence
from typing import Any, Literal

from agent6.machine import MachineError, MachineResult, journal, load_machine, schema
from agent6.sessions import ipc, layout
from agent6.viewmodel import format
from agent6.viewmodel import state as viewmodel_state
from agent6.viewmodel import tail as viewmodel_tail

# Front-ends render notifications as ephemeral surfaces, so only the tail travels.
_NOTIFY_KEEP = 20


@dataclasses.dataclass(frozen=True, slots=True)
class MachineStateView:
    """One state in the overview: its name, kind and position.

    Attributes:
        name: The state's name.
        kind: The state's kind from the spec.
        is_current: The machine is in this state, or is about to run it.
        is_visited: The machine entered or left this state.
        mark: The mark before the name, as every surface draws it.
    """

    name: str
    kind: str
    is_current: bool
    is_visited: bool
    mark: str = ""

    def __post_init__(self) -> None:
        """Fill `mark` from the position flags when the caller left it empty."""
        if not self.mark:
            mark = format.machine_state_mark(is_current=self.is_current, is_visited=self.is_visited)
            object.__setattr__(self, "mark", mark)


@dataclasses.dataclass(frozen=True, slots=True)
class TransitionView:
    """One journaled transition: state --label--> goto.

    Attributes:
        seq: The transition's sequence number.
        state: The state left.
        label: The transition's label.
        goto: The state entered.
        detail: Bounded failure evidence (a failed tool's exit code and last output
            line, a failed agent state's stop reason); "" on success.
        line: `format_transition` of the fields, as every surface prints it.
    """

    seq: int
    state: str
    label: str
    goto: str
    detail: str = ""
    line: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class NotificationView:
    """One journaled `machine.notify` (a state's `notify` message), in order."""

    ts: str
    state: str
    message: str
    level: str


@dataclasses.dataclass(frozen=True, slots=True)
class MachineState:
    """A machine instance's folded watch view.

    Attributes:
        machine: The spec's declared name.
        version: The spec's version.
        initial: The initial state's name.
        current: Where the machine is, or is about to run.
        states: Every state in spec order, position-flagged.
        transitions: The path taken, in order.
        ended: The end record, None while the machine has not ended.
        notifications: The recent `machine.notify` events, oldest first.
    """

    machine: str
    version: int
    initial: str
    current: str
    states: tuple[MachineStateView, ...]
    transitions: tuple[TransitionView, ...]
    ended: MachineResult | None
    notifications: tuple[NotificationView, ...]

    @property
    def current_kind(self) -> str | None:
        """The current state's kind ("wait" parks the machine), None at the end."""
        return next((s.kind for s in self.states if s.is_current), None)


def _transition_view(s: journal.StepEvent) -> TransitionView:
    """Return a step event as its view, with the detail and line rendered."""
    detail = _fact_detail(s)
    line = format.format_transition(s.seq, s.state, s.label, s.goto, detail)
    return TransitionView(
        seq=s.seq, state=s.state, label=s.label, goto=s.goto, detail=detail, line=line
    )


def _fact_detail(step: journal.StepEvent) -> str:
    """Return bounded failure evidence for one transition, "" on success."""
    fact = step.fact
    if isinstance(fact, journal.ToolFact) and (fact.exit_code != 0 or fact.timed_out):
        tail = next(
            (
                ln.strip()
                for ln in reversed((fact.stderr or fact.stdout).splitlines())
                if ln.strip()
            ),
            "",
        )
        head = "timed out" if fact.timed_out else f"exit {fact.exit_code}"
        return f"{head}: {tail[:160]}" if tail else head
    if isinstance(fact, journal.AgentFact) and fact.outcome != "ok":
        return f"{fact.outcome}: {fact.reason}"[:160]
    return ""


def fold_machine(spec: schema.MachineSpec, events: Sequence[object]) -> MachineState:
    """Fold a machine journal into its watch view.

    Args:
        spec: The machine's spec.
        events: The journal's events.

    Returns:
        The state: `current` is the last transition's goto, else the initial state;
        a state is visited when any transition entered or left it.
    """
    steps = [e for e in events if isinstance(e, journal.StepEvent)]
    end = next((e for e in reversed(events) if isinstance(e, journal.MachineEnd)), None)
    current = steps[-1].goto if steps else spec.initial
    visited: set[str] = set()
    for s in steps:
        visited.update((s.state, s.goto))
    states = tuple(
        MachineStateView(
            name=name,
            kind=st.kind,
            is_current=(name == current),
            is_visited=(name in visited),
        )
        for name, st in spec.states.items()
    )
    transitions = tuple(_transition_view(s) for s in steps)
    ended = MachineResult.from_end(end) if end is not None else None
    notes = [e for e in events if isinstance(e, journal.MachineNotify)]
    notifications = tuple(
        NotificationView(ts=n.ts, state=n.state, message=n.message, level=n.level)
        for n in notes[-_NOTIFY_KEEP:]
    )
    return MachineState(
        machine=spec.machine,
        version=spec.version,
        initial=spec.initial,
        current=current,
        states=states,
        transitions=transitions,
        ended=ended,
        notifications=notifications,
    )


def machine_status_word(
    ms: MachineState, *, parked: bool, alive: bool, blocked: bool = False
) -> str:
    """Decide the liveness word, so a machine that is not working never renders busy.

    The fold is pure, so the caller probes the three dir facts.

    Args:
        ms: The folded state.
        parked: A persisted wait record is armed for the current state.
        alive: The worker pid is live.
        blocked: The newest state log holds an unanswered operator prompt.

    Returns:
        The end's own status for an ended machine; "waiting" when parked, blocked or
        live in a wait state; "running" for a live worker elsewhere; "stopped" for a
        dead worker that is neither parked nor ended.
    """
    if ms.ended is not None:
        return ms.ended.status
    if parked:
        return "waiting"
    if alive:
        if blocked:
            return "waiting"
        return "waiting" if ms.current_kind == "wait" else "running"
    return "stopped"


@dataclasses.dataclass(frozen=True, slots=True)
class AgentExecution:
    """The newest agent state's execution as the operator verbs see it.

    Attributes:
        open: The execution has begun and not ended, so a steer has a reader.
        blocked_in: The state dir (`0001-attempt`) holding an unanswered approval or
            question, else "".
    """

    open: bool = False
    blocked_in: str = ""


def execution_of(state: viewmodel_state.SessionState, log: pathlib.Path) -> AgentExecution:
    """Return the execution a folded state log describes.

    Args:
        state: The log's fold.
        log: The log's path; its parent names the state dir.

    Returns:
        The execution facts.
    """
    open_prompts = [*state.pending_approvals, *state.pending_questions]
    return AgentExecution(
        open=state.started and not state.finished,
        blocked_in=log.parent.name if any(not p.answered for p in open_prompts) else "",
    )


def newest_agent_execution(machine_dir: pathlib.Path) -> AgentExecution:
    """Return the newest state log's execution from one fold of that log.

    A poll loop holds a `NewestExecutionFold` instead.

    Args:
        machine_dir: The instance's dir.

    Returns:
        The execution facts, empty when no agent state has a log yet.
    """
    log = newest_state_log(machine_dir)
    if log is None:
        return AgentExecution()
    return execution_of(
        viewmodel_state.fold_session(viewmodel_tail.tail_events(log, follow=False)), log
    )


class NewestExecutionFold:
    """The newest state log folded across a poll loop.

    Each `refresh` folds the bytes appended since the last one into the held state,
    and starts over when the machine entered a newer agent state or the log was
    rewritten, so a tick reads the log once instead of whole.

    Attributes:
        state: The held fold.
    """

    def __init__(self) -> None:
        self._log: pathlib.Path | None = None
        self._tail: viewmodel_tail.LogTail | None = None
        self.state: viewmodel_state.SessionState = viewmodel_state.initial_state()

    def refresh(self, machine_dir: pathlib.Path) -> pathlib.Path | None:
        """Fold what the newest state log gained since the last refresh.

        Args:
            machine_dir: The instance's dir.

        Returns:
            The newest state log, None when no agent state has one yet.
        """
        log = newest_state_log(machine_dir)
        if log != self._log:
            self._log, self._tail, self.state = (
                log,
                viewmodel_tail.LogTail(log) if log else None,
                viewmodel_state.initial_state(),
            )
        if self._tail is not None:
            events = self._tail.read()
            if self._tail.rewound:
                self.state = viewmodel_state.initial_state()
            for event in events:
                self.state = viewmodel_state.apply_event(self.state, event)
        return log

    @property
    def log(self) -> pathlib.Path | None:
        """The state log the held fold describes, None before any exists."""
        return self._log

    def execution(self) -> AgentExecution:
        """Return the execution facts of the held fold."""
        return execution_of(self.state, self._log) if self._log is not None else AgentExecution()


def armed_wait(machine_dir: pathlib.Path, ms: MachineState) -> journal.PendingWait | None:
    """Return the persisted wait record when it belongs to this visit of the current state.

    The engine's own test: the record names `ms.current` and its transition count.

    Args:
        machine_dir: The instance's dir.
        ms: The folded state.

    Returns:
        The record, or None, a record a death left behind an earlier visit included.

    Raises:
        JournalError: The wait record is corrupt.
    """
    pending = journal.MachineJournal(machine_dir).read_pending_wait()
    if pending is None or pending.state != ms.current or pending.seq != len(ms.transitions):
        return None
    return pending


@dataclasses.dataclass(frozen=True, slots=True)
class InstanceProbes:
    """What an instance dir says beside its fold.

    Under `--exit-on-wait` scheduling a parked machine has no live process, so a dead
    pid reads as parked, never as crashed, while a wait is armed.

    Attributes:
        alive: The worker pid is live.
        parked: A wait record is armed for the current state.
        execution: The newest agent execution, folded only for a live, unended machine.
        wait_error: The corrupt wait record's error; such a machine reads as parked.
    """

    alive: bool
    parked: bool
    execution: AgentExecution
    wait_error: str = ""

    def status_word(self, ms: MachineState) -> str:
        """Return `machine_status_word` fed these probes."""
        return machine_status_word(
            ms, parked=self.parked, alive=self.alive, blocked=bool(self.execution.blocked_in)
        )

    def refusals(self, name: str, ms: MachineState) -> dict[MachineVerb, str]:
        """Return `verb_refusals` fed these probes.

        Args:
            name: The instance's name.
            ms: The folded state.

        Returns:
            Each verb's refusal, "" where it can act; a corrupt wait record names itself
            on every verb.
        """
        if self.wait_error:
            return dict.fromkeys(MACHINE_VERBS, f"machine {name!r}: {self.wait_error}")
        return verb_refusals(
            name,
            ended=ms.ended,
            alive=self.alive,
            open_wait=self.parked,
            agent_open=self.execution.open,
            prompt_open=bool(self.execution.blocked_in),
        )


def probe_instance(
    machine_dir: pathlib.Path, ms: MachineState, *, execution: AgentExecution | None = None
) -> InstanceProbes:
    """Probe an instance dir beside its fold.

    Args:
        machine_dir: The instance's dir.
        ms: The folded state.
        execution: The newest execution from a caller's `NewestExecutionFold`; without
            it the log is folded here, for a live, unended machine.

    Returns:
        The probes.
    """
    alive = ipc.worker_is_alive(machine_dir)
    if execution is None:
        execution = (
            newest_agent_execution(machine_dir) if ms.ended is None and alive else AgentExecution()
        )
    try:
        parked = armed_wait(machine_dir, ms) is not None
    except journal.JournalError as exc:
        return InstanceProbes(alive=alive, parked=True, execution=execution, wait_error=str(exc))
    return InstanceProbes(alive=alive, parked=parked, execution=execution)


@dataclasses.dataclass(frozen=True, slots=True)
class MachineSummary:
    """One machine-instance row: what a hub or `machine list` shows, uncolored.

    Attributes:
        name: The instance dir's name.
        machine: The spec's declared name; "" when unreadable.
        current: Where the machine is; "" when unreadable.
        status: The dir-aware status word, or "unreadable".
        reason: A failed end's reason, or what a live instance waits on; else "".
        mtime: The last-activity time as epoch seconds.
    """

    name: str
    machine: str
    current: str
    status: str
    reason: str
    mtime: float


@dataclasses.dataclass(frozen=True, slots=True)
class Spend:
    """A dollar and token spend, summable so booked and live spend fold.

    Attributes:
        usd: The dollar figure.
        input_tokens: Input tokens.
        output_tokens: Output tokens.
        partial: An unpriced model contributed $0, so `usd` is a lower bound; it ORs
            across folds and the render adds the shared "~" marker.
        cache_read_tokens: The cached side of the input; 0 where never recorded.
        cache_creation_tokens: Cache-creation tokens; 0 where never recorded.
    """

    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    partial: bool = False
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: Spend) -> Spend:
        """Return the field-wise sum, `partial` ORed."""
        return Spend(
            self.usd + other.usd,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.partial or other.partial,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
        )


def read_budget_totals(log_path: pathlib.Path, *, from_offset: int = 0) -> Spend:
    """Read the latest running budget totals from an agent state's event log.

    Each `budget.update` carries cumulative totals from its call's own tracker, so
    the last one is that call's running total. This recovers the spend of a killed
    subprocess whose `result.json` never landed, and the live total of an in-flight
    state whose `StepEvent` is not written yet.

    Args:
        log_path: The state's journal.
        from_offset: Read only events appended after this byte offset; a caller
            salvaging one call on a shared log passes the size captured before its
            spawn, or a call that died before its first update double-books the prior one.

    Returns:
        The totals, or an empty `Spend` when there are none or the log is unreadable.
    """
    usd, tin, tout, cr, cc = 0.0, 0, 0, 0, 0
    partial = False
    with contextlib.suppress(OSError):
        with log_path.open("rb") as fh:
            if from_offset > 0:
                fh.seek(from_offset)
            body = fh.read().decode("utf-8", errors="replace")
        for line in body.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "budget.update":
                usd = float(e.get("usd_total", usd) or 0.0)
                tin = int(e.get("input_total", tin) or 0)
                tout = int(e.get("output_total", tout) or 0)
                cr = int(e.get("cache_read_total", cr) or 0)
                cc = int(e.get("cache_creation_total", cc) or 0)
                partial = partial or bool(e.get("usd_partial", False))
    return Spend(usd, tin, tout, partial, cr, cc)


def state_dir_seq(dir_name: str) -> int | None:
    """Return the transition seq a `<seq>-<state>` dir name encodes, None when it has none."""
    head = dir_name.split("-", 1)[0]
    return int(head) if head.isdigit() else None


def machine_spend(
    events: Sequence[object], root: pathlib.Path, *, alive: bool
) -> tuple[Spend, str]:
    """Sum a machine instance's spend, the in-flight state's live figure included.

    A state books its StepEvent only when it completes, so the running state's log
    dir carries a seq no StepEvent holds: that log is the in-flight state, folded only
    while the worker is alive so a crashed in-flight log is ignored.

    Args:
        events: The journal's events.
        root: The instance's dir.
        alive: The worker pid is live.

    Returns:
        The total and the in-flight state's name, "" when none is running.
    """
    total = Spend()
    step_seqs: set[int] = set()
    for event in events:
        if isinstance(event, journal.StepEvent):
            step_seqs.add(event.seq)
            if isinstance(event.fact, journal.AgentFact):
                total += Spend(
                    event.fact.usd,
                    event.fact.input_tokens,
                    event.fact.output_tokens,
                    event.fact.usd_partial,
                )
        elif isinstance(event, journal.AttemptSpend):
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
        elif isinstance(event, journal.MachineEnd):
            # An unbooked slice rides on the end; gating on its usd would drop an unpriced one.
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
    inflight_state = ""
    newest = newest_state_log(root) if alive else None
    if newest is not None:
        seq = state_dir_seq(newest.parent.name)
        if seq is not None and seq not in step_seqs:
            total += read_budget_totals(newest)
            inflight_state = newest.parent.name.split("-", 1)[-1]
    return total, inflight_state


def machine_mtime(machine_dir: pathlib.Path) -> float:
    """Return the instance's last activity: the journal's mtime, else the dir's, else 0.0."""
    for candidate in (machine_dir / "journal.jsonl", machine_dir):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def machine_instance_dirs(state_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return every machine instance dir under the state dir, newest first."""
    root = layout.machines_root(state_dir)
    if not root.is_dir():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "machine.asm.toml").is_file()]
    return sorted(dirs, key=machine_mtime, reverse=True)


def summarize_machine_dir(machine_dir: pathlib.Path) -> MachineSummary:
    """Fold an instance dir's spec and journal into its listing row.

    Args:
        machine_dir: The instance's dir.

    Returns:
        The row; a corrupt source or journal reads "unreadable" with the error's
        first line rather than vanishing from the listing.
    """
    mtime = machine_mtime(machine_dir)
    try:
        spec = load_machine(machine_dir / "machine.asm.toml")
        ms = fold_machine(spec, journal.MachineJournal(machine_dir).read())
    except (MachineError, OSError) as exc:
        first_line = str(exc).split("\n", 1)[0]
        return MachineSummary(machine_dir.name, "", "", "unreadable", first_line, mtime)
    probes = probe_instance(machine_dir, ms)
    reason = ms.ended.reason if ms.ended is not None and ms.ended.status == "failed" else ""
    if probes.execution.blocked_in:
        reason = f"waiting on an answer in {probes.execution.blocked_in}"
    return MachineSummary(
        name=machine_dir.name,
        machine=ms.machine,
        current=ms.current,
        status=probes.status_word(ms),
        reason=reason,
        mtime=mtime,
    )


def machine_files(cwd: pathlib.Path) -> list[pathlib.Path]:
    """Return the `.asm.toml` files a hub offers, from the cwd and its `machines/` subdir."""
    found: set[pathlib.Path] = set(cwd.glob("*.asm.toml"))
    sub = cwd / "machines"
    if sub.is_dir():
        found.update(sub.glob("*.asm.toml"))
    return sorted(found)


MachineVerb = Literal["stop", "poke", "steer", "answer"]
MACHINE_VERBS: tuple[MachineVerb, ...] = ("stop", "poke", "steer", "answer")


def verb_refusals(
    name: str,
    *,
    ended: MachineResult | None,
    alive: bool,
    open_wait: bool,
    agent_open: bool,
    prompt_open: bool,
) -> dict[MachineVerb, str]:
    """Decide why each verb cannot reach a machine, "" where it can.

    Pure, like `machine_status_word`: the probes are the caller's, so a caller holding
    the fold does not read the journal again.

    Args:
        name: The instance's name.
        ended: The end record, None while the machine runs.
        alive: The worker pid is live.
        open_wait: A wait record is armed.
        agent_open: The newest agent execution has begun and not ended.
        prompt_open: The newest agent execution holds an unanswered prompt.

    Returns:
        A refusal per verb: an ended machine consumes no input; a poke reaches only an
        open wait, a steer only an open agent state, an answer only an open prompt,
        and a stop any live worker.
    """
    if ended is not None:
        done = f"machine {name!r} already ended in {ended.state!r} ({ended.status}: {ended.reason})"
        return {
            "stop": f"{done}; nothing to stop",
            "poke": f"{done}; a poke would never be consumed",
            "steer": f"{done}; there is no state to steer",
            "answer": f"{done}; the prompt is closed",
        }
    no_wait = f"machine {name!r} has no open wait to poke"
    if not alive:
        return {
            "stop": (
                f"machine {name!r} is not running; nothing to stop (a parked instance resumes"
                " with `agent6 machine run`)"
            ),
            "steer": (
                f"machine {name!r} is not running, so no agent state would read a steer"
                " (poke it to wake a waiting machine)"
            ),
            "answer": f"machine {name!r} is not running; poke it to wake a waiting machine",
            "poke": "" if open_wait else no_wait,
        }
    if open_wait:
        steer = f"machine {name!r} is waiting; a wait state reads no steer (poke it to wake it)"
    elif agent_open:
        steer = ""
    else:
        steer = f"machine {name!r} has no open agent state to steer"
    return {
        "stop": "",
        "poke": "" if open_wait else no_wait,
        "answer": "" if prompt_open else f"machine {name!r} has no open prompt to answer",
        "steer": steer,
    }


def machine_verb_refusals(machine_dir: pathlib.Path, name: str) -> dict[MachineVerb, str]:
    """Return `verb_refusals` over an instance dir, reading the journal itself.

    Args:
        machine_dir: The instance's dir.
        name: The instance's name.

    Returns:
        A refusal per verb; an unknown or unreadable instance names itself on every verb.
    """
    if not machine_dir.is_dir():
        return dict.fromkeys(MACHINE_VERBS, f"no machine {name!r}")
    try:
        spec = load_machine(machine_dir / "machine.asm.toml")
        ms = fold_machine(spec, journal.MachineJournal(machine_dir).read())
    except (MachineError, journal.JournalError) as exc:
        return dict.fromkeys(MACHINE_VERBS, f"machine {name!r}: {exc}")
    return probe_instance(machine_dir, ms).refusals(name, ms)


def wait_line(machine_id: str, state: str, wake_at: str) -> str:
    """Return the sentence naming where a parked machine waits, when it wakes and how to wake it.

    Args:
        machine_id: The instance's name.
        state: The wait state's name.
        wake_at: The scheduled wake time, "" for a wait on a poke alone.

    Returns:
        The sentence `machine status` and the foreground `machine run` share.
    """
    poke = f"agent6 machine poke {machine_id} [--message TEXT]"
    if wake_at:
        return f"waiting in {state!r}: wakes at {wake_at}; a poke wakes it now: {poke}"
    return f"waiting in {state!r} for a poke: {poke}"


def verb_answer(machine_dir: pathlib.Path, name: str, verb: MachineVerb) -> tuple[bool, str]:
    """Answer a verb before it acts, for every surface.

    Args:
        machine_dir: The instance's dir.
        name: The instance's name.
        verb: The verb about to run.

    Returns:
        `(False, why)` when the whole journal cannot be read, `(True, refusal)` when
        the verb has nothing to act on, and `(True, "")` when it acts.
    """
    if not machine_dir.is_dir():
        return False, f"no machine {name!r}"
    try:
        journal.MachineJournal(machine_dir).read()
    except journal.JournalError as exc:
        return False, f"machine {name!r}: {exc}"
    return True, machine_verb_refusal(machine_dir, name, verb)


def machine_verb_refusal(machine_dir: pathlib.Path, name: str, verb: MachineVerb) -> str:
    """Return why one verb cannot reach a machine now, "" when it can."""
    return machine_verb_refusals(machine_dir, name)[verb]


def machine_word_for_dir(ms: MachineState, machine_dir: pathlib.Path) -> str:
    """Return the status word for an instance with a dir on disk, its probes fed in."""
    return probe_instance(machine_dir, ms).status_word(ms)


def notification_key(n: NotificationView) -> tuple[str, str, str]:
    """Return a notification's identity for dedup across the sliding window.

    Mirrors the web client's `ts|state|message`.
    """
    return (n.ts, n.state, n.message)


def newest_state_log(root: pathlib.Path) -> pathlib.Path | None:
    """Return the newest agent state's journal, the one a watcher follows live, or None."""
    states = root / "states"
    if not states.is_dir():
        return None

    def seq_of(p: pathlib.Path) -> int:
        """Return the dir's seq, -1 for a dir without one."""
        head = p.name.split("-", 1)[0]
        return int(head) if head.isdigit() else -1

    for d in sorted((p for p in states.iterdir() if p.is_dir()), key=seq_of, reverse=True):
        log = d / layout.LOGS_NAME
        if log.is_file():
            return log
    return None


def read_complete_lines(path: pathlib.Path, offset: int) -> tuple[list[str], int]:
    """Read the complete lines a file gained past a byte offset.

    Reads bytes: a poll can hit EOF inside a multibyte sequence, where a text-mode
    readline would raise. Only complete lines are decoded.

    Args:
        path: The file.
        offset: Where the last read stopped.

    Returns:
        The new lines and the new offset, the start of any partial trailing line.
    """
    lines: list[str] = []
    pos = offset
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            while True:
                pos = fh.tell()
                raw = fh.readline()
                if not raw.endswith(b"\n"):
                    break
                lines.append(raw.decode("utf-8", errors="replace"))
    except OSError:
        pass
    return lines, pos


@dataclasses.dataclass
class MachineWatchCursor:
    """What a live machine watcher has already surfaced.

    The three dedup rules every front-end agrees on: transitions by count,
    notifications by identity (the tuple is a sliding window, so a count would miss
    every notify past its cap), and the newest state log by path and byte offset.

    Attributes:
        seen_steps: How many transitions were surfaced.
        seen_notifications: The keys of the notifications surfaced; None before seeding.
        log_path: The state log being followed.
        log_offset: Where the last complete line of that log ended.
    """

    seen_steps: int = 0
    seen_notifications: set[tuple[str, str, str]] | None = None
    log_path: pathlib.Path | None = None
    log_offset: int = 0

    def seed_notifications(self, ms: MachineState) -> None:
        """Mark every recorded notification as seen, so opening a watch does not replay them."""
        self.seen_notifications = {notification_key(n) for n in ms.notifications}

    def new_transitions(self, ms: MachineState) -> list[TransitionView]:
        """Return the transitions not yet surfaced and mark them seen."""
        out = list(ms.transitions[self.seen_steps :])
        self.seen_steps = len(ms.transitions)
        return out

    def new_notifications(self, ms: MachineState) -> list[NotificationView]:
        """Return the notifications not yet surfaced and mark them seen."""
        if self.seen_notifications is None:
            self.seen_notifications = set()
        out: list[NotificationView] = []
        for n in ms.notifications:
            key = notification_key(n)
            if key not in self.seen_notifications:
                self.seen_notifications.add(key)
                out.append(n)
        return out

    def advance_log(self, root: pathlib.Path) -> tuple[pathlib.Path | None, bool]:
        """Follow the newest per-state log under the instance dir.

        Args:
            root: The instance's dir.

        Returns:
            The current log and whether it changed; on a change the caller resets its
            render state and announces the new agent state.
        """
        newest = newest_state_log(root)
        if newest != self.log_path:
            self.log_path, self.log_offset = newest, 0
            return newest, True
        return newest, False

    def read_log_lines(self) -> list[str]:
        """Return the complete lines the current state log gained since the last poll."""
        if self.log_path is None:
            return []
        lines, self.log_offset = read_complete_lines(self.log_path, self.log_offset)
        return lines


def machine_state_as_dict(
    ms: MachineState,
    machine_dir: pathlib.Path | None = None,
    *,
    execution: AgentExecution | None = None,
) -> dict[str, Any]:
    """Return the wire form of a `MachineState`, what `attach --json` and the web serialize.

    Args:
        ms: The folded state.
        machine_dir: The instance's dir; with it `status` is the dir-aware word, so a
            client can tell a parked instance from a running one, and every verb's
            refusal rides along so a front-end gates its buttons from the one decision.
        execution: The newest execution from a caller's `NewestExecutionFold`.

    Returns:
        The state's fields, plus `status`, `level`, `refusals` and, for a stopped
        machine, `worker_lost`.
    """
    d = dataclasses.asdict(ms)
    if machine_dir is not None:
        probes = probe_instance(machine_dir, ms, execution=execution)
        d["status"] = probes.status_word(ms)
        d["level"] = format.status_level(d["status"])
        if d["status"] == "stopped":
            # Resumable, and the wire says so; `ended` stays a durable MachineEnd.
            d["worker_lost"] = {"reason": "no worker running", "state": ms.current}
        # Fed from this fold's probes: `machine_verb_refusals` would fold the journal again.
        d["refusals"] = probes.refusals(machine_dir.name, ms)
    return d
