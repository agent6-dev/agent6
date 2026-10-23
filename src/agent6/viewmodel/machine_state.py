# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a machine instance's journal and spec into the watch view every front-end renders.

The machine analogue of state.py: which states exist, where the machine is, the path
taken and how it ended. The reasoning inside an agent state is itself a run log and
folds through `SessionState`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Any, Literal

from agent6.machine import MachineError, MachineResult, journal, load_machine, schema
from agent6.sessions import layout, process
from agent6.viewmodel import _machine_logs, format

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
    detail = fact_detail(s)
    line = format.format_transition(s.seq, s.state, s.label, s.goto, detail)
    return TransitionView(
        seq=s.seq, state=s.state, label=s.label, goto=s.goto, detail=detail, line=line
    )


def tool_failure(exit_code: int, timed_out: bool, output: str) -> str:
    """Return a failed tool command's exit code and last output line, "" on success.

    Args:
        exit_code: The command's exit code.
        timed_out: The command hit its timeout.
        output: Its stderr, or its stdout when stderr is empty.

    Returns:
        `exit N` or `timed out`, then the last non-blank output line cut to 160
        characters when there is one.
    """
    if exit_code == 0 and not timed_out:
        return ""
    tail = next((ln.strip() for ln in reversed(output.splitlines()) if ln.strip()), "")
    head = "timed out" if timed_out else f"exit {exit_code}"
    return f"{head}: {tail[:160]}" if tail else head


def fact_detail(step: journal.StepEvent) -> str:
    """Return bounded failure evidence for one transition, "" on success."""
    fact = step.fact
    if isinstance(fact, journal.ToolFact):
        return tool_failure(fact.exit_code, fact.timed_out, fact.stderr or fact.stdout)
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
    execution: _machine_logs.AgentExecution
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
    machine_dir: pathlib.Path,
    ms: MachineState,
    *,
    execution: _machine_logs.AgentExecution | None = None,
) -> InstanceProbes:
    """Probe an instance dir beside its fold.

    Args:
        machine_dir: The instance's dir.
        ms: The folded state.
        execution: The newest execution from a caller's `_machine_logs.NewestExecutionFold`; without
            it the log is folded here, for a live, unended machine.

    Returns:
        The probes.
    """
    alive = process.worker_is_alive(machine_dir)
    if execution is None:
        execution = (
            _machine_logs.newest_agent_execution(machine_dir)
            if ms.ended is None and alive
            else _machine_logs.AgentExecution()
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
        reason: A failed end's reason, what a live instance waits on, or "no worker" for a
            parked instance nothing is driving; else "".
        mtime: The last-activity time as epoch seconds.
    """

    name: str
    machine: str
    current: str
    status: str
    reason: str
    mtime: float


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
    elif ms.ended is None and probes.parked and not probes.alive:
        reason = "no worker"
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
                f" with `agent6 machine run {name}`)"
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


def wait_line(
    machine_id: str, state: str, wake_at: str, *, alive: bool, include_poke: bool = True
) -> str:
    """Return the sentence naming where a parked machine waits, when it wakes and how to wake it.

    Args:
        machine_id: The instance's name.
        state: The wait state's name.
        wake_at: The scheduled wake time, "" for a wait on a poke alone.
        alive: The worker pid is live; without a worker only `machine run` moves the machine.
        include_poke: Append the poke command; a polling run includes it only on its first line.

    Returns:
        The sentence `machine status` and the foreground `machine run` share.
    """
    poke = f"agent6 machine poke {machine_id} [--message TEXT]"
    if not alive:
        due = f"due at {wake_at}" if wake_at else "waiting for a poke"
        hint = f"; a poke wakes it on that run: {poke}" if include_poke else ""
        return (
            f"parked in {state!r} with no worker running ({due}): resume with"
            f" `agent6 machine run {machine_id}`{hint}"
        )
    if wake_at:
        hint = f"; a poke wakes it now: {poke}" if include_poke else ""
        return f"waiting in {state!r}: wakes at {wake_at}{hint}"
    hint = f": {poke}" if include_poke else ""
    return f"waiting in {state!r} for a poke{hint}"


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


def machine_state_as_dict(
    ms: MachineState,
    machine_dir: pathlib.Path | None = None,
    *,
    execution: _machine_logs.AgentExecution | None = None,
) -> dict[str, Any]:
    """Return the wire form of a `MachineState`, what `attach --json` and the web serialize.

    Args:
        ms: The folded state.
        machine_dir: The instance's dir; with it `status` is the dir-aware word, so a
            client can tell a parked instance from a running one, and every verb's
            refusal rides along so a front-end gates its buttons from the one decision.
        execution: The newest execution from a caller's `_machine_logs.NewestExecutionFold`.

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
