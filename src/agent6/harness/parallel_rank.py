# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rank the attempts at one task: their judges, the judges' orders, the mechanical order.

A judge is a reviewer child whose verdict is an order. `rank` starts a task's judges in the
background of its run once every attempt has ended, and writes the ranking at the turn
boundary after they end; the pure half (`judge_tasks`, `reply_of`, `combined`) also serves
`sessions compare`, which runs the same reviewers and waits for them.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from agent6 import child_records, errors, kinds, llm_json
from agent6.config import Config, ConfigError
from agent6.harness import child_repo, parallel_panel
from agent6.harness.review import verdict as review_verdict
from agent6.prompts import review
from agent6.sessions import child_dispatch, completion
from agent6.sessions import layout as sessions_layout
from agent6.tools import schema as tools_schema
from agent6.vcs.git import read as git_read

if TYPE_CHECKING:
    from agent6.harness import parallel_dispatch

# Each attempt's diff in a judge's opening; a cut is marked, and the judge's clone holds the rest.
DIFF_CAP = 60_000
# A judge's reason, as the record, the journal and the merge notice keep it.
_REASON_CAP = 2_000


@dataclasses.dataclass(frozen=True, slots=True)
class Judge:
    """One entry of `[harness.parallel].judges`, resolved.

    Attributes:
        selector: The entry as written: a role name or `[provider/]model`.
        settings: The route, effort and temperature the reviewer runs with.
    """

    selector: str
    settings: kinds.ModelSettings


@dataclasses.dataclass(frozen=True, slots=True)
class JudgeReply:
    """What one judge returned.

    Attributes:
        selector: The judge.
        order: The attempts' ids, best first; None when it gave no usable order.
        said: Its reason for the order, or why it gave none.
    """

    selector: str
    order: tuple[str, ...] | None
    said: str


@dataclasses.dataclass(slots=True)
class Judging:
    """A task's judges while they run.

    Attributes:
        record: The judges' own dispatch record.
        children: The started judges, in the record's order.
    """

    record: child_dispatch.DispatchRecord
    children: list[child_records.StartedChild]


def mechanical_ranking(candidates: Sequence[child_records.CandidateBrief]) -> tuple[str, ...]:
    """Return the candidates ranked verify-pass first, then by lower cost, stable within ties."""
    ranked = sorted(candidates, key=lambda c: (c.verification != "passed", c.cost_usd))
    return tuple(c.session_id for c in ranked)


def configured_judges(cfg: Config, *, log: Callable[[str], None]) -> tuple[Judge, ...]:
    """Resolve `[harness.parallel].judges` to the reviewers that will sit.

    Args:
        cfg: The run's config.
        log: Told of a selector that names nothing.

    Returns:
        One judge per entry with a model route; a role with none judges nothing.
    """
    unrouted = {role for role in kinds.MODEL_ROLES if cfg.models.resolve(role) is None}
    judges: list[Judge] = []
    for selector in cfg.harness.parallel.judges:
        if selector in unrouted:
            continue
        try:
            settings = cfg.child_settings(selector)
        except ConfigError as exc:
            log(f"judge {selector} names no model ({exc}); it ranks nothing")
            continue
        if settings is not None:
            judges.append(Judge(selector, settings))
    return tuple(judges)


def candidates(
    root: pathlib.Path, results: Sequence[child_records.ChildResult], *, base: str, task: str
) -> list[child_records.CandidateBrief]:
    """Return a dispatch's attempts as their judges read them.

    Args:
        root: The repository the attempts were imported into.
        results: The attempts that ended with a result by themselves.
        base: The commit they were cut from.
        task: The task they were given.

    Returns:
        One brief per attempt, in child order; a diff that cannot be read is "".
    """
    briefs: list[child_records.CandidateBrief] = []
    for res in results:
        diff = ""
        if res.imported_tip and base:
            try:
                diff = git_read.diff_range(root, base, res.imported_tip)
            except errors.GitError:
                diff = ""
        briefs.append(
            child_records.CandidateBrief(
                session_id=res.spec.session_id,
                task=task,
                diff=diff,
                verification=res.verification,
                cost_usd=res.usage.usd if res.usage is not None else 0.0,
                ref=res.imported_ref,
                tip=res.imported_tip,
                summary=res.report,
            )
        )
    return briefs


def opening(attempts: Sequence[child_records.CandidateBrief], *, base: str) -> str:
    """Return a judge's first message: the task, then each attempt with its facts and diff.

    Args:
        attempts: The attempts to rank.
        base: The commit they were cut from, the judge's checkout; "" when they share none.

    Returns:
        The message. One task is given once; attempts at different tasks each give their own.
    """
    shared = len({a.task for a in attempts}) == 1
    parts: list[str] = []
    if shared:
        parts.append(f"TASK:\n{attempts[0].task.strip()[:4000]}")
    if base:
        parts.append(f"BASE: {base[:12]} (your checkout)")
    for a in attempts:
        lines = [f"ATTEMPT {a.session_id}"]
        if not shared:
            lines.append(f"task: {a.task.strip()[:4000]}")
        if a.tip:
            # A clone names the origin's branches `origin/<branch>`; any other ref goes by its tip.
            held = f"origin/{a.ref} at " if a.ref and not a.ref.startswith("refs/") else ""
            lines.append(f"branch: {held}{a.tip[:12]}")
        lines.append(f"verify: {a.verification or 'unknown'}")
        lines.append(f"cost: ${a.cost_usd:.4f}")
        if a.summary.strip():
            lines.append("summary: " + a.summary.strip()[:2000].replace("\n", "\n  "))
        diff = a.diff[:DIFF_CAP]
        if len(a.diff) > DIFF_CAP:
            shown = f"; `git diff {base[:12]} {a.tip[:12]}` shows all of it"
            diff += f"\n[diff cut at {DIFF_CAP:,} characters{shown if base and a.tip else ''}]"
        lines.append(f"DIFF:\n{diff}" if diff.strip() else "DIFF: none")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def judge_tasks(
    judges: Sequence[Judge],
    attempts: Sequence[child_records.CandidateBrief],
    *,
    base: str,
    max_turns: int,
    tools: Sequence[str] | None,
) -> list[child_records.ChildTask]:
    """Return one reviewer child per judge, each opening on the same attempts.

    Args:
        judges: The judges, in config order.
        attempts: The attempts to rank.
        base: The commit the attempts were cut from; "" when they share none.
        max_turns: Each judge's reply cap, `review.max_turns`.
        tools: The tools the parent's own model is offered now, of which a judge gets the
            ones a reviewer's contract may carry; None, for a host with no model of its
            own, hands down all of those, and the judge's own config withholds the rest.

    Returns:
        The children to start, in judge order.
    """
    carried = tools_schema.mode_tools("reviewer").names
    handed = tuple(
        name for name in (sorted(carried) if tools is None else tools) if name in carried
    )
    commands = tools_schema.RunCommandInput.TOOL_NAME in handed
    first = opening(attempts, base=base)
    tasks: list[child_records.ChildTask] = []
    for judge in judges:
        tasks.append(
            child_records.ChildTask(
                task=f"Rank {len(attempts)} attempts",
                model=judge.selector,
                settings=judge.settings,
                contract=kinds.ConversationContract(
                    system=review.rank_prompt(max_turns=max_turns, commands=commands),
                    opening=first,
                    max_turns=max_turns,
                    temperature=judge.settings.temperature,
                    allowed_tools=handed,
                ),
            )
        )
    return tasks


def read_reply(
    selector: str,
    session_dir: pathlib.Path,
    ids: Sequence[str],
    *,
    max_turns: int,
    failed: str = "",
) -> JudgeReply:
    """Read a judge's reply from its session dir.

    Args:
        selector: The judge.
        session_dir: The judge's session dir, where its completion record is.
        ids: The attempts' session ids.
        max_turns: The judge's reply cap.
        failed: Why its run failed, "" when it did not; a judge its reply limit ended
            still leaves a usable verdict.

    Returns:
        The order with the judge's reason, or no order with why.
    """
    found, error = None, ""
    try:
        found = completion.read_completion(session_dir)
    except completion.CompletionError as exc:
        error = str(exc)
    if failed and (found is None or found.stop_reason != "max_turns") and not error:
        error = failed
    return reply_of(selector, found, ids, max_turns=max_turns, error=error)


def reply_of(
    selector: str,
    record: completion.CompletionRecord | None,
    ids: Sequence[str],
    *,
    max_turns: int,
    error: str = "",
) -> JudgeReply:
    """Read one judge's verdict: an order that names every attempt once, with its reason.

    Args:
        selector: The judge.
        record: Its completion record; None when it left none.
        ids: The attempts' session ids.
        max_turns: The judge's reply cap, named when it ran out.
        error: Why the judge has no usable record, when its run already says.

    Returns:
        The order with the judge's reason, or no order with why.
    """
    if error or record is None:
        return JudgeReply(selector, None, error or "it left no verdict")
    if record.stop_reason not in ("answered", "max_turns"):
        why = "; ".join(record.diagnostics) or record.text or record.stop_reason
        return JudgeReply(selector, None, why.strip().splitlines()[0] if why.strip() else "failed")
    obj = llm_json.extract_json(record.text, prefer=("order",))
    raw = obj.get("order") if obj is not None else None
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        if record.asked_for_tools:
            return JudgeReply(selector, None, f"no verdict within {max_turns} replies")
        return JudgeReply(selector, None, review_verdict.no_verdict_error(record))
    order = tuple(str(item) for item in raw)
    if len(order) != len(ids) or set(order) != set(ids):
        return JudgeReply(selector, None, "its order did not name every attempt exactly once")
    reason = " ".join(str(obj.get("summary", "") if obj is not None else "").split())
    return JudgeReply(selector, order, reason[:_REASON_CAP])


def combined(
    attempts: Sequence[child_records.CandidateBrief],
    replies: Sequence[JudgeReply],
    *,
    unasked: str = "",
) -> child_records.Ranking:
    """Fold the judges' orders into one ranking.

    Each usable order places every attempt; the lowest sum of places wins, and a tie goes
    to the mechanical order. With no usable order the mechanical order ranks.

    Args:
        attempts: The attempts ranked.
        replies: Each judge's reply, in judge order.
        unasked: Why no judge was asked; "" when the judges were.

    Returns:
        The ranking, without a winner: `winner` crowns one.
    """
    mechanical = mechanical_ranking(attempts)
    usable = [reply for reply in replies if reply.order is not None]
    failed = "; ".join(f"{r.selector}: {r.said}" for r in replies if r.order is None)
    if not usable:
        why = unasked or f"no judge gave a usable order ({failed})"
        return child_records.Ranking(order=mechanical, fallback=why)
    if len(usable) == 1:
        (only,) = usable
        assert only.order is not None
        order = only.order
        rationale = only.said
    else:
        total = {
            sid: sum(reply.order.index(sid) for reply in usable if reply.order is not None)
            for sid in mechanical
        }
        order = tuple(sorted(mechanical, key=lambda sid: (total[sid], mechanical.index(sid))))
        rationale = "\n".join(
            [f"lowest sum of ranks over {len(usable)} judges"]
            + [f"  {reply.selector}: {reply.said}" for reply in usable]
        )
    return child_records.Ranking(order=order, ranked_by="judge", rationale=rationale)


def winner(
    attempts: Sequence[child_records.CandidateBrief], order: Sequence[str], *, harness_gate: bool
) -> str:
    """Return the attempt an order crowns, or "" when it crowns none.

    A judge's order is model output: the first wins unless a gate ran red and none passed, and
    only a verified attempt wins when any verified. An unverified finish counts as red only
    under a harness gate, the rule one run's exit code follows.

    Args:
        attempts: The ranked attempts.
        order: Their ids, best first.
        harness_gate: Whether the harness runs the gate (`harness.verify.when` is not `never`).

    Returns:
        The winner's session id, or "".
    """
    if not order:
        return ""
    first = next(a for a in attempts if a.session_id == order[0])
    verdicts = {a.verification for a in attempts}
    if "passed" in verdicts:
        return first.session_id if first.verification == "passed" else ""
    red = {"failed", "unverified"} if harness_gate else {"failed"}
    return "" if verdicts & red else first.session_id


def judges_identity(dispatcher: parallel_dispatch.ParallelDispatcher, identity: str) -> str:
    """Return the id of the dispatch record a task's judges are written under."""
    group = identity.removeprefix(f"{dispatcher.session_id}-")
    return f"{dispatcher.session_id}-r{group}"


def replies(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    record: child_dispatch.DispatchRecord,
    ids: Sequence[str],
) -> list[JudgeReply]:
    """Read each judge's reply from its session, by the judges' completed record.

    Args:
        dispatcher: The run's dispatcher.
        record: The judges' record, each child joined.
        ids: The attempts' session ids.

    Returns:
        One reply per judge, in record order.
    """
    assert dispatcher.session_dir is not None
    state_dir = sessions_layout.layout_of(dispatcher.session_dir).state_dir
    bucket = sessions_layout.bucket_dir(state_dir, kinds.session_bucket("reviewer"))
    out: list[JudgeReply] = []
    for child in record.children:
        failed = ""
        if child.join is not None and child.join.status == "failed":
            failed = child.join.detail or "its run failed"
        out.append(
            read_reply(
                child.model or "worker",
                bucket / child.session_id,
                ids,
                max_turns=dispatcher.reviewer_turns,
                failed=failed,
            )
        )
    return out


def start(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    task: child_dispatch.DispatchRecord,
    attempts: Sequence[child_records.CandidateBrief],
    *,
    turn: int,
) -> Judging | str:
    """Start a task's judges in the background and return them, or why none started.

    Args:
        dispatcher: The run's dispatcher.
        task: The task's record.
        attempts: The attempts to rank.
        turn: The turn the judges start after.

    Returns:
        The judges in flight, or the reason the mechanical order ranks instead.
    """  # noqa: DOC501  # the spawner's fault is caught below
    assert dispatcher.child_spawner is not None
    identity = judges_identity(dispatcher, task.identity)
    group = identity.removeprefix(f"{dispatcher.session_id}-")
    deadline = dispatcher.clock() + parallel_panel.PANEL_BATCH_DEADLINE_S
    children = [
        dataclasses.replace(
            child,
            limits=child_records.ChildLimits(
                max_turns=dispatcher.reviewer_turns, deadline_at=deadline
            ),
        )
        for child in judge_tasks(
            dispatcher.judges,
            attempts,
            base=task.base,
            max_turns=dispatcher.reviewer_turns,
            tools=dispatcher.reviewer_tools(),
        )
    ]
    # Written delivered and reported: a ranking's result is its task's merge notice.
    record = child_dispatch.DispatchRecord(
        identity=identity,
        source="panel",
        join_policy="record_only",
        owner=dispatcher.execution_id,
        turn=turn,
        base=task.base,
        delivered=True,
        children=tuple(
            child_dispatch.ChildRecord(
                session_id=f"{identity}-l{i}",
                number=i,
                task=child.task,
                model=child.model,
                limits=child.limits,
                reported=True,
            )
            for i, child in enumerate(children, start=1)
        ),
    )
    try:
        dispatcher.write(record)
    except OSError as exc:
        return f"the judges' record could not be written ({exc})"
    record = dispatcher.announce(record, children)
    ceiling = dispatcher.budget.available() if dispatcher.budget is not None else None
    started: list[child_records.StartedChild] = []
    try:
        started = dispatcher.child_spawner(
            [dataclasses.replace(child, allowance=ceiling) for child in children],
            group,
            at=task.base or None,
        )
        if len(started) != len(children):
            raise child_repo.ChildRepoError(
                f"group spawner started {len(started)} judge(s) for {len(children)}"
            )
    except Exception as exc:
        for child in started:
            child.stop()
        dispatcher.failed_launch(record, exc)
        return f"the judges could not start ({exc})"
    return Judging(record, started)


def rank(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    task: parallel_dispatch.OpenDispatch,
    results: Sequence[child_records.ChildResult],
    *,
    turn: int,
) -> child_dispatch.DispatchRecord:
    """Rank a task's ended attempts, or start or await the judges that will.

    Called at a turn boundary once every attempt has ended. A stopped attempt is no
    candidate. With two candidates or more and a judge configured, the judges start in the
    background and the ranking is written at the boundary after they end; otherwise the
    mechanical order ranks at once.

    Args:
        dispatcher: The run's dispatcher.
        task: The open rank join.
        results: Its children's results.
        turn: The turn just completed.

    Returns:
        The task's record: with its ranking, or without one while its judges run or a stop
        holds them.
    """
    by_id = {res.spec.session_id: res for res in results}
    record = task.record
    if task.judging is None:
        record = dispatcher.settled(record, by_id, turn=turn)
    elif any(not judge.result.done() for judge in task.judging.children):
        return record
    attempts = candidates(
        dispatcher.chain.root,
        [
            res
            for child in record.children
            if (res := by_id.get(child.session_id)) is not None and res.ok and not res.stopped_by
        ],
        base=record.base,
        task=record.children[0].task if record.children else "",
    )
    ids = [a.session_id for a in attempts]
    told: list[JudgeReply] = []
    unasked = ""
    cost, partial = 0.0, False
    if task.judging is not None:
        judged = [dispatcher.result_of(task.judging.record, j) for j in task.judging.children]
        dispatcher.complete(task.judging.record, judged, turn=turn)
        cost = sum(res.usage.usd for res in judged if res.usage is not None)
        partial = any(res.usage is None or res.usage.usd_partial for res in judged)
        told = replies(dispatcher, _read(dispatcher, task.judging.record.identity), ids)
        task.judging = None
    elif len(attempts) < 2:
        unasked = "one candidate" if attempts else "no candidate"
    elif not dispatcher.judges:
        unasked = "no judge is configured"
    elif (earlier := _earlier(dispatcher, record.identity)) is not None:
        # An earlier execution's judges: they are never started twice.
        told = replies(dispatcher, earlier, ids)
    elif dispatcher.stop_pending():
        dispatcher.log(f"PARALLEL: group {record.identity} stopped before its ranking")
        return record
    else:
        started = start(dispatcher, record, attempts, turn=turn)
        if isinstance(started, Judging):
            task.judging = started
            record = record.advanced("ranking")
            dispatcher.write(record)
            return record
        unasked = started
    for reply in told:
        if reply.order is None:
            dispatcher.log(f"PARALLEL: judge {reply.selector} gave no usable order ({reply.said})")
    ranking = combined(attempts, told, unasked=unasked)
    ranking = dataclasses.replace(
        ranking,
        winner=winner(attempts, ranking.order, harness_gate=dispatcher.harness_gate),
        judge_cost_usd=cost,
        judge_cost_partial=partial,
    )
    record = record.model_copy(
        update={
            "ranking": child_dispatch.RankRecord(
                order=ranking.order,
                ranked_by=ranking.ranked_by,
                rationale=ranking.rationale,
                winner=ranking.winner,
                fallback=ranking.fallback,
            )
        }
    )
    # The judges' spend is on the meter now.
    dispatcher.emit_budget(turn)
    dispatcher.write(record)
    if dispatcher.rank_recorder is not None and attempts:
        dispatcher.rank_recorder(
            attempts, [by_id[sid] for sid in ids], ranking, group=record.identity
        )
    return record


def _read(
    dispatcher: parallel_dispatch.ParallelDispatcher, identity: str
) -> child_dispatch.DispatchRecord:
    """Return a dispatch record as last written."""
    assert dispatcher.session_dir is not None
    return child_dispatch.read_record(child_dispatch.record_path(dispatcher.session_dir, identity))


def _earlier(
    dispatcher: parallel_dispatch.ParallelDispatcher, identity: str
) -> child_dispatch.DispatchRecord | None:
    """Return the record of judges an earlier execution started for this task, or None."""
    assert dispatcher.session_dir is not None
    path = child_dispatch.record_path(dispatcher.session_dir, judges_identity(dispatcher, identity))
    return child_dispatch.read_record(path) if path.is_file() else None
