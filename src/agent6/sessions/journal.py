# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The journal's vocabulary: one frozen record per event type.

A writer constructs a record and hands it to `EventSink.emit`, which stamps the envelope
(`ts`, `type`, `event_id`); a field left unset is left out of the row. The formats page lists
every record here. A sequence field serializes as a JSON list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pydantic


class Event(pydantic.BaseModel):
    """One journal event; `TYPE` is the row's `type`."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    TYPE: ClassVar[str] = ""


class ApprovalAnswer(Event):
    """The operator answered an approval request."""

    TYPE: ClassVar[str] = "approval.answer"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    approved: bool = pydantic.Field(description="Whether the operator approved the request.")
    source: str = pydantic.Field(description="Where the request or result came from.")


class ApprovalPrompt(Event):
    """A command is waiting for the operator's approval."""

    TYPE: ClassVar[str] = "approval.prompt"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    prompt: str = pydantic.Field(description="The words presented to the operator.")
    standing: bool = pydantic.Field(
        description="Whether approval can cover later calls in this scope.",
    )
    call_id: int | None = pydantic.Field(
        default=None,
        description="The tool call correlated with this event.",
    )


class BtwAnswered(Event):
    """A side question received an answer."""

    TYPE: ClassVar[str] = "btw.answered"
    btw_id: str = pydantic.Field(description="Side-question identifier.")
    block: str = pydantic.Field(description="The rendered question and answer block.")


class BtwOpened(Event):
    """The operator opened a side question."""

    TYPE: ClassVar[str] = "btw.opened"
    btw_id: str = pydantic.Field(description="Side-question identifier.")
    question: str = pydantic.Field(description="The question presented to the operator.")


class BudgetUpdate(Event):
    """The cumulative token, spend and subscription readings."""

    TYPE: ClassVar[str] = "budget.update"
    usd_total: float = pydantic.Field(
        description=(
            "Cumulative USD spend across provider/model routes, including purchased credits."
        ),
    )
    input_total: int | None = pydantic.Field(default=None, description="Cumulative input tokens.")
    output_total: int | None = pydantic.Field(default=None, description="Cumulative output tokens.")
    cache_read_total: int | None = pydantic.Field(
        default=None,
        description="Cumulative cached input tokens.",
    )
    cache_creation_total: int | None = pydantic.Field(
        default=None,
        description="Cumulative cache-creation tokens.",
    )
    usd_partial: bool | None = pydantic.Field(
        default=None,
        description="Whether unpriced usage makes spend a lower bound.",
    )
    usd_cap: float | None = pydantic.Field(
        default=None,
        description="Spend cap in USD; -1 means unlimited.",
    )
    tokens_unmetered: int | None = pydantic.Field(
        default=None,
        description="Tokens with no known price.",
    )
    tokens_fallback_cap: int | None = pydantic.Field(
        default=None,
        description="Token cap used for unpriced usage.",
    )
    plan_used_percent: float | None = pydantic.Field(
        default=None,
        description="Provider-reported subscription usage percentage.",
    )
    plan_observed_points: float | None = pydantic.Field(
        default=None,
        description=(
            "The largest observed plan movement in points across providers and windows; "
            "account-wide, so includes other consumers."
        ),
    )
    plan_cap: float | None = pydantic.Field(default=None, description="Subscription points cap.")
    plan_resets_at: float | None = pydantic.Field(
        default=None,
        description="Subscription window reset time in epoch seconds.",
    )
    plan_windows: Sequence[Any] | None = pydantic.Field(
        default=None,
        description="Every provider-reported subscription window.",
    )
    credits_has: bool | None = pydantic.Field(
        default=None,
        description="Whether the account has credits.",
    )
    credits_unlimited: bool | None = pydantic.Field(
        default=None,
        description="Whether credits are unlimited.",
    )
    credits_balance: str | None = pydantic.Field(
        default=None,
        description="Provider-reported credit balance.",
    )


class CommandBackgrounded(Event):
    """A command was handed back as a background job."""

    TYPE: ClassVar[str] = "command.backgrounded"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    handle: str = pydantic.Field(description="Launcher-instance-scoped job handle.")
    seconds: float = pydantic.Field(description="Elapsed command time in seconds at handback.")


class DiffUpdated(Event):
    """A chain commit landed with its patch."""

    TYPE: ClassVar[str] = "diff.updated"
    sha: str = pydantic.Field(description="The commit identifier, possibly abbreviated.")
    patch: str = pydantic.Field(description="The commit's patch, capped at 8000 bytes.")


class ExecutionEnvironment(Event):
    """An environment lifecycle observation, independent of job settlement."""

    TYPE: ClassVar[str] = "execution.environment"
    execution_id: str | None = pydantic.Field(default=None, description="Execution identifier.")
    target_id: str | None = pydantic.Field(default=None, description="Recorded target reference.")
    outcome: str = pydantic.Field(description="left_running, stopped, unknown or not_applicable.")
    reason: str = pydantic.Field(description="The observation's reason, not stop authority.")


class ExecutionJobs(Event):
    """The command owner observed job settlement or could not confirm it."""

    TYPE: ClassVar[str] = "execution.jobs"
    execution_id: str | None = pydantic.Field(default=None, description="Execution identifier.")
    target_id: str | None = pydantic.Field(default=None, description="Recorded target reference.")
    outcome: str = pydantic.Field(description="settled or unknown.")
    reason: str = pydantic.Field(description="Why settlement is unknown, empty when confirmed.")


class ExecutionTarget(Event):
    """Immutable target facts recorded before workspace commands."""

    TYPE: ClassVar[str] = "execution.target"
    execution_id: str | None = pydantic.Field(default=None, description="Execution identifier.")
    target_id: str = pydantic.Field(
        description="Host-issued evidence reference, not launcher identity.",
    )
    target: dict[str, Any] = pydantic.Field(
        description="ExecutionTargetStamp facts; null fields stay unobserved.",
    )


class GraphUpdate(Event):
    """The task graph and its current focus."""

    TYPE: ClassVar[str] = "graph.update"
    nodes: dict[str, Any] = pydantic.Field(description="Task nodes keyed by identifier.")
    cursor: str | None = pydantic.Field(
        default=None,
        description="Current task identifier; null when none is selected.",
    )


class JailDegraded(Event):
    """The sandbox started weaker than requested or left surviving processes."""

    TYPE: ClassVar[str] = "jail.degraded"
    detail: str = pydantic.Field(
        description="The sandbox degradation or surviving-process diagnostic.",
    )


class LoopAutoCommit(Event):
    """A turn's checkpoint commit landed."""

    TYPE: ClassVar[str] = "loop.auto_commit"
    turn: int = pydantic.Field(description="The execution's turn number.")
    sha: str = pydantic.Field(description="The commit identifier, possibly abbreviated.")
    subject: str = pydantic.Field(description="The landed commit's subject.")
    task_id: str | None = pydantic.Field(default=None, description="Task identifier.")


class LoopAutoCommitFailed(Event):
    """A checkpoint commit could not land."""

    TYPE: ClassVar[str] = "loop.auto_commit.failed"
    turn: int = pydantic.Field(description="The execution's turn number.")
    error: str = pydantic.Field(description="The failure diagnostic.")
    worktree_status: str = pydantic.Field(description="Git status when the commit failed.")
    commit_subject: str = pydantic.Field(description="The attempted commit subject.")


class LoopBaseline(Event):
    """The baseline verification attempt finished."""

    TYPE: ClassVar[str] = "loop.baseline"
    ok: bool = pydantic.Field(description="Whether the operation succeeded.")
    turn: int = pydantic.Field(description="The execution's turn number.")
    verify_id: str | None = pydantic.Field(
        default=None,
        description="Verification attempt identifier.",
    )


class LoopBudget(Event):
    """A turn's usage heartbeat before its model call."""

    TYPE: ClassVar[str] = "loop.budget"
    turn: int = pydantic.Field(description="The execution's turn number.")
    input_tokens: int = pydantic.Field(description="Input tokens consumed so far.")
    output_tokens: int = pydantic.Field(description="Output token count.")
    cache_read_tokens: int = pydantic.Field(description="Cached input tokens consumed so far.")
    cost_usd: float = pydantic.Field(description="Estimated spend in USD.")


class LoopCompactDeduped(Event):
    """Repeated tool results were removed from context."""

    TYPE: ClassVar[str] = "loop.compact.deduped"
    n: int = pydantic.Field(description="Number of affected entries or notices.")
    calls: Sequence[Any] = pydantic.Field(description="The affected tool call identities.")


class LoopCompactDropped(Event):
    """Tool results were elided from context."""

    TYPE: ClassVar[str] = "loop.compact.dropped"
    n: int = pydantic.Field(description="Number of affected entries or notices.")
    calls: Sequence[Any] = pydantic.Field(description="The affected tool call identities.")


class LoopCompactGistCall(Event):
    """The model was asked to summarize files for context."""

    TYPE: ClassVar[str] = "loop.compact.gist.call"
    files: int = pydantic.Field(description="Number of files to summarize.")


class LoopCompactGistFailed(Event):
    """The file-summary call failed."""

    TYPE: ClassVar[str] = "loop.compact.gist.failed"
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopCompactGists(Event):
    """File summaries were created or demoted during compaction."""

    TYPE: ClassVar[str] = "loop.compact.gists"
    gisted: int = pydantic.Field(description="Number of file summaries created.")
    demoted: int = pydantic.Field(
        description="Number of summaries replaced by smaller placeholders.",
    )
    paths: Sequence[Any] = pydantic.Field(description="Affected repository-relative paths.")
    demoted_paths: Sequence[Any] = pydantic.Field(description="Paths whose summaries were demoted.")


class LoopCompactRefused(Event):
    """There was too little history to summarize."""

    TYPE: ClassVar[str] = "loop.compact.refused"
    reason: str = pydantic.Field(description="The reason for the outcome.")


class LoopCompactRequested(Event):
    """The operator requested a context restart."""

    TYPE: ClassVar[str] = "loop.compact.requested"
    focus: str = pydantic.Field(description="The requested summary or review focus.")


class LoopCompactRestored(Event):
    """A resumed context restored its elision and summary counts."""

    TYPE: ClassVar[str] = "loop.compact.restored"
    elided: int = pydantic.Field(description="Number of restored elision markers.")
    gists: int = pydantic.Field(description="Number of restored file summaries.")


class LoopCompactSummariseCall(Event):
    """The model was asked to summarize the conversation."""

    TYPE: ClassVar[str] = "loop.compact.summarise.call"
    messages: int = pydantic.Field(description="Number of conversation messages.")


class LoopCompactSummariseDone(Event):
    """A summary replaced the conversation history."""

    TYPE: ClassVar[str] = "loop.compact.summarise.done"
    summary_chars: int = pydantic.Field(description="Length of the summary in characters.")
    summary: str = pydantic.Field(description="The result or outcome in words.")
    kept_messages: int = pydantic.Field(
        description="Number of recent messages kept after the summary.",
    )


class LoopCompactSummariseFailed(Event):
    """The conversation-summary call failed."""

    TYPE: ClassVar[str] = "loop.compact.summarise.failed"
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopCompactThinkingDropped(Event):
    """Reasoning text was removed from context."""

    TYPE: ClassVar[str] = "loop.compact.thinking_dropped"
    messages: int = pydantic.Field(description="Number of conversation messages.")
    chars: int = pydantic.Field(description="Text length in characters.")


class LoopDecisionRecorded(Event):
    """An operator ruling was recorded in decisions.md."""

    TYPE: ClassVar[str] = "loop.decision.recorded"
    question: str = pydantic.Field(description="The question presented to the operator.")
    answer: str = pydantic.Field(description="The operator's answer.")


class LoopDecisionUnrecorded(Event):
    """An operator ruling could not be recorded or was missing at finish."""

    TYPE: ClassVar[str] = "loop.decision.unrecorded"
    error: str | None = pydantic.Field(default=None, description="The failure diagnostic.")
    missing: int | None = pydantic.Field(
        default=None,
        description="Number of operator rulings missing at finish.",
    )


class LoopEditLeftOut(Event):
    """A checkpoint left ignored or untracked edits outside its commit."""

    TYPE: ClassVar[str] = "loop.edit.left_out"
    turn: int = pydantic.Field(description="The execution's turn number.")
    ignored: Sequence[Any] = pydantic.Field(
        description="Ignored edits left outside the checkpoint.",
    )
    untracked: Sequence[Any] = pydantic.Field(
        description="Untracked edits left outside the checkpoint.",
    )


class LoopFinishContractRefused(Event):
    """The finish payload did not satisfy the machine's result contract."""

    TYPE: ClassVar[str] = "loop.finish_contract.refused"
    turn: int = pydantic.Field(description="The execution's turn number.")
    problems: Sequence[Any] = pydantic.Field(
        description="Violations of the machine's finish contract.",
    )


class LoopLoopGuardTriggered(Event):
    """Repeated identical tool calls triggered a warning."""

    TYPE: ClassVar[str] = "loop.loop_guard.triggered"
    turn: int = pydantic.Field(description="The execution's turn number.")
    tool: str = pydantic.Field(description="The repeated tool's name.")
    streak: int = pydantic.Field(description="Number of consecutive matching turns or calls.")


class LoopMediaRefused(Event):
    """An attachment was refused by its tool or model route."""

    TYPE: ClassVar[str] = "loop.media.refused"
    provider_tool_call_id: str = pydantic.Field(description="The provider's tool-use identifier.")
    reason: str = pydantic.Field(description="The reason for the outcome.")
    route: str | None = pydantic.Field(
        default=None,
        description="The model route that refused the attachment.",
    )
    text: str | None = pydantic.Field(default=None, description="The event's text.")
    attachment: dict[str, Any] | None = pydantic.Field(
        default=None,
        description="Attachment metadata without its binary content.",
    )


class LoopMemoryFlipNudged(Event):
    """Verification turned green and the model was asked to record the lesson."""

    TYPE: ClassVar[str] = "loop.memory_flip.nudged"
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopMetricAutoCall(Event):
    """An automatic metric sample started."""

    TYPE: ClassVar[str] = "loop.metric.auto_call"
    turn: int = pydantic.Field(description="The execution's turn number.")
    sha: str = pydantic.Field(description="The commit identifier, possibly abbreviated.")


class LoopMetricAutoFailed(Event):
    """An automatic metric sample failed."""

    TYPE: ClassVar[str] = "loop.metric.auto_failed"
    turn: int = pydantic.Field(description="The execution's turn number.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopMetricSample(Event):
    """A metric sample was recorded."""

    TYPE: ClassVar[str] = "loop.metric.sample"
    turn: int = pydantic.Field(description="The execution's turn number.")
    label: str = pydantic.Field(description="Metric sample label.")
    score: float | None = pydantic.Field(
        default=None,
        description="Parsed metric score; null when unavailable.",
    )
    returncode: int | None = pydantic.Field(
        default=None,
        description="Command exit code; null when unavailable.",
    )
    sha: str = pydantic.Field(description="The commit identifier, possibly abbreviated.")


class LoopMetricCeilingStop(Event):
    """The metric ceiling ended the run."""

    TYPE: ClassVar[str] = "loop.metric_ceiling.stop"
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopMetricEarlyFinishRejected(Event):
    """A metric run's early finish was returned to the model."""

    TYPE: ClassVar[str] = "loop.metric_early_finish.rejected"
    turn: int = pydantic.Field(description="The execution's turn number.")
    nudges_used: int = pydantic.Field(description="Number of notices issued for this condition.")
    budget_remaining: float | None = pydantic.Field(
        default=None,
        description="Remaining budget fraction; null without a finite cap.",
    )
    trigger: str | None = pydantic.Field(
        default=None,
        description="The action that triggered the gate or panel.",
    )


class LoopMetricPlateauNudge(Event):
    """A metric plateau prompted another attempt."""

    TYPE: ClassVar[str] = "loop.metric_plateau.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    nudges_used: int = pydantic.Field(description="Number of notices issued for this condition.")
    budget_remaining: float | None = pydantic.Field(
        default=None,
        description="Remaining budget fraction; null without a finite cap.",
    )


class LoopNoProgressNudge(Event):
    """Repeated turns without progress prompted a warning."""

    TYPE: ClassVar[str] = "loop.no_progress.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    streak: int = pydantic.Field(description="Number of consecutive matching turns or calls.")
    level: int = pydantic.Field(description="The warning's escalation level.")


class LoopNote(Event):
    """A machine-creation progress note."""

    TYPE: ClassVar[str] = "loop.note"
    text: str = pydantic.Field(description="The event's text.")


class LoopOperatorPresence(Event):
    """The operator's observed attachment state changed."""

    TYPE: ClassVar[str] = "loop.operator.presence"
    presence: str = pydantic.Field(description="The observed operator attachment and away state.")


class LoopParallelCompared(Event):
    """Parallel candidates were ranked."""

    TYPE: ClassVar[str] = "loop.parallel.compared"
    group: str = pydantic.Field(description="Parallel dispatch group identifier.")
    ranked_by: str = pydantic.Field(description="The method used to rank the candidates.")
    rationale: str = pydantic.Field(description="The ranking explanation.")
    ranking: Sequence[Any] = pydantic.Field(
        description="Candidates in ranked order with their verify and cost readings.",
    )
    judge_cost_usd: float = pydantic.Field(description="Ranking judge spend in USD.")
    judge_cost_partial: bool = pydantic.Field(
        description="Whether the judge spend is a lower bound.",
    )


class LoopParallelDispatched(Event):
    """Parallel lanes were dispatched."""

    TYPE: ClassVar[str] = "loop.parallel.dispatched"
    group: str = pydantic.Field(description="Parallel dispatch group identifier.")
    lanes: int = pydantic.Field(
        description="Lane count for dispatch, or lane results for completion.",
    )
    tasks: Sequence[Any] = pydantic.Field(description="Task texts sent to the lanes.")


class LoopParallelFailed(Event):
    """Parallel lane dispatch or execution failed."""

    TYPE: ClassVar[str] = "loop.parallel.failed"
    group: str = pydantic.Field(description="Parallel dispatch group identifier.")
    fanout: bool | None = pydantic.Field(
        default=None,
        description="Whether the failed group was a top-level fan-out.",
    )
    lanes: Sequence[Any] | None = pydantic.Field(
        default=None,
        description="Lane count for dispatch, or lane results for completion.",
    )
    error: str | None = pydantic.Field(default=None, description="The failure diagnostic.")


class LoopParallelJoined(Event):
    """Parallel lane results returned."""

    TYPE: ClassVar[str] = "loop.parallel.joined"
    group: str = pydantic.Field(description="Parallel dispatch group identifier.")
    lanes: Sequence[Any] = pydantic.Field(
        description="Lane count for dispatch, or lane results for completion.",
    )


class LoopParked(Event):
    """A standing run parked at a turn boundary."""

    TYPE: ClassVar[str] = "loop.parked"
    turn: int = pydantic.Field(description="The execution's turn number.")
    reason: str = pydantic.Field(description="The reason for the outcome.")


class LoopParkedResumed(Event):
    """A parked run received work or an operator request."""

    TYPE: ClassVar[str] = "loop.parked.resumed"
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopPinAdded(Event):
    """An operator instruction was pinned in context."""

    TYPE: ClassVar[str] = "loop.pin.added"
    text: str = pydantic.Field(description="The event's text.")
    chars: int = pydantic.Field(description="Text length in characters.")
    count: int = pydantic.Field(description="Total pinned instructions.")


class LoopPinRefused(Event):
    """An instruction exceeded the pinned-context limit."""

    TYPE: ClassVar[str] = "loop.pin.refused"
    chars: int = pydantic.Field(description="Text length in characters.")
    limit: int = pydantic.Field(description="Pinned instruction character limit.")


class LoopPinRestored(Event):
    """A resumed or forked execution restored pinned instructions."""

    TYPE: ClassVar[str] = "loop.pin.restored"
    pins: Sequence[Any] = pydantic.Field(description="The full list of pinned instructions.")
    count: int = pydantic.Field(description="Total pinned instructions.")


class LoopPlanFinishNudge(Event):
    """The model was asked to finish its plan within the remaining budget."""

    TYPE: ClassVar[str] = "loop.plan_finish.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    budget_remaining: float | None = pydantic.Field(
        default=None,
        description="Remaining budget fraction; null without a finite cap.",
    )


class LoopPlanReadFailed(Event):
    """The plan could not be read."""

    TYPE: ClassVar[str] = "loop.plan_read.failed"
    path: str = pydantic.Field(description="The plan's file path.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopPlanReread(Event):
    """The plan was read back into context."""

    TYPE: ClassVar[str] = "loop.plan_reread"
    path: str = pydantic.Field(description="The plan's file path.")
    bytes: int = pydantic.Field(description="UTF-8 text length in bytes.")


class LoopPlanWriteFailed(Event):
    """The plan could not be written."""

    TYPE: ClassVar[str] = "loop.plan_write.failed"
    path: str = pydantic.Field(description="The plan's file path.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopPlanWritten(Event):
    """The plan was written to disk."""

    TYPE: ClassVar[str] = "loop.plan_written"
    path: str = pydantic.Field(description="The plan's file path.")
    bytes: int = pydantic.Field(description="UTF-8 text length in bytes.")


class LoopProviderEmptyToolCallRetry(Event):
    """A provider response with empty tool calls was retried."""

    TYPE: ClassVar[str] = "loop.provider.empty_tool_call_retry"
    attempt: int = pydantic.Field(description="Retry attempt number.")
    stop_reason: str = pydantic.Field(description="The provider's response stop reason.")


class LoopProviderFatal(Event):
    """A non-retryable provider error stopped the call."""

    TYPE: ClassVar[str] = "loop.provider.fatal"
    status_code: int | None = pydantic.Field(
        default=None,
        description="HTTP status code when available.",
    )
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopProviderRetry(Event):
    """A provider error prompted another attempt."""

    TYPE: ClassVar[str] = "loop.provider.retry"
    attempt: int = pydantic.Field(description="Retry attempt number.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopQuestionDelivered(Event):
    """Queued answers were delivered to the model."""

    TYPE: ClassVar[str] = "loop.question.delivered"
    ids: Sequence[Any] = pydantic.Field(description="Identifiers of the delivered questions.")


class LoopQuestionNudge(Event):
    """A silent finish ended on a question and was returned to the model."""

    TYPE: ClassVar[str] = "loop.question_nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopReasoningStarvation(Event):
    """Reasoning consumed a response without usable output."""

    TYPE: ClassVar[str] = "loop.reasoning_starvation"
    turn: int = pydantic.Field(description="The execution's turn number.")
    reasoning_chars: int = pydantic.Field(description="Reasoning output length in characters.")
    output_tokens: int = pydantic.Field(description="Output token count.")
    stop_reason: str | None = pydantic.Field(
        default=None,
        description="The provider's response stop reason.",
    )


class LoopRequestRefused(Event):
    """An operator task-graph request could not be applied."""

    TYPE: ClassVar[str] = "loop.request.refused"
    kind: str = pydantic.Field(description="The request or authentication operation kind.")
    text: str = pydantic.Field(description="The event's text.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopResumeInterrupted(Event):
    """Resume found tool calls that never returned and will not be rerun."""

    TYPE: ClassVar[str] = "loop.resume.interrupted"
    turn: int = pydantic.Field(description="The execution's turn number.")
    tools: Sequence[Any] = pydantic.Field(
        description="Interrupted tool calls that will not be rerun.",
    )


class LoopReviewFindingReopened(Event):
    """An operator reopened a finding."""

    TYPE: ClassVar[str] = "loop.review.finding.reopened"
    finding_id: str = pydantic.Field(description="Finding identifier.")
    request_id: str = pydantic.Field(
        description="Request identifier used to correlate or deduplicate work.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    version: int = pydantic.Field(description="Review-state version after the action.")
    reason: str = pydantic.Field(description="The reason for the outcome.")


class LoopReviewFindingResolved(Event):
    """An operator recorded a finding's disposition."""

    TYPE: ClassVar[str] = "loop.review.finding.resolved"
    finding_id: str = pydantic.Field(description="Finding identifier.")
    request_id: str = pydantic.Field(
        description="Request identifier used to correlate or deduplicate work.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    version: int = pydantic.Field(description="Review-state version after the action.")
    resolution: dict[str, Any] = pydantic.Field(
        description="The finding's disposition and supporting reason.",
    )


class LoopReviewFixFailed(Event):
    """A finding's fix request could not be delivered."""

    TYPE: ClassVar[str] = "loop.review.fix.failed"
    finding_id: str = pydantic.Field(description="Finding identifier.")
    request_id: str = pydantic.Field(
        description="Request identifier used to correlate or deduplicate work.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    version: int = pydantic.Field(description="Review-state version after the action.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class LoopReviewFixRequested(Event):
    """An operator requested a fix for a finding."""

    TYPE: ClassVar[str] = "loop.review.fix.requested"
    finding_id: str = pydantic.Field(description="Finding identifier.")
    request_id: str = pydantic.Field(
        description="Request identifier used to correlate or deduplicate work.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    version: int = pydantic.Field(description="Review-state version after the action.")
    target: dict[str, Any] = pydantic.Field(
        description="The immutable review target, or provider request class.",
    )
    action: str = pydantic.Field(description="Fix delivery action: live-steer or resume.")


class LoopReviewFixSubmitted(Event):
    """A finding's fix request reached an execution."""

    TYPE: ClassVar[str] = "loop.review.fix.submitted"
    finding_id: str = pydantic.Field(description="Finding identifier.")
    request_id: str = pydantic.Field(
        description="Request identifier used to correlate or deduplicate work.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    version: int = pydantic.Field(description="Review-state version after the action.")
    execution_id: str = pydantic.Field(description="Execution identifier.")


class LoopReviewPanel(Event):
    """A review panel completed and recorded its evidence and blocking decision."""

    TYPE: ClassVar[str] = "loop.review.panel"
    panel_id: str = pydantic.Field(description="Review panel identifier.")
    execution_id: str = pydantic.Field(description="Execution identifier.")
    turn: int = pydantic.Field(description="The execution's turn number.")
    trigger: str = pydantic.Field(description="The action that triggered the gate or panel.")
    target: dict[str, Any] = pydantic.Field(
        description="The immutable review target, or provider request class.",
    )
    findings: Sequence[Any] = pydantic.Field(description="Grounded findings or panel observations.")
    injected_finding_ids: Sequence[Any] = pydantic.Field(
        description="Finding identifiers supplied to the worker.",
    )
    block_on: str = pydantic.Field(description="The configured panel blocking rule.")
    roster: Sequence[Any] | None = pydantic.Field(
        default=None,
        description="The seats that sat: index, model route, focus, scope and temperature.",
    )
    raw_blocked: bool = pydantic.Field(description="Whether the verdicts meet the blocking rule.")
    blocked: bool = pydantic.Field(description="Whether the armed gate blocks the finish.")
    disarmed: bool = pydantic.Field(
        description="Whether the review gate's rejection cap disarmed it.",
    )
    n_block: int = pydantic.Field(description="Number of distinct blocking provider/model routes.")
    n_abstain: int = pydantic.Field(description="Number of abstaining seats.")
    completion: str = pydantic.Field(description="Whether the panel completed or was inconclusive.")


class LoopReviewRejectedFinish(Event):
    """The review panel rejected the declared finish."""

    TYPE: ClassVar[str] = "loop.review.rejected_finish"
    turn: int = pydantic.Field(description="The execution's turn number.")
    ending: str = pydantic.Field(description="The declared finish being reviewed.")


class LoopReviewRejectionCapReached(Event):
    """The review gate reached its rejection limit."""

    TYPE: ClassVar[str] = "loop.review.rejection_cap_reached"
    turn: int = pydantic.Field(description="The execution's turn number.")
    rejections: int = pydantic.Field(description="Number of review rejections.")


class LoopReviewReused(Event):
    """A complete panel over the same trees, paths, seats and rule stood in for a new one."""

    TYPE: ClassVar[str] = "loop.review.reused"
    panel_id: str = pydantic.Field(description="Review panel identifier.")
    execution_id: str = pydantic.Field(description="Execution identifier.")
    turn: int = pydantic.Field(description="The execution's turn number.")
    trigger: str = pydantic.Field(description="The action that triggered the gate or panel.")
    reused_panel_id: str = pydantic.Field(description="The complete panel whose verdict stood in.")
    target: dict[str, Any] = pydantic.Field(
        description="The immutable review target both panels share.",
    )
    findings: int = pydantic.Field(description="How many findings the reused panel recorded.")


class LoopReviewSeat(Event):
    """A review seat returned a grounded verdict or abstained."""

    TYPE: ClassVar[str] = "loop.review.seat"
    panel_id: str = pydantic.Field(description="Review panel identifier.")
    execution_id: str = pydantic.Field(description="Execution identifier.")
    turn: int | None = pydantic.Field(default=None, description="The execution's turn number.")
    trigger: str | None = pydantic.Field(
        default=None,
        description="The action that triggered the gate or panel.",
    )
    seat_index: int = pydantic.Field(description="Zero-based review seat index.")
    focus: str = pydantic.Field(description="The requested summary or review focus.")
    model_route: str = pydantic.Field(description="The seat's configured model route.")
    verdict: str = pydantic.Field(description="Seat verdict: pass, block or abstain.")
    findings: Sequence[Any] = pydantic.Field(description="Grounded findings or panel observations.")
    summary: str = pydantic.Field(description="The result or outcome in words.")
    error: str | None = pydantic.Field(default=None, description="The failure diagnostic.")
    risk: str = pydantic.Field(description="The seat's stated risk.")
    risk_truncated: bool = pydantic.Field(description="Whether the risk text was truncated.")


class LoopReviewSkipped(Event):
    """A review panel could not run."""

    TYPE: ClassVar[str] = "loop.review.skipped"
    panel_id: str = pydantic.Field(description="Review panel identifier.")
    execution_id: str = pydantic.Field(description="Execution identifier.")
    turn: int | None = pydantic.Field(default=None, description="The execution's turn number.")
    trigger: str | None = pydantic.Field(
        default=None,
        description="The action that triggered the gate or panel.",
    )
    reason: str = pydantic.Field(description="The reason for the outcome.")
    error: str | None = pydantic.Field(default=None, description="The failure diagnostic.")
    remaining: float | None = pydantic.Field(default=None, description="Remaining budget fraction.")


class LoopReviewStart(Event):
    """A review panel began over an immutable target."""

    TYPE: ClassVar[str] = "loop.review.start"
    panel_id: str = pydantic.Field(description="Review panel identifier.")
    execution_id: str = pydantic.Field(description="Execution identifier.")
    turn: int = pydantic.Field(description="The execution's turn number.")
    trigger: str = pydantic.Field(description="The action that triggered the gate or panel.")
    target: dict[str, Any] = pydantic.Field(
        description="The immutable review target, or provider request class.",
    )
    rule_sources: Sequence[Any] = pydantic.Field(
        description="Base-tree rule files supplied to the panel.",
    )
    rules_complete: bool = pydantic.Field(description="Whether every required rule was supplied.")
    rules_reason: str | None = pydantic.Field(
        default=None,
        description="Why rule collection was incomplete; null when complete.",
    )
    seats: Sequence[Any] = pydantic.Field(description="The resolved review seat configurations.")


class LoopRunBudgetNudge(Event):
    """The model was asked to finish within its remaining run budget."""

    TYPE: ClassVar[str] = "loop.run_budget.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    budget_remaining: float | None = pydantic.Field(
        default=None,
        description="Remaining budget fraction; null without a finite cap.",
    )


class LoopSandboxToolUnreachable(Event):
    """A tool found on the host could not run inside the sandbox."""

    TYPE: ClassVar[str] = "loop.sandbox_tool_unreachable"
    binary: str = pydantic.Field(description="The unreachable executable's name.")


class LoopSilentNoWorkNudge(Event):
    """A silent finish without work was returned to the model."""

    TYPE: ClassVar[str] = "loop.silent_no_work.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    nudges_used: int = pydantic.Field(description="Number of notices issued for this condition.")


class LoopSkillsWarning(Event):
    """A skill declaration produced a warning."""

    TYPE: ClassVar[str] = "loop.skills.warning"
    warning: str = pydantic.Field(description="The skill diagnostic.")


class LoopStagnationNudged(Event):
    """Elapsed time without edits or verification prompted a warning."""

    TYPE: ClassVar[str] = "loop.stagnation.nudged"
    turn: int = pydantic.Field(description="The execution's turn number.")
    elapsed_s: int = pydantic.Field(description="Elapsed seconds without edits or verification.")


class LoopStandingResumed(Event):
    """A standing goal resumed after absorbing a soft stop."""

    TYPE: ClassVar[str] = "loop.standing.resumed"
    reason: str = pydantic.Field(description="The reason for the outcome.")
    task_id: str = pydantic.Field(description="Task identifier.")
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopStandingSet(Event):
    """The operator set a standing goal."""

    TYPE: ClassVar[str] = "loop.standing.set"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    title: str = pydantic.Field(description="Task or standing-goal text.")


class LoopSteerAborted(Event):
    """The operator's steer aborted the run."""

    TYPE: ClassVar[str] = "loop.steer.aborted"


class LoopSteerDetached(Event):
    """The operator's steer stopped the run for background resume."""

    TYPE: ClassVar[str] = "loop.steer.detached"


class LoopSteerExited(Event):
    """The operator's steer ended the run and left the terminal."""

    TYPE: ClassVar[str] = "loop.steer.exited"


class LoopSteerFeedback(Event):
    """A steer directive returned feedback without invoking the model."""

    TYPE: ClassVar[str] = "loop.steer.feedback"
    text: str = pydantic.Field(description="The feedback for the operator.")


class LoopSteerInjected(Event):
    """An operator instruction was injected into context."""

    TYPE: ClassVar[str] = "loop.steer.injected"
    chars: int = pydantic.Field(description="Text length in characters.")
    text: str = pydantic.Field(description="The event's text.")


class LoopSteerRequested(Event):
    """The loop reached an operator steer request."""

    TYPE: ClassVar[str] = "loop.steer.requested"
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopSteerSkill(Event):
    """An operator steer invoked a skill."""

    TYPE: ClassVar[str] = "loop.steer.skill"
    name: str = pydantic.Field(description="Tool or skill name.")
    args: str = pydantic.Field(
        description="Tool argument preview, or the skill invocation's argument text.",
    )


class LoopSteerUndo(Event):
    """The operator's steer requested an undo fork."""

    TYPE: ClassVar[str] = "loop.steer.undo"


class LoopTaskQueued(Event):
    """The operator queued a task."""

    TYPE: ClassVar[str] = "loop.task.queued"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    title: str = pydantic.Field(description="Task or standing-goal text.")


class LoopTaskRetired(Event):
    """The operator retired a task."""

    TYPE: ClassVar[str] = "loop.task.retired"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    title: str = pydantic.Field(description="Task or standing-goal text.")


class LoopTaskStuckNudge(Event):
    """Time on one task prompted a change of approach."""

    TYPE: ClassVar[str] = "loop.task.stuck_nudge"
    task_id: str = pydantic.Field(description="Task identifier.")
    turns: int = pydantic.Field(
        description="Number of execution turns, or turns spent on one task.",
    )
    n: int = pydantic.Field(description="Number of affected entries or notices.")


class LoopTaskSurfaced(Event):
    """A task became the current focus."""

    TYPE: ClassVar[str] = "loop.task.surfaced"
    task_id: str = pydantic.Field(description="Task identifier.")


class LoopTaskFinishGated(Event):
    """A finish with open subtasks was returned to the model."""

    TYPE: ClassVar[str] = "loop.task_finish.gated"
    turn: int = pydantic.Field(description="The execution's turn number.")
    nudges_used: int = pydantic.Field(description="Number of notices issued for this condition.")
    trigger: str = pydantic.Field(description="The action that triggered the gate or panel.")


class LoopTestOnlyGreenNotice(Event):
    """Verification passed after changes confined to tests."""

    TYPE: ClassVar[str] = "loop.test_only_green.notice"
    turn: int = pydantic.Field(description="The execution's turn number.")
    paths: Sequence[Any] = pydantic.Field(description="Affected repository-relative paths.")


class LoopToolCall(Event):
    """The loop began dispatching a tool call."""

    TYPE: ClassVar[str] = "loop.tool.call"
    name: str = pydantic.Field(description="Tool or skill name.")
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopToolErrorNudge(Event):
    """Repeated tool errors prompted a warning."""

    TYPE: ClassVar[str] = "loop.tool_error.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    streak: int = pydantic.Field(description="Number of consecutive matching turns or calls.")
    level: int = pydantic.Field(description="The warning's escalation level.")


class LoopVerifyBrokenNudge(Event):
    """An unusable verify command prompted a correction request."""

    TYPE: ClassVar[str] = "loop.verify_broken.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopVerifyFinishGated(Event):
    """A finish without green verification was returned to the model."""

    TYPE: ClassVar[str] = "loop.verify_finish.gated"
    turn: int = pydantic.Field(description="The execution's turn number.")
    nudges_used: int = pydantic.Field(description="Number of notices issued for this condition.")
    trigger: str = pydantic.Field(description="The action that triggered the gate or panel.")


class LoopVerifyHarness(Event):
    """The harness requested verification."""

    TYPE: ClassVar[str] = "loop.verify_harness"
    why: str = pydantic.Field(description="Why the harness requested verification.")
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopVerifyInferred(Event):
    """The verify command was inferred, restored, adopted or disabled."""

    TYPE: ClassVar[str] = "loop.verify_inferred"
    command: Sequence[Any] = pydantic.Field(
        description="Resolved verify argv; empty when no command is adopted.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    adopted_at: int | None = pydantic.Field(
        default=None,
        description="Turn where the gate was adopted, restored or dropped.",
    )


class LoopVerifyScoped(Event):
    """Verification was narrowed to tests near the changed files."""

    TYPE: ClassVar[str] = "loop.verify_scoped"
    paths: Sequence[Any] = pydantic.Field(description="Affected repository-relative paths.")
    turn: int = pydantic.Field(description="The execution's turn number.")


class LoopVerifySettledNudge(Event):
    """Repeated green verification without edits prompted a finish request."""

    TYPE: ClassVar[str] = "loop.verify_settled.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    idle: int = pydantic.Field(description="Consecutive settled turns.")


class LoopVerifySwapped(Event):
    """Resume replaced the saved verify command."""

    TYPE: ClassVar[str] = "loop.verify_swapped"
    was: Sequence[Any] = pydantic.Field(description="Saved verification argv.")
    now: Sequence[Any] = pydantic.Field(description="Replacement verification argv.")


class LoopWentQuietNudge(Event):
    """A response without tool calls prompted the model to continue."""

    TYPE: ClassVar[str] = "loop.went_quiet.nudge"
    turn: int = pydantic.Field(description="The execution's turn number.")
    nudges_used: int = pydantic.Field(description="Number of notices issued for this condition.")
    nudges_max: int = pydantic.Field(description="Maximum notices allowed for this condition.")
    output_tokens: int = pydantic.Field(description="Output token count.")


class McpServerUnavailable(Event):
    """A configured MCP server could not start."""

    TYPE: ClassVar[str] = "mcp.server_unavailable"
    server: str = pydantic.Field(description="Configured MCP server name.")
    error: str = pydantic.Field(description="The failure diagnostic.")


class MetricEnd(Event):
    """The metric command finished."""

    TYPE: ClassVar[str] = "metric.end"
    cmd: Sequence[Any] = pydantic.Field(description="Command argv.")
    exit_code: int | None = pydantic.Field(
        default=None,
        description="Command exit code; null if no exit was observed.",
    )
    duration_s: float = pydantic.Field(description="Command duration in seconds.")
    stdout_tail: str = pydantic.Field(description="Capped standard-output tail.")
    stderr_tail: str = pydantic.Field(description="Capped standard-error tail.")
    score: float | None = pydantic.Field(
        default=None,
        description="Parsed metric score; null when unavailable.",
    )


class MetricStart(Event):
    """The metric command started."""

    TYPE: ClassVar[str] = "metric.start"
    cmd: Sequence[Any] = pydantic.Field(description="Command argv.")


class ProviderAuth(Event):
    """A provider credential refresh, exchange or recovery finished."""

    TYPE: ClassVar[str] = "provider.auth"
    provider: str = pydantic.Field(description="Configured provider name.")
    kind: str = pydantic.Field(description="The request or authentication operation kind.")
    outcome: str = pydantic.Field(description="The operation's outcome.")


class ProviderRequest(Event):
    """An HTTP attempt was made for a model call."""

    TYPE: ClassVar[str] = "provider.request"
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The execution the call ran in; null for a call outside one.",
    )
    role: str = pydantic.Field(description="The calling model role.")
    provider: str = pydantic.Field(description="Configured provider name.")
    model: str = pydantic.Field(description="Model identifier.")
    model_call_id: str = pydantic.Field(
        description="The logical model call that made this request.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    seat_index: int | None = pydantic.Field(
        default=None,
        description="Zero-based review seat index.",
    )
    request_id: str = pydantic.Field(
        description="HTTP attempt identifier, also stored at the transcript's top level.",
    )
    target: str = pydantic.Field(
        description="The immutable review target, or provider request class.",
    )


class QuestionAnswer(Event):
    """The operator answered questions or nobody could answer."""

    TYPE: ClassVar[str] = "question.answer"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    answers: Sequence[Any] = pydantic.Field(
        description="Answers aligned with the requested questions.",
    )
    source: str = pydantic.Field(description="Where the request or result came from.")
    unseen: bool = pydantic.Field(
        description="Whether nobody could answer and no question remains queued.",
    )


class QuestionPrompt(Event):
    """Questions were presented to the operator."""

    TYPE: ClassVar[str] = "question.prompt"
    id: str = pydantic.Field(description="Prompt, task or background-job identifier.")
    questions: Sequence[Any] = pydantic.Field(description="The questions presented.")
    call_id: int | None = pydantic.Field(
        default=None,
        description="The tool call correlated with this event.",
    )
    wait: bool = pydantic.Field(description="Whether the call waits for answers.")


class RoleAttempt(Event):
    """One request attempt of a model call ended; the ending attempt's role.result follows."""

    TYPE: ClassVar[str] = "role.attempt"
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The execution the call ran in; null for a call outside one.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    role: str = pydantic.Field(description="The calling model role.")
    model: str = pydantic.Field(description="Model identifier.")
    provider: str = pydantic.Field(description="Configured provider name.")
    model_call_id: str = pydantic.Field(
        description="Identifier pairing the attempt with its model call.",
    )
    seat_index: int | None = pydantic.Field(
        default=None,
        description="Zero-based review seat index.",
    )
    attempt: int = pydantic.Field(description="The attempt's number within the call, from 1.")
    request_id: str | None = pydantic.Field(
        default=None,
        description="The HTTP attempt's id, as provider.request and the transcript name it.",
    )
    ok: bool = pydantic.Field(description="Whether the attempt returned a response.")
    tokens_in: int | None = pydantic.Field(
        default=None,
        description="Input tokens the attempt used.",
    )
    tokens_out: int | None = pydantic.Field(
        default=None,
        description="Output tokens the attempt used.",
    )
    cache_read: int | None = pydantic.Field(
        default=None,
        description="Cached input tokens the attempt used.",
    )
    cache_creation: int | None = pydantic.Field(
        default=None,
        description="Cache-creation tokens the attempt used.",
    )
    cost_usd: float | None = pydantic.Field(
        default=None,
        description="Provider-reported USD cost; zero means free, absent means unreported.",
    )
    stop_reason: str | None = pydantic.Field(
        default=None,
        description="The provider's response stop reason.",
    )
    error: str | None = pydantic.Field(default=None, description="The failure diagnostic.")


class RoleCall(Event):
    """A model call started for a role."""

    TYPE: ClassVar[str] = "role.call"
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The execution the call ran in; null for a call outside one.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    role: str = pydantic.Field(description="The calling model role.")
    model: str = pydantic.Field(description="Model identifier.")
    provider: str = pydantic.Field(description="Configured provider name.")
    model_call_id: str = pydantic.Field(
        description="Identifier pairing the model call with its result.",
    )
    seat_index: int | None = pydantic.Field(
        default=None,
        description="Zero-based review seat index.",
    )


class RoleResult(Event):
    """A model call returned or failed, once per logical call, with its last attempt's usage."""

    TYPE: ClassVar[str] = "role.result"
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The execution the call ran in; null for a call outside one.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    role: str = pydantic.Field(description="The calling model role.")
    model: str = pydantic.Field(description="Model identifier.")
    provider: str = pydantic.Field(description="Configured provider name.")
    model_call_id: str = pydantic.Field(
        description="Identifier pairing the model call with its result.",
    )
    seat_index: int | None = pydantic.Field(
        default=None,
        description="Zero-based review seat index.",
    )
    ok: bool = pydantic.Field(description="Whether the operation succeeded.")
    text: str | None = pydantic.Field(default=None, description="The event's text.")
    tokens_in: int | None = pydantic.Field(
        default=None,
        description="Input tokens used by this model call.",
    )
    tokens_out: int | None = pydantic.Field(
        default=None,
        description="Output tokens used by this model call.",
    )
    cache_read: int | None = pydantic.Field(
        default=None,
        description="Cached input tokens used by this model call.",
    )
    cache_creation: int | None = pydantic.Field(
        default=None,
        description="Cache-creation tokens used by this model call.",
    )
    cost_usd: float | None = pydantic.Field(
        default=None,
        description="Provider-reported USD cost; zero means free, absent means unreported.",
    )
    stop_reason: str | None = pydantic.Field(
        default=None,
        description="The provider's response stop reason.",
    )
    error: str | None = pydantic.Field(default=None, description="The failure diagnostic.")


class RoleTextDelta(Event):
    """A piece of assistant text streamed."""

    TYPE: ClassVar[str] = "role.text_delta"
    role: str = pydantic.Field(description="The calling model role.")
    text: str = pydantic.Field(description="The event's text.")
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The execution the call ran in; null for a call outside one.",
    )
    model_call_id: str | None = pydantic.Field(
        default=None,
        description="The model call streaming the piece; null in a journal written before ids.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    seat_index: int | None = pydantic.Field(
        default=None,
        description="The review seat's roster position; null outside a review seat.",
    )


class RoleThinkingDelta(Event):
    """A piece of assistant reasoning streamed."""

    TYPE: ClassVar[str] = "role.thinking_delta"
    role: str = pydantic.Field(description="The calling model role.")
    text: str = pydantic.Field(description="The event's text.")
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The execution the call ran in; null for a call outside one.",
    )
    model_call_id: str | None = pydantic.Field(
        default=None,
        description="The model call streaming the piece; null in a journal written before ids.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    seat_index: int | None = pydantic.Field(
        default=None,
        description="The review seat's roster position; null outside a review seat.",
    )


class SessionResume(Event):
    """Emitted at the start of every resumed execution."""

    TYPE: ClassVar[str] = "session.resume"
    session_id: str = pydantic.Field(description="Session identifier.")
    user_task: str = pydantic.Field(description="The displayed task text.")
    steer: str = pydantic.Field(description="The instruction supplied when resuming.")
    mode: str = pydantic.Field(description="The execution mode.")
    turn: int = pydantic.Field(description="The execution's turn number.")
    messages: int = pydantic.Field(description="Number of conversation messages.")
    execution_id: str = pydantic.Field(description="Execution identifier.")


class SessionStart(Event):
    """Emitted once when the session is created."""

    TYPE: ClassVar[str] = "session.start"
    session_id: str | None = pydantic.Field(default=None, description="Session identifier.")
    user_task: str = pydantic.Field(description="The displayed task text.")
    mode: str = pydantic.Field(description="The execution mode.")
    execution_id: str = pydantic.Field(description="Execution identifier.")


class SessionSteerRequested(Event):
    """Ctrl-C requested an operator steer during the run."""

    TYPE: ClassVar[str] = "session.steer_requested"
    source: str = pydantic.Field(description="Where the request or result came from.")


class SessionStop(Event):
    """The execution stopped, with its reason; the session stays resumable."""

    TYPE: ClassVar[str] = "session.stop"
    execution_id: str = pydantic.Field(description="Execution identifier; empty when unknown.")
    reason: str = pydantic.Field(description="The reason for the outcome.")
    turns: int = pydantic.Field(
        description="Number of execution turns, or turns spent on one task.",
    )
    summary: str | None = pydantic.Field(
        default=None,
        description="The result or outcome in words.",
    )
    verification: str | None = pydantic.Field(
        default=None,
        description="Final verification state: passed, failed, unverified or not_applicable.",
    )
    scoped: bool | None = pydantic.Field(
        default=None,
        description="Whether verification ran only the selected nearby tests.",
    )
    final_verify_id: str | None = pydantic.Field(
        default=None,
        description="The final verification attempt identifier.",
    )
    tree: str | None = pydantic.Field(
        default=None,
        description="The final working-tree content identifier.",
    )
    tool: str | None = pydantic.Field(default=None, description="The repeated tool's name.")
    streak: int | None = pydantic.Field(
        default=None,
        description="Number of consecutive matching turns or calls.",
    )


class SessionStranded(Event):
    """A finished run left edits uncommitted and no commit landed."""

    TYPE: ClassVar[str] = "session.stranded"


class SessionUndone(Event):
    """Undo forked the run before its last operator message."""

    TYPE: ClassVar[str] = "session.undone"
    new_session_id: str = pydantic.Field(description="Identifier of the undo fork.")
    undone_text: str = pydantic.Field(description="The operator message taken back.")


class ToolCall(Event):
    """A tool was dispatched, including calls that a guard rejects."""

    TYPE: ClassVar[str] = "tool.call"
    name: str = pydantic.Field(description="Tool or skill name.")
    args_preview: dict[str, Any] = pydantic.Field(
        description="The arguments with long values clipped, or a skill's argument text.",
    )
    call_id: int | None = pydantic.Field(
        default=None,
        description=(
            "The tool call correlated with this event; null in a journal written before ids."
        ),
    )
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The dispatching execution; null in a journal written before ids.",
    )
    model_call_id: str | None = pydantic.Field(
        default=None,
        description="The model call whose response made the call; null for a harness-run tool.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    seat_index: int | None = pydantic.Field(
        default=None,
        description="The review seat's roster position; null outside a review seat.",
    )
    provider_tool_call_id: str | None = pydantic.Field(
        default=None,
        description="The provider's tool-use id for the block; null for a harness-run tool.",
    )


class ToolResult(Event):
    """A dispatched tool returned its result or refusal."""

    TYPE: ClassVar[str] = "tool.result"
    name: str = pydantic.Field(description="Tool or skill name.")
    ok: bool = pydantic.Field(
        description="Whether the handler returned a result; a command's failure is its exit_code.",
    )
    summary: str = pydantic.Field(description="The result or outcome in words.")
    call_id: int | None = pydantic.Field(
        default=None,
        description=(
            "The tool call correlated with this event; null in a journal written before ids."
        ),
    )
    execution_id: str | None = pydantic.Field(
        default=None,
        description="The dispatching execution; null in a journal written before ids.",
    )
    model_call_id: str | None = pydantic.Field(
        default=None,
        description="The model call whose response made the call; null for a harness-run tool.",
    )
    panel_id: str | None = pydantic.Field(default=None, description="Review panel identifier.")
    seat_index: int | None = pydantic.Field(
        default=None,
        description="The review seat's roster position; null outside a review seat.",
    )
    provider_tool_call_id: str | None = pydantic.Field(
        default=None,
        description="The provider's tool-use id for the block; null for a harness-run tool.",
    )
    exit_code: int | None = pydantic.Field(
        default=None,
        description="A command's exit code; null while it runs in the background or when unknown.",
    )
    duration_s: float | None = pydantic.Field(
        default=None,
        description="A command's wall-clock seconds until the result.",
    )
    timed_out: bool | None = pydantic.Field(
        default=None,
        description="Whether the runner ended the command at its deadline.",
    )
    exec_failed: bool | None = pydantic.Field(
        default=None,
        description="Whether the command could not be started.",
    )
    background_id: str | None = pydantic.Field(
        default=None,
        description="The background job the command continues as, when handed back.",
    )
    paths: Sequence[Any] | None = pydantic.Field(
        default=None,
        description="Affected repository-relative paths.",
    )
    stdout_tail: str | None = pydantic.Field(
        default=None,
        description="Capped standard-output tail.",
    )
    stderr_tail: str | None = pydantic.Field(
        default=None,
        description="Capped standard-error tail.",
    )
    stdout_truncated: bool | None = pydantic.Field(
        default=None,
        description="Whether stdout_tail is a cut of the output; null when its end is unknown.",
    )
    stderr_truncated: bool | None = pydantic.Field(
        default=None,
        description="Whether stderr_tail is a cut of the output; null when its end is unknown.",
    )
    stopped_by_operator: bool | None = pydantic.Field(
        default=None,
        description="Whether an immediate operator stop ended the command.",
    )
    attachment: dict[str, Any] | None = pydantic.Field(
        default=None,
        description="Attachment metadata without its binary content.",
    )
    head_tail: str | None = pydantic.Field(
        default=None,
        description="A read preview or attachment summary.",
    )
    lines_total: int | None = pydantic.Field(
        default=None,
        description="Total lines in the file read.",
    )


class VerifyEnd(Event):
    """A verification attempt finished with evidence about the observed tree."""

    TYPE: ClassVar[str] = "verify.end"
    verify_id: str | None = pydantic.Field(
        default=None,
        description="Verification attempt identifier.",
    )
    execution_id: str | None = pydantic.Field(default=None, description="Execution identifier.")
    target_id: str | None = pydantic.Field(
        default=None,
        description="Recorded execution-target reference.",
    )
    cmd: Sequence[Any] = pydantic.Field(description="Command argv.")
    tree_before: str | None = pydantic.Field(
        default=None,
        description="Content identifier observed before verification.",
    )
    tree_after: str | None = pydantic.Field(
        default=None,
        description="Content identifier observed after verification.",
    )
    scoped: bool = pydantic.Field(
        description="Whether verification ran only the selected nearby tests.",
    )
    selected_paths: Sequence[Any] = pydantic.Field(
        description="Paths selected for scoped verification.",
    )
    outcome: str = pydantic.Field(description="The operation's outcome.")
    exit_code: int | None = pydantic.Field(
        default=None,
        description="Command exit code; null if no exit was observed.",
    )
    duration_s: float = pydantic.Field(description="Command duration in seconds.")
    timeout_s: float = pydantic.Field(description="Verification wall-clock limit in seconds.")
    stdout_tail: str = pydantic.Field(description="Capped standard-output tail.")
    stderr_tail: str = pydantic.Field(description="Capped standard-error tail.")
    stdout_truncated: bool | None = pydantic.Field(
        default=None,
        description="Whether standard-output capture was truncated; null when unknown.",
    )
    stderr_truncated: bool | None = pydantic.Field(
        default=None,
        description="Whether standard-error capture was truncated; null when unknown.",
    )
    baseline: bool = pydantic.Field(description="Whether the attempt measured the baseline tree.")
    reason: str = pydantic.Field(description="The reason for the outcome.")


class VerifyStart(Event):
    """A verification attempt started."""

    TYPE: ClassVar[str] = "verify.start"
    approval_required: bool | None = pydantic.Field(
        default=None,
        description="Whether the attempt needs approval.",
    )
    call_id: int | None = pydantic.Field(
        default=None,
        description="The model tool call, absent for a harness gate.",
    )
    verify_id: str | None = pydantic.Field(
        default=None,
        description="Verification attempt identifier.",
    )
    execution_id: str | None = pydantic.Field(default=None, description="Execution identifier.")
    target_id: str | None = pydantic.Field(
        default=None,
        description="Recorded execution-target reference.",
    )
    tree_before: str | None = pydantic.Field(
        default=None,
        description="Content identifier observed before verification.",
    )
    cmd: Sequence[Any] = pydantic.Field(description="Command argv.")
    scoped: bool = pydantic.Field(
        description="Whether verification ran only the selected nearby tests.",
    )
    selected_paths: Sequence[Any] = pydantic.Field(
        description="Paths selected for scoped verification.",
    )
    timeout_s: float = pydantic.Field(description="Verification wall-clock limit in seconds.")


class WorkspaceSetup(Event):
    """The workspace setup command finished."""

    TYPE: ClassVar[str] = "workspace.setup"
    execution_id: str | None = pydantic.Field(default=None, description="Execution identifier.")
    target_id: str | None = pydantic.Field(
        default=None,
        description="Recorded execution-target reference.",
    )
    cmd: Sequence[Any] = pydantic.Field(description="Command argv.")
    exit_code: int | None = pydantic.Field(
        default=None,
        description="Command exit code; null if no exit was observed.",
    )
    duration_s: float = pydantic.Field(description="Command duration in seconds.")
    exec_failed: bool = pydantic.Field(
        description="Whether the setup command could not be executed.",
    )
    stdout_tail: str = pydantic.Field(description="Capped standard-output tail.")
    stderr_tail: str = pydantic.Field(description="Capped standard-error tail.")


# Every record by its row type, in type order.
EVENT_TYPES: dict[str, type[Event]] = {
    "approval.answer": ApprovalAnswer,
    "approval.prompt": ApprovalPrompt,
    "btw.answered": BtwAnswered,
    "btw.opened": BtwOpened,
    "budget.update": BudgetUpdate,
    "command.backgrounded": CommandBackgrounded,
    "diff.updated": DiffUpdated,
    "execution.environment": ExecutionEnvironment,
    "execution.jobs": ExecutionJobs,
    "execution.target": ExecutionTarget,
    "graph.update": GraphUpdate,
    "jail.degraded": JailDegraded,
    "loop.auto_commit": LoopAutoCommit,
    "loop.auto_commit.failed": LoopAutoCommitFailed,
    "loop.baseline": LoopBaseline,
    "loop.budget": LoopBudget,
    "loop.compact.deduped": LoopCompactDeduped,
    "loop.compact.dropped": LoopCompactDropped,
    "loop.compact.gist.call": LoopCompactGistCall,
    "loop.compact.gist.failed": LoopCompactGistFailed,
    "loop.compact.gists": LoopCompactGists,
    "loop.compact.refused": LoopCompactRefused,
    "loop.compact.requested": LoopCompactRequested,
    "loop.compact.restored": LoopCompactRestored,
    "loop.compact.summarise.call": LoopCompactSummariseCall,
    "loop.compact.summarise.done": LoopCompactSummariseDone,
    "loop.compact.summarise.failed": LoopCompactSummariseFailed,
    "loop.compact.thinking_dropped": LoopCompactThinkingDropped,
    "loop.decision.recorded": LoopDecisionRecorded,
    "loop.decision.unrecorded": LoopDecisionUnrecorded,
    "loop.edit.left_out": LoopEditLeftOut,
    "loop.finish_contract.refused": LoopFinishContractRefused,
    "loop.loop_guard.triggered": LoopLoopGuardTriggered,
    "loop.media.refused": LoopMediaRefused,
    "loop.memory_flip.nudged": LoopMemoryFlipNudged,
    "loop.metric.auto_call": LoopMetricAutoCall,
    "loop.metric.auto_failed": LoopMetricAutoFailed,
    "loop.metric.sample": LoopMetricSample,
    "loop.metric_ceiling.stop": LoopMetricCeilingStop,
    "loop.metric_early_finish.rejected": LoopMetricEarlyFinishRejected,
    "loop.metric_plateau.nudge": LoopMetricPlateauNudge,
    "loop.no_progress.nudge": LoopNoProgressNudge,
    "loop.note": LoopNote,
    "loop.operator.presence": LoopOperatorPresence,
    "loop.parallel.compared": LoopParallelCompared,
    "loop.parallel.dispatched": LoopParallelDispatched,
    "loop.parallel.failed": LoopParallelFailed,
    "loop.parallel.joined": LoopParallelJoined,
    "loop.parked": LoopParked,
    "loop.parked.resumed": LoopParkedResumed,
    "loop.pin.added": LoopPinAdded,
    "loop.pin.refused": LoopPinRefused,
    "loop.pin.restored": LoopPinRestored,
    "loop.plan_finish.nudge": LoopPlanFinishNudge,
    "loop.plan_read.failed": LoopPlanReadFailed,
    "loop.plan_reread": LoopPlanReread,
    "loop.plan_write.failed": LoopPlanWriteFailed,
    "loop.plan_written": LoopPlanWritten,
    "loop.provider.empty_tool_call_retry": LoopProviderEmptyToolCallRetry,
    "loop.provider.fatal": LoopProviderFatal,
    "loop.provider.retry": LoopProviderRetry,
    "loop.question.delivered": LoopQuestionDelivered,
    "loop.question_nudge": LoopQuestionNudge,
    "loop.reasoning_starvation": LoopReasoningStarvation,
    "loop.request.refused": LoopRequestRefused,
    "loop.resume.interrupted": LoopResumeInterrupted,
    "loop.review.finding.reopened": LoopReviewFindingReopened,
    "loop.review.finding.resolved": LoopReviewFindingResolved,
    "loop.review.fix.failed": LoopReviewFixFailed,
    "loop.review.fix.requested": LoopReviewFixRequested,
    "loop.review.fix.submitted": LoopReviewFixSubmitted,
    "loop.review.panel": LoopReviewPanel,
    "loop.review.rejected_finish": LoopReviewRejectedFinish,
    "loop.review.rejection_cap_reached": LoopReviewRejectionCapReached,
    "loop.review.reused": LoopReviewReused,
    "loop.review.seat": LoopReviewSeat,
    "loop.review.skipped": LoopReviewSkipped,
    "loop.review.start": LoopReviewStart,
    "loop.run_budget.nudge": LoopRunBudgetNudge,
    "loop.sandbox_tool_unreachable": LoopSandboxToolUnreachable,
    "loop.silent_no_work.nudge": LoopSilentNoWorkNudge,
    "loop.skills.warning": LoopSkillsWarning,
    "loop.stagnation.nudged": LoopStagnationNudged,
    "loop.standing.resumed": LoopStandingResumed,
    "loop.standing.set": LoopStandingSet,
    "loop.steer.aborted": LoopSteerAborted,
    "loop.steer.detached": LoopSteerDetached,
    "loop.steer.exited": LoopSteerExited,
    "loop.steer.feedback": LoopSteerFeedback,
    "loop.steer.injected": LoopSteerInjected,
    "loop.steer.requested": LoopSteerRequested,
    "loop.steer.skill": LoopSteerSkill,
    "loop.steer.undo": LoopSteerUndo,
    "loop.task.queued": LoopTaskQueued,
    "loop.task.retired": LoopTaskRetired,
    "loop.task.stuck_nudge": LoopTaskStuckNudge,
    "loop.task.surfaced": LoopTaskSurfaced,
    "loop.task_finish.gated": LoopTaskFinishGated,
    "loop.test_only_green.notice": LoopTestOnlyGreenNotice,
    "loop.tool.call": LoopToolCall,
    "loop.tool_error.nudge": LoopToolErrorNudge,
    "loop.verify_broken.nudge": LoopVerifyBrokenNudge,
    "loop.verify_finish.gated": LoopVerifyFinishGated,
    "loop.verify_harness": LoopVerifyHarness,
    "loop.verify_inferred": LoopVerifyInferred,
    "loop.verify_scoped": LoopVerifyScoped,
    "loop.verify_settled.nudge": LoopVerifySettledNudge,
    "loop.verify_swapped": LoopVerifySwapped,
    "loop.went_quiet.nudge": LoopWentQuietNudge,
    "mcp.server_unavailable": McpServerUnavailable,
    "metric.end": MetricEnd,
    "metric.start": MetricStart,
    "provider.auth": ProviderAuth,
    "provider.request": ProviderRequest,
    "question.answer": QuestionAnswer,
    "question.prompt": QuestionPrompt,
    "role.attempt": RoleAttempt,
    "role.call": RoleCall,
    "role.result": RoleResult,
    "role.text_delta": RoleTextDelta,
    "role.thinking_delta": RoleThinkingDelta,
    "session.resume": SessionResume,
    "session.start": SessionStart,
    "session.steer_requested": SessionSteerRequested,
    "session.stop": SessionStop,
    "session.stranded": SessionStranded,
    "session.undone": SessionUndone,
    "tool.call": ToolCall,
    "tool.result": ToolResult,
    "verify.end": VerifyEnd,
    "verify.start": VerifyStart,
    "workspace.setup": WorkspaceSetup,
}

# A fresh run emits session.start; a resumed execution emits only session.resume.
SESSION_START_EVENTS = frozenset({SessionResume.TYPE, SessionStart.TYPE})
