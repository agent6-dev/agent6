"""Declare the journal's write vocabulary and the meanings used by its reference table.

Kinds describe JSON values: tuples serialize as lists and integers also satisfy float.
Optional fields may be absent or null. Required fields must be present and non-null.
The registry describes current writers, not the tolerant read-side history parser.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import Literal


@dataclasses.dataclass(frozen=True, slots=True)
class FieldSpec:
    """Describe a payload field; optional fields may be absent or null."""

    name: str
    kind: Literal["str", "int", "float", "bool", "list", "object"]
    required: bool
    meaning: str


@dataclasses.dataclass(frozen=True, slots=True)
class EventSpec:
    """Describe one journal event without its sink-owned envelope."""

    type: str
    meaning: str
    fields: tuple[FieldSpec, ...]


EVENTS: tuple[EventSpec, ...] = (
    EventSpec(
        "approval.answer",
        "The operator answered an approval request.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("approved", "bool", True, "Whether the operator approved the request."),
            FieldSpec("source", "str", True, "Where the request or result came from."),
        ),
    ),
    EventSpec(
        "approval.prompt",
        "A command is waiting for the operator's approval.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("prompt", "str", True, "The words presented to the operator."),
            FieldSpec(
                "standing", "bool", True, "Whether approval can cover later calls in this scope."
            ),
            FieldSpec("call_id", "int", False, "The tool call correlated with this event."),
        ),
    ),
    EventSpec(
        "btw.answered",
        "A side question received an answer.",
        (
            FieldSpec("btw_id", "str", True, "Side-question identifier."),
            FieldSpec("block", "str", True, "The rendered question and answer block."),
        ),
    ),
    EventSpec(
        "btw.opened",
        "The operator opened a side question.",
        (
            FieldSpec("btw_id", "str", True, "Side-question identifier."),
            FieldSpec("question", "str", True, "The question presented to the operator."),
        ),
    ),
    EventSpec(
        "budget.update",
        "The cumulative token, spend and subscription readings.",
        (
            FieldSpec(
                "usd_total",
                "float",
                True,
                "Cumulative USD spend across provider/model routes, including purchased credits.",
            ),
            FieldSpec("input_total", "int", False, "Cumulative input tokens."),
            FieldSpec("output_total", "int", False, "Cumulative output tokens."),
            FieldSpec("cache_read_total", "int", False, "Cumulative cached input tokens."),
            FieldSpec("cache_creation_total", "int", False, "Cumulative cache-creation tokens."),
            FieldSpec(
                "usd_partial", "bool", False, "Whether unpriced usage makes spend a lower bound."
            ),
            FieldSpec("usd_cap", "float", False, "Spend cap in USD; -1 means unlimited."),
            FieldSpec("tokens_unmetered", "int", False, "Tokens with no known price."),
            FieldSpec("tokens_fallback_cap", "int", False, "Token cap used for unpriced usage."),
            FieldSpec(
                "plan_used_percent",
                "float",
                False,
                "Provider-reported subscription usage percentage.",
            ),
            FieldSpec(
                "plan_consumed",
                "float",
                False,
                "The largest observed plan movement in points across providers and windows;"
                " account-wide, so includes other consumers.",
            ),
            FieldSpec("plan_cap", "float", False, "Subscription points cap."),
            FieldSpec(
                "plan_resets_at", "float", False, "Subscription window reset time in epoch seconds."
            ),
            FieldSpec(
                "plan_windows", "list", False, "Every provider-reported subscription window."
            ),
            FieldSpec("credits_has", "bool", False, "Whether the account has credits."),
            FieldSpec("credits_unlimited", "bool", False, "Whether credits are unlimited."),
            FieldSpec("credits_balance", "str", False, "Provider-reported credit balance."),
        ),
    ),
    EventSpec(
        "command.backgrounded",
        "A command was handed back as a background job.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("handle", "str", True, "Launcher-instance-scoped job handle."),
            FieldSpec("seconds", "float", True, "Elapsed command time in seconds at handback."),
        ),
    ),
    EventSpec(
        "diff.updated",
        "A chain commit landed with its patch.",
        (
            FieldSpec("sha", "str", True, "The commit identifier, possibly abbreviated."),
            FieldSpec("patch", "str", True, "The commit's patch, capped at 8000 bytes."),
        ),
    ),
    EventSpec(
        "execution.environment",
        "An environment lifecycle observation, independent of job settlement.",
        (
            FieldSpec("execution_id", "str", False, "Execution identifier."),
            FieldSpec("target_id", "str", False, "Recorded target reference."),
            FieldSpec("outcome", "str", True, "left_running, stopped, unknown or not_applicable."),
            FieldSpec("reason", "str", True, "The observation's reason, not stop authority."),
        ),
    ),
    EventSpec(
        "execution.jobs",
        "The command owner observed job settlement or could not confirm it.",
        (
            FieldSpec("execution_id", "str", False, "Execution identifier."),
            FieldSpec("target_id", "str", False, "Recorded target reference."),
            FieldSpec("outcome", "str", True, "settled or unknown."),
            FieldSpec("reason", "str", True, "Why settlement is unknown, empty when confirmed."),
        ),
    ),
    EventSpec(
        "execution.target",
        "Immutable target facts recorded before workspace commands.",
        (
            FieldSpec("execution_id", "str", False, "Execution identifier."),
            FieldSpec(
                "target_id", "str", True, "Host-issued evidence reference, not launcher identity."
            ),
            FieldSpec(
                "target", "object", True, "ExecutionTargetStamp facts; null fields stay unobserved."
            ),
        ),
    ),
    EventSpec(
        "graph.update",
        "The task graph and its current focus.",
        (
            FieldSpec("nodes", "object", True, "Task nodes keyed by identifier."),
            FieldSpec(
                "cursor", "str", False, "Current task identifier; null when none is selected."
            ),
        ),
    ),
    EventSpec(
        "jail.degraded",
        "The sandbox started weaker than requested or left surviving processes.",
        (
            FieldSpec(
                "detail", "str", True, "The sandbox degradation or surviving-process diagnostic."
            ),
        ),
    ),
    EventSpec(
        "loop.auto_commit",
        "A turn's checkpoint commit landed.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("sha", "str", True, "The commit identifier, possibly abbreviated."),
            FieldSpec("subject", "str", True, "The landed commit's subject."),
            FieldSpec("task_id", "str", False, "Task identifier."),
        ),
    ),
    EventSpec(
        "loop.auto_commit.failed",
        "A checkpoint commit could not land.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
            FieldSpec("worktree_status", "str", True, "Git status when the commit failed."),
            FieldSpec("commit_subject", "str", True, "The attempted commit subject."),
        ),
    ),
    EventSpec(
        "loop.baseline",
        "The baseline verification attempt finished.",
        (
            FieldSpec("ok", "bool", True, "Whether the operation succeeded."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("verify_id", "str", False, "Verification attempt identifier."),
        ),
    ),
    EventSpec(
        "loop.budget",
        "A turn's usage heartbeat before its model call.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("input_tokens", "int", True, "Input tokens consumed so far."),
            FieldSpec("output_tokens", "int", True, "Output token count."),
            FieldSpec("cache_read_tokens", "int", True, "Cached input tokens consumed so far."),
            FieldSpec("cost_usd", "float", True, "Estimated spend in USD."),
        ),
    ),
    EventSpec(
        "loop.compact.deduped",
        "Repeated tool results were removed from context.",
        (
            FieldSpec("n", "int", True, "Number of affected entries or notices."),
            FieldSpec("calls", "list", True, "The affected tool call identities."),
        ),
    ),
    EventSpec(
        "loop.compact.dropped",
        "Tool results were elided from context.",
        (
            FieldSpec("n", "int", True, "Number of affected entries or notices."),
            FieldSpec("calls", "list", True, "The affected tool call identities."),
        ),
    ),
    EventSpec(
        "loop.compact.gist.call",
        "The model was asked to summarize files for context.",
        (FieldSpec("files", "int", True, "Number of files to summarize."),),
    ),
    EventSpec(
        "loop.compact.gist.failed",
        "The file-summary call failed.",
        (FieldSpec("error", "str", True, "The failure diagnostic."),),
    ),
    EventSpec(
        "loop.compact.gists",
        "File summaries were created or demoted during compaction.",
        (
            FieldSpec("gisted", "int", True, "Number of file summaries created."),
            FieldSpec(
                "demoted", "int", True, "Number of summaries replaced by smaller placeholders."
            ),
            FieldSpec("paths", "list", True, "Affected repository-relative paths."),
            FieldSpec("demoted_paths", "list", True, "Paths whose summaries were demoted."),
        ),
    ),
    EventSpec(
        "loop.compact.refused",
        "There was too little history to summarize.",
        (FieldSpec("reason", "str", True, "The reason for the outcome."),),
    ),
    EventSpec(
        "loop.compact.requested",
        "The operator requested a context restart.",
        (FieldSpec("focus", "str", True, "The requested summary or review focus."),),
    ),
    EventSpec(
        "loop.compact.restored",
        "A resumed context restored its elision and summary counts.",
        (
            FieldSpec("elided", "int", True, "Number of restored elision markers."),
            FieldSpec("gists", "int", True, "Number of restored file summaries."),
        ),
    ),
    EventSpec(
        "loop.compact.summarise.call",
        "The model was asked to summarize the conversation.",
        (FieldSpec("messages", "int", True, "Number of conversation messages."),),
    ),
    EventSpec(
        "loop.compact.summarise.done",
        "A summary replaced the conversation history.",
        (
            FieldSpec("summary_chars", "int", True, "Length of the summary in characters."),
            FieldSpec("summary", "str", True, "The result or outcome in words."),
            FieldSpec(
                "kept_messages", "int", True, "Number of recent messages kept after the summary."
            ),
        ),
    ),
    EventSpec(
        "loop.compact.summarise.failed",
        "The conversation-summary call failed.",
        (FieldSpec("error", "str", True, "The failure diagnostic."),),
    ),
    EventSpec(
        "loop.compact.thinking_dropped",
        "Reasoning text was removed from context.",
        (
            FieldSpec("messages", "int", True, "Number of conversation messages."),
            FieldSpec("chars", "int", True, "Text length in characters."),
        ),
    ),
    EventSpec(
        "loop.decision.recorded",
        "An operator ruling was recorded in decisions.md.",
        (
            FieldSpec("question", "str", True, "The question presented to the operator."),
            FieldSpec("answer", "str", True, "The operator's answer."),
        ),
    ),
    EventSpec(
        "loop.decision.unrecorded",
        "An operator ruling could not be recorded or was missing at finish.",
        (
            FieldSpec("error", "str", False, "The failure diagnostic."),
            FieldSpec("missing", "int", False, "Number of operator rulings missing at finish."),
        ),
    ),
    EventSpec(
        "loop.edit.left_out",
        "A checkpoint left ignored or untracked edits outside its commit.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("ignored", "list", True, "Ignored edits left outside the checkpoint."),
            FieldSpec("untracked", "list", True, "Untracked edits left outside the checkpoint."),
        ),
    ),
    EventSpec(
        "loop.finish_contract.refused",
        "The finish payload did not satisfy the machine's result contract.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("problems", "list", True, "Violations of the machine's finish contract."),
        ),
    ),
    EventSpec(
        "loop.loop_guard.triggered",
        "Repeated identical tool calls triggered a warning.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("tool", "str", True, "The repeated tool's name."),
            FieldSpec("streak", "int", True, "Number of consecutive matching turns or calls."),
        ),
    ),
    EventSpec(
        "loop.media.refused",
        "An attachment was refused by its tool or model route.",
        (
            FieldSpec("provider_tool_call_id", "str", True, "The provider's tool-use identifier."),
            FieldSpec("reason", "str", True, "The reason for the outcome."),
            FieldSpec("route", "str", False, "The model route that refused the attachment."),
            FieldSpec("text", "str", False, "The event's text."),
            FieldSpec(
                "attachment", "object", False, "Attachment metadata without its binary content."
            ),
        ),
    ),
    EventSpec(
        "loop.memory_flip.nudged",
        "Verification turned green and the model was asked to record the lesson.",
        (FieldSpec("turn", "int", True, "The execution's turn number."),),
    ),
    EventSpec(
        "loop.metric.auto_call",
        "An automatic metric sample started.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("sha", "str", True, "The commit identifier, possibly abbreviated."),
        ),
    ),
    EventSpec(
        "loop.metric.auto_failed",
        "An automatic metric sample failed.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.metric.sample",
        "A metric sample was recorded.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("label", "str", True, "Metric sample label."),
            FieldSpec("score", "float", False, "Parsed metric score; null when unavailable."),
            FieldSpec("returncode", "int", False, "Command exit code; null when unavailable."),
            FieldSpec("sha", "str", True, "The commit identifier, possibly abbreviated."),
        ),
    ),
    EventSpec(
        "loop.metric_ceiling.stop",
        "The metric ceiling ended the run.",
        (FieldSpec("turn", "int", True, "The execution's turn number."),),
    ),
    EventSpec(
        "loop.metric_early_finish.rejected",
        "A metric run's early finish was returned to the model.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("nudges_used", "int", True, "Number of notices issued for this condition."),
            FieldSpec(
                "budget_remaining",
                "float",
                False,
                "Remaining budget fraction; null without a finite cap.",
            ),
            FieldSpec("trigger", "str", False, "The action that triggered the gate or panel."),
        ),
    ),
    EventSpec(
        "loop.metric_plateau.nudge",
        "A metric plateau prompted another attempt.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("nudges_used", "int", True, "Number of notices issued for this condition."),
            FieldSpec(
                "budget_remaining",
                "float",
                False,
                "Remaining budget fraction; null without a finite cap.",
            ),
        ),
    ),
    EventSpec(
        "loop.no_progress.nudge",
        "Repeated turns without progress prompted a warning.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("streak", "int", True, "Number of consecutive matching turns or calls."),
            FieldSpec("level", "int", True, "The warning's escalation level."),
        ),
    ),
    EventSpec(
        "loop.note",
        "A machine-creation progress note.",
        (FieldSpec("text", "str", True, "The event's text."),),
    ),
    EventSpec(
        "loop.operator.presence",
        "The operator's observed attachment state changed.",
        (FieldSpec("presence", "str", True, "The observed operator attachment and away state."),),
    ),
    EventSpec(
        "loop.parallel.compared",
        "Parallel candidates were ranked.",
        (
            FieldSpec("group", "str", True, "Parallel dispatch group identifier."),
            FieldSpec("ranked_by", "str", True, "The method used to rank the candidates."),
            FieldSpec("rationale", "str", True, "The ranking explanation."),
            FieldSpec(
                "ranking",
                "list",
                True,
                "Candidates in ranked order with their verify and cost readings.",
            ),
            FieldSpec("judge_cost_usd", "float", True, "Ranking judge spend in USD."),
            FieldSpec(
                "judge_cost_partial", "bool", True, "Whether the judge spend is a lower bound."
            ),
        ),
    ),
    EventSpec(
        "loop.parallel.dispatched",
        "Parallel lanes were dispatched.",
        (
            FieldSpec("group", "str", True, "Parallel dispatch group identifier."),
            FieldSpec(
                "lanes", "int", True, "Lane count for dispatch, or lane results for completion."
            ),
            FieldSpec("tasks", "list", True, "Task texts sent to the lanes."),
        ),
    ),
    EventSpec(
        "loop.parallel.failed",
        "Parallel lane dispatch or execution failed.",
        (
            FieldSpec("group", "str", True, "Parallel dispatch group identifier."),
            FieldSpec("fanout", "bool", False, "Whether the failed group was a top-level fan-out."),
            FieldSpec(
                "lanes", "list", False, "Lane count for dispatch, or lane results for completion."
            ),
            FieldSpec("error", "str", False, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.parallel.joined",
        "Parallel lane results returned.",
        (
            FieldSpec("group", "str", True, "Parallel dispatch group identifier."),
            FieldSpec(
                "lanes", "list", True, "Lane count for dispatch, or lane results for completion."
            ),
        ),
    ),
    EventSpec(
        "loop.parked",
        "A standing run parked at a turn boundary.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("reason", "str", True, "The reason for the outcome."),
        ),
    ),
    EventSpec(
        "loop.parked.resumed",
        "A parked run received work or an operator request.",
        (FieldSpec("turn", "int", True, "The execution's turn number."),),
    ),
    EventSpec(
        "loop.pin.added",
        "An operator instruction was pinned in context.",
        (
            FieldSpec("text", "str", True, "The event's text."),
            FieldSpec("chars", "int", True, "Text length in characters."),
            FieldSpec("count", "int", True, "Total pinned instructions."),
        ),
    ),
    EventSpec(
        "loop.pin.refused",
        "An instruction exceeded the pinned-context limit.",
        (
            FieldSpec("chars", "int", True, "Text length in characters."),
            FieldSpec("limit", "int", True, "Pinned instruction character limit."),
        ),
    ),
    EventSpec(
        "loop.pin.restored",
        "A resumed or forked execution restored pinned instructions.",
        (
            FieldSpec("pins", "list", True, "The full list of pinned instructions."),
            FieldSpec("count", "int", True, "Total pinned instructions."),
        ),
    ),
    EventSpec(
        "loop.plan_finish.nudge",
        "The model was asked to finish its plan within the remaining budget.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec(
                "budget_remaining",
                "float",
                False,
                "Remaining budget fraction; null without a finite cap.",
            ),
        ),
    ),
    EventSpec(
        "loop.plan_read.failed",
        "The plan could not be read.",
        (
            FieldSpec("path", "str", True, "The plan's file path."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.plan_reread",
        "The plan was read back into context.",
        (
            FieldSpec("path", "str", True, "The plan's file path."),
            FieldSpec("bytes", "int", True, "UTF-8 text length in bytes."),
        ),
    ),
    EventSpec(
        "loop.plan_write.failed",
        "The plan could not be written.",
        (
            FieldSpec("path", "str", True, "The plan's file path."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.plan_written",
        "The plan was written to disk.",
        (
            FieldSpec("path", "str", True, "The plan's file path."),
            FieldSpec("bytes", "int", True, "UTF-8 text length in bytes."),
        ),
    ),
    EventSpec(
        "loop.provider.empty_tool_call_retry",
        "A provider response with empty tool calls was retried.",
        (
            FieldSpec("attempt", "int", True, "Retry attempt number."),
            FieldSpec("stop_reason", "str", True, "The provider's response stop reason."),
        ),
    ),
    EventSpec(
        "loop.provider.fatal",
        "A non-retryable provider error stopped the call.",
        (
            FieldSpec("status_code", "int", False, "HTTP status code when available."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.provider.retry",
        "A provider error prompted another attempt.",
        (
            FieldSpec("attempt", "int", True, "Retry attempt number."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.question.delivered",
        "Queued answers were delivered to the model.",
        (FieldSpec("ids", "list", True, "Identifiers of the delivered questions."),),
    ),
    EventSpec(
        "loop.question_nudge",
        "A silent finish ended on a question and was returned to the model.",
        (FieldSpec("turn", "int", True, "The execution's turn number."),),
    ),
    EventSpec(
        "loop.reasoning_starvation",
        "Reasoning consumed a response without usable output.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("reasoning_chars", "int", True, "Reasoning output length in characters."),
            FieldSpec("output_tokens", "int", True, "Output token count."),
            FieldSpec("stop_reason", "str", False, "The provider's response stop reason."),
        ),
    ),
    EventSpec(
        "loop.request.refused",
        "An operator task-graph request could not be applied.",
        (
            FieldSpec("kind", "str", True, "The request or authentication operation kind."),
            FieldSpec("text", "str", True, "The event's text."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.resume.interrupted",
        "Resume found tool calls that never returned and will not be rerun.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("tools", "list", True, "Interrupted tool calls that will not be rerun."),
        ),
    ),
    EventSpec(
        "loop.review.finding.reopened",
        "An operator reopened a finding.",
        (
            FieldSpec("finding_id", "str", True, "Finding identifier."),
            FieldSpec(
                "request_id",
                "str",
                True,
                "Request identifier used to correlate or deduplicate work.",
            ),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec("version", "int", True, "Review-state version after the action."),
            FieldSpec("reason", "str", True, "The reason for the outcome."),
        ),
    ),
    EventSpec(
        "loop.review.finding.resolved",
        "An operator recorded a finding's disposition.",
        (
            FieldSpec("finding_id", "str", True, "Finding identifier."),
            FieldSpec(
                "request_id",
                "str",
                True,
                "Request identifier used to correlate or deduplicate work.",
            ),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec("version", "int", True, "Review-state version after the action."),
            FieldSpec(
                "resolution", "object", True, "The finding's disposition and supporting reason."
            ),
        ),
    ),
    EventSpec(
        "loop.review.fix.failed",
        "A finding's fix request could not be delivered.",
        (
            FieldSpec("finding_id", "str", True, "Finding identifier."),
            FieldSpec(
                "request_id",
                "str",
                True,
                "Request identifier used to correlate or deduplicate work.",
            ),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec("version", "int", True, "Review-state version after the action."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "loop.review.fix.requested",
        "An operator requested a fix for a finding.",
        (
            FieldSpec("finding_id", "str", True, "Finding identifier."),
            FieldSpec(
                "request_id",
                "str",
                True,
                "Request identifier used to correlate or deduplicate work.",
            ),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec("version", "int", True, "Review-state version after the action."),
            FieldSpec(
                "target", "object", True, "The immutable review target, or provider request class."
            ),
            FieldSpec("action", "str", True, "Fix delivery action: live-steer or resume."),
        ),
    ),
    EventSpec(
        "loop.review.fix.submitted",
        "A finding's fix request reached an execution.",
        (
            FieldSpec("finding_id", "str", True, "Finding identifier."),
            FieldSpec(
                "request_id",
                "str",
                True,
                "Request identifier used to correlate or deduplicate work.",
            ),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec("version", "int", True, "Review-state version after the action."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
        ),
    ),
    EventSpec(
        "loop.review.panel",
        "A review panel completed and recorded its evidence and blocking decision.",
        (
            FieldSpec("panel_id", "str", True, "Review panel identifier."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("trigger", "str", True, "The action that triggered the gate or panel."),
            FieldSpec(
                "target", "object", True, "The immutable review target, or provider request class."
            ),
            FieldSpec("findings", "list", True, "Grounded findings or panel observations."),
            FieldSpec(
                "injected_finding_ids", "list", True, "Finding identifiers supplied to the worker."
            ),
            FieldSpec("block_on", "str", True, "The configured panel blocking rule."),
            FieldSpec("raw_blocked", "bool", True, "Whether the verdicts meet the blocking rule."),
            FieldSpec("blocked", "bool", True, "Whether the armed gate blocks the finish."),
            FieldSpec(
                "disarmed", "bool", True, "Whether the review gate's rejection cap disarmed it."
            ),
            FieldSpec("n_block", "int", True, "Number of distinct blocking provider/model routes."),
            FieldSpec("n_abstain", "int", True, "Number of abstaining seats."),
            FieldSpec(
                "completion", "str", True, "Whether the panel completed or was inconclusive."
            ),
        ),
    ),
    EventSpec(
        "loop.review.rejected_finish",
        "The review panel rejected the declared finish.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("ending", "str", True, "The declared finish being reviewed."),
        ),
    ),
    EventSpec(
        "loop.review.rejection_cap_reached",
        "The review gate reached its rejection limit.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("rejections", "int", True, "Number of review rejections."),
        ),
    ),
    EventSpec(
        "loop.review.seat",
        "A review seat returned a grounded verdict or abstained.",
        (
            FieldSpec("panel_id", "str", True, "Review panel identifier."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
            FieldSpec("turn", "int", False, "The execution's turn number."),
            FieldSpec("trigger", "str", False, "The action that triggered the gate or panel."),
            FieldSpec("seat_index", "int", True, "Zero-based review seat index."),
            FieldSpec("focus", "str", True, "The requested summary or review focus."),
            FieldSpec("model_route", "str", True, "The seat's configured model route."),
            FieldSpec("verdict", "str", True, "Seat verdict: pass, block or abstain."),
            FieldSpec("findings", "list", True, "Grounded findings or panel observations."),
            FieldSpec("summary", "str", True, "The result or outcome in words."),
            FieldSpec("error", "str", False, "The failure diagnostic."),
            FieldSpec("risk", "str", True, "The seat's stated risk."),
            FieldSpec("risk_truncated", "bool", True, "Whether the risk text was truncated."),
        ),
    ),
    EventSpec(
        "loop.review.skipped",
        "A review panel could not run.",
        (
            FieldSpec("panel_id", "str", True, "Review panel identifier."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
            FieldSpec("turn", "int", False, "The execution's turn number."),
            FieldSpec("trigger", "str", False, "The action that triggered the gate or panel."),
            FieldSpec("reason", "str", True, "The reason for the outcome."),
            FieldSpec("error", "str", False, "The failure diagnostic."),
            FieldSpec("remaining", "float", False, "Remaining budget fraction."),
        ),
    ),
    EventSpec(
        "loop.review.start",
        "A review panel began over an immutable target.",
        (
            FieldSpec("panel_id", "str", True, "Review panel identifier."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("trigger", "str", True, "The action that triggered the gate or panel."),
            FieldSpec(
                "target", "object", True, "The immutable review target, or provider request class."
            ),
            FieldSpec("rule_sources", "list", True, "Base-tree rule files supplied to the panel."),
            FieldSpec("rules_complete", "bool", True, "Whether every required rule was supplied."),
            FieldSpec(
                "rules_reason",
                "str",
                False,
                "Why rule collection was incomplete; null when complete.",
            ),
            FieldSpec("seats", "list", True, "The resolved review seat configurations."),
        ),
    ),
    EventSpec(
        "loop.run_budget.nudge",
        "The model was asked to finish within its remaining run budget.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec(
                "budget_remaining",
                "float",
                False,
                "Remaining budget fraction; null without a finite cap.",
            ),
        ),
    ),
    EventSpec(
        "loop.sandbox_tool_unreachable",
        "A tool found on the host could not run inside the sandbox.",
        (FieldSpec("binary", "str", True, "The unreachable executable's name."),),
    ),
    EventSpec(
        "loop.silent_no_work.nudge",
        "A silent finish without work was returned to the model.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("nudges_used", "int", True, "Number of notices issued for this condition."),
        ),
    ),
    EventSpec(
        "loop.skills.warning",
        "A skill declaration produced a warning.",
        (FieldSpec("warning", "str", True, "The skill diagnostic."),),
    ),
    EventSpec(
        "loop.stagnation.nudged",
        "Elapsed time without edits or verification prompted a warning.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("elapsed_s", "int", True, "Elapsed seconds without edits or verification."),
        ),
    ),
    EventSpec(
        "loop.standing.resumed",
        "A standing goal resumed after absorbing a soft stop.",
        (
            FieldSpec("reason", "str", True, "The reason for the outcome."),
            FieldSpec("task_id", "str", True, "Task identifier."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
        ),
    ),
    EventSpec(
        "loop.standing.set",
        "The operator set a standing goal.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("title", "str", True, "Task or standing-goal text."),
        ),
    ),
    EventSpec(
        "loop.steer.aborted",
        "The operator's steer aborted the run.",
        (),
    ),
    EventSpec(
        "loop.steer.detached",
        "The operator's steer stopped the run for background resume.",
        (),
    ),
    EventSpec(
        "loop.steer.exited",
        "The operator's steer ended the run and left the terminal.",
        (),
    ),
    EventSpec(
        "loop.steer.feedback",
        "A steer directive returned feedback without invoking the model.",
        (FieldSpec("text", "str", True, "The feedback for the operator."),),
    ),
    EventSpec(
        "loop.steer.injected",
        "An operator instruction was injected into context.",
        (
            FieldSpec("chars", "int", True, "Text length in characters."),
            FieldSpec("text", "str", True, "The event's text."),
        ),
    ),
    EventSpec(
        "loop.steer.requested",
        "The loop reached an operator steer request.",
        (FieldSpec("turn", "int", True, "The execution's turn number."),),
    ),
    EventSpec(
        "loop.steer.skill",
        "An operator steer invoked a skill.",
        (
            FieldSpec("name", "str", True, "Tool or skill name."),
            FieldSpec(
                "args",
                "str",
                True,
                "Tool argument preview, or the skill invocation's argument text.",
            ),
        ),
    ),
    EventSpec(
        "loop.steer.undo",
        "The operator's steer requested an undo fork.",
        (),
    ),
    EventSpec(
        "loop.task.queued",
        "The operator queued a task.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("title", "str", True, "Task or standing-goal text."),
        ),
    ),
    EventSpec(
        "loop.task.retired",
        "The operator retired a task.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("title", "str", True, "Task or standing-goal text."),
        ),
    ),
    EventSpec(
        "loop.task.stuck_nudge",
        "Time on one task prompted a change of approach.",
        (
            FieldSpec("task_id", "str", True, "Task identifier."),
            FieldSpec(
                "turns", "int", True, "Number of execution turns, or turns spent on one task."
            ),
            FieldSpec("n", "int", True, "Number of affected entries or notices."),
        ),
    ),
    EventSpec(
        "loop.task.surfaced",
        "A task became the current focus.",
        (FieldSpec("task_id", "str", True, "Task identifier."),),
    ),
    EventSpec(
        "loop.task_finish.gated",
        "A finish with open subtasks was returned to the model.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("nudges_used", "int", True, "Number of notices issued for this condition."),
            FieldSpec("trigger", "str", True, "The action that triggered the gate or panel."),
        ),
    ),
    EventSpec(
        "loop.test_only_green.notice",
        "Verification passed after changes confined to tests.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("paths", "list", True, "Affected repository-relative paths."),
        ),
    ),
    EventSpec(
        "loop.tool.call",
        "The loop began dispatching a tool call.",
        (
            FieldSpec("name", "str", True, "Tool or skill name."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
        ),
    ),
    EventSpec(
        "loop.tool_error.nudge",
        "Repeated tool errors prompted a warning.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("streak", "int", True, "Number of consecutive matching turns or calls."),
            FieldSpec("level", "int", True, "The warning's escalation level."),
        ),
    ),
    EventSpec(
        "loop.verify_broken.nudge",
        "An unusable verify command prompted a correction request.",
        (FieldSpec("turn", "int", True, "The execution's turn number."),),
    ),
    EventSpec(
        "loop.verify_finish.gated",
        "A finish without green verification was returned to the model.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("nudges_used", "int", True, "Number of notices issued for this condition."),
            FieldSpec("trigger", "str", True, "The action that triggered the gate or panel."),
        ),
    ),
    EventSpec(
        "loop.verify_harness",
        "The harness requested verification.",
        (
            FieldSpec("why", "str", True, "Why the harness requested verification."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
        ),
    ),
    EventSpec(
        "loop.verify_inferred",
        "The verify command was inferred, restored, adopted or disabled.",
        (
            FieldSpec(
                "command", "list", True, "Resolved verify argv; empty when no command is adopted."
            ),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec(
                "adopted_at", "int", False, "Turn where the gate was adopted, restored or dropped."
            ),
        ),
    ),
    EventSpec(
        "loop.verify_scoped",
        "Verification was narrowed to tests near the changed files.",
        (
            FieldSpec("paths", "list", True, "Affected repository-relative paths."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
        ),
    ),
    EventSpec(
        "loop.verify_settled.nudge",
        "Repeated green verification without edits prompted a finish request.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("idle", "int", True, "Consecutive settled turns."),
        ),
    ),
    EventSpec(
        "loop.verify_swapped",
        "Resume replaced the saved verify command.",
        (
            FieldSpec("was", "list", True, "Saved verification argv."),
            FieldSpec("now", "list", True, "Replacement verification argv."),
        ),
    ),
    EventSpec(
        "loop.went_quiet.nudge",
        "A response without tool calls prompted the model to continue.",
        (
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("nudges_used", "int", True, "Number of notices issued for this condition."),
            FieldSpec("nudges_max", "int", True, "Maximum notices allowed for this condition."),
            FieldSpec("output_tokens", "int", True, "Output token count."),
        ),
    ),
    EventSpec(
        "mcp.server_unavailable",
        "A configured MCP server could not start.",
        (
            FieldSpec("server", "str", True, "Configured MCP server name."),
            FieldSpec("error", "str", True, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "metric.end",
        "The metric command finished.",
        (
            FieldSpec("cmd", "list", True, "Command argv."),
            FieldSpec(
                "exit_code", "int", False, "Command exit code; null if no exit was observed."
            ),
            FieldSpec("duration_s", "float", True, "Command duration in seconds."),
            FieldSpec("stdout_tail", "str", True, "Capped standard-output tail."),
            FieldSpec("stderr_tail", "str", True, "Capped standard-error tail."),
            FieldSpec("score", "float", False, "Parsed metric score; null when unavailable."),
        ),
    ),
    EventSpec(
        "metric.start",
        "The metric command started.",
        (FieldSpec("cmd", "list", True, "Command argv."),),
    ),
    EventSpec(
        "provider.auth",
        "A provider credential refresh, exchange or recovery finished.",
        (
            FieldSpec("provider", "str", True, "Configured provider name."),
            FieldSpec("kind", "str", True, "The request or authentication operation kind."),
            FieldSpec("outcome", "str", True, "The operation's outcome."),
        ),
    ),
    EventSpec(
        "provider.request",
        "An HTTP attempt was made for a model call.",
        (
            FieldSpec("role", "str", True, "The calling model role."),
            FieldSpec("provider", "str", True, "Configured provider name."),
            FieldSpec("model", "str", True, "Model identifier."),
            FieldSpec(
                "model_call_id", "str", True, "The logical model call that made this request."
            ),
            FieldSpec("panel_id", "str", False, "Review panel identifier."),
            FieldSpec("seat_index", "int", False, "Zero-based review seat index."),
            FieldSpec(
                "request_id",
                "str",
                True,
                "HTTP attempt identifier, also stored at the transcript's top level.",
            ),
            FieldSpec(
                "target", "str", True, "The immutable review target, or provider request class."
            ),
        ),
    ),
    EventSpec(
        "question.answer",
        "The operator answered questions or nobody could answer.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec("answers", "list", True, "Answers aligned with the requested questions."),
            FieldSpec("source", "str", True, "Where the request or result came from."),
            FieldSpec(
                "unseen",
                "bool",
                True,
                "Whether nobody could answer and no question remains queued.",
            ),
        ),
    ),
    EventSpec(
        "question.prompt",
        "Questions were presented to the operator.",
        (
            FieldSpec("id", "str", True, "Prompt, task or background-job identifier."),
            FieldSpec(
                "questions",
                "list",
                True,
                "The questions presented.",
            ),
            FieldSpec("call_id", "int", False, "The tool call correlated with this event."),
            FieldSpec("wait", "bool", True, "Whether the call waits for answers."),
        ),
    ),
    EventSpec(
        "role.call",
        "A model call started for a role.",
        (
            FieldSpec("panel_id", "str", False, "Review panel identifier."),
            FieldSpec("role", "str", True, "The calling model role."),
            FieldSpec("model", "str", True, "Model identifier."),
            FieldSpec("provider", "str", True, "Configured provider name."),
            FieldSpec(
                "model_call_id", "str", True, "Identifier pairing the model call with its result."
            ),
            FieldSpec("seat_index", "int", False, "Zero-based review seat index."),
        ),
    ),
    EventSpec(
        "role.result",
        "A model call returned or failed. Failed calls include usage only when reported.",
        (
            FieldSpec("panel_id", "str", False, "Review panel identifier."),
            FieldSpec("role", "str", True, "The calling model role."),
            FieldSpec("model", "str", True, "Model identifier."),
            FieldSpec("provider", "str", True, "Configured provider name."),
            FieldSpec(
                "model_call_id", "str", True, "Identifier pairing the model call with its result."
            ),
            FieldSpec("seat_index", "int", False, "Zero-based review seat index."),
            FieldSpec("ok", "bool", True, "Whether the operation succeeded."),
            FieldSpec("text", "str", False, "The event's text."),
            FieldSpec("tokens_in", "int", False, "Input tokens used by this model call."),
            FieldSpec("tokens_out", "int", False, "Output tokens used by this model call."),
            FieldSpec("cache_read", "int", False, "Cached input tokens used by this model call."),
            FieldSpec(
                "cache_creation", "int", False, "Cache-creation tokens used by this model call."
            ),
            FieldSpec(
                "cost_usd",
                "float",
                False,
                "Provider-reported USD cost; zero means free, absent means unreported.",
            ),
            FieldSpec("stop_reason", "str", False, "The provider's response stop reason."),
            FieldSpec("error", "str", False, "The failure diagnostic."),
        ),
    ),
    EventSpec(
        "role.text_delta",
        "A piece of assistant text streamed.",
        (
            FieldSpec("role", "str", True, "The calling model role."),
            FieldSpec("text", "str", True, "The event's text."),
        ),
    ),
    EventSpec(
        "role.thinking_delta",
        "A piece of assistant reasoning streamed.",
        (
            FieldSpec("role", "str", True, "The calling model role."),
            FieldSpec("text", "str", True, "The event's text."),
        ),
    ),
    EventSpec(
        "session.resume",
        "Emitted at the start of every resumed execution.",
        (
            FieldSpec("session_id", "str", True, "Session identifier."),
            FieldSpec("user_task", "str", True, "The displayed task text."),
            FieldSpec("steer", "str", True, "The instruction supplied when resuming."),
            FieldSpec("mode", "str", True, "The execution mode."),
            FieldSpec("turn", "int", True, "The execution's turn number."),
            FieldSpec("messages", "int", True, "Number of conversation messages."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
        ),
    ),
    EventSpec(
        "session.start",
        "Emitted once when the session is created.",
        (
            FieldSpec("session_id", "str", False, "Session identifier."),
            FieldSpec("user_task", "str", True, "The displayed task text."),
            FieldSpec("mode", "str", True, "The execution mode."),
            FieldSpec("execution_id", "str", True, "Execution identifier."),
        ),
    ),
    EventSpec(
        "session.steer_requested",
        "Ctrl-C requested an operator steer during the run.",
        (FieldSpec("source", "str", True, "Where the request or result came from."),),
    ),
    EventSpec(
        "session.stop",
        "The execution stopped, with its reason; the session stays resumable.",
        (
            FieldSpec("execution_id", "str", True, "Execution identifier; empty when unknown."),
            FieldSpec("reason", "str", True, "The reason for the outcome."),
            FieldSpec(
                "turns", "int", True, "Number of execution turns, or turns spent on one task."
            ),
            FieldSpec("summary", "str", False, "The result or outcome in words."),
            FieldSpec(
                "verification",
                "str",
                False,
                "Final verification state: passed, failed, unverified or not_applicable.",
            ),
            FieldSpec(
                "scoped", "bool", False, "Whether verification ran only the selected nearby tests."
            ),
            FieldSpec(
                "final_verify_id", "str", False, "The final verification attempt identifier."
            ),
            FieldSpec("tree", "str", False, "The final working-tree content identifier."),
            FieldSpec("tool", "str", False, "The repeated tool's name."),
            FieldSpec("streak", "int", False, "Number of consecutive matching turns or calls."),
        ),
    ),
    EventSpec(
        "session.stranded",
        "A finished run left edits uncommitted and no commit landed.",
        (),
    ),
    EventSpec(
        "session.undone",
        "Undo forked the run before its last operator message.",
        (
            FieldSpec("new_session_id", "str", True, "Identifier of the undo fork."),
            FieldSpec("undone_text", "str", True, "The operator message taken back."),
        ),
    ),
    EventSpec(
        "tool.call",
        "A tool was dispatched, including calls that a guard rejects.",
        (
            FieldSpec("name", "str", True, "Tool or skill name."),
            FieldSpec(
                "args",
                "object",
                True,
                "Tool argument preview, or the skill invocation's argument text.",
            ),
            FieldSpec("call_id", "int", True, "The tool call correlated with this event."),
        ),
    ),
    EventSpec(
        "tool.result",
        "A dispatched tool returned its result or refusal.",
        (
            FieldSpec("name", "str", True, "Tool or skill name."),
            FieldSpec("ok", "bool", True, "Whether the operation succeeded."),
            FieldSpec("summary", "str", True, "The result or outcome in words."),
            FieldSpec("call_id", "int", True, "The tool call correlated with this event."),
            FieldSpec("paths", "list", False, "Affected repository-relative paths."),
            FieldSpec("stdout_tail", "str", False, "Capped standard-output tail."),
            FieldSpec("stderr_tail", "str", False, "Capped standard-error tail."),
            FieldSpec(
                "stopped_by_operator",
                "bool",
                False,
                "Whether an immediate operator stop ended the command.",
            ),
            FieldSpec(
                "attachment", "object", False, "Attachment metadata without its binary content."
            ),
            FieldSpec("head_tail", "str", False, "A read preview or attachment summary."),
            FieldSpec("lines_total", "int", False, "Total lines in the file read."),
        ),
    ),
    EventSpec(
        "verify.end",
        "A verification attempt finished with evidence about the observed tree.",
        (
            FieldSpec("verify_id", "str", False, "Verification attempt identifier."),
            FieldSpec("execution_id", "str", False, "Execution identifier."),
            FieldSpec("target_id", "str", False, "Recorded execution-target reference."),
            FieldSpec("cmd", "list", True, "Command argv."),
            FieldSpec(
                "tree_before", "str", False, "Content identifier observed before verification."
            ),
            FieldSpec(
                "tree_after", "str", False, "Content identifier observed after verification."
            ),
            FieldSpec(
                "scoped", "bool", True, "Whether verification ran only the selected nearby tests."
            ),
            FieldSpec("selected_paths", "list", True, "Paths selected for scoped verification."),
            FieldSpec("outcome", "str", True, "The operation's outcome."),
            FieldSpec(
                "exit_code", "int", False, "Command exit code; null if no exit was observed."
            ),
            FieldSpec("duration_s", "float", True, "Command duration in seconds."),
            FieldSpec("timeout_s", "float", True, "Verification wall-clock limit in seconds."),
            FieldSpec("stdout_tail", "str", True, "Capped standard-output tail."),
            FieldSpec("stderr_tail", "str", True, "Capped standard-error tail."),
            FieldSpec(
                "stdout_truncated",
                "bool",
                False,
                "Whether standard-output capture was truncated; null when unknown.",
            ),
            FieldSpec(
                "stderr_truncated",
                "bool",
                False,
                "Whether standard-error capture was truncated; null when unknown.",
            ),
            FieldSpec("baseline", "bool", True, "Whether the attempt measured the baseline tree."),
            FieldSpec("reason", "str", True, "The reason for the outcome."),
        ),
    ),
    EventSpec(
        "verify.start",
        "A verification attempt started.",
        (
            FieldSpec("approval_required", "bool", False, "Whether the attempt needs approval."),
            FieldSpec("call_id", "int", False, "The model tool call, absent for a harness gate."),
            FieldSpec("verify_id", "str", False, "Verification attempt identifier."),
            FieldSpec("execution_id", "str", False, "Execution identifier."),
            FieldSpec("target_id", "str", False, "Recorded execution-target reference."),
            FieldSpec(
                "tree_before", "str", False, "Content identifier observed before verification."
            ),
            FieldSpec("cmd", "list", True, "Command argv."),
            FieldSpec(
                "scoped", "bool", True, "Whether verification ran only the selected nearby tests."
            ),
            FieldSpec("selected_paths", "list", True, "Paths selected for scoped verification."),
            FieldSpec("timeout_s", "float", True, "Verification wall-clock limit in seconds."),
        ),
    ),
    EventSpec(
        "workspace.setup",
        "The workspace setup command finished.",
        (
            FieldSpec("execution_id", "str", False, "Execution identifier."),
            FieldSpec("target_id", "str", False, "Recorded execution-target reference."),
            FieldSpec("cmd", "list", True, "Command argv."),
            FieldSpec(
                "exit_code", "int", False, "Command exit code; null if no exit was observed."
            ),
            FieldSpec("duration_s", "float", True, "Command duration in seconds."),
            FieldSpec(
                "exec_failed", "bool", True, "Whether the setup command could not be executed."
            ),
            FieldSpec("stdout_tail", "str", True, "Capped standard-output tail."),
            FieldSpec("stderr_tail", "str", True, "Capped standard-error tail."),
        ),
    ),
)

REGISTRY: Mapping[str, EventSpec] = types.MappingProxyType({event.type: event for event in EVENTS})


def spec_for(event_type: str) -> EventSpec | None:
    """Return the declared event, or None for a type outside the write vocabulary."""
    return REGISTRY.get(event_type)
