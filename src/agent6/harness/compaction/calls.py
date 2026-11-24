"""Bind compaction model calls to the prepared worker turn."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from agent6 import kinds
from agent6.harness import loop_conversation, provider_call
from agent6.harness.compaction import window
from agent6.providers import provider_calls, provider_types, request_headers


@dataclasses.dataclass(frozen=True, slots=True)
class Calls:
    """Use the same providers, metering and stop signals as the driving run."""

    worker: provider_calls.Provider
    caller: provider_call.ProviderCaller
    settings: window.CompactionSettings
    conversation: loop_conversation.Conversation
    system: str
    tools: list[provider_types.ToolDefinition]
    temperature: float | None
    execution_id: str
    turn_identity: kinds.CallContext | None
    check_budget: Callable[[], None]
    should_abort: Callable[[], bool]
    should_interrupt: Callable[[], bool]

    def check_stop(self) -> None:
        """Raise before a model call or a restart if the operator intervenes.

        Raises:
            ProviderAborted: The operator stopped the run.
            ProviderInterrupted: A steer is pending.
        """
        if self.should_abort():
            raise provider_types.ProviderAborted("operator stopped compaction")
        if self.should_interrupt():
            raise provider_types.ProviderInterrupted("operator steered during compaction")

    def call_endpoint(
        self, instruction: str, call_id: str, count: int
    ) -> provider_types.ProviderResponse:
        """Ask the endpoint for a summary without appending to its request.

        Args:
            instruction: The summary instruction outside the messages.
            call_id: The summary attempt's identity.
            count: The number of messages in the request being summarized.

        Returns:
            The metered response after the caller's retry policy.
        """
        identity = self.turn_identity or kinds.CallContext(execution_id=self.execution_id)
        return self.caller.call(
            self.system,
            self.conversation.to_wire()[:count],
            self.tools,
            self.settings.summary_max_tokens,
            initiator=self.conversation.initiator,
            call_context=dataclasses.replace(identity, model_call_id=call_id),
            compaction=provider_types.Summarize(instruction),
        )

    def call_worker(self, instruction: str, call_id: str) -> provider_types.ProviderResponse:
        """Ask the worker for text without running any returned tool calls.

        Args:
            instruction: The final user instruction.
            call_id: The journal identity for this logical call.

        Returns:
            The provider's response, without changing the conversation.
        """
        self.check_stop()
        self.check_budget()
        identity = self.turn_identity or kinds.CallContext(execution_id=self.execution_id)
        request = self.conversation.complete()
        return self.worker.call(
            system=self.system,
            messages=[*request.to_wire(), {"role": "user", "content": instruction}],
            tools=self.tools,
            max_tokens=self.settings.summary_max_tokens,
            temperature=self.temperature,
            should_abort=self.should_abort,
            should_interrupt=self.should_interrupt,
            context=request_headers.RequestContext.initiated_by(
                request.initiator,
                call_context=dataclasses.replace(identity, model_call_id=call_id),
            ),
        )

    def call_helper(
        self, system: str, prompt: str, call_id: str
    ) -> provider_types.ProviderResponse:
        """Send one tool-less request through the helper's metered provider.

        Args:
            system: The helper's system instruction.
            prompt: The helper's user input.
            call_id: The journal identity for this logical call.

        Returns:
            The helper's response.
        """
        self.check_stop()
        self.check_budget()
        return (self.settings.helper or self.worker).call(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            max_tokens=self.settings.summary_max_tokens,
            temperature=0.0,
            should_abort=self.should_abort,
            should_interrupt=self.should_interrupt,
            context=request_headers.RequestContext.initiated_by(
                self.conversation.initiator,
                call_context=kinds.CallContext(
                    model_call_id=call_id, execution_id=self.execution_id
                ),
            ),
        )
