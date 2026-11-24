from __future__ import annotations

import pathlib
from typing import Any

from agent6 import review_records
from agent6.providers import provider_types
from agent6.sessions import completion, layout

SYSTEM = " Recorded reviewer prompt.\n"
OPENING = "TASK:\nReview π\n\nREVIEW RULES (base tree):\nKeep the trailing space. \n\nDIFF:\n+x\n"
NOTICE = "[harness] reply 1 of 7."
SCHEMA = {"type": "object", "properties": {"argv": {"type": "array"}}}


def responses_body() -> dict[str, Any]:
    """Return a Responses opening with a tool description distinct from the tree's."""
    return {
        "model": "source-model",
        "instructions": SYSTEM,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
            for text in (OPENING, NOTICE)
        ],
        "tools": [
            {
                "type": "function",
                "name": "run_command",
                "description": "Recorded command description.",
                "parameters": SCHEMA,
                "strict": False,
            }
        ],
        "tool_choice": "auto",
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "low", "summary": "auto"},
        "temperature": 0.25,
        "max_output_tokens": 18000,
    }


def record_review(
    state: pathlib.Path,
    session_id: str,
    body: dict[str, Any],
    *,
    focus: str = "security",
) -> layout.SessionLayout:
    """Write a finished panel child with object-backed requests in its own evidence bucket."""
    child = layout.SessionLayout(state, session_id, "reviewers")
    sink = provider_types.TranscriptSink(child.transcripts_dir, objects_dir=child.objects_dir)
    requests: tuple[tuple[str, dict[str, Any]], ...] = (
        ("helper:summarizer", {"messages": []}),
        (f"reviewer:{focus}", body),
        (f"reviewer:{focus}", {"input": [{"type": "function_call_output", "output": "later"}]}),
    )
    for actor, request in requests:
        sink.record(
            actor=actor,
            url="https://recorded.invalid/v1/responses",
            request_headers={},
            request_body=request,
            response_status=200,
            response_body={},
        )
    completion.write_completion(
        child.session_dir,
        completion.CompletionRecord(
            session_id=session_id,
            review_binding=review_records.ReviewBinding(
                "panel", 0, review_records.ReviewTarget("b", "bt", "h", "ht", "h", (), "revisions")
            ),
            text='{"verdict":"pass","findings":[],"summary":"Final, not the opening"}',
            stop_reason="answered",
            calls=3,
            output_tokens=60,
        ),
    )
    return child
