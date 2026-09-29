# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for text-embedded OpenAI tool-call recovery."""

from __future__ import annotations

import json
import pathlib
from typing import Any, ClassVar

import pytest

from agent6 import event_log
from agent6.config import Config
from agent6.harness import _chain, _conversation, _loop_state, loop
from agent6.providers import (
    OpenAIProvider,
    ToolDefinition,
    TranscriptSink,
    _openai_parse,
    _openai_recovery,
)
from agent6.tools import dispatch
from tests.unit.turn_context import turn_context

_TOOLS = frozenset({"list_dir", "read_file"})
_READ_FILE = ToolDefinition(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
_LIST_DIR = ToolDefinition(
    name="list_dir",
    description="List a directory",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
    },
)
_RECOVERY_TOOLS = [_READ_FILE, _LIST_DIR]


class _Response:
    status_code = 200
    headers: ClassVar[dict[str, str]] = {}
    text = ""

    def __init__(self, content: str):
        self._content = content

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _transcripts(path: pathlib.Path) -> list[dict[str, Any]]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in path.glob("*.json")]
    return sorted(docs, key=lambda doc: doc["seq"])


def test_mixed_tag_and_fenced_calls_are_recovered_in_source_order() -> None:
    text = (
        '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>\n'
        '```json\n{"name":"list_dir","arguments":{"path":"src"}}\n```'
    )

    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)

    assert calls == [
        {"name": "read_file", "input": {"path": "a.py"}},
        {"name": "list_dir", "input": {"path": "src"}},
    ]
    assert remaining == ""


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('"{\\"path\\": \\"unterminated"', {"_raw_arguments": '{"path": "unterminated'}),
        ('["a.py"]', {"_raw_arguments": '["a.py"]'}),
        ("null", {"_raw_arguments": "null"}),
    ],
)
def test_tagged_call_with_malformed_arguments_is_recovered_for_an_error(
    arguments: str, expected: dict[str, object]
) -> None:
    text = f'<tool_call>{{"name":"read_file","arguments":{arguments}}}</tool_call>'

    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)

    assert calls == [{"name": "read_file", "input": expected}]
    assert remaining == ""


def test_tagged_unknown_tool_is_recovered_for_an_error() -> None:
    text = '<tool_call>{"name":"write_file","arguments":{"path":"a.py"}}</tool_call>'

    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)

    assert calls == [{"name": "write_file", "input": {"path": "a.py"}}]
    assert remaining == ""


def test_unknown_tagged_call_returns_the_dispatcher_error(tmp_path: pathlib.Path) -> None:
    model_text = '<tool_call>{"name":"write_file","arguments":{"path":"a.py"}}</tool_call>'
    response = _openai_parse.parse_response(_Response(model_text).json(), tool_names=_TOOLS)
    conversation = _conversation.Conversation()
    conversation.notice("Write a.py")
    assistant = conversation.assistant(response.raw["content"])
    dispatcher = dispatch.ToolDispatcher(root=tmp_path, config=Config())
    harness = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=Config(),
        provider=OpenAIProvider(api_key="k", model="weak-model"),
        dispatcher=dispatcher,
        logger=lambda _: None,
    )
    turn = _loop_state.TurnState(iteration=1, resp=response, assistant=assistant)

    harness._turn_dispatch_tools(  # pyright: ignore[reportPrivateUsage]
        _loop_state.LoopState(original_task="Write a.py", tool_calls=0), turn, turn_context()
    )

    results = [item for item in turn.tool_results if isinstance(item, _conversation.ToolResultItem)]
    assert len(results) == 1
    assert json.loads(results[0].content) == {"error": "Unknown tool: write_file"}


@pytest.mark.parametrize(
    "text",
    [
        (
            "````markdown\n"
            "```json\n"
            '{"name":"read_file","arguments":{"path":"example.py"}}\n'
            "```\n"
            "````"
        ),
        (
            "```xml\n"
            '<tool_call>{"name":"read_file","arguments":{"path":"example.py"}}</tool_call>\n'
            "```"
        ),
    ],
)
def test_tool_call_quoted_in_a_markdown_fence_stays_text(text: str) -> None:
    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)

    assert calls == []
    assert remaining == text


@pytest.mark.parametrize(
    "call",
    ["read_file('a.py')", "read_file(path=target)"],
)
def test_tool_code_does_not_dispatch_with_arguments_it_dropped(call: str) -> None:
    text = f"```tool_code\n{call}\n```"

    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)

    assert calls == [{"name": "read_file", "input": {"_raw_arguments": call}}]
    assert remaining == ""


@pytest.mark.parametrize("form", ["bare", "fence", "tag"])
def test_call_markup_inside_an_argument_is_not_a_second_call(form: str) -> None:
    path = "<function=list_dir><parameter=path>.</parameter></function>"
    text = f'{{"name":"read_file","arguments":{{"path":"{path}"}}}}'
    if form == "fence":
        text = f"```json\n{text}\n```"
    elif form == "tag":
        text = f"<tool_call>{text}</tool_call>"

    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)

    assert calls == [{"name": "read_file", "input": {"path": path}}]
    assert remaining == ""


def test_recovered_id_and_result_are_echoed_once_on_the_next_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    model_text = (
        '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>\n'
        '```json\n{"name":"list_dir","arguments":{"path":"."}}\n```'
    )
    responses = iter((_Response(model_text), _Response("done")))
    request_bodies: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        request_bodies.append(json.loads(kwargs["content"]))
        return next(responses)

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    transcripts = tmp_path / "transcripts"
    provider = OpenAIProvider(
        api_key="k",
        model="weak-model",
        transcript_sink=TranscriptSink(transcripts),
    )
    conversation = _conversation.Conversation()
    conversation.notice("Read a.py")
    response = provider.call(
        system="system", messages=conversation.to_wire(), tools=_RECOVERY_TOOLS
    )
    assistant = conversation.assistant(response.raw["content"])

    (tmp_path / "a.py").write_text("answer = 42\n", encoding="utf-8")
    journal = tmp_path / "logs.jsonl"
    events = event_log.EventSink(journal)
    dispatcher = dispatch.ToolDispatcher(root=tmp_path, config=Config(), events=events)
    harness = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=Config(),
        provider=provider,
        dispatcher=dispatcher,
        events=events,
        logger=lambda _: None,
    )
    turn = _loop_state.TurnState(iteration=1, resp=response, assistant=assistant)

    assert (
        harness._turn_dispatch_tools(  # pyright: ignore[reportPrivateUsage]
            _loop_state.LoopState(original_task="Read a.py", tool_calls=0), turn, turn_context()
        )
        is None
    )
    conversation.results(turn.tool_results)
    provider.call(system="system", messages=conversation.to_wire(), tools=_RECOVERY_TOOLS)

    calls = request_bodies[1]["messages"][2]["tool_calls"]
    results = request_bodies[1]["messages"][3:5]
    assert [call["id"] for call in calls] == ["call_text_0", "call_text_1"]
    assert [call["function"]["name"] for call in calls] == ["read_file", "list_dir"]
    assert calls[0]["function"]["arguments"] == '{"path": "a.py"}'
    assert [result["tool_call_id"] for result in results] == [call["id"] for call in calls]
    assert "answer = 42" in results[0]["content"]
    assert '"a.py"' in results[1]["content"]

    recorded = _transcripts(transcripts)
    assert recorded[0]["response"]["body"]["choices"][0]["message"]["content"] == model_text
    assert recorded[1]["request"]["body"] == request_bodies[1]
    journal_events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["type"] for event in journal_events] == [
        "loop.tool.call",
        "tool.call",
        "tool.result",
        "loop.tool.call",
        "tool.call",
        "tool.result",
    ]
    assert [event["name"] for event in journal_events if event["type"] == "tool.call"] == [
        "read_file",
        "list_dir",
    ]


def test_malformed_recovered_call_returns_only_its_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    model_text = '<tool_call>{"name":"read_file","arguments":"{\\"path\\": \\"a.py"}</tool_call>'

    def fake_post(url: str, **kwargs: Any) -> _Response:
        return _Response(model_text)

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = OpenAIProvider(api_key="k", model="weak-model")
    response = provider.call(
        system="system",
        messages=[{"role": "user", "content": "Read a.py"}],
        tools=[_READ_FILE],
    )
    conversation = _conversation.Conversation()
    conversation.notice("Read a.py")
    assistant = conversation.assistant(response.raw["content"])
    dispatcher = dispatch.ToolDispatcher(root=tmp_path, config=Config())
    harness = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=Config(),
        provider=provider,
        dispatcher=dispatcher,
        logger=lambda _: None,
    )
    turn = _loop_state.TurnState(iteration=1, resp=response, assistant=assistant)

    harness._turn_dispatch_tools(  # pyright: ignore[reportPrivateUsage]
        _loop_state.LoopState(original_task="Read a.py", tool_calls=0), turn, turn_context()
    )

    assert response.text == ""
    assert response.raw["content"] == [
        {
            "type": "tool_use",
            "id": "call_text_0",
            "name": "read_file",
            "input": {"_raw_arguments": '{"path": "a.py'},
        }
    ]
    results = [item for item in turn.tool_results if isinstance(item, _conversation.ToolResultItem)]
    assert len(results) == 1
    assert json.loads(results[0].content) == {
        "error": (
            "the arguments were not a JSON object. Resend the call with a"
            " single valid JSON object of arguments."
        )
    }


def test_an_unterminated_fence_hides_no_later_call() -> None:
    """An unterminated fence hides no later call.

    An opener with no closer read as a fence reaching the end of the text quotes every call after a
    code sample the model left open, or an argument value that ended mid-fence, and the run goes
    quiet.
    """
    tag = '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
    calls, remaining = _openai_recovery.coerce_text_tool_calls(
        f"A sample:\n```\nsome code\n\n{tag}", _TOOLS
    )
    assert [c["name"] for c in calls] == ["read_file"]
    assert remaining == "A sample:\n```\nsome code"


def test_a_json_fence_showing_an_object_with_a_name_key_is_not_a_call() -> None:
    """A json fence showing an object with a `name` key is not a call.

    A ```json fence is how a model shows JSON; a package manifest with a `name` field must not be
    dispatched as a tool and cut out of the answer.
    """
    text = 'The manifest:\n\n```json\n{"name": "my-pkg", "version": "1.0.0"}\n```\n\nOK?'
    assert _openai_recovery.coerce_text_tool_calls(text, _TOOLS) == ([], text)


def test_the_same_call_in_a_tag_and_a_fence_is_one_call() -> None:
    """A tag followed by the same object restated in a ```json fence dispatched twice."""
    call = '{"name":"read_file","arguments":{"path":"a.py"}}'
    text = f"<tool_call>{call}</tool_call>\nIn JSON that is:\n```json\n{call}\n```"
    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)
    assert calls == [{"name": "read_file", "input": {"path": "a.py"}}]
    assert remaining == "In JSON that is:"


def test_a_call_in_a_four_backtick_fence_is_recovered() -> None:
    """A call in a four-backtick fence is recovered.

    The fence span begins at its opener, one backtick before where the JSON fence match begins;
    otherwise a call in a longer fence reads as quoted inside itself.
    """
    text = '````json\n{"name":"read_file","arguments":{"path":"a.py"}}\n````'
    calls, _ = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)
    assert [c["name"] for c in calls] == ["read_file"]


def test_arguments_that_are_not_an_object_are_marked_malformed_as_json() -> None:
    """Arguments that are not an object are marked malformed as JSON.

    A list or null in place of the arguments object must not ride a second sentinel the dispatcher
    does not know, which reads to the model as an opaque schema error instead of "not a JSON
    object".
    """
    text = '<tool_call>{"name":"read_file","arguments":["a.py"]}</tool_call>'
    (call,) = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)[0]
    assert set(call["input"]) == {"_raw_arguments"}
    assert json.loads(call["input"]["_raw_arguments"]) == ["a.py"]


def test_a_fence_opened_inside_an_argument_quotes_nothing_after_it() -> None:
    """A fence opened inside an argument quotes nothing after it.

    Read off the raw text, a code fence a parameter value opens pairs with the next fenced block and
    quotes the call between them, which vanishes.
    """
    xml = "<function=apply_edit><parameter=new_string>\n```python\nx = 1\n</parameter></function>"
    tag = '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'
    text = f"{xml}\n{tag}\nexpected output:\n```\nok\n```"
    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS | {"apply_edit"})
    assert [c["name"] for c in calls] == ["apply_edit", "read_file"]
    assert remaining == "expected output:\n```\nok\n```"


def test_a_call_restated_in_a_second_fence_is_one_call() -> None:
    """A call restated in a second fence is one call.

    Two ```json fences holding the same call dispatch it once, as the tag-then-fence case does.
    """
    call = '{"name":"read_file","arguments":{"path":"a.py"}}'
    text = f"first\n```json\n{call}\n```\nsecond\n```json\n{call}\n```"
    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)
    assert calls == [{"name": "read_file", "input": {"path": "a.py"}}]
    assert remaining == "first\n\nsecond"


@pytest.mark.parametrize(
    "inner",
    [
        "<function=read_file><parameter=path>a.py</parameter></function>",
        '```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```',
    ],
)
def test_a_form_wrapped_in_a_tag_leaves_no_marker(inner: str) -> None:
    """A form wrapped in a tag leaves no marker.

    Qwen's template wraps its XML call in `<tool_call>` tags; the tag's markers go with the call.
    """
    calls, remaining = _openai_recovery.coerce_text_tool_calls(
        f"ok\n<tool_call>\n{inner}\n</tool_call>\ndone", _TOOLS
    )
    assert calls == [{"name": "read_file", "input": {"path": "a.py"}}]
    assert remaining == "ok\n\ndone"


def test_an_unclosed_function_ends_at_its_last_closed_parameter() -> None:
    """An unclosed function ends at its last closed parameter.

    A `<function=` block missing its closer must not run to the end of the text and swallow the
    prose after it; a truncated last parameter still runs to the end.
    """
    text = "<function=read_file><parameter=path>a.py</parameter>\nAfter that I summarise."
    calls, remaining = _openai_recovery.coerce_text_tool_calls(text, _TOOLS)
    assert calls == [{"name": "read_file", "input": {"path": "a.py"}}]
    assert remaining == "After that I summarise."
    truncated = "<function=apply_edit><parameter=path>x</parameter><parameter=new_string>abc"
    calls, remaining = _openai_recovery.coerce_text_tool_calls(truncated, _TOOLS | {"apply_edit"})
    assert calls == [{"name": "apply_edit", "input": {"path": "x", "new_string": "abc"}}]
    assert remaining == ""
