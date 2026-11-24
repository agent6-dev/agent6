# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An operator's image reaches the model as user content, and survives a snapshot."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import kinds, media, operator_input
from agent6.harness import loop_conversation, loop_state, operator_bridge, provider_call
from agent6.providers import provider_types
from agent6.sessions import attachments, ipc, journal
from tests.unit import test_harness, test_operator_images

png = test_operator_images.png
stored = test_operator_images.stored


def _image(shade: int = 0) -> kinds.MediaAttachment:
    return kinds.MediaAttachment(png(shade), "image/png", "")


def _blocks(conversation: loop_conversation.Conversation) -> list[dict[str, Any]]:
    """Return the blocks of the conversation's last user message, as a provider gets them."""
    return conversation.to_wire()[-1]["content"]


def test_a_notice_sends_its_images_as_user_content_after_its_text() -> None:
    """The image block is the one a `read_file` result carries, outside any tool result."""
    conversation = loop_conversation.Conversation()
    image = _image()
    conversation.notice("look at this", origin="operator", images=(image,))
    conversation.roll_cache_marks()

    text, block = _blocks(conversation)

    assert text == {"type": "text", "text": "look at this"}
    assert block == {**loop_conversation.image_block(image), "cache_control": {"type": "ephemeral"}}
    assert block["source"]["media_type"] == "image/png"
    # The live wire round-trips, marks included.
    again = loop_conversation.Conversation.from_wire(conversation.to_wire(snapshot=True))
    assert again.to_wire(snapshot=True) == conversation.to_wire(snapshot=True)


def test_a_snapshot_keeps_the_reference_and_a_reload_checks_the_stored_bytes(
    tmp_path: pathlib.Path,
) -> None:
    """A resume dropped the image: the snapshot holds no pixels and nothing reloaded them."""
    kept, lost = stored(tmp_path, 1), stored(tmp_path, 2)
    conversation = loop_conversation.Conversation()
    conversation.notice(
        "compare these",
        origin="operator",
        images=(attachments.load(tmp_path, kept), attachments.load(tmp_path, lost)),
    )
    snapshot = media.omit_payloads(conversation.to_wire(snapshot=True))
    assert "data" not in str(snapshot)
    assert loop_conversation.referenced_images(snapshot) == (kept, lost)
    (tmp_path / "attachments" / lost.sha256).write_bytes(png(9))

    reloaded = loop_conversation.Conversation.from_wire(
        snapshot, load_image=lambda ref: operator_bridge.stored_image(tmp_path, ref)
    )

    text, block = _blocks(reloaded)
    assert block == loop_conversation.image_block(attachments.load(tmp_path, kept))
    assert text["text"] == (
        f"compare these\n[{lost.label()} is not shown: its file changed after it was stored]"
    )


def test_a_request_keeps_the_newest_four_images_and_labels_the_older_ones() -> None:
    """Operator images share the request limit `read_file` images have."""
    conversation = loop_conversation.Conversation()
    images = [_image(shade) for shade in range(5)]
    conversation.notice("first", origin="operator", images=(images[0],))
    conversation.assistant([{"type": "text", "text": "ok"}])
    conversation.notice("more", origin="operator", images=tuple(images[1:]))

    refused = conversation.limit_attachments()

    old = loop_conversation.image_ref(images[0])
    why = "newer images took the request's limit of four images and 4 MiB"
    line = f"[{old.label()} is not shown: {why}]"
    assert refused == (("", line),)
    assert conversation.to_wire()[0]["content"] == [{"type": "text", "text": f"first\n{line}"}]
    assert len(_blocks(conversation)) == 5  # the text and the four newest images


def test_a_route_without_image_input_gets_the_refusal_in_words() -> None:
    """A text-only route received the image block and failed the call."""
    conversation = loop_conversation.Conversation()
    image = _image()
    conversation.notice("look", origin="operator", images=(image,))
    events: list[journal.Event] = []
    recovery = provider_call.MediaRecovery(
        media.InputCapabilities(kinds.ModelRoute("p", "text-only"), image="unsupported")
    )

    assert recovery.replace(conversation, None, events.append)

    ref = loop_conversation.image_ref(image)
    line = f"[{ref.label()} is not shown: p/text-only: image input is unsupported]"
    assert _blocks(conversation) == [{"type": "text", "text": f"look\n{line}"}]
    (refused,) = events
    assert isinstance(refused, journal.MediaRefused)
    assert (refused.provider_tool_call_id, refused.reason, refused.text) == (
        "",
        "unsupported",
        line,
    )
    # A provider's own rejection of images takes the same path.
    conversation.notice("again", origin="operator", images=(image,))
    rejected = provider_types.MediaInputError(modality="image", reason="too_large")
    assert recovery.replace(conversation, rejected, None)
    assert "the image exceeds the provider's size limit" in _blocks(conversation)[0]["text"]


def _steering(session_dir: pathlib.Path, events: list[journal.Event]) -> operator_bridge.Steering:
    """A steer reader over the session's real slot and inbox."""

    def prompt() -> str | None:
        return ipc.take_steer_answer(session_dir)

    def clear() -> None:
        ipc.clear_steer_answer(session_dir)
        ipc.clear_steer_request(session_dir)

    def requested() -> bool:
        ipc.promote_queued_steer(session_dir)
        return ipc.steer_request_pending(session_dir)

    return operator_bridge.Steering(
        bridge=operator_bridge.OperatorBridge(
            steer_requested=requested, steer_prompt=prompt, steer_clear=clear
        ),
        parallel=mock.MagicMock(),
        dispatcher=mock.MagicMock(resolved_skills=lambda: ()),
        record_decision=lambda _state, _question, _answer: None,
        log=lambda _line: None,
        emit=events.append,
        session_dir=session_dir,
    )


def _handle(steering: operator_bridge.Steering) -> loop_conversation.Conversation:
    conversation = loop_conversation.Conversation()
    conversation.notice("TASK:\nt", origin="operator")
    conversation.assistant([{"type": "text", "text": "working"}])
    state = loop_state.LoopState(original_task="t", execution_id="exec-1", tool_calls=0)
    assert steering.handle(conversation, 2, state) is None
    return conversation


def test_a_steer_delivers_its_images_and_records_their_references(
    tmp_path: pathlib.Path,
) -> None:
    """A steer's image never left the inbox: the loop read the slot's text alone."""
    ref = stored(tmp_path)
    message = operator_input.OperatorInput(parts=("what is wrong here?", ref))
    events: list[journal.Event] = []
    assert ipc.submit_steer(tmp_path, message.text, images=message.images)

    conversation = _handle(_steering(tmp_path, events))

    text, block = _blocks(conversation)
    assert text["text"].endswith(f"what is wrong here?\n\n[{ref.label()}]")
    assert block == loop_conversation.image_block(attachments.load(tmp_path, ref))
    injected = next(e for e in events if isinstance(e, journal.SteerInjected))
    assert injected.images == (ref,) and injected.text == message.text
    assert not (tmp_path / ipc.STEER_IMAGES_FILE).exists()


def test_a_steer_that_is_only_an_image_is_delivered(tmp_path: pathlib.Path) -> None:
    """A blank slot continues the run, so an image with no words needs its label as the text."""
    ref = stored(tmp_path)
    message = operator_input.OperatorInput(parts=(ref,))
    assert ipc.submit_steer(tmp_path, message.text, images=message.images)

    conversation = _handle(_steering(tmp_path, []))

    assert _blocks(conversation)[1] == loop_conversation.image_block(
        attachments.load(tmp_path, ref)
    )


def test_a_missing_image_is_said_to_the_model_and_the_operator(tmp_path: pathlib.Path) -> None:
    """A blob that vanished was dropped without a word to either side."""
    ref = stored(tmp_path)
    (tmp_path / "attachments" / ref.sha256).unlink()
    events: list[journal.Event] = []
    ipc.submit_steer(tmp_path, "see this", images=(ref,))

    conversation = _handle(_steering(tmp_path, events))

    line = f"[{ref.label()} is not shown: its file is missing from the session's attachments]"
    assert _blocks(conversation) == [
        {"type": "text", "text": mock.ANY},
    ]
    assert _blocks(conversation)[0]["text"].endswith(f"see this\n{line}")
    assert [e.text for e in events if isinstance(e, journal.SteerFeedback)] == [line]


def test_a_directive_takes_no_image_and_says_so(tmp_path: pathlib.Path) -> None:
    """Only an ordinary instruction carries an image; a pin's text re-enters every restart."""
    ref = stored(tmp_path)
    events: list[journal.Event] = []
    ipc.submit_steer(tmp_path, "/pin never rebase", images=(ref,))

    conversation = _handle(_steering(tmp_path, events))

    assert "image" not in {block["type"] for block in _blocks(conversation)}
    assert [e.text for e in events if isinstance(e, journal.SteerFeedback)] == [
        "1 image(s) not delivered: only an ordinary instruction carries one"
    ]


def test_an_opening_delivers_the_images_its_front_end_stored(tmp_path: pathlib.Path) -> None:
    """The task's image stayed in the session directory: the run started from words alone."""
    ref = stored(tmp_path)
    message = operator_input.OperatorInput(parts=("match this mockup", ref))
    ipc.write_launch_input(tmp_path, message)
    events: list[journal.Event] = []

    class _Events:
        path = tmp_path / "logs.jsonl"

        def emit(self, event: journal.Event) -> None:
            events.append(event)

    provider = mock.MagicMock()
    provider.call.return_value = test_harness._resp("done")
    wf = test_harness._wf(mode="run", budget=None, provider=provider, events=_Events())

    wf.run(message.text)

    first = provider.call.call_args_list[0].kwargs["messages"][0]["content"]
    assert first[0]["text"].startswith(f"TASK:\nmatch this mockup\n\n[{ref.label()}]")
    assert first[1]["type"] == "image" and first[1]["source"]["media_type"] == "image/png"
    start = next(e for e in events if isinstance(e, journal.SessionStart))
    assert start.images == (ref,)
    assert not (tmp_path / ipc.LAUNCH_INPUT_FILE).exists()


@pytest.mark.parametrize("words", ["another task", ""])
def test_an_opening_ignores_a_launch_record_written_for_other_words(
    tmp_path: pathlib.Path, words: str
) -> None:
    ref = stored(tmp_path)
    ipc.write_launch_input(tmp_path, operator_input.OperatorInput(parts=("first task", ref)))

    assert ipc.take_launch_images(tmp_path, words) == ()
