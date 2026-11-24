# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An operator message may carry images: the record, the store, and the file bridge."""

from __future__ import annotations

import pathlib
import stat
import struct
import zlib

import pytest

from agent6 import kinds, operator_input
from agent6.sessions import attachments, ipc


def png(shade: int = 0) -> bytes:
    """Return a valid one-pixel greyscale PNG; each shade is a different file."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    pixels = zlib.compress(bytes((0, shade)))
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def stored(session_dir: pathlib.Path, shade: int = 0) -> operator_input.ImageRef:
    """Store one image in the session and return its reference."""
    return attachments.store(session_dir, kinds.MediaAttachment(png(shade), "image/png", ""))


def test_an_image_is_stored_by_digest_owner_only_and_reads_back(tmp_path: pathlib.Path) -> None:
    data = png()
    ref = stored(tmp_path)

    blob = tmp_path / "attachments" / ref.sha256
    assert blob.read_bytes() == data
    assert stat.S_IMODE(blob.stat().st_mode) == 0o600
    assert (ref.media_type, ref.byte_size) == ("image/png", len(data))
    assert [p.name for p in (tmp_path / "attachments").iterdir()] == [ref.sha256]  # no temp left
    assert attachments.load(tmp_path, ref).data == data


def test_a_missing_or_changed_image_is_refused_in_words(tmp_path: pathlib.Path) -> None:
    """A reload trusted whatever the file held: the digest names the bytes the operator sent."""
    ref = stored(tmp_path)
    blob = tmp_path / "attachments" / ref.sha256

    blob.write_bytes(png(9))
    with pytest.raises(attachments.AttachmentError, match="changed after it was stored"):
        attachments.load(tmp_path, ref)
    blob.unlink()
    with pytest.raises(attachments.AttachmentError, match="missing from the session"):
        attachments.load(tmp_path, ref)


def test_copy_takes_only_the_referenced_images(tmp_path: pathlib.Path) -> None:
    source, target = tmp_path / "a", tmp_path / "b"
    kept, _left = stored(source, 1), stored(source, 2)

    attachments.copy(source, target, [kept])

    assert [p.name for p in (target / "attachments").iterdir()] == [kept.sha256]
    assert attachments.load(target, kept).data == png(1)


def test_a_reference_refuses_what_is_not_an_image_digest() -> None:
    with pytest.raises(ValueError, match="media_type"):
        operator_input.ImageRef(media_type="application/pdf", byte_size=1, sha256="0" * 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sha256"):
        operator_input.ImageRef(media_type="image/png", byte_size=1, sha256="../x")


def test_a_message_reads_as_its_words_with_a_label_where_each_image_was(
    tmp_path: pathlib.Path,
) -> None:
    first, second = stored(tmp_path, 1), stored(tmp_path, 2)
    message = operator_input.OperatorInput(parts=("compare", first, "with", second))

    assert message.images == (first, second)
    assert message.text == (
        f"compare\n\n[image image/png, 1 KiB, sha256 {first.sha256[:12]}]\n\nwith\n\n"
        f"[image image/png, 1 KiB, sha256 {second.sha256[:12]}]"
    )
    assert operator_input.OperatorInput(parts=(first,)).text == f"[{first.label()}]"


def test_a_steer_with_images_waits_in_the_inbox_and_is_promoted_with_them(
    tmp_path: pathlib.Path,
) -> None:
    """The slot holds text alone, so only the promoting worker pairs it with its images."""
    ref = stored(tmp_path)

    assert ipc.submit_steer(tmp_path, "what is this?", images=(ref,))
    assert not ipc.steer_request_pending(tmp_path)
    assert ipc.queued_steers(tmp_path) == (ipc.QueuedSteer(text="what is this?", images=(ref,)),)

    assert ipc.promote_queued_steer(tmp_path)
    assert ipc.steer_request_pending(tmp_path)
    assert ipc.take_steer_answer(tmp_path) == "what is this?"
    assert ipc.take_steer_images(tmp_path) == (ref,)
    assert ipc.take_steer_images(tmp_path) == ()  # consumed with their steer


def test_an_abandoned_or_displaced_steer_keeps_no_image_for_the_next_one(
    tmp_path: pathlib.Path,
) -> None:
    """Images left beside the slot would attach to whichever steer came next."""
    ref = stored(tmp_path)
    ipc.submit_steer(tmp_path, "look", images=(ref,))
    ipc.promote_queued_steer(tmp_path)

    # An urgent steer takes the slot; the displaced one returns to the inbox with its image.
    assert ipc.submit_steer(tmp_path, "stop that", now=True)
    assert ipc.take_steer_answer(tmp_path) == "stop that"
    assert ipc.take_steer_images(tmp_path) == ()
    assert ipc.queued_steers(tmp_path) == (ipc.QueuedSteer(text="look", images=(ref,)),)

    ipc.clear_steer_request(tmp_path)
    ipc.promote_queued_steer(tmp_path)
    ipc.clear_steer_answer(tmp_path)  # an abandoned prompt drops the steer and its images
    assert ipc.take_steer_images(tmp_path) == ()


def test_a_launch_record_gives_its_images_only_to_the_message_it_was_written_for(
    tmp_path: pathlib.Path,
) -> None:
    """A record left by a launch that never started must not attach to a later task."""
    ref = stored(tmp_path)
    message = operator_input.OperatorInput(parts=("fix the layout", ref))

    ipc.write_launch_input(tmp_path, message)
    assert ipc.take_launch_images(tmp_path, message.text) == (ref,)
    assert ipc.take_launch_images(tmp_path, message.text) == ()  # consumed

    ipc.write_launch_input(tmp_path, message)
    assert ipc.take_launch_images(tmp_path, "another task") == ()
    assert not (tmp_path / ipc.LAUNCH_INPUT_FILE).exists()
