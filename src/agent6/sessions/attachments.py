# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Store the images an operator sent with a message, in the session directory.

A file is named by the SHA-256 of its bytes and written 0600 under `attachments/`. The
session directory is outside the workspace, so no commit holds an image, and a jailed
command under `full` isolation cannot read one.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Iterable

from agent6 import kinds, operator_input, paths, portable

DIR_NAME = "attachments"


class AttachmentError(Exception):
    """A stored image is missing or does not match its reference."""


def store(session_dir: pathlib.Path, image: kinds.MediaAttachment) -> operator_input.ImageRef:
    """Write a validated image into the session's attachments.

    Args:
        session_dir: The session directory.
        image: The image, already checked by `media_files.inspect_media`.

    Returns:
        The reference that names the stored file.

    Raises:
        OSError: The file cannot be written.
        ValidationError: The media type is not an image's.
    """
    ref = operator_input.ImageRef.model_validate(
        {"media_type": image.media_type, "byte_size": image.byte_size, "sha256": image.sha256}
    )
    directory = session_dir / DIR_NAME
    paths.mkdir_for_real_user(directory)
    portable.atomic_write(directory / ref.sha256, image.data, mode=0o600)
    return ref


def load(session_dir: pathlib.Path, ref: operator_input.ImageRef) -> kinds.MediaAttachment:
    """Read a stored image back and check it against its reference.

    Args:
        session_dir: The session directory.
        ref: The reference a journal event or a checkpoint recorded.

    Returns:
        The image.

    Raises:
        AttachmentError: The file is missing, or its size or digest differs from the reference.
    """
    try:
        with (session_dir / DIR_NAME / ref.sha256).open("rb") as handle:
            data = handle.read(ref.byte_size + 1)
    except OSError as exc:
        raise AttachmentError("its file is missing from the session's attachments") from exc
    if len(data) != ref.byte_size or hashlib.sha256(data).hexdigest() != ref.sha256:
        raise AttachmentError("its file changed after it was stored")
    return kinds.MediaAttachment(data, ref.media_type, "")


def copy(
    source_dir: pathlib.Path, target_dir: pathlib.Path, refs: Iterable[operator_input.ImageRef]
) -> None:
    """Copy the referenced images from one session's attachments to another's.

    A missing or changed source file is left out: the target's `load` reports it.

    Args:
        source_dir: The session that holds the images.
        target_dir: The session that receives them.
        refs: The references to copy.

    Raises:
        OSError: A file cannot be written.
    """
    for ref in refs:
        try:
            image = load(source_dir, ref)
        except AttachmentError:
            continue
        store(target_dir, image)
