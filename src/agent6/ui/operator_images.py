# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Take the images an operator sends with a message, for the web and ACP front-ends.

A front-end hands over base64 text and the media type the client declared. Nothing here
opens a path or fetches a URL: the bytes come from the request alone.
"""

from __future__ import annotations

import base64
import pathlib
from collections.abc import Sequence

from agent6 import kinds, media, operator_input
from agent6.sessions import attachments
from agent6.tools import media_files, tool_errors

# The base64 length of the largest image, checked before any decoding.
_ENCODED_MAX = 4 * ((media.IMAGE_MAX_BYTES + 2) // 3)
# The base64 length of a message's images together, which bounds a request body.
ENCODED_TOTAL_MAX = 4 * ((media.IMAGE_TOTAL_BYTES + 2) // 3)

# A declared media type and the base64 payload, as a client sent them.
EncodedImage = tuple[str, str]


class ImageRefusedError(Exception):
    """An image cannot go with the message; the text says which one and why."""


def decode(encoded: Sequence[EncodedImage]) -> tuple[kinds.MediaAttachment, ...]:
    """Decode and check the images of one operator message, under `read_file`'s limits.

    Args:
        encoded: The images in the order given.

    Returns:
        The images: static PNG, JPEG, GIF or WebP, each at most 2 MiB, at most four and
        4 MiB together.

    Raises:
        ImageRefusedError: An image is too large, malformed, animated, not an image, or its
            container is not the declared type; or the message carries too many.
    """
    if len(encoded) > media.IMAGE_MAX_COUNT:
        raise ImageRefusedError("a message carries at most four images")
    images: list[kinds.MediaAttachment] = []
    for number, (declared, data) in enumerate(encoded, start=1):
        name = f"image {number}"
        if len(data) > _ENCODED_MAX:
            raise ImageRefusedError(f"{name} is over the 2 MiB limit")
        try:
            image = media_files.inspect_media(base64.b64decode(data, validate=True), name)
        except ValueError:
            raise ImageRefusedError(f"{name} is not valid base64") from None
        except tool_errors.ToolError:
            raise ImageRefusedError(
                f"{name} is not a static PNG, JPEG, GIF or WebP image of at most 2 MiB"
            ) from None
        if image.modality != "image":
            raise ImageRefusedError(
                f"{name} is a PDF; name its path in the workspace and the model reads it"
            )
        if image.media_type != declared:
            raise ImageRefusedError(f"{name} was sent as {declared!r} but is {image.media_type}")
        images.append(image)
    if sum(image.byte_size for image in images) > media.IMAGE_TOTAL_BYTES:
        raise ImageRefusedError("the images of a message total at most 4 MiB")
    return tuple(images)


def accept(
    session_dir: pathlib.Path, text: str, encoded: Sequence[EncodedImage]
) -> operator_input.OperatorInput:
    """Check and store a message's images, and return the message that names them.

    Args:
        session_dir: The session the message goes to; a new session's reserved directory.
        text: The operator's words; "" for a message of images alone.
        encoded: The images, after the words.

    Returns:
        The message: the words, then a reference per stored image.

    Raises:
        ImageRefusedError: An image is refused; nothing was stored.
        OSError: An image cannot be written.
    """
    images = decode(encoded)
    refs = tuple(attachments.store(session_dir, image) for image in images)
    return operator_input.OperatorInput(parts=(*((text,) if text else ()), *refs))
