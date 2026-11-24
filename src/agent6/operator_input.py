# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The operator's input: typed text and references to the images sent with it."""

from __future__ import annotations

from typing import Literal

import pydantic

ImageMediaType = Literal["image/gif", "image/jpeg", "image/png", "image/webp"]


class ImageRef(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Name one image in a session's attachments; the bytes stay in the file the digest names."""

    media_type: ImageMediaType = pydantic.Field(description="The image's media type.")
    byte_size: int = pydantic.Field(gt=0, description="The size of the stored image in bytes.")
    sha256: str = pydantic.Field(
        pattern=r"^[0-9a-f]{64}$", description="The SHA-256 of the stored bytes, in hex."
    )

    def label(self) -> str:
        """Return the words that stand for the image in text: its type, size and digest."""
        size = (self.byte_size + 1023) // 1024
        return f"image {self.media_type}, {size} KiB, sha256 {self.sha256[:12]}"


class OperatorInput(pydantic.BaseModel, frozen=True, extra="forbid"):
    """One operator message: text parts and image references, in the order they were given."""

    parts: tuple[str | ImageRef, ...] = pydantic.Field(
        default=(), description="The text parts and image references, in order."
    )
    message_id: str = pydantic.Field(
        default="", description="The id the front-end that took the message gave it, or empty."
    )

    @property
    def images(self) -> tuple[ImageRef, ...]:
        """The image references, in order."""
        return tuple(part for part in self.parts if isinstance(part, ImageRef))

    @property
    def text(self) -> str:
        """The message as words: each text part, and each image as its bracketed label."""
        return "\n\n".join(
            part if isinstance(part, str) else f"[{part.label()}]" for part in self.parts if part
        )
