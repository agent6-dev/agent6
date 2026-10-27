"""Recognize narrow media rejections without treating unrelated 400s as capability facts."""

from __future__ import annotations

import json
from typing import Any, Literal

from agent6 import media
from agent6.providers import types

_CODES: dict[
    str, tuple[Literal["image", "document"], Literal["unsupported", "invalid", "too_large"]]
] = {
    "document_input_not_supported": ("document", "unsupported"),
    "file_too_large": ("document", "too_large"),
    "image_input_not_supported": ("image", "unsupported"),
    "image_too_large": ("image", "too_large"),
    "invalid_file": ("document", "invalid"),
    "invalid_image": ("image", "invalid"),
    "invalid_image_format": ("image", "invalid"),
    "invalid_image_url": ("image", "invalid"),
}


def raise_rejection(status: int | None, text: str, body: dict[str, Any]) -> None:
    """Raise a typed rejection only for an explicit error about media actually sent.

    Args:
        status: The HTTP status, including errors from a streaming request.
        text: The error body, optionally prefixed with the API label.
        body: The actual serialized request.

    Raises:
        MediaInputError: A recognized, payload-free rejection of media in the request.
    """
    if status not in (400, 413, 422):
        return
    modalities = media.input_modalities(body)
    if not modalities:
        return
    try:
        obj = json.loads(text[text.index("{") :])
    except (ValueError, TypeError):
        obj = {}
    error = obj.get("error", obj) if isinstance(obj, dict) else {}
    code = error.get("code", "") if isinstance(error, dict) else ""
    match = _CODES.get(code) if isinstance(code, str) else None
    if match is not None and match[0] in modalities:
        raise types.MediaInputError(modality=match[0], reason=match[1], status_code=status)
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    message = message.lower()
    for modality in sorted(modalities):
        if any(
            phrase in message
            for phrase in (
                f"{modality} input is unsupported",
                f"does not support {modality} input",
                f"{modality} content blocks are not supported",
            )
        ) or (modality == "image" and "image_url is only supported by certain models" in message):
            raise types.MediaInputError(modality=modality, reason="unsupported", status_code=status)
    return
