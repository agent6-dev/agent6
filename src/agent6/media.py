"""Media retention and fixed context reservations."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
from typing import Any, Literal

from agent6 import kinds

# Anthropic vision guide: ~1,600 tokens per 1.15 MP image, at 4 chars/token; PDF is uncalibrated.
MEDIA_CONTEXT_CHARS = 6_400


def _source(block: dict[str, Any]) -> tuple[str, str] | None:
    if block.get("type") not in (
        "image",
        "document",
        "resource",
        "image_url",
        "input_image",
        "input_file",
        "file",
    ):
        return None
    source = block.get("source")
    if block.get("type") in ("image", "document"):
        if isinstance(source, dict) and source.get("type") == "base64":
            return str(source.get("media_type", "")), str(source.get("data", ""))
        if "data" in block and "mimeType" in block:
            return str(block["mimeType"]), str(block["data"])
    resource = block.get("resource")
    if block.get("type") == "resource" and isinstance(resource, dict) and "blob" in resource:
        return str(resource.get("mimeType", "")), str(resource["blob"])
    file = block.get("file")
    url = file.get("file_data") if isinstance(file, dict) else block.get("file_data")
    if block.get("type") in ("image_url", "input_image"):
        url = block.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        media_type, data = url[5:].split(";base64,", 1)
        return media_type, data
    return None


def _source_path(text: str) -> str:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("attachment"), dict):
        return str(obj["attachment"].get("source_path", ""))
    if text.startswith(("image: ", "document: ")) and text.endswith(" KiB"):
        return text.split(": ", 1)[1].rsplit(", ", 1)[0]
    if text.startswith(("image from tool call ", "document from tool call ")):
        return text.partition(": ")[2]
    return ""


def omit_payloads(value: Any) -> Any:
    """Copy a wire value with media metadata and scrub echoes of its payloads.

    Args:
        value: A request, transcript or snapshot value before serialization.

    Returns:
        A JSON-safe copy containing no recognized media payloads.
    """
    payloads: set[str] = set()

    def walk(obj: Any, path: str = "") -> Any:
        if isinstance(obj, list):
            items: list[Any] = []
            for item in obj:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    path = _source_path(item["text"]) or path
                items.append(walk(item, path))
            return items
        if not isinstance(obj, dict):
            return obj
        if source := _source(obj):
            media_type, encoded = source
            metadata: dict[str, Any] = {
                "source_path": path,
                "media_type": media_type,
                "payload_omitted": True,
            }
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                metadata["error"] = "invalid media payload"
            else:
                metadata.update(byte_size=len(data), sha256=hashlib.sha256(data).hexdigest())
                payloads.add(encoded)
            return {"type": obj["type"], "source": metadata}
        out = {key: walk(item, path) for key, item in obj.items()}
        if (
            obj.get("type") == "tool_result"
            and isinstance(obj.get("content"), list)
            and any(isinstance(b, dict) and _source(b) for b in obj["content"])
        ):
            out["content"] = "\n".join(
                str(b["text"])
                if b.get("type") == "text"
                else (
                    json.dumps(b, ensure_ascii=False)
                    + " Media payload omitted; use read_file for a fresh image."
                )
                for b in out["content"]
            )
        return out

    clean = walk(value)
    if not payloads:
        return clean
    # A provider may quote a request's base64 outside a structured media block.
    ordered = sorted(payloads, key=len, reverse=True)

    def scrub_echoes(obj: Any) -> Any:
        if isinstance(obj, str):
            for encoded in ordered:
                obj = obj.replace(encoded, "[media payload omitted]")
            return obj
        if isinstance(obj, list):
            return [scrub_echoes(item) for item in obj]
        if isinstance(obj, dict):
            return {key: scrub_echoes(item) for key, item in obj.items()}
        return obj

    return scrub_echoes(clean)


IMAGE_MAX_BYTES = 2 * 1024 * 1024
IMAGE_TOTAL_BYTES = 4 * 1024 * 1024
IMAGE_MAX_COUNT = 4
DOCUMENT_MAX_BYTES = 10 * 1024 * 1024
DOCUMENT_MAX_COUNT = 1


@dataclasses.dataclass(slots=True)
class AttachmentBudget:
    """Count accepted attachments within one turn or outbound request."""

    image_bytes: int = 0
    image_count: int = 0
    document_count: int = 0

    def refusal(self, attachment: kinds.MediaAttachment) -> str:
        """Accept an attachment or return a path-bearing limit refusal.

        Args:
            attachment: The original bytes to count.

        Returns:
            Empty text on acceptance, otherwise the limit and a recovery suggestion.
        """
        reason = ""
        if attachment.modality == "document":
            if attachment.byte_size > DOCUMENT_MAX_BYTES:
                reason = "the limit is 10 MiB per document"
            elif self.document_count >= DOCUMENT_MAX_COUNT:
                reason = "the limit is one document per turn and request"
            else:
                self.document_count += 1
        elif attachment.byte_size > IMAGE_MAX_BYTES:
            reason = "the limit is 2 MiB per image"
        elif self.image_count >= IMAGE_MAX_COUNT:
            reason = "the limit is four images per turn and request"
        elif self.image_bytes + attachment.byte_size > IMAGE_TOTAL_BYTES:
            reason = "the limit is 4 MiB of images per turn and request"
        else:
            self.image_count += 1
            self.image_bytes += attachment.byte_size
        if not reason:
            return ""
        suggestion = (
            "Extract the text with a jailed command."
            if attachment.modality == "document"
            else "Crop or convert the image with a jailed command."
        )
        return (
            f"Cannot deliver {attachment.modality} {attachment.source_path!r}: {reason}. "
            f"{suggestion}"
        )


Support = Literal["supported", "unsupported", "unknown"]


@dataclasses.dataclass(frozen=True, slots=True)
class InputCapabilities:
    """Explicit listing facts for one resolved provider/model route."""

    route: kinds.ModelRoute = dataclasses.field(default_factory=lambda: kinds.ModelRoute("", ""))
    image: Support = "unknown"
    document: Support = "unknown"


def input_modalities(value: Any) -> frozenset[Literal["image", "document"]]:
    """Return the media modalities present in a serialized request.

    Args:
        value: A native, OpenAI or MCP request value.

    Returns:
        The modalities whose inline payloads the request contains.
    """
    found: set[Literal["image", "document"]] = set()
    if isinstance(value, dict):
        if source := _source(value):
            found.add("document" if source[0] == "application/pdf" else "image")
        for item in value.values():
            found.update(input_modalities(item))
    elif isinstance(value, list):
        for item in value:
            found.update(input_modalities(item))
    return frozenset(found)
