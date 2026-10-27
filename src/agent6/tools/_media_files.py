"""Inspect bounded media containers without decoding pixels or executing a document."""

from __future__ import annotations

import zlib

from agent6 import kinds, media
from agent6.tools import errors


def _png(data: bytes) -> bool:
    pos = 8
    seen: set[bytes] = set()
    while pos + 12 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        tag = data[pos + 4 : pos + 8]
        end = pos + 12 + size
        if end > len(data) or tag == b"acTL":
            return False
        if not seen and (tag != b"IHDR" or size != 13):
            return False
        if tag == b"IHDR" and (
            size != 13
            or not int.from_bytes(data[pos + 8 : pos + 12], "big")
            or not int.from_bytes(data[pos + 12 : pos + 16], "big")
        ):
            return False
        crc = int.from_bytes(data[end - 4 : end], "big")
        if zlib.crc32(data[pos + 4 : end - 4]) != crc:
            return False
        seen.add(tag)
        if tag == b"IEND":
            return size == 0 and end == len(data) and b"IDAT" in seen
        pos = end
    return False


def _gif(data: bytes) -> bool:
    if (
        len(data) < 14
        or not int.from_bytes(data[6:8], "little")
        or not int.from_bytes(data[8:10], "little")
    ):
        return False
    pos = 13 + (3 * (2 ** ((data[10] & 7) + 1)) if data[10] & 128 else 0)
    frames = 0
    while pos < len(data):
        tag = data[pos]
        pos += 1
        if tag == 0x3B:
            return frames == 1 and pos == len(data)
        if tag == 0x2C:
            frames += 1
            if frames > 1 or pos + 10 > len(data):
                return False
            flags = data[pos + 8]
            pos += 9 + (3 * (2 ** ((flags & 7) + 1)) if flags & 128 else 0)
            pos += 1  # LZW minimum code size precedes the sub-blocks.
        elif tag == 0x21:
            if pos >= len(data):
                break
            if data[pos] == 0xFF and data[pos + 2 : pos + 13] in (b"NETSCAPE2.0", b"ANIMEXTS1.0"):
                return False
            pos += 1  # The extension label precedes the sub-blocks.
        else:
            return False
        while pos < len(data) and data[pos]:
            pos += data[pos] + 1
        if pos >= len(data):
            break
        pos += 1
    return False


def _webp(data: bytes) -> bool:
    if len(data) < 20 or int.from_bytes(data[4:8], "little") + 8 != len(data):
        return False
    pos = 12
    frames = 0
    while pos + 8 <= len(data):
        tag = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        end = pos + 8 + size
        if end > len(data) or tag in (b"ANIM", b"ANMF"):
            return False
        if tag == b"VP8X" and (size != 10 or data[pos + 8] & 2):
            return False
        if tag in (b"VP8 ", b"VP8L"):
            if size < (10 if tag == b"VP8 " else 5):
                return False
            frames += 1
        pos = end + size % 2
    return frames == 1 and pos == len(data)


def _jpeg_scan_end(data: bytes, pos: int) -> int:
    while pos < len(data):
        marker = data.find(b"\xff", pos)
        if marker < 0 or marker + 1 >= len(data):
            return len(data)
        if data[marker + 1] == 0 or 0xD0 <= data[marker + 1] <= 0xD7:
            pos = marker + 2
            continue
        return marker
    return pos


def _jpeg(data: bytes) -> bool:
    pos = 2
    frame = scan = False
    while pos < len(data):
        if data[pos] != 0xFF:
            return False
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data) or data[pos] in (0, 0xD8):
            return False
        tag = data[pos]
        pos += 1
        if tag == 0xD9:
            return frame and scan and pos == len(data)
        size = int.from_bytes(data[pos : pos + 2], "big")
        if size < 2 or pos + size > len(data):
            return False
        if tag in (0xC0, 0xC1, 0xC2):
            if (
                size < 8
                or not int.from_bytes(data[pos + 3 : pos + 5], "big")
                or not int.from_bytes(data[pos + 5 : pos + 7], "big")
            ):
                return False
            frame = True
        pos += size
        if tag == 0xDA:
            scan = True
            pos = _jpeg_scan_end(data, pos)
    return False


def inspect_media(data: bytes, source_path: str, *, start_line: int = 1) -> kinds.MediaAttachment:
    """Recognize one bounded static image or a PDF header by content, not its filename.

    Args:
        data: The complete file bytes, read under the document cap plus one byte.
        source_path: The requested contained path.
        start_line: A page other than the first refuses; a line limit is ignored for media.

    Returns:
        The attachment after container and size checks.

    Raises:
        ToolError: The file is unsupported, animated, malformed, oversized or paginated.
    """
    advice = "Crop or convert the image, or extract document text, with a jailed command."
    if start_line != 1:
        raise errors.ToolError(f"Cannot read a page of media {source_path!r}. {advice}")
    is_pdf = data.startswith(b"%PDF-")
    cap = media.DOCUMENT_MAX_BYTES if is_pdf else media.IMAGE_MAX_BYTES
    if len(data) > cap:
        raise errors.ToolError(
            f"Media {source_path!r} exceeds {cap // (1024 * 1024)} MiB. {advice}"
        )
    if is_pdf:
        return kinds.MediaAttachment(data, "application/pdf", source_path)
    checks = (
        (data.startswith(b"\x89PNG\r\n\x1a\n"), "image/png", _png),
        (data.startswith(b"\xff\xd8"), "image/jpeg", _jpeg),
        (data.startswith((b"GIF87a", b"GIF89a")), "image/gif", _gif),
        (data.startswith(b"RIFF") and data[8:12] == b"WEBP", "image/webp", _webp),
    )
    for matches, media_type, check in checks:
        if matches and check(data):
            return kinds.MediaAttachment(data, media_type, source_path)
    raise errors.ToolError(f"Unsupported, malformed or animated media {source_path!r}. {advice}")
