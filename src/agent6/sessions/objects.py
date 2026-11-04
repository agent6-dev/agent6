"""Store sanitized session values once and materialize their references for readers."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import threading
from typing import Any

from agent6 import portable

_WRITE_LOCK = threading.Lock()


class ObjectStoreError(ValueError):
    """A record references an absent or unreadable session object."""


def put(value: Any, objects_dir: pathlib.Path) -> dict[str, str]:
    """Publish a sanitized JSON value once and return its reference.

    Args:
        value: The value after credential redaction and media omission.
        objects_dir: The session's object directory.

    Returns:
        The reference to the canonical JSON's SHA-256 hash.
    """
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = objects_dir / f"{digest}.json"
    with _WRITE_LOCK:
        if not path.exists():
            portable.atomic_write(path, text)
    return {"$object": digest}


def reference_fields(record: dict[str, Any], objects_dir: pathlib.Path) -> dict[str, Any]:
    """Replace a sanitized request or snapshot's repeated values with references.

    Args:
        record: The sanitized request body or numbered snapshot.
        objects_dir: The session's object directory.

    Returns:
        A copy with message items and system, instructions and tools values referenced.
    """
    out = dict(record)
    for key in ("messages", "input"):
        if isinstance(items := out.get(key), list):
            out[key] = [put(item, objects_dir) for item in items]
    for key in ("system", "instructions", "tools"):
        if key in out:
            out[key] = put(out[key], objects_dir)
    return out


def materialize(
    record: Any, objects_dir: pathlib.Path, *, record_path: pathlib.Path | None = None
) -> Any:
    """Replace references with their stored values, leaving legacy records unchanged.

    Args:
        record: The decoded JSON record.
        objects_dir: The session's object directory.
        record_path: The record's path for diagnostics; absent for an in-memory record.

    Returns:
        The materialized record.

    Raises:
        ObjectStoreError: An object is missing, unreadable or has an invalid hash.
    """
    label = str(record_path) if record_path is not None else repr(record)[:200]

    def decode(value: Any) -> Any:
        if isinstance(value, list):
            return [decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) != {"$object"}:
            return {key: decode(item) for key, item in value.items()}
        digest = value["$object"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ObjectStoreError(f"invalid object hash {digest!r} in record {label}")
        try:
            return json.loads((objects_dir / f"{digest}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ObjectStoreError(
                f"cannot read object {digest} for record {label}: {exc}"
            ) from exc

    return decode(record)
