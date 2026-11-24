# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The path and provider binding shared by cached listing facts."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from agent6 import config, paths


def cache_path(provider_name: str) -> pathlib.Path | None:
    """Return the provider's cache file, or None when the name is not one path component.

    Provider names are config table keys; a `/` or `..` would write outside the cache dir.

    Args:
        provider_name: The provider's config name.

    Returns:
        `<cache dir>/models/<provider_name>.json`, or None.
    """
    if provider_name in ("", ".", "..") or provider_name != pathlib.Path(provider_name).name:
        return None
    return paths.cache_dir() / "models" / f"{provider_name}.json"


def binding(entry: config.ProviderEntry) -> str:
    """Return the fingerprint of the complete provider entry."""
    return hashlib.sha256(
        json.dumps(entry.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


def read(path: pathlib.Path | None) -> dict[str, Any]:
    """Read the cached listing facts.

    Args:
        path: The provider's cache path.

    Returns:
        The listing object, empty for a missing or unreadable file.
    """
    if path is None:
        return {}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}
