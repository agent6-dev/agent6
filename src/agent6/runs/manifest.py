# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read a run's manifest.json. The single reader; the writer is ``app.manifest``.

A leaf beside ``layout.py``: json + path arithmetic, no agent6 imports, so app,
the viewmodel, and the CLI all parse a run's manifest through one owner instead
of each re-deriving the read + error-catch + shape guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestError(Exception):
    """A run's manifest.json is missing, unreadable, corrupt, or not a JSON
    object. Carries the underlying cause as its message, so a caller that wants
    to surface a detail can render it."""


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """Parse ``<run_dir>/manifest.json`` into a dict, or raise ``ManifestError``.

    Catches the read (``OSError``) and the parse (any ``ValueError``: a truncated
    JSON is a ``JSONDecodeError`` and a torn-UTF-8 tail a ``UnicodeDecodeError``,
    both ``ValueError`` subclasses), so a corrupt manifest degrades through one
    typed error. A manifest that parses to a non-object also raises, so a caller
    never gets a half-typed value.
    """
    path = run_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest is not a JSON object")
    return data
