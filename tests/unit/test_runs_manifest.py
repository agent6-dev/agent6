# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""runs.manifest.read_manifest: the one manifest.json reader. Every failure
shape (missing, unreadable, corrupt JSON, torn UTF-8, non-object) degrades
through the typed ManifestError, so callers never see a raw OSError/ValueError
or a half-typed non-dict value."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.runs.manifest import ManifestError, read_manifest


def test_reads_a_valid_manifest(tmp_path: Path) -> None:
    payload = {"run_id": "r-1", "mode": "plan", "base_sha": "abc"}
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_manifest(tmp_path) == payload


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_unreadable_manifest_raises(tmp_path: Path) -> None:
    # manifest.json as a directory: read_text raises IsADirectoryError (an
    # OSError) regardless of uid, unlike a chmod-000 probe that root ignores.
    (tmp_path / "manifest.json").mkdir()
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_corrupt_json_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_torn_utf8_raises(tmp_path: Path) -> None:
    # A torn multibyte write is a UnicodeDecodeError (a ValueError), which the
    # reader folds into the same typed error instead of leaking it.
    (tmp_path / "manifest.json").write_bytes(b'{"run_id": "\x80')
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_non_object_manifest_raises(tmp_path: Path) -> None:
    for bad in ("[]", "null", '"x"', "3"):
        (tmp_path / "manifest.json").write_text(bad, encoding="utf-8")
        with pytest.raises(ManifestError, match="not a JSON object"):
            read_manifest(tmp_path)
