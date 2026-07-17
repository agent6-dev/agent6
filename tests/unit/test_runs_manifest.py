# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""runs.manifest: the typed RunManifest reader. Every failure shape (missing,
unreadable, corrupt JSON, torn UTF-8, non-object) degrades through the typed
ManifestError; every historical run dir (old ``version: 1`` shapes, the pre-v2
flat merged_* keys, the legacy ``compare.group``) still parses for rendering;
and the fork/resume ``strict_mode`` gate refuses an unknown mode rather than
falling open to write access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.runs.manifest import ManifestError, read_manifest


def _write(run_dir: Path, payload: object) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_reads_a_valid_manifest(tmp_path: Path) -> None:
    _write(tmp_path, {"run_id": "r-1", "mode": "plan", "base_sha": "abc"})
    m = read_manifest(tmp_path)
    assert m.run_id == "r-1"
    assert m.mode == "plan"
    assert m.base_sha == "abc"


def test_missing_fields_default_so_any_old_dir_renders(tmp_path: Path) -> None:
    # An almost-empty manifest still parses: every field defaults.
    _write(tmp_path, {})
    m = read_manifest(tmp_path)
    assert m.version == 2
    assert m.mode == "run"
    assert m.run_branch is None
    assert m.models.worker is None
    assert m.merged is None and m.compare is None


def test_legacy_version_1_and_missing_profile(tmp_path: Path) -> None:
    # A real pre-reshape dir: version 1, workflow without `profile`.
    _write(
        tmp_path,
        {
            "version": 1,
            "mode": "run",
            "user_task": "do a thing",
            "workflow": {"critic": "off", "revise_prompt": "off"},
        },
    )
    m = read_manifest(tmp_path)
    assert m.version == 1
    assert m.user_task == "do a thing"
    assert m.workflow.profile == ""


def test_legacy_flat_merge_keys_fold_into_merged(tmp_path: Path) -> None:
    # A run merged before this reshape recorded flat merged_into/_sha/_ts.
    _write(
        tmp_path,
        {"run_branch": "agent6/r", "merged_into": "main", "merged_sha": "abc123", "merged_ts": "t"},
    )
    m = read_manifest(tmp_path)
    assert m.merged is not None
    assert m.merged.into == "main"
    assert m.merged.sha == "abc123"
    assert m.merged.ts == "t"


def test_legacy_compare_group_is_ignored(tmp_path: Path) -> None:
    # The pre-dedup stamp carried a `group` key (same fact as parallel_id); it is
    # dropped on read (extra="ignore"), the rest of the stamp survives.
    _write(
        tmp_path,
        {"compare": {"group": "fan", "rank": 1, "of": 2, "winner": True, "ranked_by": "judge"}},
    )
    m = read_manifest(tmp_path)
    assert m.compare is not None
    assert m.compare.rank == 1 and m.compare.winner is True
    assert not hasattr(m.compare, "group")


def test_strict_mode_accepts_the_two_known_modes(tmp_path: Path) -> None:
    for mode in ("run", "plan"):
        _write(tmp_path, {"mode": mode})
        assert read_manifest(tmp_path).strict_mode() == mode


def test_strict_mode_refuses_an_unknown_mode(tmp_path: Path) -> None:
    # The security gate: a damaged mode must NOT silently fall open to write
    # ("run") access; strict_mode refuses loudly. Rendering still reads it raw.
    _write(tmp_path, {"mode": "wat"})
    m = read_manifest(tmp_path)
    assert m.mode == "wat"  # lenient render read
    with pytest.raises(ManifestError, match="unknown run mode"):
        m.strict_mode()


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


def test_stamp_rewrite_restamps_version_to_the_shape_written(tmp_path: Path) -> None:
    # A manifest written by a NEWER agent6 (version 3, unknown keys) that this
    # binary stamp-rewrites loses the keys it doesn't know (extra="ignore"), so
    # the write path must re-stamp version: the on-disk claim matches the shape
    # actually written, never the shape that was lost.
    from agent6.app.manifest import write_manifest
    from agent6.runs.manifest import MANIFEST_VERSION

    _write(tmp_path, {"version": 3, "run_id": "r-1", "future_key": {"x": 1}})
    m = read_manifest(tmp_path)
    write_manifest(tmp_path / "manifest.json", m)
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == MANIFEST_VERSION
    assert "future_key" not in on_disk
    assert on_disk["run_id"] == "r-1"
