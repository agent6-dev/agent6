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

from agent6.runs.manifest import MANIFEST_VERSION, ManifestError, read_manifest

_DATA = Path(__file__).parent / "data"


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
    assert m.version == MANIFEST_VERSION
    assert m.mode == "run"
    assert m.run_branch is None
    assert m.models.worker is None
    assert m.merged is None and m.compare is None


def test_legacy_version_1_and_missing_profile(tmp_path: Path) -> None:
    # A real pre-reshape dir: version 1, workflow without `preset`.
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
    assert m.workflow.preset == ""


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


def test_write_manifest_bytes_fresh(tmp_path: Path) -> None:
    # Byte pin of the writer's emitted JSON (the read side is pinned above; this
    # pins the EXACT bytes write_manifest lands on disk: key set, key order,
    # indent, null shape, trailing newline). A fresh run: no fork/merge/compare.
    from agent6.app.manifest import write_manifest
    from agent6.runs.manifest import ModelBrief, ModelsBrief, RunManifest, WorkflowStamp

    m = RunManifest(
        agent6_version="0.1.0",
        run_id="r-fresh01",
        mode="run",
        start_ts="2026-07-16T00:00:00.000000+00:00",
        user_task="add a feature",
        base_sha="0" * 40,
        base_branch="master",
        run_branch="agent6/r-fresh01",
        models=ModelsBrief(
            worker=ModelBrief(provider="anthropic", model="claude-x"),
            reviewer=ModelBrief(provider="anthropic", model="claude-y"),
        ),
        workflow=WorkflowStamp(critic="off", revise_prompt="on", preset="strict"),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    assert path.read_text(encoding="utf-8") == (_DATA / "golden_manifest_fresh.json").read_text(
        encoding="utf-8"
    )


def test_write_manifest_bytes_stamped_lane(tmp_path: Path) -> None:
    # Byte pin of a fully-stamped fan-out lane: fork lineage + merge stamp +
    # parallel_id/lane + compare, so every optional nested stamp's serialized
    # shape is frozen, not just the fresh subset.
    from agent6.app.manifest import write_manifest
    from agent6.runs.manifest import (
        CompareStamp,
        MergeStamp,
        ModelBrief,
        ModelsBrief,
        RunManifest,
        WorkflowStamp,
    )

    m = RunManifest(
        agent6_version="0.1.0",
        run_id="r-lane02",
        mode="run",
        start_ts="2026-07-16T00:00:00.000000+00:00",
        user_task="fan-out lane",
        base_sha="1" * 40,
        base_branch="master",
        run_branch="agent6/r-lane02",
        models=ModelsBrief(worker=ModelBrief(provider="openai", model="gpt-z")),
        workflow=WorkflowStamp(critic="on", revise_prompt="off", preset=""),
        parent_run_id="r-parent",
        forked_from_turn=7,
        forked_from_sha="2" * 40,
        merged=MergeStamp(
            into="master",
            sha="3" * 40,
            ts="2026-07-16T01:00:00.000000+00:00",
            tip="4" * 40,
        ),
        parallel_id="p-abc",
        lane=1,
        compare=CompareStamp(
            rank=1,
            of=3,
            winner=True,
            ranked_by="judge",
            rationale="cleanest diff",
            judge_cost_usd=0.0102,
            judge_cost_partial=True,
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    golden = (_DATA / "golden_manifest_stamped.json").read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == golden
    # The pinned bytes round-trip back to an equal model (writer <-> reader).
    assert read_manifest(tmp_path) == m


def test_rewriting_a_newer_manifest_is_refused(tmp_path: Path) -> None:
    """Reads stay tolerant so every historical run keeps rendering, but a
    REWRITE of a manifest a newer agent6 wrote is refused: extra="ignore" drops
    the keys this binary doesn't know, and a merge/compare stamp would silently
    downgrade the record it was only supposed to annotate."""
    from agent6.app.manifest import write_manifest

    _write(tmp_path, {"version": 4, "run_id": "r-1", "future_key": {"x": 1}})
    m = read_manifest(tmp_path)  # reading it is fine
    assert m.run_id == "r-1"
    with pytest.raises(ManifestError, match="version 4"):
        write_manifest(tmp_path / "manifest.json", m)
    # Untouched on disk: the newer record keeps its version AND its keys.
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == 4
    assert on_disk["future_key"] == {"x": 1}


def test_rewriting_an_older_manifest_upgrades_it(tmp_path: Path) -> None:
    """An OLDER manifest has no keys this binary can lose, so a stamp rewrite
    upgrades the version claim to the shape it actually wrote."""
    from agent6.app.manifest import write_manifest
    from agent6.runs.manifest import MANIFEST_VERSION

    _write(tmp_path, {"version": 1, "run_id": "r-old"})
    write_manifest(tmp_path / "manifest.json", read_manifest(tmp_path))
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == MANIFEST_VERSION
    assert on_disk["run_id"] == "r-old"


def test_merge_and_lane_stamps_survive_a_newer_manifest(tmp_path: Path) -> None:
    """Both rewrite paths degrade instead of crashing on a manifest they may not
    rewrite: the merge already happened and the lane import already stands, so
    each leaves the newer record untouched and (for the lane) reports it."""
    from agent6.app.merge import record_merge_in_manifest
    from agent6.app.parallel import _stamp  # pyright: ignore[reportPrivateUsage]
    from agent6.runs.layout import RunLayout

    run_dir = tmp_path / "runs" / "r-newer"
    run_dir.mkdir(parents=True)
    payload = {"version": 4, "run_id": "r-newer", "future_key": 1}
    _write(run_dir, payload)

    layout = RunLayout(state_dir=tmp_path, run_id="r-newer")
    record_merge_in_manifest(layout, merged_into="main", merged_sha="abc123")
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) == payload

    err = _stamp(run_dir, lane=2)
    assert err is not None and "version 4" in err
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) == payload


def test_legacy_profile_stamp_reads_as_preset(tmp_path: Path) -> None:
    """A run recorded before the preset rename keeps its strategy: replay_preset
    feeds resume and fork, so dropping the old spelling would silently restart
    an ultra run without its blocking veto panel."""
    _write(
        tmp_path,
        {"version": 2, "workflow": {"profile": "ultra", "profile_from_flag": True}},
    )
    stamp = read_manifest(tmp_path).workflow
    assert stamp.preset == "ultra"
    assert stamp.preset_from_flag is True
    assert stamp.replay_preset == "ultra"


def test_legacy_config_selected_profile_is_not_replayed_as_a_flag(tmp_path: Path) -> None:
    # Same rule the current spelling gets: only a FLAG-selected preset replays.
    _write(tmp_path, {"version": 2, "workflow": {"profile": "quick"}})
    stamp = read_manifest(tmp_path).workflow
    assert stamp.preset == "quick"
    assert stamp.replay_preset == ""
