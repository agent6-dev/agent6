# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""runs.manifest: the typed SessionManifest reader.

Every failure shape (missing, unreadable, corrupt JSON, torn UTF-8, non-object) degrades through the
typed ManifestError; every historical run dir (old ``version: 1`` shapes, the pre-v2 flat merged_*
keys, the legacy ``compare.group``) still parses for rendering; and the fork/resume ``session_mode``
gate refuses an unknown mode rather than falling open to write access.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6.sessions import manifest as sessions_manifest

_DATA = pathlib.Path(__file__).parent / "data"


def _write(session_dir: pathlib.Path, payload: object) -> None:
    (session_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_reads_a_valid_manifest(tmp_path: pathlib.Path) -> None:
    _write(tmp_path, {"session_id": "r-1", "mode": "plan", "base_sha": "abc"})
    m = sessions_manifest.read_manifest(tmp_path)
    assert m.session_id == "r-1"
    assert m.mode == "plan"
    assert m.base_sha == "abc"


def test_missing_fields_default_so_any_old_dir_renders(tmp_path: pathlib.Path) -> None:
    # An almost-empty manifest still parses: every field defaults.
    _write(tmp_path, {})
    m = sessions_manifest.read_manifest(tmp_path)
    assert m.version == sessions_manifest.MANIFEST_VERSION
    # mode has NO default: it is the privilege gate's input (see the
    # fall-open test below), so an absent key stays absent.
    assert m.mode == ""
    assert m.run_branch is None
    assert m.models.driver is None
    assert m.merged is None and m.compare is None


def test_legacy_version_1_and_missing_profile(tmp_path: pathlib.Path) -> None:
    # A real pre-reshape dir: version 1, harness without `preset`.
    _write(
        tmp_path,
        {
            "version": 1,
            "mode": "run",
            "user_task": "do a thing",
            "harness": {"critic": "off", "revise_prompt": "off"},
        },
    )
    m = sessions_manifest.read_manifest(tmp_path)
    assert m.version == 1
    assert m.user_task == "do a thing"
    assert m.harness.preset == ""


def test_unknown_keys_are_dropped_never_folded(tmp_path: pathlib.Path) -> None:
    """Superseded or foreign merge keys are ignored, not converted.

    A manifest carrying only flat merged_* keys reads as unmerged, the safe direction.
    """
    _write(
        tmp_path,
        {"run_branch": "agent6/r", "merged_into": "main", "merged_sha": "abc123", "merged_ts": "t"},
    )
    m = sessions_manifest.read_manifest(tmp_path)
    assert m.merged is None
    assert m.run_branch == "agent6/r"


def test_legacy_compare_group_is_ignored(tmp_path: pathlib.Path) -> None:
    # The pre-dedup stamp carried a `group` key (the lineage's fact); it is
    # dropped on read (extra="ignore"), the rest of the stamp survives.
    _write(
        tmp_path,
        {"compare": {"group": "fan", "rank": 1, "of": 2, "winner": True, "ranked_by": "judge"}},
    )
    m = sessions_manifest.read_manifest(tmp_path)
    assert m.compare is not None
    assert m.compare.rank == 1 and m.compare.winner is True
    assert not hasattr(m.compare, "group")


def test_session_mode_accepts_the_two_known_modes(tmp_path: pathlib.Path) -> None:
    for mode in ("run", "plan"):
        _write(tmp_path, {"mode": mode})
        assert sessions_manifest.read_manifest(tmp_path).session_mode() == mode


def test_session_mode_refuses_an_unknown_mode(tmp_path: pathlib.Path) -> None:
    # The security gate: a damaged mode must NOT silently fall open to write
    # ("run") access; session_mode refuses loudly. Rendering still reads it raw.
    _write(tmp_path, {"mode": "wat"})
    m = sessions_manifest.read_manifest(tmp_path)
    assert m.mode == "wat"  # lenient render read
    with pytest.raises(sessions_manifest.ManifestError, match="unknown session mode"):
        m.session_mode()


def test_missing_manifest_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(sessions_manifest.ManifestError):
        sessions_manifest.read_manifest(tmp_path)


def test_unreadable_manifest_raises(tmp_path: pathlib.Path) -> None:
    # manifest.json as a directory: read_text raises IsADirectoryError (an
    # OSError) regardless of uid, unlike a chmod-000 probe that root ignores.
    (tmp_path / "manifest.json").mkdir()
    with pytest.raises(sessions_manifest.ManifestError):
        sessions_manifest.read_manifest(tmp_path)


def test_corrupt_json_raises(tmp_path: pathlib.Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(sessions_manifest.ManifestError):
        sessions_manifest.read_manifest(tmp_path)


def test_torn_utf8_raises(tmp_path: pathlib.Path) -> None:
    # A torn multibyte write is a UnicodeDecodeError (a ValueError), which the
    # reader folds into the same typed error instead of leaking it.
    (tmp_path / "manifest.json").write_bytes(b'{"session_id": "\x80')
    with pytest.raises(sessions_manifest.ManifestError):
        sessions_manifest.read_manifest(tmp_path)


def test_non_object_manifest_raises(tmp_path: pathlib.Path) -> None:
    for bad in ("[]", "null", '"x"', "3"):
        (tmp_path / "manifest.json").write_text(bad, encoding="utf-8")
        with pytest.raises(sessions_manifest.ManifestError, match="not a JSON object"):
            sessions_manifest.read_manifest(tmp_path)


def test_write_manifest_bytes_fresh(tmp_path: pathlib.Path) -> None:
    # Byte pin of the writer's emitted JSON (the read side is pinned above; this
    # pins the EXACT bytes write_manifest lands on disk: key set, key order,
    # indent, null shape, trailing newline). A fresh run: no fork/merge/compare.
    from agent6.app import manifest as app_manifest

    m = sessions_manifest.SessionManifest(
        agent6_version="0.1.0",
        session_id="r-fresh01",
        mode="run",
        start_ts="2026-07-16T00:00:00.000000+00:00",
        user_task="add a feature",
        base_sha="0" * 40,
        base_branch="master",
        run_branch="agent6/r-fresh01",
        models=sessions_manifest.ModelsBrief(
            driver=sessions_manifest.ModelBrief(provider="anthropic", model="claude-x"),
            reviewer=sessions_manifest.ModelBrief(provider="anthropic", model="claude-y"),
        ),
        harness=sessions_manifest.HarnessStamp(
            review_trigger="off", revise_prompt="on", preset="strict"
        ),
    )
    path = tmp_path / "manifest.json"
    app_manifest.write_manifest(path, m)
    assert path.read_text(encoding="utf-8") == (_DATA / "golden_manifest_fresh.json").read_text(
        encoding="utf-8"
    )


def test_write_manifest_bytes_stamped_lane(tmp_path: pathlib.Path) -> None:
    # Byte pin of a fully-stamped fan-out lane: fork lineage + merge stamp +
    # parallel lineage + compare, so every optional nested stamp's serialized
    # shape is frozen, not just the fresh subset.
    from agent6.app import manifest as app_manifest

    m = sessions_manifest.SessionManifest(
        agent6_version="0.1.0",
        session_id="r-lane02",
        mode="run",
        start_ts="2026-07-16T00:00:00.000000+00:00",
        user_task="fan-out lane",
        base_sha="1" * 40,
        base_branch="master",
        run_branch="agent6/r-lane02",
        models=sessions_manifest.ModelsBrief(
            driver=sessions_manifest.ModelBrief(provider="openai", model="gpt-z")
        ),
        harness=sessions_manifest.HarnessStamp(review_trigger="on", revise_prompt="off", preset=""),
        parent_session_id="r-parent",
        forked_from_turn=7,
        forked_from_sha="2" * 40,
        merged=sessions_manifest.MergeStamp(
            into="master",
            sha="3" * 40,
            ts="2026-07-16T01:00:00.000000+00:00",
            tip="4" * 40,
        ),
        parallel=sessions_manifest.ParallelLineage(group="p-abc", lane=1, coordinator="p-abc"),
        compare=sessions_manifest.CompareStamp(
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
    app_manifest.write_manifest(path, m)
    golden = (_DATA / "golden_manifest_stamped.json").read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == golden
    # The pinned bytes round-trip back to an equal model (writer <-> reader).
    assert sessions_manifest.read_manifest(tmp_path) == m


def test_rewriting_a_newer_manifest_is_refused(tmp_path: pathlib.Path) -> None:
    """A rewrite of a manifest a newer agent6 wrote is refused; reads stay tolerant.

    `extra="ignore"` drops the keys this binary does not know, so a stamp would silently
    downgrade the record it was meant to annotate.
    """
    from agent6.app import manifest as app_manifest

    _write(
        tmp_path,
        {
            "version": sessions_manifest.MANIFEST_VERSION + 1,
            "session_id": "r-1",
            "future_key": {"x": 1},
        },
    )
    m = sessions_manifest.read_manifest(tmp_path)  # reading it is fine
    assert m.session_id == "r-1"
    with pytest.raises(
        sessions_manifest.ManifestError, match=f"version {sessions_manifest.MANIFEST_VERSION + 1}"
    ):
        app_manifest.write_manifest(tmp_path / "manifest.json", m)
    # Untouched on disk: the newer record keeps its version AND its keys.
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == sessions_manifest.MANIFEST_VERSION + 1
    assert on_disk["future_key"] == {"x": 1}


def test_rewriting_an_older_manifest_upgrades_it(tmp_path: pathlib.Path) -> None:
    """A stamp rewrite of an older manifest upgrades the version claim to the shape it wrote."""
    from agent6.app import manifest as app_manifest

    _write(tmp_path, {"version": 1, "session_id": "r-old"})
    app_manifest.write_manifest(
        tmp_path / "manifest.json", sessions_manifest.read_manifest(tmp_path)
    )
    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == sessions_manifest.MANIFEST_VERSION
    assert on_disk["session_id"] == "r-old"


def test_merge_and_lane_stamps_survive_a_newer_manifest(tmp_path: pathlib.Path) -> None:
    """Both rewrite paths degrade on a manifest they may not rewrite, and the lane reports it."""
    from agent6.app import (
        merge,
        parallel,  # pyright: ignore[reportPrivateUsage]
    )
    from agent6.sessions import layout as sessions_layout

    session_dir = tmp_path / "sessions" / "runs" / "r-newer"
    session_dir.mkdir(parents=True)
    payload = {
        "version": sessions_manifest.MANIFEST_VERSION + 1,
        "session_id": "r-newer",
        "future_key": 1,
    }
    _write(session_dir, payload)

    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="r-newer")
    merge.record_merge_in_manifest(layout, merged_into="main", merged_sha="abc123")
    assert json.loads((session_dir / "manifest.json").read_text(encoding="utf-8")) == payload

    err = parallel._stamp(session_dir, lane=2)
    assert err is not None and f"version {sessions_manifest.MANIFEST_VERSION + 1}" in err
    assert json.loads((session_dir / "manifest.json").read_text(encoding="utf-8")) == payload


def test_plan_run_stamps_the_planner_as_its_driver(tmp_path: pathlib.Path) -> None:
    """`sessions show` reads one field for "the model that drove this run".

    Reading the worker unconditionally, a plan run (driven by the planner) displays a model that
    never ran, disagreeing with both the web (which reads the role events) and its own cost block.
    """
    from agent6.app import manifest as app_manifest
    from agent6.config import Config
    from agent6.sessions import layout as sessions_layout

    cfg = Config.model_validate(
        {
            "providers": {"anthropic": {"api_format": "anthropic"}},
            "models": {
                "worker": {"provider": "anthropic", "model": "worker-model"},
                "planner": {"provider": "anthropic", "model": "planner-model"},
            },
        }
    )
    for mode, expected in (("plan", "planner-model"), ("run", "worker-model")):
        layout = sessions_layout.SessionLayout(state_dir=tmp_path / mode, session_id="r")
        layout.ensure()
        app_manifest.write_session_manifest(
            layout,
            session_id="r",
            user_task="t",
            base_sha="",
            base_branch="main",
            run_branch=None,
            cfg=cfg,
            mode=mode,
        )
        driver = sessions_manifest.read_manifest(layout.session_dir).models.driver
        assert driver is not None and driver.model == expected


def test_write_session_manifest_stores_the_operators_words(tmp_path: pathlib.Path) -> None:
    """`user_task` is the operator's words, not the composed prompt.

    `run --skill` and `--from` prepend a skill block and a digest to the engine's task, and the
    composed prompt reached every listing as the task.
    """
    from agent6 import task_text
    from agent6.app import manifest as app_manifest
    from agent6.config import Config
    from agent6.sessions import layout as sessions_layout

    words = "fix the parser " * 20  # past the event's 200-char clip
    composed = (
        f'{task_text.SKILLS_PREAMBLE}\n<skill name="tidy">be tidy</skill>\n---\n'
        f'<prior-run id="r-earlier">what it found</prior-run>\n\n{words}'
    )
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="r-words")
    layout.ensure()
    app_manifest.write_session_manifest(
        layout,
        session_id="r-words",
        user_task=composed,
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=Config(),
        mode="run",
    )
    assert sessions_manifest.read_manifest(layout.session_dir).user_task == words.strip()


def test_a_manifest_with_no_mode_key_does_not_fall_open_to_run(tmp_path: pathlib.Path) -> None:
    """A manifest with no mode key is refused, like an unknown mode value.

    The field defaulted to "run", so a manifest that lost its mode resumed or forked with the
    write-tool surface, the escalation session_mode exists to stop.
    """
    _write(tmp_path, {"version": 3, "session_id": "r", "user_task": "t"})
    m = sessions_manifest.read_manifest(tmp_path)
    with pytest.raises(sessions_manifest.ManifestError, match="unknown session mode"):
        m.session_mode()


def test_a_plan_manifest_still_gates_as_plan(tmp_path: pathlib.Path) -> None:
    for mode in ("run", "plan"):
        _write(tmp_path, {"version": 3, "mode": mode})
        assert sessions_manifest.read_manifest(tmp_path).session_mode() == mode


def test_the_gate_is_pinned_with_where_it_came_from(tmp_path: pathlib.Path) -> None:
    """A run records the verify gate it is judged by and its origin.

    A later edit to the file an inferred gate came from cannot move it, and any surface can say
    whether an operator or the repo chose it.
    """
    from agent6.app import manifest as app_manifest

    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "run", "session_id": "r"}), encoding="utf-8"
    )
    assert (
        sessions_manifest.read_manifest(tmp_path).harness.verify_origin == ""
    )  # gateless until pinned
    app_manifest.stamp_verify_gate(tmp_path, ("uv", "run", "pytest"), "inferred")
    wf = sessions_manifest.read_manifest(tmp_path).harness
    assert wf.verify_command == ("uv", "run", "pytest")
    assert wf.verify_origin == "inferred"
    # Re-stamping is what adoption does; it must not disturb the rest.
    assert sessions_manifest.read_manifest(tmp_path).mode == "run"


@pytest.mark.parametrize(
    ("configured", "has_gate", "pinned", "expected"),
    [
        (True, True, "inferred", "configured"),  # config outranks the pin
        (False, True, "adopted", "adopted"),  # an adopted gate stays adopted
        (False, True, "", "inferred"),  # the execution had to re-infer
        (False, False, "inferred", ""),  # gateless execution claims nothing
        (True, False, "inferred", ""),  # a dropped gate claims nothing, even over config
    ],
)
def test_a_resumed_execution_reports_whose_gate_it_used(
    configured: bool, has_gate: bool, pinned: str, expected: str
) -> None:
    """An operator's config outranks the pinned gate, and the pin outranks re-inference.

    The manifest names which one this execution ran under.
    """
    from agent6.app import resume

    assert (
        resume.execution_gate_origin(configured=configured, has_gate=has_gate, pinned=pinned)
        == expected
    )


def test_a_known_mode_is_never_reported_as_an_unknown_one(tmp_path: pathlib.Path) -> None:
    """The bug this vocabulary exists to prevent.

    "What kind of session is this" used to be answered in a dozen places, each
    re-deriving it from a bare string -- and two of them disagreed: the
    manifest's own list refused `machine` and `agent` outright while
    `mode_tools` happily built a tool surface for both, so a real mode was
    reported as damaged data. One table now, and the two failures are
    distinguished: a mode this agent6 does not know, and a known mode that
    resume cannot pick up.
    """
    from agent6 import kinds

    for name, kind in kinds.SESSION_KINDS.items():
        session_dir = tmp_path / name
        session_dir.mkdir()
        _write(session_dir, {"mode": name})
        manifest = sessions_manifest.read_manifest(session_dir)
        if kind.resumable:
            assert manifest.session_mode() == name
            continue
        with pytest.raises(sessions_manifest.ManifestError, match="not resumable"):
            manifest.session_mode()


def test_each_mode_gets_its_own_tool_surface() -> None:
    """Read off the record, not re-derived per call site."""
    from agent6 import kinds
    from agent6.tools import schema

    assert schema.mode_tools("machine").extras == schema.MACHINE_EXTRA_TOOLS
    assert schema.mode_tools("agent").extras == schema.MACHINE_EXTRA_TOOLS
    assert schema.mode_tools("ask").extras == schema.ASK_EXTRA_TOOLS
    for name, kind in kinds.SESSION_KINDS.items():
        names = schema.mode_tools(name).names
        assert ("apply_edit" in names) is kind.edits, name
        assert ("run_command" in names) is kind.runs_commands, name
    with pytest.raises(kinds.UnknownSessionKindError):
        schema.mode_tools("wat")


def test_a_execution_restamps_a_config_selected_preset(tmp_path: pathlib.Path) -> None:
    """A plain resume replaces the prior execution's preset name with the one it re-resolved."""
    from agent6.app import manifest as app_manifest
    from agent6.config import Config

    _write(
        tmp_path,
        {
            "version": sessions_manifest.MANIFEST_VERSION,
            "session_id": "executions-preset-A1",
            "mode": "run",
            "harness": {"preset": "old-config", "preset_from_flag": False},
        },
    )

    app_manifest.stamp_execution(tmp_path, Config(preset="new-config"), "run", "strict")

    harness = sessions_manifest.read_manifest(tmp_path).harness
    assert harness.preset == "new-config"
    assert harness.preset_from_flag is False


def test_a_execution_restamps_the_models_and_policy_it_runs_under(tmp_path: pathlib.Path) -> None:
    """The sandbox and model stamps describe this execution, not execution 1.

    Written once at run start, `agent6 exec` joined a recorded unsandboxed policy against a
    jailed agent, and every policy surface named a model another one answered for.
    """
    from agent6.app import manifest as app_manifest
    from agent6.config import Config

    _write(
        tmp_path,
        {
            "version": sessions_manifest.MANIFEST_VERSION,
            "session_id": "executions-run-A1",
            "mode": "run",
            "user_task": "t",
            "models": {"driver": {"provider": "openai", "model": "old-model"}},
            "policy": {"run_commands": "yes", "isolation": "none", "network": "auto"},
        },
    )
    cfg = Config.model_validate(
        {
            "models": {"worker": {"provider": "anthropic", "model": "new-model"}},
            "sandbox": {"run_commands": "ask"},
            "git": {"commit_per_step": False},
        }
    )

    app_manifest.stamp_execution(tmp_path, cfg, "run", "strict")

    m = sessions_manifest.read_manifest(tmp_path)
    assert m.policy.isolation == "strict"
    assert m.policy.run_commands == "ask"
    assert m.policy.commit_per_step is False
    assert m.models.driver is not None and m.models.driver.model == "new-model"


def test_a_version_3_manifest_keeps_its_stamp_under_the_old_key(tmp_path: pathlib.Path) -> None:
    """A version-3 manifest's `workflow` stamp is read under the `harness` key.

    `extra="ignore"` dropped it silently, so a resumed session lost its preset and gate pin.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "old-run-AAAAAA",
                "mode": "run",
                "workflow": {"review_trigger": "off", "preset": "strict", "preset_from_flag": True},
            }
        ),
        encoding="utf-8",
    )
    stamp = sessions_manifest.read_manifest(tmp_path).harness
    assert (stamp.review_trigger, stamp.preset, stamp.replay_preset) == ("off", "strict", "strict")
