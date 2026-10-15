# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The per-repo memory store: one fact per file with a rendered index."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
from typing import Any

import pytest

from agent6 import memory, portable


def test_add_writes_file_and_renders_entry(tmp_path: pathlib.Path) -> None:
    path = memory.add(tmp_path, "build-quirk", "The build needs FOO=1.\nDetails here.")
    assert path == memory.memory_dir(tmp_path) / "build-quirk.md"
    assert path.read_text() == "The build needs FOO=1.\nDetails here.\n"
    assert sorted(p.name for p in path.parent.iterdir()) == ["build-quirk.md"]
    assert memory.index_text(tmp_path) == "- build-quirk: The build needs FOO=1."


@pytest.mark.parametrize("by_hand", [False, True])
def test_add_keeps_the_full_first_line(tmp_path: pathlib.Path, *, by_hand: bool) -> None:
    """Added and hand-written facts render their full first line."""
    first = (
        "Every public calc function rejects a non-numeric operand with "
        "ValueError('inputs must be numbers'), the way multiply does; "
        "a new function copies that check and tests it. Keep this sentence too."
    )
    body = first + "\nDetails stay in the fact file."
    name = "input-validation"
    if by_hand:
        directory = memory.memory_dir(tmp_path)
        directory.mkdir()
        (directory / f"{name}.md").write_text(body + "\n", encoding="utf-8")
        with pytest.raises(memory.MemoryStoreError, match="exists"):
            memory.add(tmp_path, name, "Do not replace the stored fact.")
    else:
        memory.add(tmp_path, name, body)
    assert memory.show(tmp_path, name) == body + "\n"
    assert memory.index_text(tmp_path) == f"- {name}: {first}"


@pytest.mark.parametrize("first", ["# Title", "## Title", "Title"])
@pytest.mark.parametrize("rest", ["", "\nDetails stay in the fact file."])
def test_index_strips_heading_markers(tmp_path: pathlib.Path, first: str, rest: str) -> None:
    """The index keeps the first line's text, without heading markers or a body fallback."""
    memory.add(tmp_path, "fact", first + rest)
    assert memory.index_entries(tmp_path) == (("fact", "Title"),)
    assert memory.index_text(tmp_path) == "- fact: Title"


def test_add_refuses_duplicate_and_bad_names(tmp_path: pathlib.Path) -> None:
    memory.add(tmp_path, "one", "fact")
    with pytest.raises(memory.MemoryStoreError, match="exists"):
        memory.add(tmp_path, "one", "other")
    for bad in ("Has-Caps", "sl/ash", "..", "-lead", "a" * 65, "looks-valid\n"):
        with pytest.raises(memory.MemoryStoreError, match="bad memory name"):
            memory.add(tmp_path, bad, "x")
    with pytest.raises(memory.MemoryStoreError, match="non-empty"):
        memory.add(tmp_path, "empty", "   ")


def test_remove_deletes_file_and_rendered_entry(tmp_path: pathlib.Path) -> None:
    memory.add(tmp_path, "keep", "kept fact")
    memory.add(tmp_path, "drop", "dropped fact")
    memory.remove(tmp_path, "drop")
    assert not (memory.memory_dir(tmp_path) / "drop.md").exists()
    assert memory.index_text(tmp_path) == "- keep: kept fact"
    with pytest.raises(memory.MemoryStoreError, match="no memory named"):
        memory.remove(tmp_path, "drop")


def test_show_reads_one_entry(tmp_path: pathlib.Path) -> None:
    memory.add(tmp_path, "fact", "body text")
    assert memory.show(tmp_path, "fact") == "body text\n"
    with pytest.raises(memory.MemoryStoreError, match="no memory named"):
        memory.show(tmp_path, "absent")


def test_a_bad_byte_costs_one_character_not_the_index(tmp_path: pathlib.Path) -> None:
    """A byte that is not UTF-8 costs one character, never the index."""
    assert memory.index_text(tmp_path) == ""
    memory.add(tmp_path, "alpha", "the parser is generated")
    memory.add(tmp_path, "beta", "the cache is per repo")
    (memory.memory_dir(tmp_path) / "gamma.md").write_bytes(b"caf\xe9 a latin-1 byte\n")

    kept = memory.index_text(tmp_path)

    assert "alpha" in kept and "beta" in kept and "gamma" in kept

    memory.add(tmp_path, "delta", "a third fact")

    after = memory.index_text(tmp_path)
    assert all(name in after for name in ("alpha", "beta", "gamma", "delta")), after


def test_index_renders_entries_in_name_order(tmp_path: pathlib.Path) -> None:
    """Entries render in name order, not add order, with no blank lines between them."""
    memory.add(tmp_path, "one", "first fact")
    memory.add(tmp_path, "two", "second fact")
    memory.add(tmp_path, "three", "third fact")
    assert memory.index_text(tmp_path) == (
        "- one: first fact\n- three: third fact\n- two: second fact"
    )


def test_record_decision_appends_verbatim_and_the_text_clips_to_the_newest(
    tmp_path: pathlib.Path,
) -> None:
    """The harness-owned decisions.md.

    Append-only entries (question, answer verbatim with continuation lines indented, session, UTC
    time); the injected text keeps the newest rulings behind a pointer past the cap.
    """
    assert memory.decisions_text(tmp_path) == ""
    first = memory.record_decision(
        tmp_path, question="Keep the modal?", answer="No.\nInline item.", session="s1", when=0
    )
    second = memory.record_decision(
        tmp_path, question="Port?", answer="8931", session="s2", when=60
    )
    text = memory.decisions_path(tmp_path).read_text(encoding="utf-8")
    assert text == first + second
    assert first == "- 1970-01-01 00:00Z [s1] Q: Keep the modal?\n  A: No.\n  Inline item.\n"
    assert memory.decisions_text(tmp_path) == text.strip()
    for i in range(200):
        memory.record_decision(
            tmp_path, question=f"q{i} " + "x" * 40, answer="y" * 40, session="s", when=0
        )
    clipped = memory.decisions_text(tmp_path)
    assert len(clipped) <= memory.DECISIONS_INJECT_CAP
    # The marker names the operator's command, not a file the model cannot open.
    marker = "... (earlier rulings clipped; the operator sees them all: `agent6 memory decisions`)"
    assert clipped.startswith(marker + "\n- ") and clipped.rstrip().endswith("y" * 40)
    assert "decisions.md" not in clipped
    assert "q199" in clipped and "Keep the modal" not in clipped


def test_merge_decisions_appends_a_lanes_rulings(tmp_path: pathlib.Path) -> None:
    """A fan-out lane's rulings land in the coordinator's decisions.md after its own.

    A lane that recorded none writes nothing.
    """
    lane, origin = tmp_path / "lane", tmp_path / "origin"
    assert memory.merge_decisions(lane, origin) == (0, 0)
    assert not memory.decisions_path(origin).exists()
    memory.record_decision(origin, question="q0?", answer="a0", session="run")
    memory.record_decision(lane, question="q1?", answer="a1\n\nsecond paragraph", session="l1")
    assert memory.merge_decisions(lane, origin) == (1, 0)
    text = memory.decisions_path(origin).read_text(encoding="utf-8")
    assert text.index("Q: q0?") < text.index("Q: q1?")
    assert "[l1]" in text
    assert text.endswith("  A: a1\n  \n  second paragraph\n"), text


def test_merge_decisions_skips_a_ruling_the_origin_already_holds(tmp_path: pathlib.Path) -> None:
    """N lanes asked the same question and got the same answer.

    One ruling, recorded once, however many session tags it arrived under.
    """
    l1, l2, origin = tmp_path / "l1", tmp_path / "l2", tmp_path / "origin"
    memory.record_decision(l1, question="Tabs?", answer="spaces", session="fan-l1", when=0)
    memory.record_decision(l2, question="Tabs?", answer="spaces", session="fan-l2", when=60)
    memory.record_decision(l2, question="Tabs?", answer="tabs", session="fan-l2", when=120)
    assert memory.merge_decisions(l1, origin) == (1, 0)
    assert memory.merge_decisions(l2, origin) == (1, 1)
    assert memory.merge_decisions(l2, origin) == (0, 2)
    text = memory.decisions_path(origin).read_text(encoding="utf-8")
    assert text.count("Q: Tabs?") == 2
    assert text.count("A: spaces") == 1 and "[fan-l1]" in text and "A: tabs" in text


def test_a_ruling_larger_than_the_cap_shows_its_tail(tmp_path: pathlib.Path) -> None:
    """A ruling longer than the injection cap shows its tail.

    The tail is the newest words the operator said; the block never holds the clip marker alone.
    """
    memory.record_decision(
        tmp_path,
        question="Style?",
        answer="y" * (memory.DECISIONS_INJECT_CAP + 500),
        session="s",
        when=0,
    )
    text = memory.decisions_text(tmp_path)
    assert len(text) <= memory.DECISIONS_INJECT_CAP
    assert text.startswith("... (earlier rulings clipped")
    assert text.rstrip().endswith("y" * 100)


def test_record_decision_dedupes_a_ruling_already_recorded(tmp_path: pathlib.Path) -> None:
    """Repeated delivery of one question and answer records one ruling."""
    first = memory.record_decision(
        tmp_path, question="Tabs?", answer="spaces", session="s1", when=0
    )
    second = memory.record_decision(
        tmp_path, question="Tabs?", answer="spaces", session="s2", when=60
    )
    assert second == first
    assert memory.decisions_path(tmp_path).read_text(encoding="utf-8").count("Q: Tabs?") == 1


def test_merge_decisions_skips_a_repeat_within_the_source(tmp_path: pathlib.Path) -> None:
    """A repeat within the source is carried over once.

    A ruling the origin recorded long ago, under another session, is a skip too.
    """
    lane, origin = tmp_path / "lane", tmp_path / "origin"
    memory.record_decision(origin, question="Tabs?", answer="spaces", session="old", when=0)
    memory.record_decision(lane, question="Tabs?", answer="spaces", session="lane", when=3600)
    memory.record_decision(lane, question="Lint?", answer="ruff", session="lane", when=3660)
    # Preserve a duplicate written by an older version or hand edit; the
    # current writer itself dedupes it.
    with memory.decisions_path(lane).open("a", encoding="utf-8") as fh:
        fh.write("- 1970-01-01 01:02Z [lane] Q: Lint?\n  A: ruff\n")
    assert memory.merge_decisions(lane, origin) == (1, 2)
    text = memory.decisions_path(origin).read_text(encoding="utf-8")
    assert text.count("Q: Tabs?") == 1 and text.count("Q: Lint?") == 1


def test_merge_decisions_starts_on_its_own_line(tmp_path: pathlib.Path) -> None:
    """A merged ruling starts on its own line.

    A destination cut short of its trailing newline (a partial write, a hand edit) gets one before
    the first appended entry.
    """
    lane, origin = tmp_path / "lane", tmp_path / "origin"
    path = memory.decisions_path(origin)
    path.parent.mkdir(parents=True)
    path.write_text("- 2026-01-01 00:00Z [x] Q: a?\n  A: b", encoding="utf-8")
    memory.record_decision(lane, question="c?", answer="d", session="lane", when=0)
    assert memory.merge_decisions(lane, origin) == (1, 0)
    assert path.read_text(encoding="utf-8") == (
        "- 2026-01-01 00:00Z [x] Q: a?\n  A: b\n- 1970-01-01 00:00Z [lane] Q: c?\n  A: d\n"
    )


def _seeded_lane(tmp_path: pathlib.Path, origin: pathlib.Path, name: str = "lane") -> pathlib.Path:
    lane = tmp_path / name
    memory.seed_store(origin, lane)
    return lane


def test_merge_memory_lands_new_facts_and_leaves_untouched_copies_alone(
    tmp_path: pathlib.Path,
) -> None:
    """A lane's store is a copy of the origin's.

    At import a fact the lane added lands with its rendered entry; the copies it never touched are
    nothing to report (every one read as "already recorded" before), and the same content on both
    sides is nothing either.
    """
    origin = tmp_path / "origin"
    memory.add(origin, "repo-fact", "The build needs BUILD_ID set.")
    lane = _seeded_lane(tmp_path, origin)
    assert list(json.loads(memory.seed_path(lane).read_text(encoding="utf-8"))) == ["repo-fact"]
    memory.add(lane, "lane-fact", "The flaky test is test_clock.")
    held = tmp_path / "held"
    assert memory.merge_memory(lane, origin, held_dir=held) == memory.MemoryMerge(
        carried=("lane-fact",)
    )
    assert "test_clock" in memory.show(origin, "lane-fact")
    assert memory.index_text(origin).splitlines() == [
        "- lane-fact: The flaky test is test_clock.",
        "- repo-fact: The build needs BUILD_ID set.",
    ]
    assert not held.exists()
    assert memory.merge_memory(lane, origin, held_dir=held) == memory.MemoryMerge()


def test_merge_memory_fast_forwards_a_lanes_edit_and_deletion(tmp_path: pathlib.Path) -> None:
    """A lane's edit and deletion fast-forward over a copy the origin has not touched.

    An edit replaces the file and a deletion removes it, as with the branch.
    """
    origin = tmp_path / "origin"
    memory.add(origin, "a-fact", "A first.")
    memory.add(origin, "b-fact", "B first.")
    memory.add(origin, "c-fact", "C first.")
    lane = _seeded_lane(tmp_path, origin)
    (memory.memory_dir(lane) / "a-fact.md").write_text("A second, refined.\n", encoding="utf-8")
    memory.remove(lane, "c-fact")
    held = tmp_path / "held"
    assert memory.merge_memory(lane, origin, held_dir=held) == memory.MemoryMerge(
        updated=("a-fact",), deleted=("c-fact",)
    )
    assert memory.show(origin, "a-fact") == "A second, refined.\n"
    assert not (memory.memory_dir(origin) / "c-fact.md").exists()
    assert memory.index_text(origin).splitlines() == [
        "- a-fact: A second, refined.",
        "- b-fact: B first.",
    ]
    assert not held.exists()


def test_merge_memory_holds_back_a_change_on_both_sides(tmp_path: pathlib.Path) -> None:
    """Changed in the lane and in the origin since seeding.

    The origin keeps its version, the lane's is kept under held_dir, and both names are reported. A
    lane deletion over an origin edit is held the same way, with nothing to keep.
    """
    origin = tmp_path / "origin"
    memory.add(origin, "a-fact", "A first.")
    memory.add(origin, "d-fact", "D first.")
    lane = _seeded_lane(tmp_path, origin)
    (memory.memory_dir(lane) / "a-fact.md").write_text("A per the lane.\n", encoding="utf-8")
    (memory.memory_dir(origin) / "a-fact.md").write_text("A per the origin.\n", encoding="utf-8")
    memory.remove(lane, "d-fact")
    (memory.memory_dir(origin) / "d-fact.md").write_text("D per the origin.\n", encoding="utf-8")
    held = tmp_path / "held"
    assert memory.merge_memory(lane, origin, held_dir=held) == memory.MemoryMerge(
        held=("a-fact", "d-fact")
    )
    assert memory.show(origin, "a-fact") == "A per the origin.\n"
    assert memory.show(origin, "d-fact") == "D per the origin.\n"
    assert (held / "a-fact.md").read_text(encoding="utf-8") == "A per the lane.\n"
    assert not (held / "d-fact.md").exists()


def test_merge_memory_holds_back_a_name_two_lanes_invented(tmp_path: pathlib.Path) -> None:
    """Two lanes recording different facts under one new name.

    The first lands, the second is held with its version kept, so neither silently overwrites the
    other.
    """
    origin = tmp_path / "origin"
    l1, l2 = _seeded_lane(tmp_path, origin, "l1"), _seeded_lane(tmp_path, origin, "l2")
    memory.add(l1, "shared-name", "What lane one saw.")
    memory.add(l2, "shared-name", "What lane two saw.")
    assert memory.merge_memory(l1, origin, held_dir=tmp_path / "h1") == memory.MemoryMerge(
        carried=("shared-name",)
    )
    assert memory.merge_memory(l2, origin, held_dir=tmp_path / "h2") == memory.MemoryMerge(
        held=("shared-name",)
    )
    assert (tmp_path / "h2" / "shared-name.md").read_text(
        encoding="utf-8"
    ) == "What lane two saw.\n"
    assert memory.show(origin, "shared-name") == "What lane one saw.\n"


def test_merge_memory_skips_a_name_outside_the_stores_rule(tmp_path: pathlib.Path) -> None:
    """A name outside the store's rule is skipped on both sides.

    A seed key or a stray file whose name breaks the rule (a path fragment, an upper-case name) is
    not a memory and names no path; a seed key of `../escape` matching a file beside the store would
    otherwise read as "deleted in the lane" and be unlinked outside the store.
    """
    origin = tmp_path / "origin"
    memory.add(origin, "a-fact", "A first.")
    lane = _seeded_lane(tmp_path, origin)
    outside = origin / "escape.md"
    outside.write_text("outside\n", encoding="utf-8")
    seeds = json.loads(memory.seed_path(lane).read_text(encoding="utf-8"))
    seeds["../escape"] = hashlib.sha256(b"outside\n").hexdigest()
    seeds["Bad Name"] = "0" * 64
    memory.seed_path(lane).write_text(json.dumps(seeds), encoding="utf-8")
    (memory.memory_dir(lane) / "Bad Name.md").write_text("stray\n", encoding="utf-8")
    held = tmp_path / "held"
    merged = memory.merge_memory(lane, origin, held_dir=held)
    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert merged == memory.MemoryMerge()
    assert not (memory.memory_dir(origin) / "Bad Name.md").exists() and not held.exists()


def test_seed_store_keeps_the_digests_of_an_earlier_seeding(tmp_path: pathlib.Path) -> None:
    """A second seeding keeps the digests of an earlier one.

    A manifest rewritten with this seeding's files alone makes every earlier copy read as new at
    import and be held back.
    """
    origin = tmp_path / "origin"
    memory.add(origin, "a-fact", "A first.")
    lane = _seeded_lane(tmp_path, origin)
    memory.add(origin, "b-fact", "B first.")
    memory.seed_store(origin, lane)
    assert sorted(json.loads(memory.seed_path(lane).read_text(encoding="utf-8"))) == [
        "a-fact",
        "b-fact",
    ]
    assert memory.merge_memory(lane, origin, held_dir=tmp_path / "held") == memory.MemoryMerge()


def test_a_seed_manifest_that_is_not_an_object_reads_as_no_seeds(tmp_path: pathlib.Path) -> None:
    """A seed manifest that is not an object reads as no seeds.

    One reader owns the manifest's shape; a hand-edited JSON array raises neither TypeError out of
    seed_store nor AttributeError out of merge_memory.
    """
    origin = tmp_path / "origin"
    memory.add(origin, "a-fact", "A first.")
    lane = _seeded_lane(tmp_path, origin)
    memory.seed_path(lane).write_text("[]\n", encoding="utf-8")
    memory.add(origin, "b-fact", "B first.")
    memory.seed_store(origin, lane)
    assert sorted(json.loads(memory.seed_path(lane).read_text(encoding="utf-8"))) == ["b-fact"]
    memory.seed_path(lane).write_text("[]\n", encoding="utf-8")
    assert memory.merge_memory(lane, origin, held_dir=tmp_path / "held") == memory.MemoryMerge()


def test_merge_memory_skips_an_identical_destination_copy(tmp_path: pathlib.Path) -> None:
    """An identical destination copy is visible without another merge write."""
    origin = tmp_path / "origin"
    lane = _seeded_lane(tmp_path, origin)
    memory.add(lane, "lane-fact", "The parser is generated.")
    dst = memory.memory_dir(origin)
    dst.mkdir(parents=True)
    (dst / "lane-fact.md").write_bytes((memory.memory_dir(lane) / "lane-fact.md").read_bytes())

    assert memory.merge_memory(lane, origin, held_dir=tmp_path / "held") == memory.MemoryMerge()
    assert memory.index_text(origin) == "- lane-fact: The parser is generated."


def test_seed_store_recovers_a_copy_published_before_its_digest(tmp_path: pathlib.Path) -> None:
    """A seed copy published before its digest is recorded on retry.

    A crash between copying a seed fact and publishing the manifest must not lose the digest.
    """
    origin, lane = tmp_path / "origin", tmp_path / "lane"
    memory.add(origin, "a-fact", "A first.")
    dst = memory.memory_dir(lane)
    dst.mkdir(parents=True)
    (dst / "a-fact.md").write_bytes((memory.memory_dir(origin) / "a-fact.md").read_bytes())

    memory.seed_store(origin, lane)

    assert set(memory.seed_digests(lane)) == {"a-fact"}


def test_add_publishes_a_fact_atomically(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash in the durable publish must not expose a partial fact file."""
    real_atomic_write = portable.atomic_write

    def crash(path: pathlib.Path, data: str | bytes) -> None:
        if path.name == "crash.md":
            raise OSError("simulated crash")
        real_atomic_write(path, data)

    monkeypatch.setattr(portable, "atomic_write", crash)
    with pytest.raises(OSError, match="simulated crash"):
        memory.add(tmp_path, "crash", "A complete fact.")
    assert not (memory.memory_dir(tmp_path) / "crash.md").exists()


def test_record_use_keeps_who_wrote_and_who_read(tmp_path: pathlib.Path) -> None:
    """The use record keeps who wrote and who read.

    The creating write stays as `created`, the latest as `updated`; reads accumulate with the last
    reader.
    """
    memory.record_use(
        tmp_path, session="run-a", wrote=("fact",), created=("fact",), read={}, when=0.0
    )
    memory.record_use(tmp_path, session="run-b", wrote=("fact",), read={"fact": 2}, when=3600.0)
    memory.record_use(
        tmp_path, session="run-c", wrote=(), read={"fact": 1, "other": 1}, when=7200.0
    )
    memory.record_use(tmp_path, session="run-b", wrote=("fact",), read={}, when=10800.0)
    memory.record_use(tmp_path, session="run-d", wrote=("older",), read={}, when=14400.0)
    use = memory.read_use(tmp_path)
    assert use["fact"] == memory.MemoryUse(
        created=memory.Touch("run-a", "1970-01-01 00:00Z"),
        writes=(
            memory.Touch("run-a", "1970-01-01 00:00Z"),
            memory.Touch("run-b", "1970-01-01 01:00Z"),
            memory.Touch("run-b", "1970-01-01 03:00Z"),
        ),
        reads=3,
        last_read=memory.Touch("run-c", "1970-01-01 02:00Z"),
    )
    assert use["fact"].writers == ("run-a", "run-b")
    assert use["fact"].updated == memory.Touch("run-b", "1970-01-01 03:00Z")
    # A write of a fact the record never saw created (hand-written, or older
    # than the record) is an edit: no creation is claimed.
    assert use["older"] == memory.MemoryUse(writes=(memory.Touch("run-d", "1970-01-01 04:00Z"),))
    # A deletion ends the entry, as `memory rm` does; alone, it still writes.
    memory.record_use(
        tmp_path, session="run-e", wrote=(), read={"older": 1}, deleted=("older",), when=18000.0
    )
    assert "older" not in memory.read_use(tmp_path)  # a read of its old life brings nothing back
    # A read of a fact nobody recorded writing (hand-written, or older than
    # the record) still counts, with no writer named.
    assert use["other"] == memory.MemoryUse(
        reads=1, last_read=memory.Touch("run-c", "1970-01-01 02:00Z")
    )
    assert use["other"].created is None
    assert memory.use_path(tmp_path) == tmp_path / "memory-use.json"


def test_operator_add_and_rm_keep_the_use_record_in_step(tmp_path: pathlib.Path) -> None:
    memory.add(tmp_path, "quirk", "The build needs FOO=1.")
    use = memory.read_use(tmp_path)
    assert use["quirk"].writers == ("operator",)
    assert use["quirk"].created is not None and use["quirk"].created.session == "operator"
    assert use["quirk"].reads == 0
    memory.remove(tmp_path, "quirk")
    assert "quirk" not in memory.read_use(tmp_path)


def test_read_use_tolerates_a_missing_or_misshapen_record(tmp_path: pathlib.Path) -> None:
    assert memory.read_use(tmp_path) == {}
    memory.use_path(tmp_path).write_text("[1, 2]", encoding="utf-8")
    assert memory.read_use(tmp_path) == {}
    memory.use_path(tmp_path).write_text(
        '{"fact": {"reads": "many", "writes": []}, "ok": {"reads": 2, "writes": []},'
        ' "w": {"reads": 1}}',
        encoding="utf-8",
    )
    assert memory.read_use(tmp_path) == {"ok": memory.MemoryUse(reads=2)}
    memory.use_path(tmp_path).write_text(
        '{"n": {"created": {"session": "run-a", "at": "2026-01-01 00:00Z"},'
        ' "writes": [{"session": "run-a", "at": "2026-01-01 00:00Z"}], "reads": 1,'
        ' "last_read": {"session": "run-b", "at": "2026-01-02 00:00Z"}},'
        ' "e": {"writes": [{"session": "run-a", "at": "2026-01-01 00:00Z"}]},'
        ' "bad": {"writes": [{"session": 1}]}, "worse": {"writes": "run-a"},'
        ' "worst": {"created": "run-a", "writes": []}, "old": {"created_by": "run-a"},'
        ' "unstamped": {"writes": [{"session": "run-a"}]},'
        ' "blank": {"writes": [{"session": "run-a", "at": ""}]}}',
        encoding="utf-8",
    )
    assert memory.read_use(tmp_path) == {
        "n": memory.MemoryUse(
            created=memory.Touch("run-a", "2026-01-01 00:00Z"),
            writes=(memory.Touch("run-a", "2026-01-01 00:00Z"),),
            reads=1,
            last_read=memory.Touch("run-b", "2026-01-02 00:00Z"),
        ),
        "e": memory.MemoryUse(writes=(memory.Touch("run-a", "2026-01-01 00:00Z"),)),
    }


def test_merge_use_carries_a_lanes_record_into_the_origin(tmp_path: pathlib.Path) -> None:
    """A --parallel lane's use record died with its state dir.

    The origin never named the lane as a fact's writer or reader.
    """
    origin, lane = tmp_path / "origin", tmp_path / "lane"
    memory.record_use(
        origin,
        session="run-o",
        wrote=("shared",),
        created=("shared",),
        read={"shared": 1},
        when=0.0,
    )
    memory.record_use(
        lane,
        session="lane-1",
        wrote=("shared", "fresh"),
        created=("shared", "fresh"),  # the lane's store had neither: both created there
        read={"shared": 2, "gone": 1},
        when=60.0,
    )
    merged, reads = memory.merge_use(lane, origin, written=("shared", "fresh"))
    assert (merged, reads) == (2, 1)  # two writers carried; one fact read (gone was never held)
    use = memory.read_use(origin)
    assert use["shared"].writers == ("run-o", "lane-1")
    assert (use["shared"].reads, use["shared"].last_read) == (
        3,
        memory.Touch("lane-1", "1970-01-01 00:01Z"),
    )
    assert use["shared"].created == memory.Touch(
        "run-o", "1970-01-01 00:00Z"
    )  # the origin's stands
    assert use["fresh"].created == memory.Touch("lane-1", "1970-01-01 00:01Z")
    assert "gone" not in use  # a read of a fact the origin never held travels nowhere
    # Nothing to carry leaves the origin alone.
    assert memory.merge_use(tmp_path / "empty", origin, written=()) == (0, 0)


def test_carry_back_lands_a_lanes_use_record(tmp_path: pathlib.Path) -> None:
    """The lane import carries the use record along with the facts and the rulings.

    Otherwise it goes with the lane's state dir.
    """
    from agent6.app import parallel, reporter

    origin, lane, dest = tmp_path / "origin", tmp_path / "lane", tmp_path / "dest"
    memory.add(lane, "lane-fact", "A fact the lane found.")
    memory.record_use(lane, session="lane-1", wrote=("lane-fact",), read={"lane-fact": 2})
    said: list[str] = []
    parallel.carry_back(
        lane, origin, dest, lane=1, reporter=reporter.Reporter(out=said.append, err=said.append)
    )
    use = memory.read_use(origin)
    assert use["lane-fact"].writers == ("operator", "lane-1")
    assert use["lane-fact"].reads == 2
    assert any("use record carried for 1" in line for line in said)


def _legacy_rulings(state_dir: pathlib.Path, text: str = "- old ruling\n") -> pathlib.Path:
    legacy = memory.memory_dir(state_dir) / "DECISIONS.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(text, encoding="utf-8")
    return legacy


def test_migrate_legacy_decisions_renames_the_file_once(tmp_path: pathlib.Path) -> None:
    """An old `memory/DECISIONS.md` with no `decisions.md` beside the store is renamed, once."""
    legacy = _legacy_rulings(tmp_path)
    target = memory.decisions_path(tmp_path)
    assert not target.is_relative_to(memory.memory_dir(tmp_path))
    assert memory.migrate_legacy_decisions(tmp_path) == (
        f"operator rulings moved from {legacy} to {target}"
    )
    assert not legacy.exists()
    assert target.read_text(encoding="utf-8") == "- old ruling\n"
    assert memory.migrate_legacy_decisions(tmp_path) == ""
    assert memory.migrate_legacy_decisions(tmp_path / "no-store") == ""


def test_a_store_write_under_the_old_rulings_name_never_reaches_the_rulings(
    tmp_path: pathlib.Path,
) -> None:
    """A `memory/DECISIONS.md` beside an existing `decisions.md` is reported and left alone.

    The store is the model's write grant; a merge would launder its content into the rulings.
    """
    memory.ensure_decisions_file(tmp_path)
    target = memory.decisions_path(tmp_path)
    assert target.read_bytes() == b""
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    memory.record_decision(tmp_path, question="Q?", answer="A", session="s", when=0)
    memory.ensure_decisions_file(tmp_path)  # a second run start keeps what is there
    before = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600  # one mode, whichever writer came first
    stray = _legacy_rulings(tmp_path, "- 1970-01-01 00:00Z [x] Q: Planted?\n  A: yes\n")
    assert memory.migrate_legacy_decisions(tmp_path) == (
        f"ignored a stray {stray}: rulings live at {target}; delete it to silence this"
    )
    assert stray.exists()
    assert target.read_bytes() == before
    assert "Planted" not in memory.decisions_text(tmp_path)


def test_seed_store_copies_the_origins_rulings_beside_the_lanes_store(
    tmp_path: pathlib.Path,
) -> None:
    """A lane is seeded with the origin's rulings, outside its store and outside the count.

    A lane holding rulings keeps its own; an origin with no store leaves the lane storeless.
    """
    origin, lane = tmp_path / "origin", tmp_path / "lane"
    memory.record_decision(origin, question="Q?", answer="A", session="s", when=0)
    memory.add(origin, "a-fact", "A first.")
    assert memory.seed_store(origin, lane) == 1  # the fact
    assert memory.decisions_text(lane) == memory.decisions_text(origin)
    assert stat.S_IMODE(memory.decisions_path(lane).stat().st_mode) == 0o600
    assert set(memory.seed_digests(lane)) == {"a-fact"}
    other = tmp_path / "other"
    memory.record_decision(other, question="Mine?", answer="B", session="s", when=0)
    assert memory.seed_store(origin, other) == 1
    assert memory.decisions_text(other) == "- 1970-01-01 00:00Z [s] Q: Mine?\n  A: B"
    bare, storeless = tmp_path / "bare", tmp_path / "storeless"
    bare.mkdir()
    memory.decisions_path(bare).write_text("- ruling\n", encoding="utf-8")
    assert memory.seed_store(bare, storeless) == 0
    assert memory.decisions_text(storeless) == "- ruling"
    assert not memory.memory_dir(storeless).exists()
    assert memory.seed_store(tmp_path / "nothing", tmp_path / "untouched") == 0
    assert not (tmp_path / "untouched").exists()


def test_merge_memory_skips_a_stray_rulings_file_in_the_lanes_store(tmp_path: pathlib.Path) -> None:
    """`DECISIONS` fails the name rule: a lane's `memory/DECISIONS.md` never travels as a fact."""
    assert not memory.is_memory_name("DECISIONS")
    origin = tmp_path / "origin"
    lane = _seeded_lane(tmp_path, origin)
    memory.add(lane, "lane-fact", "A lane fact.")
    stray = _legacy_rulings(lane, "- planted\n")
    assert memory.merge_memory(lane, origin, held_dir=tmp_path / "held") == memory.MemoryMerge(
        carried=("lane-fact",)
    )
    assert stray.exists()
    assert not (memory.memory_dir(origin) / "DECISIONS.md").exists()
    assert memory.merge_decisions(lane, origin) == (0, 0)
    assert memory.decisions_text(origin) == ""


def test_index_shows_each_facts_first_non_empty_line(tmp_path: pathlib.Path) -> None:
    """Leading blank lines are skipped; a fact with no text reads `(empty)`, so it still shows."""
    store = memory.memory_dir(tmp_path)
    store.mkdir()
    (store / "zebra.md").write_text("\n  \nActual first line.\nDetails.\n", encoding="utf-8")
    (store / "empty.md").write_text("  \n\n", encoding="utf-8")
    assert memory.index_entries(tmp_path) == (("empty", "(empty)"), ("zebra", "Actual first line."))
    assert memory.index_text(tmp_path) == "- empty: (empty)\n- zebra: Actual first line."


def test_index_follows_the_files(tmp_path: pathlib.Path) -> None:
    """An edit shows at the next read and a deletion drops the entry; no store reads as nothing."""
    assert memory.index_text(tmp_path) == ""
    memory.add(tmp_path, "zebra", "A first.")
    (memory.memory_dir(tmp_path) / "zebra.md").write_text("Edited.\n", encoding="utf-8")
    assert memory.index_text(tmp_path) == "- zebra: Edited."
    (memory.memory_dir(tmp_path) / "zebra.md").unlink()
    assert memory.index_text(tmp_path) == ""


def test_index_skips_what_is_not_a_fact_file(tmp_path: pathlib.Path) -> None:
    """One rule for a fact: `<name>.md` under the name rule, a regular file.

    A stray `MEMORY.md`, a name outside the rule, another suffix, a directory and a symlink (the
    store is model-writable, and a fact feeds every prompt) are ignored and named as such.
    """
    store = memory.memory_dir(tmp_path)
    store.mkdir()
    (store / "ok.md").write_text("x\n", encoding="utf-8")
    (store / "MEMORY.md").write_text("- ok: stale\n", encoding="utf-8")
    (store / "Draft.md").write_text("x\n", encoding="utf-8")
    (store / "notes.txt").write_text("x\n", encoding="utf-8")
    (store / "directory.md").mkdir()
    (store / "linked.md").symlink_to(store / "ok.md")
    assert memory.index_text(tmp_path) == "- ok: x"
    assert memory.ignored_names(tmp_path) == (
        "Draft.md",
        "MEMORY.md",
        "directory.md",
        "linked.md",
        "notes.txt",
    )
    assert memory.ignored_names(tmp_path / "no-store") == ()


def test_index_skips_a_fact_it_cannot_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fact deleted between the listing and the read, or unreadable, is skipped, never raised.

    The store is model-writable, and memory is context: a prompt build never fails on it.
    """
    store = memory.memory_dir(tmp_path)
    store.mkdir()
    (store / "kept.md").write_text("Kept.\n", encoding="utf-8")
    vanished = store / "vanished.md"
    vanished.write_text("Gone before the read.\n", encoding="utf-8")
    real_read_text = pathlib.Path.read_text

    def read_after_a_deletion(self: pathlib.Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "vanished.md":
            self.unlink()
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_after_a_deletion)
    unreadable = store / "unreadable.md"
    unreadable.write_text("Hidden.\n", encoding="utf-8")
    unreadable.chmod(0)
    try:
        expect = (("unreadable", "Hidden."),) if os.geteuid() == 0 else ()  # root reads any mode
        assert memory.index_entries(tmp_path) == (("kept", "Kept."), *expect)
    finally:
        unreadable.chmod(0o600)
    assert not vanished.exists()


def test_a_store_that_cannot_be_listed_reads_as_empty(tmp_path: pathlib.Path) -> None:
    """A store path that is not a directory lists nothing rather than failing the prompt build."""
    memory.memory_dir(tmp_path).write_text("not a directory\n", encoding="utf-8")
    assert memory.index_entries(tmp_path) == ()
    assert memory.ignored_names(tmp_path) == ()


def test_seed_and_merge_share_the_fact_rule(tmp_path: pathlib.Path) -> None:
    """Only fact files are seeded into a lane, and only fact files travel back at import."""
    origin = tmp_path / "origin"
    memory.add(origin, "a-fact", "A first.")
    (memory.memory_dir(origin) / "MEMORY.md").write_text("- a-fact: stale\n", encoding="utf-8")
    (memory.memory_dir(origin) / "linked.md").symlink_to(memory.memory_dir(origin) / "a-fact.md")
    lane = _seeded_lane(tmp_path, origin)
    assert sorted(p.name for p in memory.memory_dir(lane).iterdir()) == ["a-fact.md"]
    assert set(memory.seed_digests(lane)) == {"a-fact"}
    (memory.memory_dir(lane) / "lane-link.md").symlink_to(memory.memory_dir(lane) / "a-fact.md")
    assert memory.merge_memory(lane, origin, held_dir=tmp_path / "held") == memory.MemoryMerge()
    assert not (memory.memory_dir(origin) / "lane-link.md").exists()
