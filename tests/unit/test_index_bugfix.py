# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for SymbolIndex out-of-band staleness and thread-safety.

Covers two fixes:
  * stat-based self-healing when a file is changed/deleted outside
    apply_edit (e.g. a run_command formatter, ``rm``, ``git mv``, ``sed``)
    without any mark_changed/mark_deleted call.
  * a lock around the public readers/mutators so the index can be shared
    across the concurrent explore-review seats.
"""

from __future__ import annotations

import threading
from pathlib import Path

from agent6.tools.index import SymbolIndex


def _bump_mtime(p: Path) -> None:
    """Force a distinct (mtime_ns, size) so the stat check fires even on a
    coarse-resolution clock / same-tick write."""
    st = p.stat()
    import os

    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_index_self_heals_on_out_of_band_edit(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text("def alpha():\n    pass\n", encoding="utf-8")
    idx = SymbolIndex(tmp_path)

    defs = idx.find_definition("alpha")
    assert len(defs) == 1
    assert idx.find_definition("beta") == []

    # Rewrite the file WITHOUT calling mark_changed (simulates a formatter or
    # sed run via run_command).
    src.write_text("def beta():\n    pass\n", encoding="utf-8")
    _bump_mtime(src)

    # The index must self-heal off the on-disk stat, not serve stale symbols.
    assert idx.find_definition("alpha") == []
    beta = idx.find_definition("beta")
    assert len(beta) == 1
    assert beta[0].name == "beta"


def test_index_evicts_deleted_file(tmp_path: Path) -> None:
    src = tmp_path / "gone.py"
    src.write_text("def doomed():\n    pass\n", encoding="utf-8")
    idx = SymbolIndex(tmp_path)
    assert len(idx.find_definition("doomed")) == 1

    # Delete out of band (e.g. run_command rm / git mv) with no mark_deleted.
    src.unlink()

    assert idx.find_definition("doomed") == []
    # The phantom path must not survive in any reader.
    assert all(p != src.resolve() for p in idx.file_outlines())


def test_index_concurrent_readers_do_not_raise(tmp_path: Path) -> None:
    # Many files so iteration over _symbols.values() is non-trivial and the
    # outline on-demand path keeps adding new keys while other threads read.
    for i in range(40):
        (tmp_path / f"f{i}.py").write_text(f"def fn_{i}():\n    helper()\n", encoding="utf-8")
    idx = SymbolIndex(tmp_path)

    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                idx.find_definition("helper")
                idx.find_references("helper")
                idx.hot_symbols()
                idx.file_outlines()
                # outline() adds new keys on demand -> mutates while others read
                idx.outline(tmp_path / "f0.py")
        except BaseException as exc:
            errors.append(exc)

    def mutator() -> None:
        try:
            n = 0
            while not stop.is_set():
                p = tmp_path / "churn.py"
                p.write_text(f"def churn_{n}():\n    helper()\n", encoding="utf-8")
                idx.mark_changed(p)
                n += 1
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads.append(threading.Thread(target=mutator))
    for t in threads:
        t.start()
    threading.Event().wait(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"concurrent access raised: {errors!r}"
