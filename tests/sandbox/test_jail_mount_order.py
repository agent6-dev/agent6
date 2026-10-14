# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What `strict` mounts on top when an operator grant contains the workspace."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds
from agent6.sandbox import jail

pytestmark = pytest.mark.needs_namespaces


@pytest.mark.parametrize(
    ("protect", "target"),
    [(".git", ".git/config"), ("ops/deploy.asm.toml", "ops/deploy.asm.toml")],
    ids=["directory", "single-file"],
)
def test_a_write_grant_containing_the_workspace_keeps_protect_paths_read_only(
    tmp_path: pathlib.Path, protect: str, target: str
) -> None:
    """A write grant on the workspace's parent does not make a protect path writable.

    A grant bound over the workspace covers the read-only binds of `.git` and of a machine
    bundle file; the rest of the workspace and of the grant stay writable.
    """
    outer = tmp_path / "outer"
    ws = outer / "ws"
    protected = ws / target
    protected.parent.mkdir(parents=True)
    protected.write_text("original\n", encoding="utf-8")

    probe = (
        f"{{ echo tamper >> {target}; }} 2>/dev/null && echo PROTECT-WRITABLE || echo refused; "
        "echo w > note.txt && echo ws-written; "
        "echo g > ../beside.txt && echo grant-written"
    )
    res = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=ws,
            argv=("sh", "-c", probe),
            isolation="strict",
            extra_rw_paths=(outer,),
            extra_protect_paths=(ws / protect,),
            timeout_s=20.0,
        )
    )
    out = res.stdout + res.stderr
    assert "refused" in res.stdout and "PROTECT-WRITABLE" not in res.stdout, out
    assert protected.read_text(encoding="utf-8") == "original\n"
    assert "ws-written" in res.stdout, out
    assert (ws / "note.txt").read_text(encoding="utf-8") == "w\n"
    assert "grant-written" in res.stdout, out
    assert (outer / "beside.txt").read_text(encoding="utf-8") == "g\n"
