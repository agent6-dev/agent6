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


def test_a_read_grant_containing_the_workspace_leaves_it_writable(tmp_path: pathlib.Path) -> None:
    """A read grant on the workspace's parent leaves the workspace writable.

    A read-only grant bound over the workspace refuses every jailed write inside it. `.git`
    and the rest of the grant stay read-only.
    """
    outer = tmp_path / "outer"
    ws = outer / "ws"
    config = ws / ".git" / "config"
    config.parent.mkdir(parents=True)
    config.write_text("original\n", encoding="utf-8")
    (outer / "seed.txt").write_text("seeded\n", encoding="utf-8")

    probe = (
        "echo w > note.txt && echo ws-written; "
        "{ echo tamper >> .git/config; } 2>/dev/null && echo PROTECT-WRITABLE || echo refused; "
        "{ echo g > ../beside.txt; } 2>/dev/null && echo GRANT-WRITABLE || echo grant-read-only; "
        "cat ../seed.txt"
    )
    res = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=ws,
            argv=("sh", "-c", probe),
            isolation="strict",
            extra_ro_paths=(outer,),
            extra_protect_paths=(ws / ".git",),
            timeout_s=20.0,
        )
    )
    out = res.stdout + res.stderr
    assert "ws-written" in res.stdout, out
    assert (ws / "note.txt").read_text(encoding="utf-8") == "w\n"
    assert "refused" in res.stdout and "PROTECT-WRITABLE" not in res.stdout, out
    assert config.read_text(encoding="utf-8") == "original\n"
    assert "grant-read-only" in res.stdout and "seeded" in res.stdout, out
    assert not (outer / "beside.txt").exists()


def test_a_read_grant_inside_the_workspace_holds_under_a_containing_write_grant(
    tmp_path: pathlib.Path,
) -> None:
    """A read grant inside the workspace stays read-only under a write grant on the parent.

    Bound before the workspace or before the containing grant, the read grant is covered and
    a jailed command writes the tree the operator granted read-only.
    """
    outer = tmp_path / "outer"
    ws = outer / "ws"
    vendored = ws / "vendor" / "lib.txt"
    vendored.parent.mkdir(parents=True)
    vendored.write_text("original\n", encoding="utf-8")

    probe = (
        "{ echo tamper >> vendor/lib.txt; } 2>/dev/null && echo VENDOR-WRITABLE || echo refused; "
        "echo w > note.txt && echo ws-written"
    )
    res = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=ws,
            argv=("sh", "-c", probe),
            isolation="strict",
            extra_ro_paths=(ws / "vendor",),
            extra_rw_paths=(outer,),
            timeout_s=20.0,
        )
    )
    out = res.stdout + res.stderr
    assert "refused" in res.stdout and "VENDOR-WRITABLE" not in res.stdout, out
    assert vendored.read_text(encoding="utf-8") == "original\n"
    assert "ws-written" in res.stdout, out


def test_a_workspace_named_through_a_symlink_is_mounted_at_its_real_path(
    tmp_path: pathlib.Path,
) -> None:
    """A workspace given through a symlink keeps `.git` read-only at its real path.

    Mounted under the symlink's name, the workspace and its read-only `.git` bind sat beside
    a write grant on the real parent, which showed the same `.git/config` writable.
    """
    outer = tmp_path / "outer"
    ws = outer / "ws"
    config = ws / ".git" / "config"
    config.parent.mkdir(parents=True)
    config.write_text("original\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(ws)

    probe = (
        f"{{ echo tamper >> {config}; }} 2>/dev/null && echo REAL-WRITABLE || echo real-refused; "
        "{ echo tamper >> .git/config; } 2>/dev/null && echo CWD-WRITABLE || echo cwd-refused; "
        "echo w > note.txt && echo ws-written; "
        "pwd"
    )
    res = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=link,
            argv=("sh", "-c", probe),
            isolation="strict",
            extra_rw_paths=(outer,),
            extra_protect_paths=(link / ".git",),
            timeout_s=20.0,
        )
    )
    out = res.stdout + res.stderr
    assert "real-refused" in res.stdout and "REAL-WRITABLE" not in res.stdout, out
    assert "cwd-refused" in res.stdout and "CWD-WRITABLE" not in res.stdout, out
    assert config.read_text(encoding="utf-8") == "original\n"
    assert "ws-written" in res.stdout, out
    assert (ws / "note.txt").read_text(encoding="utf-8") == "w\n"
    assert res.stdout.splitlines()[-1] == str(ws.resolve()), out


def test_a_hidden_path_named_through_a_symlink_is_masked_at_its_real_path(
    tmp_path: pathlib.Path,
) -> None:
    """A hidden path spelled through a symlink is masked where the workspace is mounted.

    The mask went on the spelled name only, which the jail does not hold: the workspace is
    mounted at its real path, and the hidden directory stayed readable there.
    """
    ws = tmp_path / "ws"
    (ws / "secret").mkdir(parents=True)
    (ws / "secret" / "key").write_text("key\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(ws)

    for cwd in (ws, link):
        res = jail.run_in_jail(
            kinds.JailPolicy(
                cwd=cwd,
                argv=("sh", "-c", "ls -A secret | wc -l"),
                isolation="strict",
                hide_paths=(link / "secret",),
                timeout_s=20.0,
            )
        )
        assert res.stdout.strip() == "0", (cwd, res.stdout + res.stderr)
