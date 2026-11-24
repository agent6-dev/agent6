"""The benchmark arms change only the advertised tool or its description."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tarfile
from typing import Any

from agent6.tools import schema

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARMS = ROOT / "bench/tools/arms"


def test_archived_arm_surfaces(tmp_path: pathlib.Path) -> None:
    """A withheld name is not callable, and shorter descriptions keep the argument schemas."""
    surfaces: dict[str, dict[str, Any]] = {}
    for arm in ("control", "no-docs", "no-patch", "short"):
        tree = tmp_path / arm
        tree.mkdir()
        with (tree / "src.tar").open("wb+") as archive:
            subprocess.run(["git", "archive", "HEAD", "src"], cwd=ROOT, stdout=archive, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tar:
                tar.extractall(tree, filter="data")
        patch = ARMS / f"{arm}.patch"
        assert patch.exists(), f"no patch for {arm}"
        subprocess.run(["git", "apply", "--allow-empty", patch], cwd=tree, check=True)
        read = subprocess.check_output(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys,json; sys.path.insert(0, sys.argv[1]); from agent6.tools import schema;"
                "print(json.dumps({mode: {"
                "'tools': {t.TOOL_NAME: [t.TOOL_DESCRIPTION,schema.wire_schema(t)] "
                "for t in (*s.base,*s.extras)}, 'permitted': sorted(s.permitted)} "
                "for mode in ['run','ask'] for s in [schema.mode_tools(mode)]}))",
                tree / "src",
            ],
            cwd=tree,
            text=True,
        )
        surfaces[arm] = json.loads(read)

    base = surfaces["control"]
    for arm, mode, withheld in (
        ("no-docs", "ask", "agent6_docs"),
        ("no-patch", "run", "apply_patch"),
    ):
        tools = base[mode]["tools"].copy()
        del tools[withheld]
        assert surfaces[arm][mode]["tools"] == tools
        assert withheld not in surfaces[arm][mode]["permitted"]

    short = surfaces["short"]["run"]["tools"]
    assert short.keys() == base["run"]["tools"].keys()
    for name, (description, parameters) in base["run"]["tools"].items():
        assert short[name][1] == parameters
        if name in {"apply_edit", "ask_user", "parallel"}:
            assert len(short[name][0]) < len(description)
        else:
            assert short[name][0] == description
    assert "apply_patch" in schema.mode_tools("run").names
