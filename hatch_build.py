# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hatch build hook that compiles the `agent6-jail` Rust binary and bundles
it inside the wheel under `agent6/sandbox/_bin/agent6-jail`.

Runs whenever hatchling builds a wheel — including the editable wheel that
`uv sync` produces — so a contributor only needs `uv sync` to get a working
dev environment, with no separate `cargo build` step. The compiled binary
is copied into the package tree (`src/agent6/sandbox/_bin/`) so the wheel
contains it, and `_locate_jail_binary` finds it relative to the installed
package at runtime.

Because the embedded binary is platform-specific, a wheel built by this
hook is also platform-specific: `pure_python` is set to False and
`infer_tag` to True, which makes hatchling stamp the wheel with the
current interpreter + ABI + platform tag. For PyPI distribution this hook
should be run inside a manylinux container (e.g. via cibuildwheel) so the
binary links against an old enough glibc to satisfy `manylinux_2_28` or
similar. The binary is Linux-only (Landlock + seccomp + namespaces are
Linux kernel features), so no macOS/Windows wheels are produced.

The hook is deliberately tolerant:

  - If `cargo` is not on PATH (e.g. a user pip-installs an sdist on a
    Rust-less host) we skip with a clear stderr note and produce a
    pure-Python wheel without the binary. `agent6 check sandbox` will
    then tell the user to build the binary themselves.
  - If `AGENT6_SKIP_JAIL_BUILD=1` is set we also skip. Useful when the
    CI pipeline builds the jail in a dedicated job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_BIN_NAME = "agent6-jail"
# Destination inside the package tree. Must match the lookup in
# src/agent6/sandbox/jail.py::_locate_jail_binary.
_BUNDLE_REL = Path("src") / "agent6" / "sandbox" / "_bin" / _BIN_NAME


class JailBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        dest = root / _BUNDLE_REL

        if os.environ.get("AGENT6_SKIP_JAIL_BUILD") == "1":
            print(
                "[hatch_build] AGENT6_SKIP_JAIL_BUILD=1, skipping cargo build",
                file=sys.stderr,
            )
            self._maybe_mark_platform_wheel(build_data, dest)
            return

        manifest = root / "src" / "agent6" / "jail" / "Cargo.toml"
        if not manifest.is_file():
            print(
                f"[hatch_build] {manifest} not found, skipping cargo build",
                file=sys.stderr,
            )
            self._maybe_mark_platform_wheel(build_data, dest)
            return

        cargo = shutil.which("cargo")
        if cargo is None:
            print(
                "[hatch_build] cargo not on PATH, skipping; the agent6-jail "
                "binary must be built separately (see SECURITY.md)",
                file=sys.stderr,
            )
            self._maybe_mark_platform_wheel(build_data, dest)
            return

        print("[hatch_build] cargo build --release --locked", file=sys.stderr)
        subprocess.run(
            [cargo, "build", "--release", "--locked", "--manifest-path", str(manifest)],
            check=True,
        )

        built = root / "src" / "agent6" / "jail" / "target" / "release" / _BIN_NAME
        if not built.is_file():
            print(
                f"[hatch_build] cargo build succeeded but {built} is missing",
                file=sys.stderr,
            )
            self._maybe_mark_platform_wheel(build_data, dest)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, dest)
        dest.chmod(0o755)
        print(f"[hatch_build] bundled {dest.relative_to(root)}", file=sys.stderr)
        self._maybe_mark_platform_wheel(build_data, dest)

    @staticmethod
    def _maybe_mark_platform_wheel(build_data: dict[str, Any], dest: Path) -> None:
        """Stamp the wheel as platform-specific iff we actually have a binary.

        Without this, hatchling builds a `py3-none-any` wheel which is wrong
        when we embed a native binary. With it, we set an explicit
        `py3-none-linux_x86_64` tag — the only native artifact is the bundled
        `agent6-jail` binary, which does not depend on the Python ABI, so a
        single wheel covers every interpreter satisfying `requires-python`
        (3.12, 3.13, 3.14, …). The CI workflow then retags `linux_*` to
        `manylinux_2_34_*` for PyPI.
        """
        if dest.is_file():
            build_data["pure_python"] = False
            build_data["infer_tag"] = False
            # py3-none-<plat>: any Python 3, no ABI dependency, host platform.
            build_data["tag"] = f"py3-none-linux_{os.uname().machine}"
