# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail build hook is host-gated: non-Linux hosts skip the crate and install the pure wheel.

Keyed on cargo's presence alone, a mac with cargo failed the sdist install on Linux-only symbols.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_hook_module() -> ModuleType:
    # hatchling is absent from the dev venv; the subject is the hook's gate, so a bare base does.
    iface = ModuleType("hatchling.builders.hooks.plugin.interface")
    iface.BuildHookInterface = object  # type: ignore[attr-defined]
    for name in (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.builders.hooks.plugin",
    ):
        sys.modules.setdefault(name, ModuleType(name))
    sys.modules["hatchling.builders.hooks.plugin.interface"] = iface
    spec = importlib.util.spec_from_file_location("hatch_build", _ROOT / "hatch_build.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_non_linux_host_never_invokes_cargo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_hook_module()
    monkeypatch.setattr(mod.sys, "platform", "darwin")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cargo must not run on a non-linux host")

    monkeypatch.setattr(mod.subprocess, "run", _boom)

    def _cargo_present(name: str) -> str:
        return "/usr/local/bin/cargo"

    monkeypatch.setattr(mod.shutil, "which", _cargo_present)
    monkeypatch.delenv("AGENT6_SKIP_JAIL_BUILD", raising=False)
    hook = mod.JailBuildHook.__new__(mod.JailBuildHook)
    hook.__dict__["root"] = str(_ROOT)
    build_data: dict[str, Any] = {}
    hook.initialize("standard", build_data)  # returns without touching cargo
