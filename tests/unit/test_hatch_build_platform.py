# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The jail build hook is host-gated.

The agent6-jail crate is Linux-only (namespaces, Landlock, seccomp), but the
hook keyed the build on cargo's PRESENCE alone: a mac with cargo installed
failed the sdist install on Linux-only symbols while a bare one happened to
pass. Non-Linux hosts skip the build and install the pure wheel (isolation
"none" is their documented posture)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_hook_module() -> ModuleType:
    # hatchling is a build-time dependency, absent from the dev venv; the
    # pin's subject is the hook's own gate logic, so a bare base suffices.
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
