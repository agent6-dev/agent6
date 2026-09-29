# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A launcher binary the kernel will not execute is a refusal, not a crash."""

from __future__ import annotations

import errno
import os
import pathlib
import re

import pytest

from agent6 import kinds
from agent6.sandbox import jail


def test_an_unusable_launcher_binary_is_refused_with_the_remedy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexecutable launcher binary is named in a refusal, not a bare OSError."""
    fake = tmp_path / "agent6-jail"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o644)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))
    policy = kinds.JailPolicy(cwd=tmp_path, argv=("/bin/true",), isolation="strict", timeout_s=5.0)
    with pytest.raises(jail.JailUnavailableError, match=re.escape(str(fake))) as info:
        jail.run_in_jail(policy)
    said = str(info.value)
    assert "Permission denied" in said
    assert "uv sync --reinstall-package agent6" in said and "AGENT6_JAIL_BIN" in said


def test_a_fork_or_descriptor_failure_is_not_blamed_on_the_binary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ENOEXEC and EACCES speak about the binary; EAGAIN, EMFILE and ENOMEM pass through.

    Through the cached strict probe, a transient fork failure at startup would resolve `auto` for
    the whole run.
    """
    fake = tmp_path / "agent6-jail"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))

    def _fork_fails(*_a: object, **_k: object) -> object:
        raise OSError(errno.EAGAIN, os.strerror(errno.EAGAIN))

    monkeypatch.setattr(jail.subprocess, "Popen", _fork_fails)
    policy = kinds.JailPolicy(cwd=tmp_path, argv=("/bin/true",), isolation="strict", timeout_s=5.0)
    with pytest.raises(OSError) as info:
        jail.run_in_jail(policy)
    assert not isinstance(info.value, jail.JailUnavailableError)
    assert info.value.errno == errno.EAGAIN
