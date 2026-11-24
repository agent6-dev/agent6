# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch and the schemas exposed to the LLM.

A layer: it imports `sandbox` and the shared packages, never `harness`, `app` or `ui`.
The package itself loads nothing: import the submodule you need.
Main modules: `schema`, `dispatch`, `command_tools`, `fs_tools`, `policy`, `results`.
"""

from __future__ import annotations
