# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the engine into the run, resume, fork and machine lifecycles.

`agent6.app` sits below `ui/` and never imports it: the lifecycles (`run`, `resume`, `fork`,
`machine`) and the `--parallel` fan-out (`parallel`, `compare`) drive `harness`, `git_ops`,
`sessions` and the headless `viewmodel`. What needs a terminal, a live view, a detached
`agent6` process or the run-dir bridge is injected by `ui/cli` as frozen values of callables
(`frontend.SessionFrontend`, `parallel.LaneRuntime`). Output goes through the injected
`reporter.Reporter` (default `STDIO_REPORTER`), never a direct `print`.
"""

from __future__ import annotations
