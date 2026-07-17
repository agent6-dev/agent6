# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Application pipelines that compose the engine but are not a front-end.

`agent6.app` sits beside `ui/` in the layering: it drives the run/resume
lifecycle (`run`, `resume`, with their `preflight`/`manifest`/`merge`/
`finalize`/`providers` pieces) and multi-run orchestration (the `--parallel`
fan-out and the coordinator's `/parallel` dispatch, `parallel`/`compare`) over
`workflows`, `git_ops`, `runs`, and the headless `viewmodel`, and never imports
`agent6.ui`. What it cannot do itself -- own a terminal, render a live view,
spawn a detached `agent6` process, or drive the run-dir bridge -- is injected
by the front-end (`ui/cli`) as frozen values of callables (`run.RunFrontend`,
`parallel.LaneRuntime`), so the pipelines stay testable and ui-free.
"""

from __future__ import annotations
