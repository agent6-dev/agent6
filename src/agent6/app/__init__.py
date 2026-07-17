# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Application pipelines that compose the engine but are not a front-end.

`agent6.app` sits beside `ui/` in the layering: it drives multi-run
orchestration (the `--parallel` fan-out and the coordinator's `/parallel`
dispatch) over `workflows`, `git_ops`, `runs`, and the headless `viewmodel`,
and never imports `agent6.ui`. The one thing it cannot do itself -- spawn a
detached `agent6` process and drive the run-dir bridge -- is injected by the
front-end (`ui/cli`) as a small set of callables, so the pipeline stays
testable and ui-free.
"""

from __future__ import annotations
