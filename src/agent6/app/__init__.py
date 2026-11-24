# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose executions into the run, resume, fork and machine lifecycles.

App owns what spans executions: the manifest, the writer lock and the providers.
The loop and every primitive it calls in a turn belong to `harness`.
App imports `harness` and the layers below it, never `ui`.
A front-end injects what app cannot do itself: `frontend.SessionFrontend` and
`parallel.children.ChildRuntime`.
Output goes through the injected `reporter.Reporter`, never a direct `print`.
Main modules: `run`, `resume`, `fork`, `execution`, `parallel`, `merge`, `finalize`, `stop`.
"""

from __future__ import annotations
