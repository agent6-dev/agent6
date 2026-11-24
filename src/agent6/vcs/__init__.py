# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Version control for agent6's own commits, refs and merges.

A shared package: it imports `sandbox` and no layer above it.
Git lives in the `git` sub-package, and only this package's modules import its runner.
The package spells no push, `--force`, history rewrite or `reset --hard`.
Main modules: `commit_message`, `commit_graph`, `review`, `review_rule_reader`.
"""

from __future__ import annotations
