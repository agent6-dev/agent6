# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Version control for agent6's own commits, refs and merges.

Git today: the operations (`git_ops`), the hardening flags and hook policy (`git_policy`),
the commit message composer (`commit_message`) and the private runner (`_git_command`),
which only `git_ops` imports. All leaves; import the submodules directly.
"""

from __future__ import annotations
