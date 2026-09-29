# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Session identity and on-disk state.

Ids and prefix resolution (`id`), the state directory layout (`layout`), the manifest
(`manifest`), the single-writer flock (`lock`) and the answer-file contract between a
front-end and the harness (`ipc`). All leaves; import the submodules directly.
"""

from __future__ import annotations
