# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 is a coding agent that jails model commands and uses editable state machines.

The layers are `ui`, `app`, `harness`, `tools` and `sandbox`, and each imports only below itself.
Every other package and the modules at this level are shared: any layer may import them.
`__version__` here is the one version every surface reads.
Main modules: `kinds`, `paths`, `errors`, `budget`, `directive`, `evidence`, `pack`.
"""

from __future__ import annotations

__version__ = "0.0.45"
