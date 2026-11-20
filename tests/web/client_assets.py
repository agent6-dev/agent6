# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Select the delivered client subjects a DOM fixture exercises."""

from __future__ import annotations

from importlib import resources

from agent6.ui.web import page


def script(*subjects: str) -> str:
    """Return the named subjects in the same order as the page's inline script."""
    names = {f"client_{subject}.js" for subject in subjects}
    assert names.issubset(page._CLIENT_FILES), names.difference(page._CLIENT_FILES)
    web = resources.files("agent6.ui.web")
    return "".join(
        web.joinpath(name).read_text(encoding="utf-8")
        for name in page._CLIENT_FILES
        if name in names
    )
