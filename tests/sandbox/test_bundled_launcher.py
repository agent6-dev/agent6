from __future__ import annotations

import hashlib
import pathlib
import tempfile
from typing import TYPE_CHECKING

from agent6 import kinds
from agent6.sandbox import container, jail

if TYPE_CHECKING:
    import pytest


def check_bundled_launcher(root: pathlib.Path) -> None:
    """Run the installed package's launcher and check its reported identity."""
    bundled = pathlib.Path(jail.__file__).resolve().parent / "_bin" / "agent6-jail"
    assert jail.require_jail_binary() == bundled
    digest = "sha256:" + hashlib.sha256(bundled.read_bytes()).hexdigest()
    selected = container.discover(
        root,
        container.ContainerOptions(exec=("/usr/bin/env", "-i", "HOME=/tmp", "PATH=/usr/bin:/bin")),
    )
    session = container.prepare(selected)
    try:
        result = session.run(("/bin/echo", "bundled-launcher"))
        assert isinstance(result, kinds.CommandResult), result
        assert result.returncode == 0 and result.stdout == "bundled-launcher\n", result
        assert session.target is not None and session.target.launcher_instance
        assert session.target.launcher_digest == digest
    finally:
        assert not session.close()


def test_bundled_launcher(
    tmp_path: pathlib.Path, static_launcher: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected launcher comes from the package, not an environment override."""
    import pytest

    if static_launcher.parent != pathlib.Path(jail.__file__).resolve().parent / "_bin":
        pytest.skip("the static launcher is outside the package")
    monkeypatch.delenv("AGENT6_JAIL_BIN", raising=False)
    check_bundled_launcher(tmp_path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        check_bundled_launcher(pathlib.Path(directory))
