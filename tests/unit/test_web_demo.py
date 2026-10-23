from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from unittest import mock

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots" / "web_demo.py"


@pytest.fixture
def web_demo() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("web_demo", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("mode", "width", "height"),
    [("hero", 1600, 900), ("desktop", 1280, 800), ("phone", 390, 844), ("machine", 1280, 800)],
)
def test_recording_uses_the_mode_viewport(
    web_demo: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    mode: str,
    width: int,
    height: int,
) -> None:
    """The hero records at 16:9 without resizing the desktop, phone or machine tours."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--url",
            "http://localhost:8987",
            "--out",
            str(tmp_path / "out.webm"),
            "--mode",
            mode,
        ],
    )
    monkeypatch.setenv("AGENT6_WEB_SECRET", "test-secret")
    playwright = mock.MagicMock()
    browser = playwright.__enter__.return_value.chromium.launch.return_value
    context = browser.new_context.return_value
    context.request.post.return_value.json.return_value = {"token": "test-token"}
    context.new_page.return_value.video = None
    monkeypatch.setattr(web_demo, "sync_playwright", lambda: playwright)
    drive = mock.Mock(return_value=0.0)
    drive_machine = mock.Mock(return_value=0.0)
    monkeypatch.setattr(web_demo, "drive", drive)
    monkeypatch.setattr(web_demo, "drive_machine", drive_machine)

    web_demo.main()

    kwargs = browser.new_context.call_args.kwargs
    assert kwargs["viewport"] == {"width": width, "height": height}
    assert kwargs["record_video_size"] == {"width": width, "height": height}
    if mode == "machine":
        assert drive_machine.call_count == 1
        drive.assert_not_called()
    else:
        assert drive.call_args.args[2] == ("desktop" if mode == "hero" else mode)
        drive_machine.assert_not_called()
