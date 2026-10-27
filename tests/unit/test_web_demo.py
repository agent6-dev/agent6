from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import types
from unittest import mock

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots" / "web_demo.py"


@pytest.fixture
def web_demo(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.syspath_prepend(str(_SCRIPT.parent))
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
    playwright_api = pytest.importorskip("playwright.sync_api")
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
    monkeypatch.setattr(playwright_api, "sync_playwright", lambda: playwright)
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


@pytest.mark.parametrize("trim_s", [0.0, 0.05, 0.25])
def test_web_recording_always_converts_to_bt709(
    web_demo: types.ModuleType, tmp_path: pathlib.Path, trim_s: float
) -> None:
    """A negligible trim still converts and tags the browser recording."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    raw = tmp_path / "raw.webm"
    out = tmp_path / "out.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0f1414:s=64x64:r=4",
            "-t",
            "0.5",
            "-c:v",
            "libvpx-vp9",
            str(raw),
        ],
        check=True,
    )
    web_demo.write_trimmed(raw, out, trim_s)
    tags = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_space,color_primaries,color_transfer",
                "-of",
                "json",
                str(out),
            ],
            text=True,
        )
    )["streams"][0]
    assert tags == {
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
    }
    assert not raw.exists()


def test_missing_encoder_keeps_the_raw_recording(
    web_demo: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A missing encoder fails rather than publishing an untagged raw recording."""
    raw = tmp_path / "raw.webm"
    raw.write_bytes(b"raw recording")
    out = tmp_path / "out.webm"
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        web_demo.write_trimmed(raw, out, 0.0)
    assert raw.read_bytes() == b"raw recording"
    assert not out.exists()
