from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tomllib
import types

import pytest

_SCREENSHOTS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "screenshots"


@pytest.fixture
def raw_video(tmp_path: pathlib.Path) -> pathlib.Path:
    """Make an untagged recording with the terminal's cyan accent."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    raw = tmp_path / "raw.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x06f5f3:s=64x64:r=4",
            "-t",
            "0.5",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            str(raw),
        ],
        check=True,
    )
    return raw


@pytest.fixture
def overlay(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.syspath_prepend(str(_SCREENSHOTS))
    spec = importlib.util.spec_from_file_location(
        "keystroke_overlay", _SCREENSHOTS / "keystroke_overlay.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _tags(video: pathlib.Path) -> dict[str, str]:
    result = subprocess.check_output(
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
            str(video),
        ],
        text=True,
    )
    return json.loads(result)["streams"][0]


@pytest.mark.parametrize("keys", ["", "Enter\n"])
def test_overlay_converts_and_tags_even_without_keys(
    overlay: types.ModuleType, raw_video: pathlib.Path, tmp_path: pathlib.Path, keys: str
) -> None:
    """An overlay output carries BT.709 even when there are no toasts to draw."""
    tape = tmp_path / "clip.tape"
    tape.write_text(keys + "Sleep 500ms\n", encoding="utf-8")
    out = tmp_path / "out.webm"
    overlay.overlay(tape, raw_video, out)
    assert _tags(out) == {
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
    }


def test_vhs_conversion_checks_its_output(raw_video: pathlib.Path) -> None:
    """VHS output is converted in place and the checker rejects untagged recordings."""
    script = _SCREENSHOTS / "webm.py"
    rejected = subprocess.run(
        ["python3", str(script), "--check", str(raw_video)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "expected color_space=bt709" in rejected.stderr
    before = _pixel(raw_video)
    subprocess.run(["python3", str(script), str(raw_video)], check=True)
    assert max(abs(a - b) for a, b in zip(before, _pixel(raw_video), strict=True)) <= 4
    assert _tags(raw_video) == {
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
    }
    subprocess.run(["python3", str(script), "--check", str(raw_video)], check=True)


def _pixel(video: pathlib.Path) -> bytes:
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )


def test_hero_run_enables_the_tapes_baseline_approval() -> None:
    """The hero config requests the baseline approval that opens the tape."""
    script = (_SCREENSHOTS / "hero_run.sh").read_text(encoding="utf-8")
    config_text = script.split('cat > "$XDG_CONFIG_HOME/agent6/config.toml" <<EOF\n', 1)[1]
    harness = tomllib.loads(config_text.split("\nEOF\n", 1)[0])["harness"]
    assert harness.get("verify_at_start") is True
    assert harness.get("verify_when") == "finish"


def _run_hero(
    tmp_path: pathlib.Path, space: str, gif_bytes: int = 7
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run hero.sh over stub tools; ffprobe reports `space`, ffmpeg writes `gif_bytes` per gif."""
    screenshots = tmp_path / "docs" / "screenshots"
    out = screenshots / "out"
    out.mkdir(parents=True)
    for name in ("hero.sh", "webm.py"):
        shutil.copy2(_SCREENSHOTS / name, screenshots / name)
    for name in ("tui", "cli", "web"):
        (out / f"hero-src-{name}.webm").write_text("recording", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tools = {
        "git": f"#!/bin/sh\nprintf '%s\\n' '{tmp_path}'\n",
        "ffprobe": (
            f'#!/bin/sh\ncase "$*" in *format=duration*) echo 9.0 ;; *) echo {space} ;; esac\n'
        ),
        "ffmpeg": (
            f"#!{sys.executable}\nimport json, pathlib, sys\n"
            f"with open({str(tmp_path / 'encodes.jsonl')!r}, 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            f"size = {gif_bytes} if sys.argv[-1].endswith('.gif') else 7\n"
            "pathlib.Path(sys.argv[-1]).write_bytes(b'x' * size)\n"
        ),
    }
    for name, text in tools.items():
        tool = bin_dir / name
        tool.write_text(text, encoding="utf-8")
        tool.chmod(0o755)
    result = subprocess.run(
        ["bash", str(screenshots / "hero.sh")],
        env=os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )
    encodes = [json.loads(line) for line in (tmp_path / "encodes.jsonl").read_text().splitlines()]
    return result, encodes


@pytest.mark.parametrize("space", ["bt709", "unknown"])
def test_hero_checks_each_encode_before_building_gifs(tmp_path: pathlib.Path, space: str) -> None:
    """A failed colour-space check fails the hero build and prevents its GIF encode."""
    result, encodes = _run_hero(tmp_path, space)
    if space == "unknown":
        assert result.returncode != 0
        assert "expected color_space=bt709" in result.stderr
        assert not any(args[-1].endswith(".gif") for args in encodes)
        return
    assert result.returncode == 0, result.stderr
    assert len(encodes) == 6
    for args in encodes:
        if args[-1].endswith(".gif"):
            assert args[args.index("-vf") + 1] == (
                "fps=10,split[a][b];[a]palettegen=max_colors=96[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=5"
            )
            assert "-colorspace" not in args
        else:
            for flag in ("-colorspace", "-color_primaries", "-color_trc"):
                assert args[args.index(flag) + 1] == "bt709"
            assert (
                "scale=out_color_matrix=bt709:out_range=tv"
                in args[args.index("-filter_complex") + 1]
            )


def test_hero_refuses_a_gif_over_the_image_proxy_cap(tmp_path: pathlib.Path) -> None:
    """A gif over 5 MiB fails the build: GitHub's image proxy would answer 404 for it."""
    result, encodes = _run_hero(tmp_path, "bt709", gif_bytes=5 * 1024 * 1024 + 1)
    assert result.returncode != 0
    assert "image proxy serves at most 5242880" in result.stderr
    assert len(encodes) == 6
