"""Convert recordings to BT.709 and verify their colour-space tags."""

from __future__ import annotations

import argparse
import pathlib
import subprocess


class WebMError(RuntimeError):
    """A recording does not carry its required colour space."""


def check(path: pathlib.Path) -> None:
    """Require a BT.709 colour-space tag on the first video stream.

    Args:
        path: The encoded recording.

    Raises:
        WebMError: The stream has no BT.709 colour-space tag.
    """
    space = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_space",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    ).strip()
    if space != "bt709":
        raise WebMError(f"{path}: expected color_space=bt709, got {space or 'missing'}")


def encode(
    source: pathlib.Path, out: pathlib.Path, *, filters: str = "", trim_s: float = 0
) -> None:
    """Encode limited-range BT.709 and check the output tag.

    Args:
        source: The raw recording.
        out: The output WebM path.
        filters: Filters applied before the colour conversion.
        trim_s: Seconds to trim from the start.
    """
    filters = f"{filters}," if filters else ""
    # Frame tags keep the filter output from overriding the encoder's colour tags.
    filters += (
        "scale=out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ss",
            f"{trim_s:.3f}",
            "-vf",
            filters,
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "32",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-an",
            str(out),
        ],
        check=True,
    )
    check(out)


def main() -> None:
    """Convert VHS output in place, or check an already encoded recording."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--check", action="store_true", help="Check without encoding.")
    parser.add_argument("path", type=pathlib.Path)
    args = parser.parse_args()
    if args.check:
        check(args.path)
        return
    converted = args.path.with_suffix(".bt709.webm")
    try:
        encode(args.path, converted)
        converted.replace(args.path)
    finally:
        converted.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
