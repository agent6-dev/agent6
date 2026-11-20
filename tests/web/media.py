from __future__ import annotations

import base64
import pathlib
import wave

# Three seconds of a blue 64 by 48 pixel VP8 frame, at one frame per second.
VIDEO = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAIxEU2bdLpNu4tTq4QVSalmU6yB"
    "oU27i1OrhBZUrmtTrIHGTbuMU6uEElTDZ1OsggETTbuMU6uEHFO7a1OsggIb7AEAAAAAAABZAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAVSalmoCrXsYMPQkBNgIRMYXZmV0GETGF2ZkSJiECncAAAAAAAFlSua8iuAQAAAAAAAD/XgQFz"
    "xYgAAAAAAAAAAZyBACK1nIN1bmSIgQCGhVZfVlA4g4EBI+ODhDuaygDgkLCBQLqBMJqBAlWwhFW5gQESVMNn0HNz"
    "zWPAi2PFiAAAAAAAAAABZ8iYRaOHRU5DT0RFUkSHi0xhdmMgbGlidnB4Z8ihRaOIRFVSQVRJT05Eh5MwMDowMDow"
    "My4wMDAwMDAwMDAAH0O2dUCt54EAo8KBAACAUAMAnQEqQAAwAABHCIWFiJmEiAICAnWqA/gCBwjg/fQmzy4A/v9N"
    "Ev/8SqfxKp/Eqn/xKp/9GH8MP4Yf3hCjsYED6AARAgABEBAAGAAaT/QMABT6qqiA/v9NEv/8SqfxKp/Eqn/xKp/9"
    "GH8MP4Yf3hCjsYEH0AARAgABEBAAGAAaT/QMABT6qqiA/v9NEv/8SqfxKp/Eqn/xKp/9GH8MP4Yf3hAcU7trkbuP"
    "s4EAt4r3gQHxggFo8IED"
)


def write(repo: pathlib.Path) -> None:
    """Write short media fixtures without an external encoder."""
    (repo / "clip.webm").write_bytes(VIDEO)
    with wave.open(str(repo / "sound.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(1)
        output.setframerate(8000)
        output.writeframes(b"\x80" * 24000)
