"""Keep the receipt and sequence watermark when session artifacts are pruned."""

from __future__ import annotations

import pathlib
from typing import Literal

import pydantic


class RetentionError(ValueError):
    """The prune receipt cannot be read safely."""


class Pruned(pydantic.BaseModel):
    """Record an explicit removal of a session's retained artifacts.

    Attributes:
        pruned_at: The UTC time pruning began.
        counts: The file counts selected for removal in each directory.
        last_transcript_seq: The highest removed sequence, including earlier prunes.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    pruned_at: pydantic.AwareDatetime
    counts: dict[Literal["transcripts", "checkpoints", "objects"], pydantic.NonNegativeInt]
    last_transcript_seq: pydantic.NonNegativeInt

    def notice(self, session_id: str) -> str:
        """Return the shared transcript and fork notice."""
        timestamp = self.model_dump(mode="json")["pruned_at"]
        return (
            f"session {session_id}: transcripts, checkpoints and objects were pruned at {timestamp}"
            f" (transcript sequences through {self.last_transcript_seq})"
        )


def read_pruned(session_dir: pathlib.Path) -> Pruned | None:
    """Read the prune receipt, returning None only when it is absent.

    Args:
        session_dir: The session's directory.

    Returns:
        The validated receipt, or None.

    Raises:
        RetentionError: The receipt exists but cannot be read.
    """
    path = session_dir / "pruned.json"
    try:
        return Pruned.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RetentionError(f"cannot read prune receipt {path}: {exc}") from exc


def last_sequence(transcripts_dir: pathlib.Path) -> int:
    """Return the highest transcript sequence on disk or in the prune receipt.

    Args:
        transcripts_dir: The session's transcript directory, possibly absent.

    Returns:
        The watermark from which a resumed sink continues.
    """
    pruned = read_pruned(transcripts_dir.parent)
    seqs = [pruned.last_transcript_seq if pruned else 0]
    seqs.extend(
        int(tail)
        for path in transcripts_dir.glob("*.json")
        if (tail := path.stem.rsplit("-", 1)[-1]).isdigit()
    )
    return max(seqs)
