"""Define the measurements exported for an enhanced completed repetition."""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CompletedRepetition:
    """
    Measurements recorded over one inclusive repetition interval.

    Frame fields are integer video-frame identities, angle fields are degrees,
    and ``duration_frames`` is ``end_frame - start_frame + 1``. Alignment
    values contain only valid observations from that interval, so missing
    evidence is represented by absence rather than a placeholder or stale
    value. Classification is performed separately by ``RepetitionClassifier``.
    """

    rep_id: int

    start_frame: int
    bottom_frame: int
    end_frame: int

    start_top_angle: float
    minimum_elbow_angle: float
    end_top_angle: float

    duration_frames: int

    # Valid smoothed body-alignment observations collected during
    # the repetition. Missing observations are not inserted.
    alignment_angles: Tuple[float, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        """Return a dictionary suitable for repetition-level logging."""
        return asdict(self)
