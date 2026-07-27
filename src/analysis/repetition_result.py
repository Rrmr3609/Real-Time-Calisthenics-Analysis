from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CompletedRepetition:
    """
    Measurements recorded for one temporally completed repetition.

    Classification is performed separately by RepetitionClassifier.
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
        return asdict(self)