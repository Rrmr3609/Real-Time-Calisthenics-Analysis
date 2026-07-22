from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class CompletedRepetition:
    """
    Measurements recorded for one temporally completed repetition.

    This object does not assign a form class. Classification will be
    implemented separately after repetition segmentation is stable.
    """

    rep_id: int
    start_frame: int
    bottom_frame: int
    end_frame: int

    start_top_angle: float
    minimum_elbow_angle: float
    end_top_angle: float

    duration_frames: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)