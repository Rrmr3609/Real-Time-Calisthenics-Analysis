from dataclasses import replace
from typing import List, Optional

from analysis.repetition_result import CompletedRepetition


ACTIVE_PHASES = {
    "descending",
    "bottom",
    "ascending",
}


class RepetitionFeatureAggregator:
    """
    Collect frame-level body-alignment measurements for one active
    repetition and attach them to its CompletedRepetition object.
    """

    def __init__(self):
        self._alignment_angles: List[float] = []

    def update(
        self,
        phase: str,
        phase_changed: bool,
        body_alignment_angle: Optional[float],
        completed_repetition: Optional[CompletedRepetition],
    ) -> Optional[CompletedRepetition]:
        """
        Update repetition-level feature aggregation.

        Returns an enriched CompletedRepetition only when a repetition
        has completed. Otherwise returns None.
        """

        # A newly confirmed descending phase begins a new attempt.
        if phase_changed and phase == "descending":
            self.reset()

        # Collect valid alignment values while the repetition is active.
        if (
            phase in ACTIVE_PHASES
            and body_alignment_angle is not None
        ):
            self._alignment_angles.append(
                float(body_alignment_angle)
            )

        # On the completion frame, the state machine has already moved
        # back to "top". Include the final alignment observation once.
        if completed_repetition is not None:
            if body_alignment_angle is not None:
                self._alignment_angles.append(
                    float(body_alignment_angle)
                )

            enriched = replace(
                completed_repetition,
                alignment_angles=tuple(
                    self._alignment_angles
                ),
            )

            self.reset()
            return enriched

        # If an attempt returns to top without completion, or is reset
        # to waiting after missing frames, discard its accumulated data.
        if (
            phase_changed
            and phase in {"top", "waiting"}
        ):
            self.reset()

        return None

    def reset(self) -> None:
        self._alignment_angles = []