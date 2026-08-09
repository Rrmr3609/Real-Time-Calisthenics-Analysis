"""Collect repetition-level features over state-machine windows."""

from dataclasses import replace
from typing import List, Optional

from analysis.repetition_result import CompletedRepetition


class RepetitionFeatureAggregator:
    """
    Collect body-alignment measurements over the state machine's
    inclusive repetition window.

    The window starts at the genuine top observation that supplies
    ``start_top_angle`` and ends at the frame that confirms the return to top.
    Missing values remain absent from ``alignment_angles``, while
    ``duration_frames`` retains every frame in the inclusive interval.
    """

    def __init__(self):
        self._window_start_frame: Optional[int] = None
        self._alignment_angles: List[float] = []

    def update(
        self,
        frame_index: int,
        repetition_window_start_frame: Optional[int],
        body_alignment_angle: Optional[float],
        completed_repetition: Optional[CompletedRepetition],
    ) -> Optional[CompletedRepetition]:
        """
        Collect one frame for the current inclusive repetition window.

        Valid alignment angles are measured in degrees. A completed repetition
        is returned with those observations attached only when the state
        machine completes the same window; otherwise the method returns
        ``None``.
        """

        if repetition_window_start_frame is None:
            self.reset()
            return None

        if (
            self._window_start_frame
            != repetition_window_start_frame
        ):
            self.reset()
            self._window_start_frame = (
                repetition_window_start_frame
            )

        if (
            frame_index >= repetition_window_start_frame
            and body_alignment_angle is not None
        ):
            self._alignment_angles.append(
                float(body_alignment_angle)
            )

        if completed_repetition is not None:
            if (
                completed_repetition.start_frame
                != repetition_window_start_frame
            ):
                raise ValueError(
                    "Completed repetition start frame does not match "
                    "the active aggregation window"
                )

            enriched = replace(
                completed_repetition,
                alignment_angles=tuple(
                    self._alignment_angles
                ),
            )

            # The completion frame is also the state machine's next top
            # anchor. Seed the tentative next window so the shared
            # boundary frame is available if no later top replaces it.
            self.reset()
            self._window_start_frame = frame_index
            if body_alignment_angle is not None:
                self._alignment_angles.append(
                    float(body_alignment_angle)
                )

            return enriched

        return None

    def reset(self) -> None:
        """Discard the active window and its collected observations."""
        self._window_start_frame = None
        self._alignment_angles = []
