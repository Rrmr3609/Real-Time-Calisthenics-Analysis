from enum import Enum
from typing import Optional

from analysis.repetition_result import CompletedRepetition


class PushUpPhase(str, Enum):
    WAITING = "waiting"
    TOP = "top"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class PushUpPhaseStateMachine:
    """
    Temporal push-up phase and repetition detector.

    This enhanced counter uses smoothed elbow angles, hysteresis and
    consecutive-frame confirmation.

    Segmentation thresholds are deliberately more permissive than
    final form-quality thresholds. This allows attempted repetitions
    with insufficient depth or incomplete extension to be segmented
    before they are classified.
    """

    def __init__(
        self,
        top_region_threshold: float = 130.0,
        bottom_region_threshold: float = 120.0,
        hysteresis: float = 5.0,
        confirmation_frames: int = 3,
        missing_grace_frames: int = 5,
        minimum_rep_frames: int = 8,
    ):
        if bottom_region_threshold >= top_region_threshold:
            raise ValueError(
                "bottom_region_threshold must be lower than "
                "top_region_threshold"
            )

        if hysteresis < 0.0:
            raise ValueError("hysteresis cannot be negative")

        if confirmation_frames < 1:
            raise ValueError(
                "confirmation_frames must be at least 1"
            )

        if missing_grace_frames < 0:
            raise ValueError(
                "missing_grace_frames cannot be negative"
            )

        if minimum_rep_frames < 1:
            raise ValueError(
                "minimum_rep_frames must be at least 1"
            )

        self.top_region_threshold = float(
            top_region_threshold
        )
        self.bottom_region_threshold = float(
            bottom_region_threshold
        )
        self.hysteresis = float(hysteresis)
        self.confirmation_frames = confirmation_frames
        self.missing_grace_frames = missing_grace_frames
        self.minimum_rep_frames = minimum_rep_frames

        self.phase = PushUpPhase.WAITING
        self.rep_count = 0

        self._candidate_phase: Optional[PushUpPhase] = None
        self._candidate_count = 0
        self._missing_count = 0

        self._rep_start_frame: Optional[int] = None
        self._bottom_frame: Optional[int] = None

        self._start_top_angle: Optional[float] = None
        self._minimum_elbow_angle: Optional[float] = None
        self._end_top_angle: Optional[float] = None

    def _clear_candidate(self) -> None:
        self._candidate_phase = None
        self._candidate_count = 0

    def _confirm_transition(
        self,
        target_phase: PushUpPhase,
        condition: bool,
    ) -> bool:
        """
        Confirm a transition only when its condition remains true for
        the configured number of consecutive valid frames.
        """
        if not condition:
            self._clear_candidate()
            return False

        if self._candidate_phase == target_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = target_phase
            self._candidate_count = 1

        if self._candidate_count >= self.confirmation_frames:
            self._clear_candidate()
            return True

        return False

    def _clear_current_attempt(self) -> None:
        self._rep_start_frame = None
        self._bottom_frame = None

        self._start_top_angle = None
        self._minimum_elbow_angle = None
        self._end_top_angle = None

        self._clear_candidate()

    def _reset_to_waiting(self) -> None:
        self.phase = PushUpPhase.WAITING
        self._missing_count = 0
        self._clear_current_attempt()

    def _return_to_top_without_counting(
        self,
        angle: float,
    ) -> None:
        """
        Return to a ready top state after an incomplete attempt that
        never entered the bottom region.
        """
        self.phase = PushUpPhase.TOP
        self._clear_current_attempt()
        self._start_top_angle = angle

    def _complete_repetition(
        self,
        frame_index: int,
        angle: float,
    ) -> Optional[CompletedRepetition]:
        if (
            self._rep_start_frame is None
            or self._bottom_frame is None
            or self._start_top_angle is None
            or self._minimum_elbow_angle is None
        ):
            self._return_to_top_without_counting(angle)
            return None

        duration_frames = (
            frame_index - self._rep_start_frame + 1
        )

        if duration_frames < self.minimum_rep_frames:
            self._return_to_top_without_counting(angle)
            return None

        self.rep_count += 1

        completed = CompletedRepetition(
            rep_id=self.rep_count,
            start_frame=self._rep_start_frame,
            bottom_frame=self._bottom_frame,
            end_frame=frame_index,
            start_top_angle=self._start_top_angle,
            minimum_elbow_angle=self._minimum_elbow_angle,
            end_top_angle=max(
                angle,
                self._end_top_angle
                if self._end_top_angle is not None
                else angle,
            ),
            duration_frames=duration_frames,
        )

        self.phase = PushUpPhase.TOP
        self._clear_current_attempt()
        self._start_top_angle = angle

        return completed

    def update(
        self,
        elbow_angle: Optional[float],
        frame_index: int,
    ) -> dict:
        """
        Update the phase machine with one smoothed elbow angle.

        Returns current phase, count and an optional completed
        repetition object.
        """
        previous_phase = self.phase
        completed_repetition = None

        if elbow_angle is None:
            self._missing_count += 1
            self._clear_candidate()

            if self._missing_count > self.missing_grace_frames:
                self._reset_to_waiting()

            return {
                "phase": self.phase.value,
                "phase_changed": self.phase != previous_phase,
                "rep_count": self.rep_count,
                "completed_repetition": None,
                "missing_angle_frames": self._missing_count,
            }

        angle = float(elbow_angle)
        self._missing_count = 0

        top_descent_boundary = (
            self.top_region_threshold - self.hysteresis
        )

        bottom_ascent_boundary = (
            self.bottom_region_threshold + self.hysteresis
        )

        if self.phase == PushUpPhase.WAITING:
            if self._confirm_transition(
                PushUpPhase.TOP,
                angle >= self.top_region_threshold,
            ):
                self.phase = PushUpPhase.TOP
                self._start_top_angle = angle

        elif self.phase == PushUpPhase.TOP:
            if self._start_top_angle is None:
                self._start_top_angle = angle
            else:
                self._start_top_angle = max(
                    self._start_top_angle,
                    angle,
                )

            if self._confirm_transition(
                PushUpPhase.DESCENDING,
                angle <= top_descent_boundary,
            ):
                self.phase = PushUpPhase.DESCENDING

                self._rep_start_frame = max(
                    0,
                    frame_index
                    - self.confirmation_frames
                    + 1,
                )

                self._minimum_elbow_angle = angle
                self._bottom_frame = frame_index

        elif self.phase == PushUpPhase.DESCENDING:
            if (
                self._minimum_elbow_angle is None
                or angle < self._minimum_elbow_angle
            ):
                self._minimum_elbow_angle = angle
                self._bottom_frame = frame_index

            if angle <= self.bottom_region_threshold:
                if self._confirm_transition(
                    PushUpPhase.BOTTOM,
                    True,
                ):
                    self.phase = PushUpPhase.BOTTOM

            elif angle >= self.top_region_threshold:
                # The movement returned to the top without entering
                # the provisional bottom region.
                if self._confirm_transition(
                    PushUpPhase.TOP,
                    True,
                ):
                    self._return_to_top_without_counting(
                        angle
                    )

            else:
                self._clear_candidate()

        elif self.phase == PushUpPhase.BOTTOM:
            if (
                self._minimum_elbow_angle is None
                or angle < self._minimum_elbow_angle
            ):
                self._minimum_elbow_angle = angle
                self._bottom_frame = frame_index

            if self._confirm_transition(
                PushUpPhase.ASCENDING,
                angle >= bottom_ascent_boundary,
            ):
                self.phase = PushUpPhase.ASCENDING
                self._end_top_angle = angle

        elif self.phase == PushUpPhase.ASCENDING:
            if self._end_top_angle is None:
                self._end_top_angle = angle
            else:
                self._end_top_angle = max(
                    self._end_top_angle,
                    angle,
                )

            if angle >= self.top_region_threshold:
                if self._confirm_transition(
                    PushUpPhase.TOP,
                    True,
                ):
                    completed_repetition = (
                        self._complete_repetition(
                            frame_index=frame_index,
                            angle=angle,
                        )
                    )

            elif angle <= self.bottom_region_threshold:
                # The subject moved back down before completing the
                # return to the top.
                if self._confirm_transition(
                    PushUpPhase.BOTTOM,
                    True,
                ):
                    self.phase = PushUpPhase.BOTTOM

            else:
                self._clear_candidate()

        return {
            "phase": self.phase.value,
            "phase_changed": self.phase != previous_phase,
            "rep_count": self.rep_count,
            "completed_repetition": completed_repetition,
            "missing_angle_frames": self._missing_count,
        }

    def reset(self) -> None:
        self.rep_count = 0
        self._reset_to_waiting()