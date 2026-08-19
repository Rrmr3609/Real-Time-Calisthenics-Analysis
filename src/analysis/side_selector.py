"""Provide temporally stable left/right feature side selection."""

from typing import Optional


class StableSideSelector:
    """
    Select and retain one body side using visibility scores.

    Acquisition and switching both require stable evidence across consecutive
    frames. Once acquired, the selected side remains sticky unless the other
    side is sufficiently stronger for long enough, or the current side remains
    unavailable beyond its grace period. This deliberate stability prevents
    frame-to-frame side jitter from contaminating temporal measurements.

    Thresholds and confirmation lengths are configurable; the selector makes
    no assumptions about the feature represented by its left/right scores.
    """

    def __init__(
        self,
        minimum_score: float = 0.5,
        acquisition_frames: int = 3,
        switch_frames: int = 5,
        switch_margin: float = 0.10,
        missing_grace_frames: int = 5,
    ):
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")

        if acquisition_frames < 1:
            raise ValueError("acquisition_frames must be at least 1")

        if switch_frames < 1:
            raise ValueError("switch_frames must be at least 1")

        if switch_margin < 0.0:
            raise ValueError("switch_margin cannot be negative")

        if missing_grace_frames < 0:
            raise ValueError("missing_grace_frames cannot be negative")

        self.minimum_score = minimum_score
        self.acquisition_frames = acquisition_frames
        self.switch_frames = switch_frames
        self.switch_margin = switch_margin
        self.missing_grace_frames = missing_grace_frames

        self.selected_side = "none"

        self._candidate_side = "none"
        self._candidate_count = 0
        self._missing_count = 0

    def _valid_score(self, score: Optional[float]) -> Optional[float]:
        if score is None:
            return None

        score = float(score)

        if score < self.minimum_score:
            return None

        return score

    def _best_side(
        self,
        left_score: Optional[float],
        right_score: Optional[float],
    ) -> str:
        if left_score is None and right_score is None:
            return "none"

        if left_score is None:
            return "right"

        if right_score is None:
            return "left"

        if right_score > left_score:
            return "right"

        return "left"

    def _update_candidate(self, side: str) -> None:
        if side == self._candidate_side:
            self._candidate_count += 1
        else:
            self._candidate_side = side
            self._candidate_count = 1

    def _clear_candidate(self) -> None:
        self._candidate_side = "none"
        self._candidate_count = 0

    def update(
        self,
        left_score: Optional[float],
        right_score: Optional[float],
    ) -> str:
        """Update selection from the current frame's visibility evidence.

        Scores below the minimum are treated as unavailable. The returned value
        is ``"left"``, ``"right"`` or ``"none"`` and is retained across calls.
        """
        left_score = self._valid_score(left_score)
        right_score = self._valid_score(right_score)

        scores = {
            "left": left_score,
            "right": right_score,
        }

        #no side has been acquired yet
        if self.selected_side == "none":
            best_side = self._best_side(left_score, right_score)

            if best_side == "none":
                self._clear_candidate()
                return self.selected_side

            self._update_candidate(best_side)

            if self._candidate_count >= self.acquisition_frames:
                self.selected_side = best_side
                self._missing_count = 0
                self._clear_candidate()

            return self.selected_side

        current_side = self.selected_side
        other_side = "right" if current_side == "left" else "left"

        current_score = scores[current_side]
        other_score = scores[other_side]

        #the current side remains usable
        if current_score is not None:
            self._missing_count = 0

            #switch only if the other side is clearly stronger
            if (
                other_score is not None
                and other_score >= current_score + self.switch_margin
            ):
                self._update_candidate(other_side)

                if self._candidate_count >= self.switch_frames:
                    self.selected_side = other_side
                    self._clear_candidate()
            else:
                self._clear_candidate()

            return self.selected_side

        #the current side is temporarily missing
        self._missing_count += 1

        if other_score is not None:
            self._update_candidate(other_side)

            if self._candidate_count >= self.switch_frames:
                self.selected_side = other_side
                self._missing_count = 0
                self._clear_candidate()

            return self.selected_side

        self._clear_candidate()

        #keep the identity briefly during missing-landmark frames
        if self._missing_count > self.missing_grace_frames:
            self.selected_side = "none"
            self._missing_count = 0

        return self.selected_side

    def reset(self) -> None:
        """Return to the unacquired state and clear candidate history."""
        self.selected_side = "none"
        self._candidate_side = "none"
        self._candidate_count = 0
        self._missing_count = 0
