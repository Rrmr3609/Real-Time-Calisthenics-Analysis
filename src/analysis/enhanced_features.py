"""Extract visibility aware, smoothed features for enhanced processing."""

from typing import Dict

from analysis.side_selector import StableSideSelector
from analysis.smoothing import ExponentialMovingAverage
from features.angles import calculate_angle
from pose.landmarks import (
    feature_landmarks_available,
    feature_visibility_score,
    get_point,
)


class EnhancedFeatureProcessor:
    """
    Extract confidence aware angles while retaining temporal feature state.

    A sticky selector uses elbow-landmark visibility to choose one body side.
    Elbow and alignment angles are smoothed independently, and both smoothers
    reset when the selected side changes so observations from different sides
    are never combined. Missing or insufficiently visible landmarks produce
    absent output measurements rather than stale smoothed values.

    Push-up phase detection and repetition counting are handled separately.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        minimum_visibility: float = 0.5,
        acquisition_frames: int = 3,
        switch_frames: int = 5,
        switch_margin: float = 0.10,
        missing_grace_frames: int = 5,
    ):
        self.minimum_visibility = minimum_visibility

        self.side_selector = StableSideSelector(
            minimum_score=minimum_visibility,
            acquisition_frames=acquisition_frames,
            switch_frames=switch_frames,
            switch_margin=switch_margin,
            missing_grace_frames=missing_grace_frames,
        )

        self.elbow_smoother = ExponentialMovingAverage(alpha=smoothing_alpha)
        self.alignment_smoother = ExponentialMovingAverage(alpha=smoothing_alpha)

        self.previous_selected_side = "none"

    def update(self, landmarks: Dict[str, dict]) -> dict:
        """Process one frame of extracted pose landmarks.

        Visibility scores retain MediaPipe style visibility units. Returned
        raw and smoothed angles are in degrees, or ``None`` when the selected
        side lacks the landmarks required for that feature.
        """
        left_elbow_score = feature_visibility_score(
            landmarks,
            side="left",
            feature="elbow",
        )

        right_elbow_score = feature_visibility_score(
            landmarks,
            side="right",
            feature="elbow",
        )

        left_alignment_score = feature_visibility_score(
            landmarks,
            side="left",
            feature="alignment",
        )

        right_alignment_score = feature_visibility_score(
            landmarks,
            side="right",
            feature="alignment",
        )

        selected_side = self.side_selector.update(
            left_score=left_elbow_score,
            right_score=right_elbow_score,
        )

        side_changed = selected_side != self.previous_selected_side

        #do not mix historical angles from different body sides
        if side_changed:
            self.elbow_smoother.reset()
            self.alignment_smoother.reset()

        self.previous_selected_side = selected_side

        raw_elbow_angle = None
        raw_alignment_angle = None

        elbow_feature_valid = False
        alignment_feature_valid = False
        opposite_alignment_feature_valid = False

        if selected_side != "none":
            opposite_side = "right" if selected_side == "left" else "left"

            opposite_alignment_feature_valid = feature_landmarks_available(
                landmarks,
                side=opposite_side,
                feature="alignment",
                minimum_visibility=self.minimum_visibility,
            )

            elbow_feature_valid = feature_landmarks_available(
                landmarks,
                side=selected_side,
                feature="elbow",
                minimum_visibility=self.minimum_visibility,
            )

            if elbow_feature_valid:
                shoulder = get_point(
                    landmarks,
                    f"{selected_side}_shoulder",
                )
                elbow = get_point(
                    landmarks,
                    f"{selected_side}_elbow",
                )
                wrist = get_point(
                    landmarks,
                    f"{selected_side}_wrist",
                )

                raw_elbow_angle = calculate_angle(
                    shoulder,
                    elbow,
                    wrist,
                )

            alignment_feature_valid = feature_landmarks_available(
                landmarks,
                side=selected_side,
                feature="alignment",
                minimum_visibility=self.minimum_visibility,
            )

            if alignment_feature_valid:
                shoulder = get_point(
                    landmarks,
                    f"{selected_side}_shoulder",
                )
                hip = get_point(
                    landmarks,
                    f"{selected_side}_hip",
                )
                ankle = get_point(
                    landmarks,
                    f"{selected_side}_ankle",
                )

                raw_alignment_angle = calculate_angle(
                    shoulder,
                    hip,
                    ankle,
                )

        #do not output stale smoothed values on missing feature frames
        smoothed_elbow_angle = None
        smoothed_alignment_angle = None

        if raw_elbow_angle is not None:
            smoothed_elbow_angle = self.elbow_smoother.update(raw_elbow_angle)

        if raw_alignment_angle is not None:
            smoothed_alignment_angle = self.alignment_smoother.update(
                raw_alignment_angle
            )

        return {
            "selected_side": selected_side,
            "selected_elbow_side": selected_side,
            "side_changed": side_changed,
            "left_elbow_visibility_score": left_elbow_score,
            "right_elbow_visibility_score": right_elbow_score,
            "left_alignment_visibility_score": (left_alignment_score),
            "right_alignment_visibility_score": (right_alignment_score),
            "elbow_feature_valid": elbow_feature_valid,
            "alignment_feature_valid": alignment_feature_valid,
            "opposite_alignment_feature_valid": (opposite_alignment_feature_valid),
            "raw_elbow_angle": raw_elbow_angle,
            "smoothed_elbow_angle": smoothed_elbow_angle,
            "raw_alignment_angle": raw_alignment_angle,
            "smoothed_alignment_angle": smoothed_alignment_angle,
        }

    def reset(self) -> None:
        """Discard the selected side and all retained smoothing state."""
        self.side_selector.reset()
        self.elbow_smoother.reset()
        self.alignment_smoother.reset()
        self.previous_selected_side = "none"
