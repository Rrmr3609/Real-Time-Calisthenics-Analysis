from analysis.repetition_counter import BasicRepetitionCounter
from analysis.form_rules import baseline_form_warnings


class BaselinePushUpAnalyser:
    """
    Baseline push-up analyser.

    Uses raw frame-level angle thresholds only.
    No smoothing, no hysteresis and no enhanced temporal logic.
    """

    def __init__(
        self,
        top_elbow_angle: float = 150.0,
        bottom_elbow_angle: float = 100.0,
        top_extension_warning_threshold: float = 150.0,
        depth_warning_threshold: float = 100.0,
        alignment_warning_minimum: float = 160.0,
    ):
        self.counter = BasicRepetitionCounter(
            top_elbow_angle=top_elbow_angle,
            bottom_elbow_angle=bottom_elbow_angle,
        )
        self.top_extension_warning_threshold = float(
            top_extension_warning_threshold
        )
        self.depth_warning_threshold = float(
            depth_warning_threshold
        )
        self.alignment_warning_minimum = float(
            alignment_warning_minimum
        )

    def update(self, elbow_angle, body_alignment_angle):
        rep_count, position = self.counter.update(elbow_angle)

        warnings = baseline_form_warnings(
            elbow_angle=elbow_angle,
            body_alignment_angle=body_alignment_angle,
            position=position,
            top_extension_threshold=(
                self.top_extension_warning_threshold
            ),
            depth_threshold=self.depth_warning_threshold,
            body_alignment_minimum=(
                self.alignment_warning_minimum
            ),
        )

        return {
            "rep_count": rep_count,
            "position": position,
            "warnings": warnings,
        }
