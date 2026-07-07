from analysis.repetition_counter import BasicRepetitionCounter
from analysis.form_rules import baseline_form_warnings


class BaselinePushUpAnalyser:
    """
    Baseline push-up analyser.

    Uses raw frame-level angle thresholds only.
    No smoothing, no hysteresis and no enhanced temporal logic.
    """

    def __init__(self):
        self.counter = BasicRepetitionCounter(
            top_elbow_angle=150.0,
            bottom_elbow_angle=100.0,
        )

    def update(self, elbow_angle, body_alignment_angle):
        rep_count, position = self.counter.update(elbow_angle)

        warnings = baseline_form_warnings(
            elbow_angle=elbow_angle,
            body_alignment_angle=body_alignment_angle,
            position=position,
        )

        return {
            "rep_count": rep_count,
            "position": position,
            "warnings": warnings,
        }