"""Apply deterministic form rules to enhanced completed repetitions."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from analysis.repetition_result import CompletedRepetition


class RepetitionClass(str, Enum):
    """Single-label outcomes supported by enhanced evaluation."""

    CORRECT = "correct"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    INCOMPLETE_EXTENSION = "incomplete_extension"
    ALIGNMENT_DEVIATION = "alignment_deviation"
    UNSCORABLE = "unscorable"


@dataclass(frozen=True)
class RepetitionClassification:
    """Immutable classification result with rule and evidence diagnostics.

    Angle fields are in degrees. Alignment validity describes evidence
    availability over the repetition window; it is distinct from whether the
    available evidence indicates an alignment deviation.
    """

    rep_id: int
    predicted_class: str

    insufficient_depth_triggered: bool
    incomplete_extension_triggered: bool
    alignment_deviation_triggered: bool

    triggered_rules: Tuple[str, ...]
    multiple_rules_triggered: bool

    top_extension_angle: float
    minimum_elbow_angle: float

    minimum_alignment_angle: Optional[float]
    alignment_valid_frames: int
    alignment_valid_ratio: float
    alignment_deviation_frames: int
    alignment_deviation_ratio: float

    classification_reason: str

    def to_dict(self) -> Dict[str, object]:
        """Return a dictionary suitable for repetition-level logging."""
        return asdict(self)


class RepetitionClassifier:
    """
    Classify one completed repetition using transparent operational
    thresholds.

    These are project-defined categories for the controlled evaluation.
    They are not universal or clinical definitions of push-up form.

    Formal repetition classes apply to enhanced completed repetitions; the
    baseline exposes only diagnostic frame warnings. When several rules apply,
    the single label follows the fixed priority: insufficient depth, incomplete
    extension, inadequate alignment evidence (unscorable), alignment deviation
    and then correct. Every triggered deviation rule remains available in the
    result even when a higher-priority label is selected.
    """

    def __init__(
        self,
        depth_threshold: float = 100.0,
        extension_threshold: float = 150.0,
        alignment_minimum: float = 160.0,
        alignment_deviation_min_frames: int = 3,
        alignment_deviation_min_ratio: float = 0.20,
        minimum_alignment_valid_ratio: float = 0.50,
    ):
        if alignment_deviation_min_frames < 1:
            raise ValueError("alignment_deviation_min_frames must be at least 1")

        if not 0.0 <= alignment_deviation_min_ratio <= 1.0:
            raise ValueError("alignment_deviation_min_ratio must be between 0 and 1")

        if not 0.0 <= minimum_alignment_valid_ratio <= 1.0:
            raise ValueError("minimum_alignment_valid_ratio must be between 0 and 1")

        self.depth_threshold = float(depth_threshold)
        self.extension_threshold = float(extension_threshold)
        self.alignment_minimum = float(alignment_minimum)

        self.alignment_deviation_min_frames = alignment_deviation_min_frames
        self.alignment_deviation_min_ratio = alignment_deviation_min_ratio
        self.minimum_alignment_valid_ratio = minimum_alignment_valid_ratio

    def classify(
        self,
        repetition: CompletedRepetition,
    ) -> RepetitionClassification:
        """Classify one completed inclusive repetition deterministically.

        Alignment coverage uses all frames in ``duration_frames`` as its
        denominator and only valid alignment observations as its numerator.
        Insufficient coverage makes alignment evidence unscorable, although a
        higher-priority elbow rule may still determine the final label.
        """
        # Extension quality is defined by the returned top posture after the
        # repetition's bottom, not by the posture before descent began.
        top_extension_angle = repetition.end_top_angle

        insufficient_depth = repetition.minimum_elbow_angle > self.depth_threshold

        incomplete_extension = top_extension_angle < self.extension_threshold

        alignment_values = list(repetition.alignment_angles)

        alignment_valid_frames = len(alignment_values)

        alignment_valid_ratio = (
            alignment_valid_frames / repetition.duration_frames
            if repetition.duration_frames > 0
            else 0.0
        )

        minimum_alignment_angle = min(alignment_values) if alignment_values else None

        alignment_deviation_frames = sum(
            value < self.alignment_minimum for value in alignment_values
        )

        alignment_deviation_ratio = (
            alignment_deviation_frames / alignment_valid_frames
            if alignment_valid_frames > 0
            else 0.0
        )

        alignment_scorable = alignment_valid_ratio >= self.minimum_alignment_valid_ratio

        alignment_deviation = (
            alignment_scorable
            and alignment_deviation_frames >= self.alignment_deviation_min_frames
            and alignment_deviation_ratio >= self.alignment_deviation_min_ratio
        )

        triggered_rules = []

        if insufficient_depth:
            triggered_rules.append(RepetitionClass.INSUFFICIENT_DEPTH.value)

        if incomplete_extension:
            triggered_rules.append(RepetitionClass.INCOMPLETE_EXTENSION.value)

        if alignment_deviation:
            triggered_rules.append(RepetitionClass.ALIGNMENT_DEVIATION.value)

        multiple_rules_triggered = len(triggered_rules) > 1

        # The evaluation uses one intended class per repetition.
        # This deterministic priority is used only when several rules
        # trigger simultaneously. All triggers remain in the log.
        if insufficient_depth:
            predicted_class = RepetitionClass.INSUFFICIENT_DEPTH.value
            reason = "Minimum elbow angle remained above the depth threshold."

        elif incomplete_extension:
            predicted_class = RepetitionClass.INCOMPLETE_EXTENSION.value
            reason = "Return-to-top extension remained below the extension threshold."

        elif not alignment_scorable:
            predicted_class = RepetitionClass.UNSCORABLE.value
            reason = (
                "Insufficient valid body-alignment observations "
                "for a correct or alignment-deviation decision."
            )

        elif alignment_deviation:
            predicted_class = RepetitionClass.ALIGNMENT_DEVIATION.value
            reason = (
                "Body alignment remained below the operational "
                "threshold for enough valid frames."
            )

        else:
            predicted_class = RepetitionClass.CORRECT.value
            reason = "None of the three predefined deviation rules was triggered."

        if multiple_rules_triggered:
            reason += (
                " Multiple rules triggered; the documented "
                "single-label priority was applied."
            )

        return RepetitionClassification(
            rep_id=repetition.rep_id,
            predicted_class=predicted_class,
            insufficient_depth_triggered=(insufficient_depth),
            incomplete_extension_triggered=(incomplete_extension),
            alignment_deviation_triggered=(alignment_deviation),
            triggered_rules=tuple(triggered_rules),
            multiple_rules_triggered=(multiple_rules_triggered),
            top_extension_angle=top_extension_angle,
            minimum_elbow_angle=(repetition.minimum_elbow_angle),
            minimum_alignment_angle=(minimum_alignment_angle),
            alignment_valid_frames=(alignment_valid_frames),
            alignment_valid_ratio=(alignment_valid_ratio),
            alignment_deviation_frames=(alignment_deviation_frames),
            alignment_deviation_ratio=(alignment_deviation_ratio),
            classification_reason=reason,
        )
