"""Generate diagnostic frame-level warnings for the simple baseline."""


def baseline_form_warnings(
    elbow_angle,
    body_alignment_angle,
    position,
    top_extension_threshold: float = 150.0,
    depth_threshold: float = 100.0,
    body_alignment_minimum: float = 160.0,
):
    """
    Return direct frame-level warnings from raw angles in degrees.

    These are provisional operational thresholds for development.
    They are not universal definitions of correct push-up form.
    The messages are diagnostics and must not be interpreted as formal
    repetition-level classifications.
    """
    warnings = []

    if elbow_angle is None:
        warnings.append("Elbow not visible")
        return warnings

    if position == "top" and elbow_angle < top_extension_threshold:
        warnings.append("Incomplete elbow extension")

    if position == "bottom" and elbow_angle > depth_threshold:
        warnings.append("Insufficient depth")

    if (
        body_alignment_angle is not None
        and body_alignment_angle < body_alignment_minimum
    ):
        warnings.append("Body alignment deviation")

    return warnings
