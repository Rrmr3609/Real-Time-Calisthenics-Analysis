"""Extract pose coordinates and visibility evidence for required features."""

from typing import Dict, Optional, Tuple


Point2D = Tuple[float, float]


LANDMARK_NAMES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def extract_landmarks(
    results,
    image_width: int,
    image_height: int,
) -> Dict[str, dict]:
    """
    Extract selected MediaPipe landmarks into image-pixel coordinates.

    MediaPipe's normalized x/y values are scaled using the supplied image
    dimensions. Visibility values are forwarded unchanged as MediaPipe-style
    confidence evidence. An unavailable pose produces an empty dictionary.
    """
    if not results.pose_landmarks:
        return {}

    extracted = {}

    for name, index in LANDMARK_NAMES.items():
        landmark = results.pose_landmarks.landmark[index]
        extracted[name] = {
            "x": landmark.x * image_width,
            "y": landmark.y * image_height,
            "visibility": landmark.visibility,
        }

    return extracted


def get_point(landmarks: Dict[str, dict], name: str) -> Optional[Point2D]:
    """Return a landmark's image-pixel coordinates, or ``None`` if absent."""
    if name not in landmarks:
        return None

    return landmarks[name]["x"], landmarks[name]["y"]


def get_visibility(landmarks: Dict[str, dict], name: str) -> Optional[float]:
    """Return a landmark's visibility value, or ``None`` if absent."""
    if name not in landmarks:
        return None

    return landmarks[name]["visibility"]


FEATURE_LANDMARKS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "alignment": ("shoulder", "hip", "ankle"),
}


def feature_visibility_values(
    landmarks: Dict[str, dict],
    side: str,
    feature: str,
) -> Optional[list[float]]:
    """
    Return MediaPipe-style visibility values required for one feature.

    ``None`` indicates that a required landmark is unavailable. An unknown
    feature name raises ``ValueError`` rather than silently changing the
    feature definition.
    """
    if feature not in FEATURE_LANDMARKS:
        raise ValueError(f"Unknown feature: {feature}")

    values = []

    for body_part in FEATURE_LANDMARKS[feature]:
        name = f"{side}_{body_part}"
        visibility = get_visibility(landmarks, name)

        if visibility is None:
            return None

        values.append(float(visibility))

    return values


def feature_landmarks_available(
    landmarks: Dict[str, dict],
    side: str,
    feature: str,
    minimum_visibility: float = 0.5,
) -> bool:
    """Return whether every required landmark meets the visibility threshold."""
    values = feature_visibility_values(landmarks, side, feature)

    if values is None:
        return False

    return all(value >= minimum_visibility for value in values)


def feature_visibility_score(
    landmarks: Dict[str, dict],
    side: str,
    feature: str,
) -> Optional[float]:
    """
    Return the weakest required landmark visibility for a feature.

    The minimum is used because the feature is constrained by its
    least reliable required landmark.
    """
    values = feature_visibility_values(landmarks, side, feature)

    if not values:
        return None

    return min(values)


def select_best_elbow_side(
    landmarks: Dict[str, dict],
    minimum_visibility: float = 0.5,
) -> str:
    """
    Select the side with the stronger shoulder-elbow-wrist visibility.

    This function is deliberately stateless and selects independently for each
    call, as required by the simple baseline. The enhanced processor instead
    uses ``StableSideSelector`` for temporal stability. ``"none"`` is returned
    if neither side reaches the minimum threshold.
    """
    scores = {}

    for side in ("left", "right"):
        score = feature_visibility_score(
            landmarks=landmarks,
            side=side,
            feature="elbow",
        )

        if score is not None and score >= minimum_visibility:
            scores[side] = score

    if not scores:
        return "none"

    return max(scores, key=scores.get)
