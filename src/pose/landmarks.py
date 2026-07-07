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


def extract_landmarks(results, image_width: int, image_height: int) -> Dict[str, dict]:
    """
    Extract selected MediaPipe landmarks into pixel coordinates.

    Returns a dictionary containing x, y and visibility values.
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
    if name not in landmarks:
        return None

    return landmarks[name]["x"], landmarks[name]["y"]


def get_visibility(landmarks: Dict[str, dict], name: str) -> Optional[float]:
    if name not in landmarks:
        return None

    return landmarks[name]["visibility"]


def side_landmarks_available(
    landmarks: Dict[str, dict],
    side: str,
    minimum_visibility: float = 0.5,
) -> bool:
    """
    Check whether the selected side has enough visible landmarks.

    side must be 'left' or 'right'.
    """
    required = [
        f"{side}_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
        f"{side}_hip",
        f"{side}_ankle",
    ]

    for name in required:
        visibility = get_visibility(landmarks, name)

        if visibility is None or visibility < minimum_visibility:
            return False

    return True