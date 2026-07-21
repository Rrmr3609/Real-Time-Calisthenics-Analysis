from pose.landmarks import (
    feature_landmarks_available,
    select_best_elbow_side,
)


def make_landmarks(
    left_arm=0.9,
    right_arm=0.8,
    left_hip=0.9,
    left_ankle=0.9,
    right_hip=0.8,
    right_ankle=0.8,
):
    def point(visibility):
        return {
            "x": 100.0,
            "y": 100.0,
            "visibility": visibility,
        }

    return {
        "left_shoulder": point(left_arm),
        "left_elbow": point(left_arm),
        "left_wrist": point(left_arm),
        "left_hip": point(left_hip),
        "left_ankle": point(left_ankle),
        "right_shoulder": point(right_arm),
        "right_elbow": point(right_arm),
        "right_wrist": point(right_arm),
        "right_hip": point(right_hip),
        "right_ankle": point(right_ankle),
    }


def test_selects_side_with_stronger_arm_visibility():
    landmarks = make_landmarks(left_arm=0.7, right_arm=0.9)

    selected_side = select_best_elbow_side(
        landmarks,
        minimum_visibility=0.5,
    )

    assert selected_side == "right"


def test_returns_none_when_neither_arm_is_visible():
    landmarks = make_landmarks(left_arm=0.3, right_arm=0.4)

    selected_side = select_best_elbow_side(
        landmarks,
        minimum_visibility=0.5,
    )

    assert selected_side == "none"


def test_elbow_feature_remains_available_when_ankle_is_hidden():
    landmarks = make_landmarks(
        left_arm=0.9,
        left_ankle=0.2,
    )

    assert feature_landmarks_available(
        landmarks,
        side="left",
        feature="elbow",
        minimum_visibility=0.5,
    )


def test_alignment_feature_is_unavailable_when_ankle_is_hidden():
    landmarks = make_landmarks(
        left_arm=0.9,
        left_ankle=0.2,
    )

    assert not feature_landmarks_available(
        landmarks,
        side="left",
        feature="alignment",
        minimum_visibility=0.5,
    )