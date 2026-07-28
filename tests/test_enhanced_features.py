from analysis.enhanced_features import EnhancedFeatureProcessor


def point(x, y, visibility):
    return {
        "x": float(x),
        "y": float(y),
        "visibility": float(visibility),
    }


def make_landmarks(
    left_arm=0.9,
    right_arm=0.8,
    left_hip=0.9,
    left_ankle=0.9,
    right_hip=0.8,
    right_ankle=0.8,
):
    return {
        "left_shoulder": point(0, 0, left_arm),
        "left_elbow": point(1, 0, left_arm),
        "left_wrist": point(2, 0, left_arm),
        "left_hip": point(0, 1, left_hip),
        "left_ankle": point(0, 2, left_ankle),
        "right_shoulder": point(0, 0, right_arm),
        "right_elbow": point(1, 0, right_arm),
        "right_wrist": point(2, 0, right_arm),
        "right_hip": point(0, 1, right_hip),
        "right_ankle": point(0, 2, right_ankle),
    }


def make_processor():
    return EnhancedFeatureProcessor(
        acquisition_frames=1,
        switch_frames=1,
    )


def test_logs_elbow_and_alignment_visibility_for_both_sides():
    processor = make_processor()

    result = processor.update(
        make_landmarks(
            left_arm=0.9,
            right_arm=0.8,
            left_hip=0.2,
            left_ankle=0.2,
            right_hip=0.9,
            right_ankle=0.9,
        )
    )

    assert result["left_elbow_visibility_score"] == 0.9
    assert result["right_elbow_visibility_score"] == 0.8
    assert result["left_alignment_visibility_score"] == 0.2
    assert result["right_alignment_visibility_score"] == 0.8


def test_reports_opposite_side_alignment_rescue_without_switching():
    processor = make_processor()

    result = processor.update(
        make_landmarks(
            left_arm=0.9,
            right_arm=0.8,
            left_hip=0.2,
            left_ankle=0.2,
            right_hip=0.9,
            right_ankle=0.9,
        )
    )

    assert result["selected_side"] == "left"
    assert result["selected_elbow_side"] == "left"
    assert not result["alignment_feature_valid"]
    assert result["opposite_alignment_feature_valid"]


def test_opposite_alignment_flag_is_false_when_opposite_is_invalid():
    processor = make_processor()

    result = processor.update(
        make_landmarks(
            left_arm=0.9,
            right_arm=0.8,
            left_hip=0.9,
            left_ankle=0.9,
            right_hip=0.2,
            right_ankle=0.2,
        )
    )

    assert result["selected_elbow_side"] == "left"
    assert result["alignment_feature_valid"]
    assert not result["opposite_alignment_feature_valid"]
