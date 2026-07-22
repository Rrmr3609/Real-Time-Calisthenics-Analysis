from analysis.side_selector import StableSideSelector


def acquire_left(selector):
    for _ in range(selector.acquisition_frames):
        selected = selector.update(
            left_score=0.9,
            right_score=0.6,
        )

    return selected


def test_side_is_acquired_after_required_frames():
    selector = StableSideSelector(
        acquisition_frames=3,
    )

    assert selector.update(0.9, 0.6) == "none"
    assert selector.update(0.9, 0.6) == "none"
    assert selector.update(0.9, 0.6) == "left"


def test_one_stronger_opposite_frame_does_not_switch_side():
    selector = StableSideSelector(
        acquisition_frames=1,
        switch_frames=3,
        switch_margin=0.1,
    )

    assert acquire_left(selector) == "left"

    selected = selector.update(
        left_score=0.6,
        right_score=0.9,
    )

    assert selected == "left"


def test_sustained_stronger_opposite_side_causes_switch():
    selector = StableSideSelector(
        acquisition_frames=1,
        switch_frames=3,
        switch_margin=0.1,
    )

    assert acquire_left(selector) == "left"

    assert selector.update(0.6, 0.9) == "left"
    assert selector.update(0.6, 0.9) == "left"
    assert selector.update(0.6, 0.9) == "right"


def test_brief_missing_scores_keep_selected_side():
    selector = StableSideSelector(
        acquisition_frames=1,
        missing_grace_frames=2,
    )

    assert acquire_left(selector) == "left"

    assert selector.update(None, None) == "left"
    assert selector.update(None, None) == "left"


def test_prolonged_missing_scores_release_selected_side():
    selector = StableSideSelector(
        acquisition_frames=1,
        missing_grace_frames=2,
    )

    assert acquire_left(selector) == "left"

    selector.update(None, None)
    selector.update(None, None)
    selected = selector.update(None, None)

    assert selected == "none"