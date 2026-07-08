from analysis.repetition_counter import BasicRepetitionCounter


def test_top_bottom_top_counts_one_repetition():
    counter = BasicRepetitionCounter(top_elbow_angle=150.0, bottom_elbow_angle=100.0)

    counter.update(160.0)  # top
    counter.update(90.0)   # bottom
    rep_count, position = counter.update(160.0)  # top again

    assert rep_count == 1
    assert position == "top"


def test_shallow_movement_does_not_count_repetition():
    counter = BasicRepetitionCounter(top_elbow_angle=150.0, bottom_elbow_angle=100.0)

    counter.update(160.0)  # top
    counter.update(130.0)  # not bottom
    rep_count, position = counter.update(160.0)  # top again

    assert rep_count == 0
    assert position == "top"


def test_none_angle_does_not_crash_or_change_count():
    counter = BasicRepetitionCounter(top_elbow_angle=150.0, bottom_elbow_angle=100.0)

    rep_count, position = counter.update(None)

    assert rep_count == 0
    assert position == "unknown"


def test_counter_does_not_duplicate_count_while_staying_top():
    counter = BasicRepetitionCounter(top_elbow_angle=150.0, bottom_elbow_angle=100.0)

    counter.update(160.0)
    counter.update(90.0)
    counter.update(160.0)
    rep_count, position = counter.update(165.0)

    assert rep_count == 1
    assert position == "top"