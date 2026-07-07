from features.angles import calculate_angle


def test_calculate_90_degree_angle():
    angle = calculate_angle((1, 0), (0, 0), (0, 1))
    assert round(angle, 2) == 90.00


def test_calculate_180_degree_angle():
    angle = calculate_angle((-1, 0), (0, 0), (1, 0))
    assert round(angle, 2) == 180.00


def test_returns_none_for_missing_point():
    angle = calculate_angle(None, (0, 0), (1, 0))
    assert angle is None


def test_returns_none_for_zero_length_vector():
    angle = calculate_angle((0, 0), (0, 0), (1, 0))
    assert angle is None