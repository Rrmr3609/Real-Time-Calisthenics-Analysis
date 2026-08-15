import pytest

from analysis.smoothing import ExponentialMovingAverage


def test_first_observation_initialises_average():
    smoother = ExponentialMovingAverage(alpha=0.3)

    result = smoother.update(120.0)

    assert result == 120.0
    assert smoother.value == 120.0


def test_exponential_average_uses_previous_value():
    smoother = ExponentialMovingAverage(alpha=0.25)

    smoother.update(100.0)
    result = smoother.update(140.0)

    # 0.25 * 140 + 0.75 * 100 = 110
    assert result == pytest.approx(110.0)


def test_none_does_not_change_stored_value():
    smoother = ExponentialMovingAverage(alpha=0.3)

    smoother.update(130.0)
    result = smoother.update(None)

    assert result == 130.0
    assert smoother.value == 130.0


def test_reset_removes_stored_value():
    smoother = ExponentialMovingAverage(alpha=0.3)

    smoother.update(130.0)
    smoother.reset()

    assert smoother.value is None


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        ExponentialMovingAverage(alpha=0.0)
