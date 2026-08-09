"""Provide stateful exponential smoothing for enhanced measurements."""

from typing import Optional


class ExponentialMovingAverage:
    """
    Exponential moving average for a stream of numeric values.

    The first valid observation initialises the smoothed value.

    Missing observations do not modify the stored state or extrapolate a new
    value. Callers that must expose absence explicitly can skip ``update`` for
    such frames, as the enhanced feature processor does.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be greater than 0 and at most 1")

        self.alpha = float(alpha)
        self._value: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        """Return the current smoothed value without modifying state."""
        return self._value

    def update(self, observation: Optional[float]) -> Optional[float]:
        """
        Add one observation and return the current smoothed value.

        If ``observation`` is ``None``, the stored value is returned unchanged;
        no new value is inferred.
        """
        if observation is None:
            return self._value

        observation = float(observation)

        if self._value is None:
            self._value = observation
        else:
            self._value = (
                self.alpha * observation
                + (1.0 - self.alpha) * self._value
            )

        return self._value

    def reset(self) -> None:
        """Remove the previously stored smoothing state."""
        self._value = None
