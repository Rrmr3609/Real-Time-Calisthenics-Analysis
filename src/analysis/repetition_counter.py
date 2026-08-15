"""Count baseline repetitions using raw elbow-angle threshold crossings."""


class BasicRepetitionCounter:
    """
    Simple baseline repetition counter.

    Counts one repetition when the movement goes top -> bottom -> top.

    Positions are sticky: angles between the top and bottom thresholds retain
    the previous position. This is intentionally simple and does not use
    smoothing, hysteresis, missing-frame tolerance or temporal confirmation.
    """

    def __init__(
        self,
        top_elbow_angle: float = 150.0,
        bottom_elbow_angle: float = 100.0,
    ):
        self.top_elbow_angle = top_elbow_angle
        self.bottom_elbow_angle = bottom_elbow_angle

        self.position = "unknown"
        self.has_reached_bottom = False
        self.rep_count = 0

    def update(self, elbow_angle):
        """Update the sticky position from one raw angle in degrees.

        ``None`` leaves all state unchanged. The cumulative count increases
        only after an observed top-to-bottom transition returns to top.
        """
        if elbow_angle is None:
            return self.rep_count, self.position

        previous_position = self.position

        if elbow_angle >= self.top_elbow_angle:
            self.position = "top"
        elif elbow_angle <= self.bottom_elbow_angle:
            self.position = "bottom"

        if previous_position == "top" and self.position == "bottom":
            self.has_reached_bottom = True

        if (
            self.has_reached_bottom
            and previous_position == "bottom"
            and self.position == "top"
        ):
            self.rep_count += 1
            self.has_reached_bottom = False

        return self.rep_count, self.position
