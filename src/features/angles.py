"""Calculate geometric joint angles from two-dimensional points."""

import math
from typing import Optional, Tuple

Point2D = Tuple[float, float]


def calculate_angle(
    a: Point2D,
    b: Point2D,
    c: Point2D,
) -> Optional[float]:
    """
    Return the angle ABC in degrees, with ``b`` as the vertex.

    The result lies between 0 and 180 degrees. ``None`` is returned when a
    point is unavailable or either vector from the vertex has zero length.
    """
    if a is None or b is None or c is None:
        return None

    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    norm_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    norm_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))
