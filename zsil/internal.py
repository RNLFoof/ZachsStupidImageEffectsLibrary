import math

from numpy import pi

tau = pi * 2

class PotentialLine:
    """

    """
    def __init__(self, start_point, end_points):
        self.start_point = start_point
        # Get the closest end_point
        # Slow af. could probably be sped way up by iterating through
        # increasingly distant point_count and checking if they're there, instead of checking every point to every other
        # point.
        self.dis = float("inf")
        for potentialend_point in end_points:
            dis = point_distance(self.start_point[0], self.start_point[1], potentialend_point[0],
                                 potentialend_point[1])
            if dis < self.dis:
                self.dis = dis
                self.end_point = potentialend_point
        # Get direction, obviously
        self.dir = point_direction(self.start_point[0], self.start_point[1], self.end_point[0], self.end_point[1])


def lengthdir_x(length: float, direction: float) -> float:
    """Returns the x component of a point that is a given distance in a given direction from the origin.

    Parameters
    ----------
    length
        The length or distance.
    direction
        The direction, in degrees.

    Returns
    -------
    float
        The x component of a point that is a given distance in a given direction from the origin.
    """
    return round(math.cos(math.radians(direction)) * length, 10)


def lengthdir_y(length: float, direction: float) -> float:
    """Returns the y component of a point that is a given distance in a given direction from the origin.

    Parameters
    ----------
    length : float
        The length or distance.
    direction : float
        The direction, in degrees.

    Returns
    -------
    float
        The y component of a point that is a given distance in a given direction from the origin.
    """
    return round(math.cos(math.radians(direction - 90)) * length, 10)


def point_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Returns the distance between two points.

    Parameters
    ----------
    x1
        The x component of the first point.
    y1
        The y component of the first point.
    x2
        The x component of the second point.
    y2
        The y component of the second point.

    Returns
    -------
        The distance between the two points.
    """
    return ((abs(x2 - x1) ** 2) + (abs(y2 - y1) ** 2)) ** 0.5


def point_direction(x1: float, y1: float, x2: float, y2: float) -> float:
    """Get the direction from the first point to the second point in radians between 0 and tau.

    Treats negatives as up, like they are in images, and not down like on graphs."""
    # % tau is necessary because atan2 returns between pi and -pi
    return math.atan2(-y2 - -y1, x2 - x1) % tau


def get_distances_to_points(start_points: set[tuple[float, float]], end_points: set[tuple[float, float]]) -> list[PotentialLine]:
    """Returns a list of objects indicating the closest end_point to each start_point."""

    # Get all potential lines
    start_points -= end_points
    potential_lines = []
    for start_point in start_points:
        potential_lines.append(PotentialLine(start_point))

    return potential_lines
