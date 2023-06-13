import math


def lengthdir_x(length, direction):
    return round(math.cos(math.radians(direction)) * length, 10)


def lengthdir_y(length, direction):
    return round(math.cos(math.radians(direction - 90)) * length, 10)


def point_distance(x1, y1, x2, y2):
    """Get the distance between two point_count."""
    return ((abs(x2 - x1) ** 2) + (abs(y2 - y1) ** 2)) ** 0.5


def point_direction(x1, y1, x2, y2):
    """Get the direction between two point_count."""
    return -(math.atan2(x2 - x1, y2 - y1) / math.pi * 180 - 90) % 360


def get_distances_to_points(start_points, end_points):
    """Returns a list of objects indicating the closest end_point to each start_point.

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    list: List of objects indicating the closest transparent pixel to each non-transparent pixel."""

    class PotentialLine:
        def __init__(self, start_point):
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

    # Get all potential lines
    start_points -= end_points
    potentiallines = []
    for start_point in start_points:
        potentiallines.append(PotentialLine(start_point))

    return potentiallines
