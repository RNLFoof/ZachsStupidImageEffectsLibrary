import math

def lengthdir_x(length,dir):
    return round(math.cos(math.radians(dir))*length, 10)


def lengthdir_y(length,dir):
    return round(math.cos(math.radians(dir-90))*length, 10)


def point_distance(x1, y1, x2, y2):
    """Get the distance between two points."""
    return ((abs(x2 - x1)**2) + (abs(y2 - y1)**2))**0.5


def point_direction(x1, y1, x2, y2):
    """Get the direction between two points."""
    return -(math.atan2(x2 - x1, y2 - y1) / math.pi * 180 - 90) % 360


def getdistancestopoints(startpoints, endpoints):
    """Returns a list of objects indicating the closest endpoint to each startpoint.

    Parameters:
    img (PIL.Image): The image to analyze.

    Returns:
    list: List of objects indicating the closest transparent pixel to each non-transparent pixel."""
    class PotentialLine:
        def __init__(self, startpoint):
            self.startpoint = startpoint
            # Get closest endpoint
            # Slow af. could probably be sped way up by iterating through
            # increasingly distant points and checking if they're there, instead of checking every point to every other
            # point.
            self.dis = float("inf")
            for potentialendpoint in endpoints:
                dis = point_distance(self.startpoint[0], self.startpoint[1], potentialendpoint[0], potentialendpoint[1])
                if dis < self.dis:
                    self.dis = dis
                    self.endpoint = potentialendpoint
            # Get dir, obviously
            self.dir = point_direction(self.startpoint[0], self.startpoint[1], self.endpoint[0], self.endpoint[1])

    # Get all potential lines
    startpoints -= endpoints
    potentiallines = []
    for startpoint in startpoints:
        potentiallines.append(PotentialLine(startpoint))

    return potentiallines