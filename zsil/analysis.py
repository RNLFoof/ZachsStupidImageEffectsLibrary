"""
    Functions that provide information about an image without actually affecting anything.
"""
import math
import typing
from typing import Callable, Optional, Generator, Sequence, Self

from PIL import Image, PyAccess

from zsil.internal import get_distances_to_points, PotentialLine


class Vector(tuple[float, float]):
    def __new__(cls, *args):
        if len(args) == 1:
            return Vector(*args[0])
        elif len(args) == 2:
            for arg in args:
                if not isinstance(arg, float) and not isinstance(arg, int):
                    raise Exception(f"WHAT ARE YOU FEEDING ME (got {arg}, which isn't an int or float)")
            return super(Vector, cls).__new__(cls, args)
        raise Exception("WHAT")

    def __add__(self, other: Self) -> Self:
        return Vector(map(sum, zip(self, other)))

    def __mul__(self, other: float) -> Self:
        return Vector(x * other for x in self)

    def __truediv__(self, other: float) -> Self:
        return Vector(x / other for x in self)

    def __floordiv__(self, other: float) -> Self:
        return Vector(x // other for x in self)

    def __round__(self) -> Self:
        return Vector(round(x) for x in self)

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

Path = Sequence[Vector]


def get_all_opaque_pixels(image: Image.Image) -> set[tuple[int, int]]:
    """Returns a set of all pixels whose alpha is larger than zero.

    Parameters
    ----------
    image
        The image to analyze.

    Returns
    -------
        All pixels whose alpha is larger than zero."""
    alpha_band = image.split()[image.getbands().index("A")]
    alpha_band_loaded = alpha_band.load()
    points = set()
    for x in range(image.width):
        for y in range(image.height):
            if alpha_band_loaded[(x, y)] > 0:
                points.add(Vector(x, y))
    return points


def get_all_transparent_pixels(image: Image.Image) -> set[tuple[int, int]]:
    """Returns a set of all pixels whose alpha is zero.

    Parameters
    ----------
    image
        The image to analyze.

    Returns
    -------
        All pixels whose alpha is zero."""
    alpha_band = image.split()[image.getbands().index("A")]
    alpha_band_loaded = alpha_band.load()
    points = set()
    for x in range(image.width):
        for y in range(image.height):
            if alpha_band_loaded[(x, y)] == 0:
                points.add((x, y))
    return points


def get_edge_pixels(image: Image.Image) -> set[tuple[int, int]]:
    """Returns a set of all opaque pixels right next to transparent ones.

    Parameters
    ----------
    image
        The image to analyze.

    Returns
    -------
        All opaque pixels right next to transparent ones."""
    from zsil.cool_stuff import inner_outline
    inner_outline_image = inner_outline(image, 1, (255, 0, 0), return_only=True)
    return get_all_opaque_pixels(inner_outline_image)


def get_distances_to_edges(image: Image.Image) -> list[PotentialLine]:
    """Returns a list of objects indicating the closest transparent pixel to each non-transparent pixel.

    Parameters
    ----------
    image
        The image to analyze.

    Returns
    -------
        List of objects indicating the closest transparent pixel to each non-transparent pixel."""
    # Get possible end_points
    end_points = get_edge_pixels(image)

    # Get possible starting point_count
    start_points = get_all_opaque_pixels(image)

    return get_distances_to_points(start_points, end_points)


def get_edge_points(image: Image.Image) -> set[Vector]:
    """Returns all integer points on a border between a transparent pixel and an opaque one.

    Parameters
    ----------
    image
        The image to analyze.

    Returns
    -------
        All integer points on a border between a transparent pixel and an opaque one."""
    from zsil.cool_stuff import round_alpha
    image = round_alpha(image)
    edge_pixels = get_edge_pixels(image)
    transparent_pixels = get_all_transparent_pixels(image)
    edge_points = set()
    for edge_pixel in edge_pixels:
        for x_corner in [-1, 1]:
            for y_corner in [-1, 1]:
                addable_offset = (0.5 + x_corner / 2, 0.5 + y_corner / 2)
                addable = Vector([int(ep + ao) for ep, ao in zip(edge_pixel, addable_offset)])
                addable = typing.cast(tuple[int, int], addable)

                # No need to waste processing time if this one is already in there
                if addable in edge_points:
                    continue

                # Is it next to the infinite transparent void outside the image?
                if 0 >= addable[0] or image.width <= addable[0] or 0 >= addable[1] or image.height <= addable[1]:
                    edge_points.add(addable)
                    continue

                # The main man
                for checking_offset in [
                    (x_corner, 0),
                    (x_corner, y_corner),
                    (0, y_corner),
                ]:
                    if tuple([ep + co for ep, co in zip(edge_pixel, checking_offset)]) in transparent_pixels:
                        edge_points.add(addable)
                        break
    return edge_points


def pixel_filter(
        function: Callable[[tuple[int, int]], bool],
        image: Image.Image,
        image_data: Optional[PyAccess.PyAccess] = None
) -> Generator[tuple[int, int], None, None]:
    """Returns the coordinates of all pixels that match a function.

    Parameters
    ----------
    function
        A function that takes a tuple as input and returns a boolean. Determines what pixels to return.
    image
        How many pixels out the outline stretches.
    image_data
        The result of image.load(), in case you had it loaded already. Generated if not provided.

    Yields
    ------
        All coordinates for which the provided function returned True."""
    # Create image_data if it's not provided
    if image_data is None:
        image_data = image.load()

    # The actual filtering
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if function(image_data[x, y]):
                yield x, y
