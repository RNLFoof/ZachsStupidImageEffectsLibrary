from PIL import ImageDraw, Image

from zsil.internal import get_distances_to_points


def get_all_opaque_pixels(image: Image.Image) -> set[tuple[int, int]]:
    """Returns a set of all pixels whose alpha is larger than zero.

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    set[tuple[int, int]]: All pixels whose alpha is larger than zero."""
    alpha_band = image.split()[image.getbands().index("A")]
    alpha_band_loaded = alpha_band.load()
    points = set()
    for x in range(image.width):
        for y in range(image.height):
            if alpha_band_loaded[(x, y)] > 0:
                points.add((x, y))
    return points


def get_all_transparent_pixels(image: Image.Image) -> set[tuple[int, int]]:
    """Returns a set of all pixels whose alpha is zero.

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    set[tuple[int, int]]: All pixels whose alpha is zero."""
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

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    set: All opaque pixels right next to transparent ones."""
    from zsil.cool_stuff import inner_outline
    inner_outline_image = inner_outline(image, 1, (255, 0, 0), return_only=True)
    return get_all_opaque_pixels(inner_outline_image)


def get_distances_to_edges(image: Image.Image):
    """Returns a list of objects indicating the closest transparent pixel to each non-transparent pixel.

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    list: List of objects indicating the closest transparent pixel to each non-transparent pixel."""
    # Get possible end_points
    end_points = get_edge_pixels(image)
    draw = ImageDraw.Draw(image)
    #
    # for pixel in end_points:
    #     draw.point(pixel, (0, 0, 0))
    #
    # image.show()

    # Get possible starting point_count
    start_points = get_all_opaque_pixels(image)

    return get_distances_to_points(start_points, end_points)


def get_edge_points(image: Image.Image) -> set[tuple[int, int]]:
    """

    Parameters
    ----------
    image : Image.Image


    Returns
    -------
    set[tuple[int, int]]


    """
    from zsil.cool_stuff import round_alpha
    image = round_alpha(image)
    edge_pixels = get_edge_pixels(image)
    transparent_pixels = get_all_transparent_pixels(image)
    edge_points = set()
    for edge_pixel in edge_pixels:
        for x_corner in [-1, 1]:
            for y_corner in [-1, 1]:
                addable_offset = (0.5 + x_corner / 2, 0.5 + y_corner / 2)
                addable = tuple([int(ep + ao) for ep, ao in zip(edge_pixel, addable_offset)])

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


def pixel_filter(function, img, imgdata=None):
    """Returns the coordinates of all pixels that match a function.

    Parameters:
    function (function): A function that takes a tuple as input and returns a boolean. Determines what pixels to return.
    img (PIL.Image): How many pixels out the outline stretches.
    imgdata (PIL.PixelAccess): The result of img.load(), in case you had it loaded already. Generated if not provided.

    Yields:
    tuple: All coordinates to which function returned true."""
    # Create imgdata if it's not provided
    if imgdata is None:
        imgdata = img.load()

    # Gee I fuckin' wonder
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if function(imgdata[x, y]):
                yield x, y