from PIL import ImageDraw
from PIL.Image import Image

from zsil.internal import get_distances_to_points


def get_all_opaque_pixels(image: Image) -> set[tuple[int, int]]:
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


def get_all_transparent_pixels(image: Image) -> set[tuple[int, int]]:
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


def get_edge_pixels(image: Image):
    """Returns a set of all opaque pixels right next to transparent ones.

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    set: All opaque pixels right next to transparent ones."""
    from zsil.cool_stuff import inner_outline
    inner_outline_image = inner_outline(image, 1, (255, 0, 0), return_only=True)
    return get_all_opaque_pixels(inner_outline_image)


def get_center_pixels(image: Image):
    """Returns a set of pixels that form a line along the "center" of opaque sections. Calculated like this:
    - Get edge distance for all opaque pixels
    - All pixels that equal the largest distance are selected
    - The next set of equal distance largest distance pixels are selected
    -

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    set: All opaque pixels right next to transparent ones."""

    # Gets pixels only touching one other pixel
    # def getlineendpixels(pixels):
    #     for pixel in pixels:
    #         count = 0
    #         for surroundingpixel in getsurroundingpixels(pixel):
    #             if surroundingpixel in pixels:
    #                 count += 1
    #                 if count >= 2:
    #                     break
    #         else:
    #             yield pixel

    def getlineendpixels(pixels):
        for pixel in pixels:
            left = False
            right = False
            up = False
            down = False
            count = 0

            for surroundingpixel in getsurroundingpixels(pixel):
                if surroundingpixel in pixels:
                    px, py = pixel
                    spx, spy = surroundingpixel

                    left = left or spx == px - 1
                    right = right or spx == px + 1
                    if left and right:
                        break

                    up = up or spy == py - 1
                    down = down or spy == py + 1
                    if up and down:
                        break

                    count += 1
                    if count >= 5:
                        break
            else:
                yield pixel

    def getnearlykissingpixels(pixels):
        for pixel in pixels:
            left = False
            right = False
            up = False
            down = False
            count = 0

            for surroundingpixel in getsurroundingpixels(pixel):
                if surroundingpixel in pixels:
                    px, py = pixel
                    spx, spy = surroundingpixel

                    left = left or spx == px - 1
                    right = right or spx == px + 1

                    up = up or spy == py - 1
                    down = down or spy == py + 1

                    # count += 1
                    # if count >= 3:
                    #     break
            else:
                yield pixel

    def getsurroundingpixels(pixel):
        x, y = pixel
        for xplus in range(-1, 2):
            for yplus in range(-1, 2):
                check = (x + xplus, y + yplus)
                if check != pixel:
                    yield check

    distancestoedges = get_distances_to_edges(image)
    distancestodteobjects = {}
    draw = ImageDraw.Draw(image)
    for distancetoedge in distancestoedges:
        distancestodteobjects.setdefault(distancetoedge.dis, [])
        distancestodteobjects[distancetoedge.dis].append(distancetoedge)
        draw.point(distancetoedge.start_point, (255, 255, 0))
    image.show()

    selected = set()
    avoid = set()
    for n, distance in enumerate(sorted(list(distancestodteobjects.keys()))[::-1]):
        distancestoedges = distancestodteobjects[distance]
        pendingselected = set()
        pendingavoid = set()
        for distancetoedges in distancestoedges:
            x, y = distancetoedges.start_point
            touching = 0
            for check in getsurroundingpixels(distancetoedges.start_point):
                if check in selected or check in avoid:
                    touching += 1
            if touching >= 1:
                pendingavoid.add((x, y))
            else:
                pendingselected.add((x, y))

        selected |= pendingselected
        avoid |= pendingavoid

    for n, distance in enumerate(sorted(list(distancestodteobjects.keys()))[::-1]):
        distancestoedges = distancestodteobjects[distance]
        lineendpixels = list(getlineendpixels(selected))

        extras = set()
        for distancetoedges in distancestoedges:
            if any(sp in lineendpixels for sp in getsurroundingpixels(distancetoedges.start_point)):
                extras.add(distancetoedges.start_point)

        selected |= extras
        for pixel in selected:
            draw.point(pixel, (0, 0, 255))
        for pixel in extras:
            draw.point(pixel, (255, 0, 0))
        for pixel in lineendpixels:
            draw.point(pixel, (0, 255, 0))
        if n % 10 == 0:
            image.show()
    image.save("ffWACK.png")
    return selected


def get_distances_to_edges(image: Image):
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


def get_edge_points(image: Image):
    from zsil.cool_stuff import roundalpha
    image = roundalpha(image)
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
