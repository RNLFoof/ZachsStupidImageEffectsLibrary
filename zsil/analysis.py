from PIL import ImageDraw
from PIL.Image import Image

from zsil.internal import getdistancestopoints


def get_all_opaque_pixels(img):
    """Returns a set of all pixels whose alpha is larger than zero."""
    alpha_band = img.split()[img.getbands().index("A")]
    alpha_band_loaded = alpha_band.load()
    points = set()
    for x in range(img.width):
        for y in range(img.height):
            if alpha_band_loaded[(x, y)] > 0:
                points.add((x, y))
    return points


def get_all_transparent_pixels(img):
    """Returns a set of all pixels whose alpha is zero."""
    alpha_band = img.split()[img.getbands().index("A")]
    alpha_band_loaded = alpha_band.load()
    points = set()
    for x in range(img.width):
        for y in range(img.height):
            if alpha_band_loaded[(x, y)] == 0:
                points.add((x, y))
    return points


def getedgepixels(img):
    """Returns a set of all opaque pixels right next to transparent ones.

    Parameters:
    img (PIL.Image): The image to analyze.

    Returns:
    set: All opaque pixels right next to transparent ones."""
    from zsil.coolstuff import inneroutline
    inneroutlineimg = inneroutline(img, 1, (255, 0, 0), retonly=True)
    return get_all_opaque_pixels(inneroutlineimg)


def getcenterpixels(img):
    """Returns a set of pixels that form a line along the "center" of opaque sections. Calculated like this:
    - Get edge distance for all opaque pixels
    - All pixels that equal the largest distance are selected
    - The next set of equal distance largest distance pixels are selected
    -

    Parameters:
    img (PIL.Image): The image to analyze.

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

                    left = left or spx == px-1
                    right = right or spx == px+1
                    if left and right:
                        break

                    up = up or spy == py-1
                    down = down or spy == py+1
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

                    left = left or spx == px-1
                    right = right or spx == px+1

                    up = up or spy == py-1
                    down = down or spy == py+1


                    # count += 1
                    # if count >= 3:
                    #     break
            else:
                yield pixel

    def getsurroundingpixels(pixel):
        x,y = pixel
        for xplus in range(-1, 2):
            for yplus in range(-1, 2):
                check = (x+xplus, y+yplus)
                if check != pixel:
                    yield check

    distancestoedges = getdistancestoedges(img)
    distancestodteobjects = {}
    draw = ImageDraw.Draw(img)
    for distancetoedge in distancestoedges:
        distancestodteobjects.setdefault(distancetoedge.dis, [])
        distancestodteobjects[distancetoedge.dis].append(distancetoedge)
        draw.point(distancetoedge.startpoint, (255, 255, 0))
    img.show()

    selected = set()
    avoid = set()
    for n, distance in enumerate(sorted(list(distancestodteobjects.keys()))[::-1]):
        distancestoedges = distancestodteobjects[distance]
        pendingselected = set()
        pendingavoid = set()
        for distancetoedges in distancestoedges:
            x, y = distancetoedges.startpoint
            touching = 0
            for check in getsurroundingpixels(distancetoedges.startpoint):
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
            if any(sp in lineendpixels for sp in getsurroundingpixels(distancetoedges.startpoint)):
                extras.add(distancetoedges.startpoint)

        selected |= extras
        for pixel in selected:
            draw.point(pixel, (0, 0, 255))
        for pixel in extras:
            draw.point(pixel, (255, 0, 0))
        for pixel in lineendpixels:
            draw.point(pixel, (0, 255, 0))
        if n % 10 == 0:
            img.show()
    img.save("ffWACK.png")
    return selected


def getdistancestoedges(img):
    """Returns a list of objects indicating the closest transparent pixel to each non-transparent pixel.

    Parameters:
    img (PIL.Image): The image to analyze.

    Returns:
    list: List of objects indicating the closest transparent pixel to each non-transparent pixel."""
    # Get possible endpoints
    endpoints = getedgepixels(img)
    draw = ImageDraw.Draw(img)
    #
    # for pixel in endpoints:
    #     draw.point(pixel, (0, 0, 0))
    #
    # img.show()

    # Get possible starting points
    startpoints = get_all_opaque_pixels(img)

    return getdistancestopoints(startpoints, endpoints)


def get_edge_points(img: Image):
    from zsil.coolstuff import roundalpha
    img = roundalpha(img)
    edge_pixels = getedgepixels(img)
    transparent_pixels = get_all_transparent_pixels(img)
    edge_points = set()
    for edge_pixel in edge_pixels:
        for x_corner in [-1, 1]:
            for y_corner in [-1, 1]:
                addable_offset = (0.5+x_corner/2, 0.5+y_corner/2)
                addable = tuple([int(ep+ao) for ep, ao in zip(edge_pixel, addable_offset)])

                # No need to waste processing time if this one is already in there
                if addable in edge_points:
                    continue

                # Is it next to the infinite transparent void outside the image?
                if 0 >= addable[0] or img.width <= addable[0] or 0 >= addable[1] or img.height <= addable[1]:
                    edge_points.add(addable)
                    continue

                # The main man
                for checking_offset in [
                    (x_corner, 0),
                    (x_corner, y_corner),
                    (0, y_corner),
                ]:
                    if tuple([ep+co for ep, co in zip(edge_pixel, checking_offset)]) in transparent_pixels:
                        edge_points.add(addable)
                        break
    return edge_points
