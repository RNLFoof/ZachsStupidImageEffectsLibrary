from ZachsStupidImageEffectsLibrary.internal import getdistancestopoints


def getallopaquepixels(img):
    """Returns a set of all pixels whose alpha is larger than zero."""
    alphaband = img.split()[img.getbands().index("A")]
    alphabandloaded = alphaband.load()
    points = set()
    for x in range(img.width):
        for y in range(img.height):
            if alphabandloaded[(x, y)] > 0:
                points.add((x, y))
    return points


def getedgepixels(img):
    """Returns a set of all opaque pixels right next to transparent ones.

    Parameters:
    img (PIL.Image): The image to analyze.

    Returns:
    set: All opaque pixels right next to transparent ones."""
    from ZachsStupidImageEffectsLibrary.coolstuff import inneroutline
    inneroutlineimg = inneroutline(img, 1, (255, 0, 0), retonly=True)
    return getallopaquepixels(inneroutlineimg)


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
    distancestoedges = getdistancestoedges(img)
    d = {}
    for distancetoedges in distancestoedges:
        d.setdefault(distancetoedges.dis, [])
        d[distancetoedges.dis].append(distancetoedges)

    selected = set()
    avoid = set()
    for k in sorted(list(d.keys())):
        i = d[k]
        for distancetoedges in i:
            pendingselected = set()
            pendingavoid = set()

            x, y = distancetoedges.startpoint
            touching = 0
            for xplus in range(-1, 2):
                for yplus in range(-1, 2):
                    check = (x+xplus, y+yplus)
                    if check in selected or check in avoid:
                        touching += 1
            if touching >= 1:
                pendingavoid.add((x, y))
            else:
                pendingselected.add((x, y))



def getdistancestoedges(img):
    """Returns a list of objects indicating the closest transparent pixel to each non-transparent pixel.

    Parameters:
    img (PIL.Image): The image to analyze.

    Returns:
    list: List of objects indicating the closest transparent pixel to each non-transparent pixel."""
    # Get possible endpoints
    endpoints = getedgepixels(img)

    # Get possible starting points
    startpoints = getallopaquepixels(img)

    return getdistancestopoints(startpoints, endpoints)