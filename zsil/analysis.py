from PIL import ImageDraw, Image
from scipy import spatial
from scipy.spatial import voronoi_plot_2d

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


def get_center_points(image: Image.Image) -> set[tuple[int, int]]:
    """Returns a set of pixels that form a line along the "center" of opaque sections. Calculated like this:
    - Get edge distance for all opaque pixels
    - All pixels that equal the largest distance are selected
    - The next set of equal distance largest distance pixels are selected
    -

    Parameters:
    image (PIL.Image): The image to analyze.

    Returns:
    set: All opaque pixels right next to transparent ones."""
    from zsil.cool_stuff import voronoi_edges
    edge_pixels = get_edge_pixels(image)

    # bigger = [tuple(y*100 for y in x) for x in edge_pixels]
    # print(bigger)
    # voronoi_edges(Image.new("RGB", (image.width * 100, image.height * 100), color=0xffffff), bigger, color=0)
    import matplotlib.pyplot as plt


    voronoi = spatial.Voronoi(list(edge_pixels), furthest_site=False)
    fig = voronoi_plot_2d(voronoi)
    furthest_for_n_guys = []
    non_furthest_for_n_guys = []
    points_and_their_connections: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for region in voronoi.regions:
        points_on_line = []
        for vert_index in region:
            if vert_index != -1:
                points_on_line.append(tuple(voronoi.vertices[vert_index]))
            else:
                points_on_line.append(None)
        if len(points_on_line) == 0:
            continue
        previous_point = None
        for point_on_line in points_on_line + points_on_line[:1]:
            if point_on_line is not None:
                pixel_for_point = tuple(round(x) for x in point_on_line)
                try:
                    if image.getpixel(pixel_for_point)[3] == 0:
                        point_on_line = None
                except IndexError: # If it's outside of the image it's considered transparent
                    point_on_line = None
            if point_on_line is not None:
                points_and_their_connections.setdefault(point_on_line, set())
                if previous_point is not None:
                    points_and_their_connections[point_on_line].add(previous_point)
                    points_and_their_connections[previous_point].add(point_on_line)
            previous_point = point_on_line

    start_points = set()
    splits_somewhere = False
    for point, connections in points_and_their_connections.items():
        if len(connections) == 1:
            start_points.add(point)
        if len(connections) >= 3:
            splits_somewhere = True

    remove_these = set(start_points)
    remove_these_length_last_time = -1
    while splits_somewhere and len(remove_these) != remove_these_length_last_time:
        remove_these_length_last_time = len(remove_these)
        for removed in list(remove_these):
            connections = points_and_their_connections[removed]
            if len(connections) > 2:
                continue
            for connection in connections:
                connections_of_the_connection = points_and_their_connections[connection]
                if len(connections_of_the_connection) > 2:
                    continue
                remove_these.add(connection)

    for remove_this in remove_these:# + non_furthest_for_n_guys:
        plt.plot(*tuple(remove_this), "bo")
        print(remove_this)
    # print(center_points)
    # for center_point in center_points:
    #     print(center_point)
        #image.putpixel([round(x) for x in center_point], 0xff0000ff)

    #image.show()
    #plt.plot()
    plt.show()
    return center_points


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


def get_edge_points(image: Image.Image):
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