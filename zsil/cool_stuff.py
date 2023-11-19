import concurrent
import io
import math
import os
import random
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from re import match, IGNORECASE
from statistics import mean
from typing import Iterable, Optional, Callable, Generator

import quads
from PIL import Image, ImageChops, ImageMath, ImageFilter
from PIL import ImageDraw
from wand.image import Image as WandImage

from zsil import colors
from zsil.analysis import Path, Vector
from zsil.internal import lengthdir_x, lengthdir_y, get_distances_to_points, point_direction, point_distance_from_origin

#  Constants for simpleshape.
SHAPE_TRIANGLE = 0
SHAPE_SQUARE = 1
SHAPE_PENTAGON = 2
SHAPE_HEXAGON = 3
SHAPE_HEPTAGON = 4
SHAPE_OCTAGON = 5
SHAPE_CIRCLE = 6
SHAPE_DIAMOND = 7
SHAPE_STAR = 8


def outline(image: Image.Image, radius: float, color, return_only: bool = False, pattern: Optional[Image.Image] = None,
            minimum_radius: Optional[float] = None, maximum_radius: Optional[float] = None,
            skip_chance=0,
            lowest_radius_of=1, color_randomize=0, position_deviation=0, shape=SHAPE_CIRCLE, shape_rotation=0,
            breeze=None):
    """Draws an outline around an image with a transparent background.

    Parameters
    ----------
    image
        The image to draw the outline around.
    radius (int): How many pixels out the outline stretches.
    aaaaaaaaaaaa I care so little aaaaaaaaaa

    Returns
    -------
    int:Returning value

   """
    if breeze is None:
        breeze = {}
    if minimum_radius is None:
        minimum_radius = radius
    if maximum_radius is None:
        maximum_radius = radius
    if pattern:
        color_randomize = 0

    breeze.setdefault("up", 0)
    breeze.setdefault("down", 0)
    breeze.setdefault("left", 0)
    breeze.setdefault("right", 0)

    outline_image = Image.new("RGBA", image.size)
    outline_image_draw = ImageDraw.Draw(outline_image)
    image_data = image.load()

    bounding_box = image.getbbox()
    if bounding_box is not None:
        points = [(x, y) for x in range(bounding_box[0], bounding_box[2]) for y in
                  range(bounding_box[1], bounding_box[3])]
        if color_randomize:
            random.shuffle(points)

        for x, y in points:
            if image_data[x, y][3] > 127:
                radius = random.uniform(minimum_radius, maximum_radius)
                if radius == 0:
                    continue
                if skip_chance and random.random() <= skip_chance:
                    radius = minimum_radius
                else:
                    for _ in range(lowest_radius_of - 1):
                        radius = min(radius, random.uniform(minimum_radius, maximum_radius))
                    working_deviation = random.uniform(-position_deviation, position_deviation)
                    # It's breezy.
                    if minimum_radius == maximum_radius:
                        breeze_radius_factor = 1
                    else:
                        breeze_radius_factor = (1 - (radius - minimum_radius) / (maximum_radius - minimum_radius))
                    working_deviation += working_deviation * (
                        breeze["left"] if working_deviation < 0 else breeze["right"]) \
                                         * breeze_radius_factor \
                                         * min([random.random() for _ in range(3)])
                    x += working_deviation

                    working_deviation = random.uniform(-position_deviation, position_deviation)
                    working_deviation += working_deviation * (breeze["up"] if working_deviation < 0 else breeze["down"]) \
                                         * breeze_radius_factor \
                                         * min([random.random() for _ in range(3)])
                    y += working_deviation

                if color_randomize:
                    current_color = tuple(
                        min(255, max(0, c - color_randomize + random.randint(0, color_randomize * 2))) for c in color)
                else:
                    current_color = color

                simple_shape(outline_image_draw, (x, y), radius, current_color, shape, rotation=shape_rotation)
        if pattern:
            outline_image = transfer_alpha(outline_image, pattern)

    if not return_only:
        outline_image.alpha_composite(image)
    return outline_image


def inner_outline(image: Image.Image, radius, color, return_only=False, pattern=None):
    # Create a slightly larger image that can for sure hold the white outline coming up.
    slightlylargerimage = image.crop((-1, -1, image.width + 1, image.height + 1))

    # Get a white silhouette of the slightly larger image
    a = round_alpha(slightlylargerimage).getchannel("A")
    bwimage = Image.merge("RGBA", (a, a, a, a))

    # Get a white outline of the image
    outline(bwimage, 1, (0, 0, 0, 0))
    thinblackoutline = bwimage.getchannel("R")
    two = Image.eval(thinblackoutline, lambda x: 255 - x)  # Thin white outline
    two = Image.merge("RGBA", (two, two, two, two))

    # Apply the outline to this line, get only the lowest alphas
    inneroutline = outline(two, radius, color, return_only=True, pattern=pattern)
    inneroutline = lowestoftwoalphaas(inneroutline, slightlylargerimage)

    # Convert the slightly larger image back to its original size
    inneroutline = inneroutline.crop((1, 1, inneroutline.width - 1, inneroutline.height - 1))
    print(image.size, inneroutline.size)

    # Return
    if not return_only:
        image.alpha_composite(inneroutline)
        inneroutline = image
    return inneroutline


def transfer_alpha(alphaman, colorman):
    colorman = colorman.resize(alphaman.size)
    a = alphaman.getchannel("A")
    r, g, b = colorman.split()[:3]
    return Image.merge("RGBA", (r, g, b, a))


def round_alpha(image: Image.Image):
    if "A" not in image.mode:
        raise Exception("Alpha channel required!")
    r, g, b, a = image.split()
    a = Image.eval(a, lambda x: round(x / 255) * 255)
    return Image.merge("RGBA", (r, g, b, a))


def threshholdalpha(image, threshhold):
    r, g, b, a = image.split()
    a = Image.eval(a, lambda x: 0 if x < threshhold else 255)
    return Image.merge("RGBA", (r, g, b, a))


def lowestoftwoalphaas(returnme, otherimage):
    a = returnme.getchannel("A")
    b = otherimage.getchannel("A")
    a = ImageMath.eval("convert(min(a, b), 'L')", a=a, b=b)
    r, b, g, nerd = returnme.split()
    return Image.merge("RGBA", (r, g, b, a))


def indent(image: Image.Image):
    # Open
    indentsdir = "images/indents"
    while True:
        indentfile = random.choice(os.listdir(indentsdir))
        indentimage = Image.open(os.path.join(indentsdir, indentfile)).convert("L")
        if indentimage.size[0] >= image.size[0] and indentimage.size[1] >= image.size[1]:
            break
    if indentimage.size[0] > indentimage.size[1]:
        indentimage.thumbnail((999999, max(image.size) * 2))
    else:
        indentimage.thumbnail((max(image.size) * 2, 999999))

    # Crop
    startx = random.randint(0, indentimage.size[0] - image.size[0])
    starty = random.randint(0, indentimage.size[1] - image.size[1])
    indentimage = indentimage.crop((startx, starty, startx + image.size[0], starty + image.size[1]))

    # Get minmax
    mincolor, maxcolor = indentimage.getextrema()
    colordif = maxcolor - mincolor

    # Stretch the values
    disfromcenter = 255
    indentimage = Image.eval(indentimage,
                             lambda x: ((x - mincolor) / colordif * disfromcenter) - (disfromcenter // 2) + 127)

    indentimagedata = indentimage.load()
    l = []
    for x in range(indentimage.size[0]):
        for y in range(indentimage.size[1]):
            l.append([x, y, indentimagedata[x, y]])
    l = sorted(l, key=lambda x: x[2])
    for n, z in enumerate(l):
        x, y, cum = z
        indentimagedata[x, y] = int(n / len(l) * 255)
    # indentimage.show()

    indentimage = indentimage.convert("RGBA")
    r, g, b, a = indentimage.split()
    a = Image.eval(r, lambda x: (
            (abs(x - 127) / (disfromcenter // 2)) * 2  # Convert black and white to distances from center
            * 16 ** 2  # Pronounce the tip
            / 256 * 32  # max
        #  //16*16  # Round
    ))
    bwband = Image.eval(r, lambda x: round(x / 255) * 255)
    indentimage = Image.merge("RGBA", (bwband, bwband, bwband, a))

    invertband = Image.eval(r, lambda x: 255 - x)
    indentimageinverted = Image.merge("RGBA", (invertband, invertband, invertband, a))
    indentimageinverted = ImageChops.offset(indentimageinverted, 5, 5)
    image = image.convert("RGBA")
    a = image.getchannel("A")
    image.alpha_composite(indentimage)
    r, g, b, WAAAAAA = image.split()
    image = Image.merge("RGBA", (r, g, b, a))

    return image


def offset_edge(image: Image.Image, xoff, yoff):
    image = round_alpha(image)
    a = image.getchannel("A")
    all_white_image = Image.new("L", a.size, 255)
    all_black_image = Image.new("L", a.size, 0)
    w = Image.merge("RGBA", (all_white_image, all_white_image, all_white_image, a))
    b = Image.merge("RGBA", (all_black_image, all_black_image, all_black_image, a))
    b = ImageChops.offset(b, xoff, yoff)
    w.alpha_composite(b)
    all_black_image = all_black_image.convert("RGBA")
    all_black_image.alpha_composite(w)
    this_band_is_going_to_be_every_band = all_black_image.getchannel("R")
    ret = Image.merge("RGBA", (
        this_band_is_going_to_be_every_band, this_band_is_going_to_be_every_band, this_band_is_going_to_be_every_band,
        this_band_is_going_to_be_every_band))
    return ret


def crop_to_content(image: Image.Image, force_top=None, force_left=None, force_bottom=None, force_right=None):
    """Returns an image cropped to its bounding box.

    Parameters
    ----------
    image
        The image to be cropped.
    force_top (int): Replaces the top of the bounding box.
    force_left (int): Replaces the left of the bounding box.
    force_bottom (int): Replaces the bottom of the bounding box.
    force_right (int): Replaces the right of the bounding box.

    Returns
    -------
    (PIL.Image): Cropped image."""
    # Get bounding box
    bb = image.getbbox()
    # If it's None, it's empty
    if bb is None:
        bb = [0, 0, 1, 1]
    else:
        # Convert because tuple indexes can't be modified
        bb = list(bb)
        # Turns out the bounding box is off by one pixel in each direction, so fix that
        for index, offset in enumerate([-1, -1, 1, 1]):
            bb[index] += offset
    # Replace bits of it, if needed
    if force_top is not None or force_left is not None or force_bottom is not None or force_right is not None:
        # Check each possible replacement and use it if needed
        for force, index in (
                (force_left, 0),
                (force_top, 1),
                (force_right, 2),
                (force_bottom, 3),
        ):
            if force is not None:
                bb[index] = force
    # Crop and return
    image = image.crop(bb)
    return image


def resize_and_crop(image: Image.Image, size):
    """Matches one side by resizing and the other by cropping."""
    old_width, old_height = image.size
    new_width, new_height = size

    width_multiplier = new_width / old_width
    height_multiplier = new_height / old_height
    multiplier = max(width_multiplier, height_multiplier)

    stretch_size = (round(old_width * multiplier), round(old_height * multiplier))
    stretched_image = image.resize(stretch_size)

    stretched_width, stretched_height = stretched_image.size
    offset_x = abs(new_width - stretched_width) // 2
    offset_y = abs(new_height - stretched_height) // 2
    cropped_image = stretched_image.crop(box=(offset_x, offset_y, offset_x + new_width, offset_y + new_height))
    return cropped_image


def shading(image: Image.Image, off_x, off_y, size, shrink, grow_back, color, alpha, blocker_multiplier=1,
            blurring=False):
    """Outline-based shading."""
    has_not_blurred_yet = True
    # Get side to shade, and also the opposite side
    light = offset_edge(image, off_x, off_y)
    dark = lowestoftwoalphaas(Image.new("RGBA", light.size, (0, 0, 0, 255)), offset_edge(image, -off_x, -off_y))
    # Make it bigger, one at a time, from both sides. The dark cancels out the light.
    # The lights from each step are added together.
    light_to_make_clones_of = light.copy()
    for x in range(1, size + 1):
        working_light = light_to_make_clones_of.copy()
        working_dark = dark.copy()
        working_light = outline(working_light, x, (255, 255, 255, 255))
        working_dark = outline(working_dark, x * blocker_multiplier, (0, 0, 0, 255))
        working_light.alpha_composite(working_dark)
        a = working_light.getchannel("R")
        light.alpha_composite(Image.merge("RGBA", (a, a, a, a)))
    # Blur?
    if not shrink and not grow_back and blurring:
        light = light.filter(ImageFilter.GaussianBlur(radius=5))
        has_not_blurred_yet = False
    light = lowestoftwoalphaas(light, image)
    # Shrink it down
    if shrink:
        light = inner_outline(light, shrink, (0, 0, 0, 255))
    # Remove stray alpha nonsense
    all_black = Image.new("RGBA", light.size, color=(0, 0, 0, 255))
    all_black.alpha_composite(light)
    r = all_black.getchannel("R")
    light = Image.merge("RGBA", (r, r, r, r))
    # Blur?
    if not grow_back and blurring and has_not_blurred_yet:
        light = light.filter(ImageFilter.GaussianBlur(radius=5))
        has_not_blurred_yet = False
    # Grow it back
    if grow_back:
        light = outline(light, grow_back, (255, 255, 255, 255))
    # Remove stray alpha nonsense 2
    r = light.getchannel("R")
    lighttrans = Image.eval(r, lambda x: min(x, alpha))
    # Set color and alpha
    light = Image.merge("RGBA", (r, r, r, lighttrans))
    colorimage = Image.new("RGBA", light.size, color)
    light = lowestoftwoalphaas(colorimage, light)
    # Blur?
    if blurring and has_not_blurred_yet:
        light = light.filter(ImageFilter.GaussianBlur(radius=5))
    # Overlay and return
    image.alpha_composite(light)
    return image


def sharplight(image: Image.Image, disout, size, blocker_multiplier=1, blurring=False):
    """Sticks out from the edge, not rounded."""
    return shading(image, 1, 1, size + disout, disout, 0, colors.random_white(), 192,
                   blocker_multiplier=blocker_multiplier,
                   blurring=blurring)


def roundedlight(image: Image.Image, disout, size, blocker_multiplier=1, blurring=False):
    """Sticks out from the edge, rounded."""
    return shading(image, 1, 1, size + disout, ((size + disout) / 2) - 1, disout / 2 - 1, colors.random_white(), 127,
                   blocker_multiplier=blocker_multiplier, blurring=blurring)


def edgelight(image: Image.Image, size, blocker_multiplier=1, blurring=False):
    """Doesn't stick out from the edge."""
    return shading(image, 1, 1, size, 0, 0, colors.random_white(), 127, blocker_multiplier=blocker_multiplier,
                   blurring=blurring)


def shadow(image: Image.Image, size, blocker_multiplier=1, blurring=False):
    return shading(image, -1, -1, size, 0, 0, (0, 0, 0), 64, blocker_multiplier=blocker_multiplier, blurring=blurring)


def shadingwrapper(image: Image.Image):
    choice = random.randint(0, 4)
    blurring = random.randint(0, 1)
    if choice == 0:
        image = sharplight(image, 4, 7, blurring=blurring)
        image = shadow(image, 7, blurring=blurring)
    elif choice == 1:
        image = roundedlight(image, 4, 7, blurring=blurring)
        image = shadow(image, 7, blurring=blurring)
    elif choice == 2:
        image = edgelight(image, 7, blurring=blurring)
        image = shadow(image, 7, blurring=blurring)
    elif choice == 3:
        image = edgelight(image, 20, blurring=blurring)
        image = shadow(image, 20, blurring=blurring)
    elif choice == 4:
        image = edgelight(image, 20, blurring=blurring, blocker_multiplier=2)
        image = shadow(image, 20, blurring=blurring, blocker_multiplier=2)
    return image


def directional_shading(image: Image.Image):
    """Draws gradient-y shading based on direction from the light source. Does this by repeatedly drawing the largest
    still-possible shortest-distance lines from inside the structure to an outline along the outside."""
    from analysis import get_all_opaque_pixels, get_edge_pixels, get_center_pixels

    # Maybe don't force this?
    # image = threshholdalpha(image, 127)

    # Get possible end_points
    edgeend_points = get_edge_pixels(image)
    centerend_points = get_center_pixels(image)
    bothend_points = centerend_points | edgeend_points

    # Get possible starting point_count
    start_points = get_all_opaque_pixels(image)
    # start_points -= bothend_points

    # Get potential lines
    edgepotentiallines = {}
    centerpotentiallines = {}
    for potentiallines, end_points in [
        (edgepotentiallines, edgeend_points),
        (centerpotentiallines, centerend_points),
    ]:
        for potentialline in get_distances_to_points(start_points, end_points):
            potentiallines[potentialline.start_point] = potentialline
    allpotentiallines = {**edgepotentiallines, **centerpotentiallines}

    # Draw along the longest potential lines
    shadingcolors = Image.new("LA", image.size)
    draw = ImageDraw.Draw(shadingcolors)
    n = 0
    goalpoints = len(start_points | bothend_points)
    uncombinedpoints = {}
    while goalpoints * 0.6 > len(get_all_opaque_pixels(shadingcolors)) and n < 100:
        print(len(get_all_opaque_pixels(shadingcolors)), goalpoints)
        # Get current longest
        # longest = -1
        # for start_point in start_points:
        #     potentialline = edgepotentiallines[start_point]
        #     longest = max(longest, potentialline.dis)
        #     potentialline = centerpotentiallines[start_point]
        #     longest = max(longest, potentialline.dis)
        # if longest <= 8:
        #     break
        # Draw on all of that length
        for start_point in list(start_points):
            edgepotentialline = edgepotentiallines[start_point]
            centerpotentialline = centerpotentiallines[start_point]
            # Use whichever is longer because it will presumably be more accurate
            if edgepotentialline.dis > centerpotentialline.dis:
                potentialline = edgepotentialline
                reversemaybe = 0
            else:
                potentialline = centerpotentialline
                reversemaybe = 180
            if True:  # potentialline.dis == longest:
                absolute = round(abs((potentialline.dir + 135 + reversemaybe) % 360 - 180) / 180 * 255)
                using = absolute  # round(absolute/255)*255
                # draw.line((potentialline.start_point, potentialline.end_point), (using, 2))
                # draw.point(potentialline.start_point, (using, 255))
                difx = potentialline.start_point[0] - potentialline.end_point[0]
                deltax = abs(difx)
                dify = abs(potentialline.start_point[1] - potentialline.end_point[1])
                deltay = abs(dify)
                maxdelta = max(deltax, deltay)
                for n in range(maxdelta + 1):
                    point = (
                        round(potentialline.start_point[0] - difx / maxdelta * n),
                        round(potentialline.start_point[1] - dify / maxdelta * n),
                    )
                    print(potentialline.start_point, point, potentialline.end_point)
                    uncombinedpoints.setdefault(point, [])
                    uncombinedpoints[point].append(using)
                start_points.remove(potentialline.start_point)
                # n += 1
                # if n**0.5 == round(n**0.5):
                #     image.alpha_composite(shadingcolors.convert("RGBA"))
                #     image.show()
        # Remove everything that's now filled in
        # shadingcolorspoints = getallalphapoints(shadingcolors)
        # print(potentialline.start_point in shadingcolorspoints)
        # start_points -= getallalphapoints(shadingcolors)
        # print(len(start_points))
        n += 1

    for point, usings in uncombinedpoints.items():
        draw.point(point, (round(mean(usings)), 255))

    # shadingcolors = shadingcolors.convert("RGBA")
    # for x in range(100):
    #     blurryboy = shadingcolors.copy()
    #     blurryboy = blurryboy.filter(ImageFilter.GaussianBlur)
    #     blurryboy.alpha_composite(shadingcolors)
    #     shadingcolors = blurryboy
    shadingcolorsdata = shadingcolors.load()
    for x in range(shadingcolors.width):
        for y in range(shadingcolors.height):
            if shadingcolorsdata[x, y][1] == 0:
                pass

    shadingcolors.show()
    shadingcolors.split()[shadingcolors.getbands().index("A")].convert("RGBA").save("WACK.png")
    shadingcolors.putalpha(image.split()[image.getbands().index("A")])
    image.alpha_composite(shadingcolors.convert("RGBA"))
    image.show()


def metallic_directional_shading(image: Image.Image):
    """Draws gradient-y shading based on direction from the light source. Does this by repeatedly drawing the largest
    still-possible shortest-distance lines from inside the structure to an outline along the outside."""
    from analysis import get_all_opaque_pixels, get_edge_pixels, get_center_pixels

    # Maybe don't force this?
    # image = threshholdalpha(image, 127)

    # Get possible end_points
    edge_end_points = get_edge_pixels(image)
    center_end_points = get_center_pixels(image)
    both_end_points = center_end_points | edge_end_points

    # Get possible starting point_count
    start_points = get_all_opaque_pixels(image)
    # start_points -= both_end_points

    # Get potential lines
    edge_potential_lines = {}
    center_potential_lines = {}
    for potentiallines, end_points in [
        (edge_potential_lines, edge_end_points),
        (center_potential_lines, center_end_points),
    ]:
        for potential_line in get_distances_to_points(start_points, end_points):
            potentiallines[potential_line.start_point] = potential_line
    allpotentiallines = {**edge_potential_lines, **center_potential_lines}

    # Draw along the longest potential lines
    shading_colors = Image.new("LA", image.size)
    draw = ImageDraw.Draw(shading_colors)
    n = 0
    goal_points = len(start_points | both_end_points)
    while goal_points * 0.6 > len(get_all_opaque_pixels(shading_colors)) and n < 100:
        print(len(get_all_opaque_pixels(shading_colors)), goal_points)
        # Get current longest
        # longest = -1
        # for start_point in start_points:
        #     potential_line = edge_potential_lines[start_point]
        #     longest = max(longest, potential_line.dis)
        #     potential_line = center_potential_lines[start_point]
        #     longest = max(longest, potential_line.dis)
        # if longest <= 8:
        #     break
        # Draw on all of that length
        for start_point in list(start_points):
            edgepotentialline = edge_potential_lines[start_point]
            centerpotentialline = center_potential_lines[start_point]
            # Use whichever is longer because it will presumably be more accurate
            if edgepotentialline.dis > centerpotentialline.dis:
                potential_line = edgepotentialline
                reversemaybe = 0
            else:
                potential_line = centerpotentialline
                reversemaybe = 180
            if True:  # potential_line.dis == longest:
                absolute = round(abs((potential_line.dir + 135 + reversemaybe) % 360 - 180) / 180 * 255)
                using = absolute  # round(absolute/255)*255
                draw.line((potential_line.start_point, potential_line.end_point), (using, 2))
                # draw.point(potential_line.start_point, (using, 255))
                start_points.remove(potential_line.start_point)
                # n += 1
                # if n**0.5 == round(n**0.5):
                #     image.alpha_composite(shading_colors.convert("RGBA"))
                #     image.show()
        # Remove everything that's now filled in
        # shadingcolorspoints = getallalphapoints(shading_colors)
        # print(potential_line.start_point in shadingcolorspoints)
        # start_points -= getallalphapoints(shading_colors)
        # print(len(start_points))
        n += 1

    # shading_colors = shading_colors.convert("RGBA")
    # for x in range(100):
    #     blurryboy = shading_colors.copy()
    #     blurryboy = blurryboy.filter(ImageFilter.GaussianBlur)
    #     blurryboy.alpha_composite(shading_colors)
    #     shading_colors = blurryboy
    shadingcolorsdata = shading_colors.load()
    for x in range(shading_colors.width):
        for y in range(shading_colors.height):
            if shadingcolorsdata[x, y][1] == 0:
                pass

    shading_colors.putalpha(image.split()[image.getbands().index("A")])
    shading_colors.show()
    image.alpha_composite(shading_colors.convert("RGBA"))
    image.show()


def go_deep_dream_yourself(image: Image.Image, max_dim=None, steps=100):
    """A lazy wrapper for something made by somebody cooler than me."""
    import tensorflow as tf
    import numpy as np

    import IPython.display as display
    import PIL.Image

    # Normalize an image
    def deprocess(image: Image.Image):
        image = 255 * (image + 1.0) / 2.0
        return tf.cast(image, tf.uint8)

    # Display an image
    def show(image: Image.Image):
        display.display(PIL.Image.fromarray(np.array(image)))

    def backtopil(image: Image.Image):
        return PIL.Image.fromarray(np.array(image))

    # Downsizing the image makes it easier to work with.
    image = image.convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim))
    original_image = np.array(image)

    # show(original_image)
    display.display(display.HTML(
        'Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # Maximize the activations of these layers
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    def calc_loss(image: Image.Image, model):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        image_batch = tf.expand_dims(image, axis=0)
        layer_activations = model(image_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        losses = []
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)

        return tf.reduce_sum(losses)

    class DeepDream(tf.Module):
        def __init__(self, model):
            self.model = model

        @tf.function(
            input_signature=(
                    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                    tf.TensorSpec(shape=[], dtype=tf.int32),
                    tf.TensorSpec(shape=[], dtype=tf.float32),)
        )
        def __call__(self, image, steps, step_size):
            print("Tracing")
            loss = tf.constant(0.0)
            for n in tf.range(steps):
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `image`
                    # `GradientTape` only watches `tf.Variable`s by default
                    tape.watch(image)
                    loss = calc_loss(image, self.model)

                # Calculate the gradient of the loss with respect to the pixels of the input image.
                gradients = tape.gradient(loss, image)

                # Normalize the gradients.
                gradients /= tf.math.reduce_std(gradients) + 1e-8

                # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
                # You can update the image by directly adding the gradients (because they're the same shape!)
                image = image + gradients * step_size
                image = tf.clip_by_value(image, -1, 1)

            print("Done tracing.")
            return loss, image

    deepdream = DeepDream(dream_model)

    def run_deep_dream_simple(image, steps=100, step_size=0.01):
        # Convert from uint8 to the range expected by the model.
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        image = tf.convert_to_tensor(image)
        step_size = tf.convert_to_tensor(step_size)
        steps_remaining = steps
        step = 0
        while steps_remaining:
            if steps_remaining > 100:
                run_steps = tf.constant(100)
            else:
                run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            loss, image = deepdream(image, run_steps, tf.constant(step_size))

            display.clear_output(wait=True)
            show(deprocess(image))
            print("Step {}, loss {}".format(step, loss))

        result = deprocess(image)
        display.clear_output(wait=True)
        # show(result)

        im = backtopil(result)

        return im

    dream_image = run_deep_dream_simple(image=original_image,
                                        steps=steps, step_size=0.01)
    return dream_image


def draw_star(draw: ImageDraw.Draw, xy: tuple[int, int], directions, point_count, inner_radius, outer_radius, fill):
    """Draw a star. Meant to invoke PIL's geometry drawing functions."""
    x, y = xy
    inneroffset = 360 / point_count / 2
    for d in range(point_count):
        workingdir = directions + (360 / point_count * d)
        draw.polygon([
            (x + lengthdir_x(inner_radius, workingdir - inneroffset),
             y + lengthdir_y(inner_radius, workingdir - inneroffset)),
            (x + lengthdir_x(outer_radius, workingdir), y + lengthdir_y(outer_radius, workingdir)),
            (x + lengthdir_x(inner_radius, workingdir + inneroffset),
             y + lengthdir_y(inner_radius, workingdir + inneroffset)),
        ],
            fill=fill)
    draw.ellipse((x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius), fill)


def repaint(image: Image.Image, function, growth_chance=0.5, recalculation_rate=10):
    """Takes an image(A), and creates a blank image(B). Until none exist, picks pixels that are transparent(alpha 0) in
    Image B but not image A, and runs function on Image B using the information from that pixel on Image A.
    I suppose it could be thought of repainting image using function as the brush.

    Parameters
    ----------
    image
        The original image.
    function (function): A function that takes the following arguments:
        image (PIL.ImageDraw.ImageDraw): Image.Image B.
        xy (Tuple): Where to draw on Image B.
        size (int): How large the shape to draw on Image B is.
        color (Tuple): What color to draw on Image B.
    growth_chance (float): Chance of higher sizes. Must be less than one.
    recalculation_rate (int): How many "strokes" will be done at once before checking which pixels are still transparent.

    Returns
    -------
    tuple: All coordinates to which function returned true."""
    # Make sure the growth chance isn't 1 or greater to avoid infinite loops
    if growth_chance >= 1:
        raise Exception("Growth chance must be less than 1.")

    # Set shit up
    original_image = image.convert("RGBA")
    new_image = Image.new("RGBA", original_image.size)
    new_image.thumbnail()
    original_image_data = original_image.load()
    new_image_data = new_image.load()
    original_transparent_pixels = set(pixel_filter(lambda x: x[3] == 0, original_image, imagedata=original_image_data))

    # Main loop
    while True:
        # Get remaining transparent pixels
        remainingtransparentpixels = list(
            set(pixel_filter(lambda x: x[3] == 0, new_image, imagedata=new_image_data)) \
            - original_transparent_pixels
        )

        # Bail if finished
        if not remainingtransparentpixels:
            break

        # Iterate through them
        random.shuffle(remainingtransparentpixels)
        for xy in remainingtransparentpixels[:min(len(remainingtransparentpixels), recalculation_rate)]:
            # Get size
            size = 1
            while random.random() <= growth_chance:
                size += 1

            # Get color
            color = original_image_data[xy[0], xy[1]]

            # Do the function
            function(new_image, xy, size, color)

    # Return
    return new_image


def simple_shape(image_or_draw, xy, radius, color, shape, rotation=None):
    """Wrapper for more complicated shape drawing functions so they can be more easily interchanged.

    Parameters
    ----------
    image (PIL.ImageDraw.ImageDraw or PIL.Image): The Image or ImageDraw to draw on.
    xy (Tuple): Center point of the shape.
    radius (float): Distance from center the shape reaches.
    color (Tuple): Color/fill of the shape.
    rotation: Angle of the shape. Random if not provided.

    Returns
    -------
    PIL.ImageDraw.ImageDraw or PIL.Image: Image.Image. Doesn't make a copy, only returns for convenience."""
    x, y = xy
    if rotation is None:
        rotation = random.random() * 360

    if type(image_or_draw) == Image.Image:
        draw = ImageDraw.Draw(image_or_draw)
    else:
        draw = image_or_draw

    if type(shape) is not int:
        shape = random.choice(shape)

    if radius > 0:
        # Draw triangle
        if shape == SHAPE_TRIANGLE:
            draw.regular_polygon((x, y, radius), 3, rotation=rotation, fill=color)
        # Draw straight square
        elif shape == SHAPE_SQUARE and rotation % 90 == 0:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
        # Draw any other square
        elif shape == SHAPE_SQUARE:
            draw.regular_polygon((x, y, radius), 4, rotation=rotation, fill=color)
        # Draw pentagon
        elif shape == SHAPE_PENTAGON:
            draw.regular_polygon((x, y, radius), 5, rotation=rotation, fill=color)
        # Draw hexagon
        elif shape == SHAPE_HEXAGON:
            draw.regular_polygon((x, y, radius), 6, rotation=rotation, fill=color)
        # Draw heptagon
        elif shape == SHAPE_HEPTAGON:
            draw.regular_polygon((x, y, radius), 7, rotation=rotation, fill=color)
        # Draw octagon
        elif shape == SHAPE_OCTAGON:
            draw.regular_polygon((x, y, radius), 8, rotation=rotation, fill=color)
        # Draw circle
        elif shape == SHAPE_CIRCLE:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), color)
        # Draw diamond
        elif shape == SHAPE_DIAMOND:
            draw.polygon((
                (
                    x + lengthdir_x(radius, 0 + rotation),
                    y + lengthdir_y(radius, 0 + rotation)
                ),
                (
                    x + lengthdir_x(radius / 2, 90 + rotation),
                    y + lengthdir_y(radius / 2, 90 + rotation)
                ),
                (
                    x + lengthdir_x(radius, 180 + rotation),
                    y + lengthdir_y(radius, 180 + rotation)
                ),
                (
                    x + lengthdir_x(radius / 2, 270 + rotation),
                    y + lengthdir_y(radius / 2, 270 + rotation)
                )
            ), fill=color)
        # Draw star
        elif shape == SHAPE_STAR:
            draw_star(draw, (x, y), random.random() * 360, 5, radius * random.uniform(0.15, 0.6),
                      radius, color)
        # Draw NOTHING!!!
        else:
            raise Exception("Invalid shape!")

        # elif shape == "many shapes":
        #     draw.regular_polygon((x, y, radius), random.choice([3, 4, 5, 6, 8]),
        #                                 rotation=random.random() * 360, fill=color)
    return image_or_draw


def predict_thumbnail_size(original_size: tuple[int, int], new_size: Image.Image):
    """Figures out what size an image of original_size would become if you used it to make a thumbnail of new_size.

    Parameters
    ----------
    original_size (Tuple): Original size(duh).
    new_size (Tuple): New size(duh).

    Returns
    -------
    Tuple: Predicted thumbnail size."""
    original_width, original_height = original_size
    x, y = map(math.floor, new_size)
    if x >= original_width and y >= original_height:
        return original_size

    def round_aspect_ratio(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    # preserve aspect ratio
    aspect = original_width / original_height
    if x / y >= aspect:
        x = round_aspect_ratio(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect_ratio(
            x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n)
        )
    return x, y


def multiline_textsize_but_it_works(text, font, max_width: int = 1500):
    # Should probably implement other parameters.
    # Also, I think the Pillow version works now?
    image = Image.new('RGBA', (max_width, 1200), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((0, 0), text, font=font, fill="black", align="center")

    bbox = image.getbbox()
    if bbox:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    else:
        return 0, 0


def auto_breaking_text(image_or_draw, allowed_width, xy, text, fill=None, font=None, anchor=None, spacing=4,
                       align='left',
                       direction=None, features=None, language=None, stroke_width=0, stroke_fill=None,
                       embedded_color=False):
    """Draws PIL text and adds line breaks whenever it gets too wide.

    Parameters
    ----------
    original_size (Tuple): Original size(duh).
    new_size (Tuple): New size(duh).

    Returns
    -------
    Tuple: Predicted thumbnail size."""
    if type(image_or_draw) == Image.Image:
        draw = ImageDraw.Draw(image_or_draw)
    else:
        draw = image_or_draw

    # Attempts to draw each extra word with a space, and if it's too wide, does it with a line break.
    # Only tries the last line, because otherwise, if a word is longer than the allowed width, every subsequent line
    # would break the limit.
    # Only splits with spaces and not with breaks to allow forced breaks.
    # Only does spaces and not other whitespace because I'm lazy.
    # If there's a forced line break followed by a word that's longer than the limit, that'll make the word before the
    # line break get its own line even though it doesn't need it. But again, lazy.
    textsplit = text.split(" ")
    teststring = ""  # What needs to pass Inspection
    usingstring = ""  # What will be drawn at the end
    for word in textsplit:
        tempteststring = teststring + " " + word
        if multiline_textsize_but_it_works(tempteststring, font, max_width=allowed_width + 20)[0] <= allowed_width:
            teststring = tempteststring
            usingstring += " " + word
        else:
            teststring = word
            usingstring += "\n" + word

    # Actually draw
    draw.multiline_text(xy, usingstring, fill=fill, font=font)


def pil_to_wand(image: Image.Image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return WandImage(blob=stream.getvalue())


def wand_to_pil(image: Image.Image):
    return Image.open(io.BytesIO(image.make_blob("png")))


def enlargeable_thumbnail(image: Image.Image, size, resample=None):
    """Same as PIL.Image.thumbnail, but can also grow to fit the required size."""
    if image.size[0] > size[0] or image.size[1] > size[1]:
        image.thumbnail(size, resample=resample)
        return image
    possiblemultipliers = sorted([
        size[0] / image.size[0],
        size[1] / image.size[1],
    ])[::-1]
    for multiplier in possiblemultipliers:
        newwidth = round(image.size[0] * multiplier)
        newheight = round(image.size[1] * multiplier)
        if newwidth <= size[0] and newheight <= size[1]:
            return image.resize((newwidth, newheight), resample=resample)


def square_crop(image: Image.Image):
    """Adds extra size on both sides of an image to form a square."""
    if image.width > image.height:
        # These are separate so that one can be one pixel larger if it's an odd number
        extrasizeononeside = (image.width - image.height) // 2
        extrasizeontheother = (image.width - image.height) - extrasizeononeside
        image = image.crop((0, -extrasizeononeside, image.width, image.height + extrasizeontheother))
    elif image.width < image.height:
        # These are separate so that one can be one pixel larger if it's an odd number
        extrasizeononeside = (image.height - image.width) // 2
        extrasizeontheother = (image.height - image.width) - extrasizeononeside
        image = image.crop((-extrasizeononeside, 0, image.width + extrasizeontheother, image.height))
    return image


def text_image(text: str):
    """Generates a simple square image with text in the middle."""
    image = Image.new("RGBA", (1200, 1200))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((5, 5), text)
    image = crop_to_content(image)
    image = outline(image, 1, (0, 0, 0))
    image = square_crop(image)
    return image


def dynamically_sized_text_image(text: str, size: tuple[int, int], font=None, fill="black"):
    """Generates an image with text whose lines are resized such that the longer lines are smaller."""
    width, height = size
    textimage = Image.new("RGBA", size)

    # Set up the lines
    text_line_data = []
    for pos, textline in enumerate(text.split("\n")):
        text_line_data.append({
            "s": textline,
            "pos": pos,
            "baby": match(r"^(\s|-|of|or|in|my|o|o'|the|\.|mr|mrs|\w{,2}\.)*$", textline, flags=IGNORECASE) is not None
        })
    text_line_data.sort(key=lambda x: -len(x["s"]))
    text_line_data.sort(key=lambda x: -x["baby"])
    print(text_line_data)

    # Generate thumbnails
    room_taken = 0
    buildup = {}
    normal_lines_remaining = len(list(filter(lambda x: not x["baby"], text_line_data)))
    baby_lines_remaining = len(text_line_data) - normal_lines_remaining
    baby_line_divider = 3
    for n, tld in enumerate(text_line_data):
        current_line_divider = baby_line_divider if tld["baby"] else 1
        room_remaining = height - room_taken
        there_is_space_for_n_lines_required = normal_lines_remaining + (baby_line_divider / baby_line_divider)

        text_line_image = Image.new("RGBA", (width * 18, height * 4))
        text_line_draw = ImageDraw.Draw(text_line_image)
        text_line_draw.text((text_line_image.size[0] // 2, text_line_image.size[1] // 2), tld["s"], font=font,
                            fill=fill,
                            align="center")
        text_line_image = crop_to_content(text_line_image)
        text_line_image.thumbnail((width, room_remaining / there_is_space_for_n_lines_required // current_line_divider),
                                  resample=Image.LANCZOS)

        buildup[tld["pos"]] = text_line_image
        room_taken += text_line_image.size[1]
        if tld["baby"]:
            baby_lines_remaining -= 1
        else:
            normal_lines_remaining -= 1

    # Combine them
    draw_at_y = 0
    for n in range(len(buildup)):
        text_line_image = buildup[n]
        textimage.paste(text_line_image, (abs(text_line_image.size[0] - textimage.size[0]) // 2, draw_at_y))
        draw_at_y += text_line_image.size[1] + 5

    return textimage


def shift_hue_by(image: Image.Image, by: int) -> Image:
    image = image.convert("HSV")
    h, s, v = image.split()
    h = Image.eval(h, lambda x: (x + by) % 255)
    return Image.merge("HSV", (h, s, v))


def shift_hue_towards(image: Image.Image, towards: int) -> Image:
    image = image.convert("HSV")
    average_hue = colors.average_color(image.getchannel("H"))
    by = towards - average_hue
    return shift_hue_by(image, by)


def shift_bands_by(image: Image.Image, by: Iterable[int]) -> Image:
    bands = []
    for index, band in enumerate(image.split()):
        if image.mode[index] == "H":  # Pretty sure H is the only one that "loops"?
            band = Image.eval(band, lambda x: (x + by[index]) % 255)
        else:
            band = Image.eval(band, lambda x: min(255, max(0, (x + by[index]))))
        bands.append(band)
    return Image.merge(image.mode, bands)


def shift_bands_towards(image: Image.Image, towards: Iterable[int]) -> Image:
    average_color = colors.average_color(image)
    return shift_bands_by(image, [x[0] - x[1] for x in zip(towards, average_color)])


def buttonize(image: Image.Image, distance_from_edge: int = 5, outermost_color=(0, 0, 0),
              innermost_color=(255, 255, 255)):
    for current_distance_from_edge in list(range(distance_from_edge))[::-1]:
        image = inner_outline(image, current_distance_from_edge, colors.merge_colors(outermost_color, innermost_color,
                                                                                     current_distance_from_edge / distance_from_edge))
    return image


def double_points_along_path(path: Path) -> Path:
    input_path = path
    halved_path = []
    for point_index, point in enumerate(input_path):
        halved_path.append(point)

        next_point_index = (point_index + 1) % len(input_path)
        next_point = input_path[next_point_index]

        point_in_the_middle = (point + next_point) / 2
        halved_path.append(point_in_the_middle)
    return halved_path


def subdivide_path(path: Path, repetitions=1):
    if repetitions > 1:
        for repetition in range(repetitions):
            path = subdivide_path(path, 1)
        return path

    input_path = double_points_along_path(path)
    subdivided_path = []
    for point_index, point in enumerate(input_path):
        point: Vector
        next_point_index = (point_index + 1) % len(input_path)
        next_point = input_path[next_point_index]
        previous_point_index = (point_index - 1) % len(input_path)
        previous_point = input_path[previous_point_index]

        averaged_point = point * 0.5 + next_point * 0.25 + previous_point * 0.25
        subdivided_path.append(averaged_point)
    return subdivided_path


@dataclass
class GenerateFromNearestKeyParams:
    image: Image.Image
    coordinates: tuple[int, int]
    nearest_point: quads.Point

    def offset_to_origin(self):
        return self._global_offset_to_origin(self.coordinates, self.nearest_point)

    @staticmethod
    @cache
    def _global_offset_to_origin(coordinates: tuple[int], nearest_point: quads.Point):
        return (
            nearest_point.x - coordinates[0],
            nearest_point.y - coordinates[1],
        )

    def direction(self) -> Optional[float]:
        return point_direction(*self.coordinates, self.nearest_point.x,
                               self.nearest_point.y)

    def distance(self) -> Optional[float]:
        return self._distance_from_origin(self.offset_to_origin())

    @staticmethod
    @cache
    def _distance_from_origin(vector: tuple[float, float]) -> float:
        return point_distance_from_origin(*vector)


def generate_from_nearest_iterable(image: Image, points: Iterable[Iterable[int]],
                                   key: Callable[[GenerateFromNearestKeyParams], None | tuple[int, ...]],
                                   coordinates_to_go_over: Optional[Iterable[tuple[int, int]]] = None) -> Generator[
    tuple[int, int, int], None, None]:
    tree = quads.QuadTree((image.width // 2, image.height // 2), *image.size)
    for point in points:
        tree.insert(point)
    draw = ImageDraw.ImageDraw(image)

    output = {}

    def process_key(coordinates_within_key):
        nearest_point = tree.nearest_neighbors(coordinates_within_key, 1)[0]

        params = GenerateFromNearestKeyParams(
            image=image,
            coordinates=coordinates_within_key,
            nearest_point=nearest_point,
        )

        result = key(params)
        if result is not None:
            draw.point(coordinates_within_key, result)

    if coordinates_to_go_over is None:
        coordinates_to_go_over = []
        for x in range(image.width):
            for y in range(image.height):
                coordinates = (x, y)
                coordinates_to_go_over.append(coordinates)

    with ThreadPoolExecutor() as executor:
        future_to_coordinates = {executor.submit(process_key, coordinates): coordinates for coordinates in
                                 coordinates_to_go_over}

    for future in concurrent.futures.as_completed(future_to_coordinates):
        yield_this = future.result()
        yield yield_this


def generate_from_nearest(image: Image, points: Iterable[Iterable[int]],
                          key: Callable[[GenerateFromNearestKeyParams], None | tuple[int, ...]],
                          coordinates_to_go_over: Optional[Iterable[tuple[int, int]]] = None):
    for _ in generate_from_nearest_iterable(image, points, key,
                                            coordinates_to_go_over=coordinates_to_go_over):
        pass
