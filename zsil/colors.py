import colorsys
from typing import Union, Tuple, Iterator

import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageShow

import random as rando
from statistics import mean
import numpy
from math import sqrt


def convert_1_to_255(col):
    c = list(col)
    for n, x in enumerate(c):
        c[n] = round(x * 255)
    c = tuple(c)
    return c


def convert_255_to_1(col):
    c = list(col)
    for n, x in enumerate(c):
        c[n] = x / 255
    c = tuple(c)
    return c


def random_white(take_hue_from=None):
    if take_hue_from:
        hue = colorsys.rgb_to_hsv(*convert_255_to_1(take_hue_from))[0]
    else:
        hue = None

    if rando.randint(0, 1) and not (hue is not None and (hue < 180 / 360 or hue > 246 / 360)):
        if hue is None:
            hue = rando.uniform(180 / 360, 246 / 360)
        c = colorsys.hsv_to_rgb(
            hue,
            rando.random() * 0.1,
            1 - rando.random() * 0.03,
        )
    else:
        if hue is None:
            hue = rando.random()
        c = colorsys.hsv_to_rgb(
            hue,
            rando.random() * 0.04,
            1 - rando.random() * 0.06,
        )

    c = list(c)
    for n, x in enumerate(c):
        c[n] = round(x * 255)
    c.append(255)
    c = tuple(c)
    return c


def random_black():
    if rando.randint(0, 1):
        c = colorsys.hsv_to_rgb(
            rando.uniform(180 / 360, 246 / 360),
            rando.random() * 0.1,
            rando.random() * 0.03,
        )
    else:
        c = colorsys.hsv_to_rgb(
            rando.random(),
            rando.random() * 0.05,
            rando.random() * 0.07,
        )

    c = list(c)
    for n, x in enumerate(c):
        c[n] = round(x * 255)
    c.append(255)
    c = tuple(c)
    return c


def merge_colors(color_1, color_2, amount=None):
    if amount is None:
        amount = rando.uniform(0.4, 0.6)
    newcolor = []
    for n, x in enumerate(color_1):
        newcolor.append(round(sqrt(x ** 2 * (1 - amount) + color_2[n] ** 2 * amount)))
    return tuple(newcolor)


def variant_of(color):
    a = None
    if len(color) > 3:
        a = color[3]
        color = color[:3]
    h, s, v = colorsys.hsv_to_rgb(*convert_255_to_1(color))
    choice = rando.randint(0, 2)
    if choice == 0:
        h = (h + rando.random() / 5) % 1
    elif choice == 1:
        s = rando.random()
    else:
        v = rando.random()
    if a is not None:
        return convert_1_to_255((h, s, v, a / 255))
    else:
        return convert_1_to_255((h, s, v))


def color_mind(input):
    print(input)
    url = "http://colormind.io/api/"
    data = {
        "model": "default",
        "input": [list(x) if type(x) is tuple else x for x in input]
    }
    data = str(data).replace("'", '"')
    r = requests.post(url, data=data)
    ret = [tuple(x) for x in r.json()["result"]]
    ret = ret * (1 + len(input) // 5)
    return ret


def average_color(colors: Union[Iterator | Image.Image], mode: str = "RGBA", alpha: Image.Image = None) -> Tuple[int, ]:
    """Averages out several colors.

    Parameters:
    colors (list | PIL.Image): List of color tuples, or an image.

    Returns:
    tuple: Average color."""

    ret = []
    if isinstance(colors, Image.Image):
        mode = colors.getbands()
        colors = list(colors.getdata())
    mode = mode[:len(colors[0])]

    if hasattr(colors[0],'__iter__'):
        weights = None
        a_band = None
        if alpha is not None:
            alpha_data = alpha.getdata()
            weights = [color / 255 for color in alpha_data]
        elif "A" in mode:
            a_band = mode.find("A")
            weights = [color[a_band] / 255 for color in colors]

        for index, band in zip(range(len(colors[0])), mode):
            if band == "H":
                ret.append(average_hue(colors))
            else:
                if weights is not None:
                    print(set(weights))
                    ret.append(numpy.average(
                        list(x[index] for x in colors),
                        weights=weights
                    ))
                else:
                    ret.append(round(mean(x[index] for x in colors)))

        return tuple(ret)
    else:
        return mean(colors)

def average_hue(colors: Union[Iterator | Image.Image]) -> int:
    """Gets the average hue from several colors. Weighs pixels differently based on their saturation, value, and alpha.

    Parameters:
    colors (list | PIL.Image): List of color tuples, or an image.

    Returns:
    int: Average hue."""
    if isinstance(colors, Image.Image):
        colors = numpy.hstack(colors.convert("HSV").getdata(), colors.getchannel('A').getdata())

    if hasattr(colors[0], '__iter__'):
        weights = []
        for color in colors:
            weights.append(numpy.prod(list(band/255 for band in color[1:])))
        return numpy.average(
            list(color[0] for color in colors),
            weights=weights
        )
    else:
        return mean(colors)


def show_palette(cols, size=64):
    image = Image.new("RGB", (size * len(cols), size))
    draw = ImageDraw.Draw(image)
    for n, x in enumerate(cols):
        draw.rectangle((n * size, 0, (n + 1) * size, size), tuple(x))
    ImageShow.show(image)


def show_palette_cube(cols, divider=2, back=False):
    image = Image.new("RGBA", (256 * 2 // divider, 256 * 3 // divider))
    draw = ImageDraw.Draw(image)

    top = tuple(x // divider for x in (255, 0))
    print(top)
    top_left = tuple(x // divider for x in (0, 127))
    top_right = tuple(x // divider for x in (255 * 2, 127))
    center = tuple(x // divider for x in (255, 255))
    bottom_left = tuple(x // divider for x in (0, 255 + 127))
    bottom_right = tuple(x // divider for x in (255 * 2, 255 + 127))
    bottom = tuple(x // divider for x in (255, 255 * 2))

    draw.polygon([top, top_left, bottom_left, bottom, bottom_right, top_right], fill=(128, 128, 128, 32),
                 outline=(0, 0, 0, 255))
    imagedata = image.load()
    multiplier = -1 if back else 1
    for n, rgb in enumerate(sorted(cols, key=lambda c: multiplier * sum(c))):
        r, g, b = rgb[:3]
        # x = round(0 + r + g)
        # y = round(255+127 - r/2 + g/2 - b)
        # x,y = bottom
        # x += r - g
        # y -= r//2 + g // 2 + b
        x, y = center
        x += (-(255 - r) + (255 - g)) // divider
        y += (-(255 - r) // 2 - (255 - g) // 2 + (255 - b)) // divider
        print(x, y)
        imagedata[x, y] = (r, g, b, 255)

    draw.polygon([top, top_left, center, top_right], fill=None, outline=(0, 0, 0, 255))
    draw.polygon([top_left, center, bottom, bottom_left], fill=None, outline=(0, 0, 0, 255))
    draw.polygon([top_right, center, bottom, bottom_right], fill=None, outline=(0, 0, 0, 255))

    ImageShow.show(image)


def get_color_difference(rgb1, rgb2, bands="rgbh"):
    """Returns an increasingly high number the more different two colors are."""
    total = 0
    # RGB
    for n in range(3):
        total += abs(rgb1[n] - rgb2[n])
    total += abs(get_hue(rgb1) - get_hue(rgb2))
    return total


def get_hue(rgb):
    r, g, b = rgb
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 255 * 2 + rc - bc
    else:
        h = 255 * 4 + gc - rc
    h = (h / 255 * 6) % 255
    return h


def get_color_usage(image: Image):
    """Counts how many times each rgba value is used. Or whatever mode the image is in.

    Parameters:
    image (PIL.Image): The image you want info on.

    Returns:
    dict: rgba(?) as key, quantity as value."""
    image_data = image.load()
    quantities = {}
    for x in range(image.width):
        for y in range(image.height):
            pixel = image_data[x, y]
            quantities.setdefault(pixel, 0)
            quantities[pixel] += 1
    return quantities


def get_most_common_colors(image: Image, fraction=0.1):
    """Returns a list of the most common colors in an image.

    Parameters:
    image (PIL.Image): The image you want info on.
    fraction (float): A decimal between 0 and 1 indicating how far from the most common colors you want.
                      For example, 0.1 returns the fewest colors that, combined, make up 10% of the image.

    Returns:
    list: The most common colors."""
    color_usage = get_color_usage(image)
    total_desired = image.width * image.height * fraction
    total = 0
    most_common = []
    for color, quantity in sorted(color_usage.items(), key=lambda x: -x[1]):
        most_common.append(color)
        total += quantity
        if total >= total_desired:
            break
    return most_common


def get_most_representative_colors(image: Image, common_fraction=0.1, representative_fraction=0.1) -> list[tuple[int, int, int]]:
    """Returns a list of the most "representative" colors in an image, determined by their proximity to the most common
    colors of the image liquid rescaled to half size.

    Parameters:
    image (PIL.Image): The image you want info on.
    common_fraction (float): A decimal between 0 and 1 indicating how far from the most common colors you want.
                      For example, 0.1 returns the fewest colors that, combined, make up 10% of the image.
    representative_fraction (float): A decimal between 0 and 1 indicating how far from the most representative colors you want.
                      For example, 0.1 returns the fewest colors that, combined, make up 10% of the image.

    Returns:
    list: The most representative colors."""
    from zsil.cool_stuff import pil_to_wand, wand_to_pil

    quantities = get_color_usage(image)
    total_desired = image.width * image.height * representative_fraction

    mode = image.mode
    resized_image = image.copy()
    resized_image.thumbnail((64, 64))
    # wandimage = piltowand(resized_image)
    # wandimage.liquid_rescale(width=resized_image.height//2, height=resized_image.height//2)
    # resized_image = wandtopil(wandimage).convert(mode)
    most_common_colors = get_most_common_colors(resized_image, fraction=common_fraction)

    color_not_representativeness = {}  # Increases as a color becomes less representative of the whole
    for color in quantities.keys():
        number_of_bands = len(color)
        color_not_representativeness[color] = 0
        for common_color in most_common_colors:
            color_not_representativeness[color] += get_color_difference(color, common_color)

    representative_colors = []
    total = 0
    for color, not_representativeness in sorted(color_not_representativeness.items()):
        representative_colors.append(color)
        total += quantities[color]
        if total >= total_desired:
            break
    return representative_colors

def tuple_to_hex(color: tuple[int, ]):
    return "".join([f"{x:x}".ljust(2, "0") for x in color])