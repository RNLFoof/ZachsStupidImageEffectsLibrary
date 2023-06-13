import math
import os
import random
from re import match, IGNORECASE
from statistics import mean
from typing import Iterable

import numpy as np
from PIL import Image, ImageChops, ImageMath, ImageFilter
from PIL import ImageDraw

import colors
from internal import lengthdir_x, lengthdir_y, getdistancestopoints

import io

from PIL import ImageFont
from wand.image import Image as WandImage

import scipy.spatial as spatial
from typing import Sequence

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


def outline(img, radius, color, retonly=False, pattern=None, minradius=None, maxradius=None, skipchance=0,
            lowestradiusof=1, colorrandomize=0, positiondeviation=0, shape=SHAPE_CIRCLE, shaperotation=0, breeze={}):
    """Draws an outline around an image with a transparent background.

    Parameters:
    img (PIL.Image): The image to draw the outline around.
    radius (int): How many pixels out the outline stretches.
    aaaaaaaaaaaa I care so little aaaaaaaaaa

    Returns:
    int:Returning value

   """
    if minradius is None:
        minradius = radius
    if maxradius is None:
        maxradius = radius
    if pattern:
        colorrandomize = 0

    breeze.setdefault("up", 0)
    breeze.setdefault("down", 0)
    breeze.setdefault("left", 0)
    breeze.setdefault("right", 0)

    outline = Image.new("RGBA", img.size)
    outlinedraw = ImageDraw.Draw(outline)
    imgdata = img.load()

    bbox = img.getbbox()
    if bbox is not None:
        points = [(x, y) for x in range(bbox[0], bbox[2]) for y in range(bbox[1], bbox[3])]
        if colorrandomize:
            random.shuffle(points)

        for x, y in points:
            if imgdata[x, y][3] > 127:
                radius = random.uniform(minradius, maxradius)
                if radius == 0:
                    continue
                if skipchance and random.random() <= skipchance:
                    radius = minradius
                else:
                    for _ in range(lowestradiusof - 1):
                        radius = min(radius, random.uniform(minradius, maxradius))
                    workingdev = random.uniform(-positiondeviation, positiondeviation)
                    # It's breezy.
                    if minradius == maxradius:
                        breezeradiusfactor = 1
                    else:
                        breezeradiusfactor = (1 - (radius - minradius) / (maxradius - minradius))
                    workingdev += workingdev * (breeze["left"] if workingdev < 0 else breeze["right"]) \
                                  * breezeradiusfactor \
                                  * min([random.random() for _ in range(3)])
                    x += workingdev

                    workingdev = random.uniform(-positiondeviation, positiondeviation)
                    workingdev += workingdev * (breeze["up"] if workingdev < 0 else breeze["down"]) \
                                  * breezeradiusfactor \
                                  * min([random.random() for _ in range(3)])
                    y += workingdev

                if colorrandomize:
                    currentcolor = tuple(
                        min(255, max(0, c - colorrandomize + random.randint(0, colorrandomize * 2))) for c in color)
                else:
                    currentcolor = color

                simpleshape(outlinedraw, (x, y), radius, currentcolor, shape, rotation=shaperotation)
        if pattern:
            outline = transferalpha(outline, pattern)

    if not retonly:
        outline.alpha_composite(img)
    return outline


def inneroutline(img, radius, color, retonly=False, pattern=None):
    # Create a slightly larger image that can for sure hold the white outline coming up.
    slightlylargerimg = img.crop((-1, -1, img.width + 1, img.height + 1))

    # Get a white silhouette of the slightly larger img
    a = roundalpha(slightlylargerimg).getchannel("A")
    bwimg = Image.merge("RGBA", (a, a, a, a))

    # Get a white outline of the image
    outline(bwimg, 1, (0, 0, 0, 0))
    thinblackoutline = bwimg.getchannel("R")
    two = Image.eval(thinblackoutline, lambda x: 255 - x)  # Thin white outline
    two = Image.merge("RGBA", (two, two, two, two))

    # Apply the outline to this line, get only the lowest alphas
    inneroutline = outline(two, radius, color, retonly=True, pattern=pattern)
    inneroutline = lowestoftwoalphaas(inneroutline, slightlylargerimg)

    # Convert the slightly larger img back to its original size
    inneroutline = inneroutline.crop((1, 1, inneroutline.width - 1, inneroutline.height - 1))
    print(img.size, inneroutline.size)

    # Return
    if not retonly:
        img.alpha_composite(inneroutline)
        inneroutline = img
    return inneroutline


def transferalpha(alphaman, colorman):
    colorman = colorman.resize(alphaman.size)
    a = alphaman.getchannel("A")
    r, g, b = colorman.split()[:3]
    return Image.merge("RGBA", (r, g, b, a))


def roundalpha(img):
    r, g, b, a = img.split()
    a = Image.eval(a, lambda x: round(x / 255) * 255)
    return Image.merge("RGBA", (r, g, b, a))


def threshholdalpha(img, threshhold):
    r, g, b, a = img.split()
    a = Image.eval(a, lambda x: 0 if x < threshhold else 255)
    return Image.merge("RGBA", (r, g, b, a))


def lowestoftwoalphaas(returnme, otherimage):
    a = returnme.getchannel("A")
    b = otherimage.getchannel("A")
    a = ImageMath.eval("convert(min(a, b), 'L')", a=a, b=b)
    r, b, g, nerd = returnme.split()
    return Image.merge("RGBA", (r, g, b, a))


def indent(img):
    # Open
    indentsdir = "images/indents"
    while True:
        indentfile = random.choice(os.listdir(indentsdir))
        indentimg = Image.open(os.path.join(indentsdir, indentfile)).convert("L")
        if indentimg.size[0] >= img.size[0] and indentimg.size[1] >= img.size[1]:
            break
    if indentimg.size[0] > indentimg.size[1]:
        indentimg.thumbnail((999999, max(img.size) * 2))
    else:
        indentimg.thumbnail((max(img.size) * 2, 999999))

    # Crop
    startx = random.randint(0, indentimg.size[0] - img.size[0])
    starty = random.randint(0, indentimg.size[1] - img.size[1])
    indentimg = indentimg.crop((startx, starty, startx + img.size[0], starty + img.size[1]))

    # Get minmax
    mincolor, maxcolor = indentimg.getextrema()
    colordif = maxcolor - mincolor

    # Stretch the values
    disfromcenter = 255
    indentimg = Image.eval(indentimg,
                           lambda x: ((x - mincolor) / colordif * disfromcenter) - (disfromcenter // 2) + 127)

    indentimgdata = indentimg.load()
    l = []
    for x in range(indentimg.size[0]):
        for y in range(indentimg.size[1]):
            l.append([x, y, indentimgdata[x, y]])
    l = sorted(l, key=lambda x: x[2])
    for n, z in enumerate(l):
        x, y, cum = z
        indentimgdata[x, y] = int(n / len(l) * 255)
    # indentimg.show()

    indentimg = indentimg.convert("RGBA")
    r, g, b, a = indentimg.split()
    a = Image.eval(r, lambda x: (
            (abs(x - 127) / (disfromcenter // 2)) * 2  # Convert black and white to distances from center
            * 16 ** 2  # Pronounce the tip
            / 256 * 32  # max
        #  //16*16  # Round
    ))
    bwband = Image.eval(r, lambda x: round(x / 255) * 255)
    indentimg = Image.merge("RGBA", (bwband, bwband, bwband, a))

    invertband = Image.eval(r, lambda x: 255 - x)
    indentimginverted = Image.merge("RGBA", (invertband, invertband, invertband, a))
    indentimginverted = ImageChops.offset(indentimginverted, 5, 5)
    img = img.convert("RGBA")
    a = img.getchannel("A")
    img.alpha_composite(indentimg)
    r, g, b, WAAAAAA = img.split()
    img = Image.merge("RGBA", (r, g, b, a))

    return img


def offsetedge(img, xoff, yoff):
    img = roundalpha(img)
    a = img.getchannel("A")
    allwhiteimg = Image.new("L", a.size, 255)
    allblackimg = Image.new("L", a.size, 0)
    w = Image.merge("RGBA", (allwhiteimg, allwhiteimg, allwhiteimg, a))
    b = Image.merge("RGBA", (allblackimg, allblackimg, allblackimg, a))
    b = ImageChops.offset(b, xoff, yoff)
    w.alpha_composite(b)
    allblackimg = allblackimg.convert("RGBA")
    allblackimg.alpha_composite(w)
    thisbandisgoingtobeeveryband = allblackimg.getchannel("R")
    ret = Image.merge("RGBA", (thisbandisgoingtobeeveryband, thisbandisgoingtobeeveryband, thisbandisgoingtobeeveryband,
                               thisbandisgoingtobeeveryband))
    return ret


def croptocontent(img, forcetop=None, forceleft=None, forcebottom=None, forceright=None):
    """Returns an image cropped to its bounding box.

    Parameters:
    img (PIL.Image): The image to be cropped.
    forcetop (int): Replaces the top of the bounding box.
    forceleft (int): Replaces the left of the bounding box.
    forcebottom (int): Replaces the bottom of the bounding box.
    forceright (int): Replaces the right of the bounding box.

    Returns:
    (PIL.Image): Cropped image."""
    # Get bounding box
    bb = img.getbbox()
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
    if forcetop is not None or forceleft is not None or forcebottom is not None or forceright is not None:
        # Check each possible replacement and use it if needed
        for force, index in (
                (forceleft, 0),
                (forcetop, 1),
                (forceright, 2),
                (forcebottom, 3),
        ):
            if force is not None:
                bb[index] = force
    # Crop and return
    img = img.crop(bb)
    return img


def resizeandcrop(img, size):
    """Matches one side by resizing and the other by cropping."""
    oldwidth, oldheight = img.size
    newwidth, newheight = size

    widthmultiplier = newwidth / oldwidth
    heightmultiplier = newheight / oldheight
    multiplier = max(widthmultiplier, heightmultiplier)

    stretchsize = (round(oldwidth * multiplier), round(oldheight * multiplier))
    stretchedimg = img.resize(stretchsize)

    stretchedwidth, stretchedheight = stretchedimg.size
    offsetx = abs(newwidth - stretchedwidth) // 2
    offsety = abs(newheight - stretchedheight) // 2
    croppedimg = stretchedimg.crop(box=(offsetx, offsety, offsetx + newwidth, offsety + newheight))
    return croppedimg


def shading(img, offx, offy, size, shrink, growback, color, alpha, blockermultiplier=1, blurring=False):
    """Outline-based shading."""
    hasntblurredyet = True
    # Get side to shade, and also the opposite side
    light = offsetedge(img, offx, offy)
    dark = lowestoftwoalphaas(Image.new("RGBA", light.size, (0, 0, 0, 255)), offsetedge(img, -offx, -offy))
    # Make it bigger, one at a time, from both sides. The dark cancels out the light.
    # The lights from each step are added together.
    lighttomakeclonesof = light.copy()
    for x in range(1, size + 1):
        workinglight = lighttomakeclonesof.copy()
        workingdark = dark.copy()
        workinglight = outline(workinglight, x, (255, 255, 255, 255))
        workingdark = outline(workingdark, x * blockermultiplier, (0, 0, 0, 255))
        workinglight.alpha_composite(workingdark)
        a = workinglight.getchannel("R")
        light.alpha_composite(Image.merge("RGBA", (a, a, a, a)))
    # Blur?
    if not shrink and not growback and blurring:
        light = light.filter(ImageFilter.GaussianBlur(radius=5))
        hasntblurredyet = False
    light = lowestoftwoalphaas(light, img)
    # Shrink it down
    if shrink:
        light = inneroutline(light, shrink, (0, 0, 0, 255))
    # Remove stray alpha nonsense
    allblack = Image.new("RGBA", light.size, color=(0, 0, 0, 255))
    allblack.alpha_composite(light)
    r = allblack.getchannel("R")
    light = Image.merge("RGBA", (r, r, r, r))
    # Blur?
    if not growback and blurring and hasntblurredyet:
        light = light.filter(ImageFilter.GaussianBlur(radius=5))
        hasntblurredyet = False
    # Grow it back
    if growback:
        light = outline(light, growback, (255, 255, 255, 255))
    # Remove stray alpha nonsense 2
    r = light.getchannel("R")
    lighttrans = Image.eval(r, lambda x: min(x, alpha))
    # Set color and alpha
    light = Image.merge("RGBA", (r, r, r, lighttrans))
    colorimg = Image.new("RGBA", light.size, color)
    light = lowestoftwoalphaas(colorimg, light)
    # Blur?
    if blurring and hasntblurredyet:
        light = light.filter(ImageFilter.GaussianBlur(radius=5))
    # Overlay and return
    img.alpha_composite(light)
    return img


def sharplight(img, disout, size, blockermultiplier=1, blurring=False):
    """Sticks out from the edge, not rounded."""
    return shading(img, 1, 1, size + disout, disout, 0, colors.randomwhite(), 192, blockermultiplier=blockermultiplier,
                   blurring=blurring)


def roundedlight(img, disout, size, blockermultiplier=1, blurring=False):
    """Sticks out from the edge, rounded."""
    return shading(img, 1, 1, size + disout, ((size + disout) / 2) - 1, disout / 2 - 1, colors.randomwhite(), 127,
                   blockermultiplier=blockermultiplier, blurring=blurring)


def edgelight(img, size, blockermultiplier=1, blurring=False):
    """Doesn't stick out from the edge."""
    return shading(img, 1, 1, size, 0, 0, colors.randomwhite(), 127, blockermultiplier=blockermultiplier,
                   blurring=blurring)


def shadow(img, size, blockermultiplier=1, blurring=False):
    return shading(img, -1, -1, size, 0, 0, (0, 0, 0), 64, blockermultiplier=blockermultiplier, blurring=blurring)


def shadingwrapper(img):
    choice = random.randint(0, 4)
    blurring = random.randint(0, 1)
    if choice == 0:
        img = sharplight(img, 4, 7, blurring=blurring)
        img = shadow(img, 7, blurring=blurring)
    elif choice == 1:
        img = roundedlight(img, 4, 7, blurring=blurring)
        img = shadow(img, 7, blurring=blurring)
    elif choice == 2:
        img = edgelight(img, 7, blurring=blurring)
        img = shadow(img, 7, blurring=blurring)
    elif choice == 3:
        img = edgelight(img, 20, blurring=blurring)
        img = shadow(img, 20, blurring=blurring)
    elif choice == 4:
        img = edgelight(img, 20, blurring=blurring, blockermultiplier=2)
        img = shadow(img, 20, blurring=blurring, blockermultiplier=2)
    return img


def directionalshading(img):
    """Draws gradient-y shading based on direction from the light source. Does this by repeatedly drawing the largest
    still-possible shortest-distance lines from inside the structure to an outline along the outside."""
    from ZachsStupidImageLibrary.analysis import get_all_opaque_pixels, getedgepixels, getcenterpixels

    # Maybe don't force this?
    # img = threshholdalpha(img, 127)

    # Get possible endpoints
    edgeendpoints = getedgepixels(img)
    centerendpoints = getcenterpixels(img)
    bothendpoints = centerendpoints | edgeendpoints

    # Get possible starting points
    startpoints = get_all_opaque_pixels(img)
    # startpoints -= bothendpoints

    # Get potential lines
    edgepotentiallines = {}
    centerpotentiallines = {}
    for potentiallines, endpoints in [
        (edgepotentiallines, edgeendpoints),
        (centerpotentiallines, centerendpoints),
    ]:
        for potentialline in getdistancestopoints(startpoints, endpoints):
            potentiallines[potentialline.startpoint] = potentialline
    allpotentiallines = {**edgepotentiallines, **centerpotentiallines}

    # Draw along the longest potential lines
    shadingcolors = Image.new("LA", img.size)
    draw = ImageDraw.Draw(shadingcolors)
    n = 0
    goalpoints = len(startpoints | bothendpoints)
    uncombinedpoints = {}
    while goalpoints * 0.6 > len(get_all_opaque_pixels(shadingcolors)) and n < 100:
        print(len(get_all_opaque_pixels(shadingcolors)), goalpoints)
        # Get current longest
        # longest = -1
        # for startpoint in startpoints:
        #     potentialline = edgepotentiallines[startpoint]
        #     longest = max(longest, potentialline.dis)
        #     potentialline = centerpotentiallines[startpoint]
        #     longest = max(longest, potentialline.dis)
        # if longest <= 8:
        #     break
        # Draw on all of that length
        for startpoint in list(startpoints):
            edgepotentialline = edgepotentiallines[startpoint]
            centerpotentialline = centerpotentiallines[startpoint]
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
                # draw.line((potentialline.startpoint, potentialline.endpoint), (using, 2))
                # draw.point(potentialline.startpoint, (using, 255))
                difx = potentialline.startpoint[0] - potentialline.endpoint[0]
                deltax = abs(difx)
                dify = abs(potentialline.startpoint[1] - potentialline.endpoint[1])
                deltay = abs(dify)
                maxdelta = max(deltax, deltay)
                for n in range(maxdelta + 1):
                    point = (
                        round(potentialline.startpoint[0] - difx / maxdelta * n),
                        round(potentialline.startpoint[1] - dify / maxdelta * n),
                    )
                    print(potentialline.startpoint, point, potentialline.endpoint)
                    uncombinedpoints.setdefault(point, [])
                    uncombinedpoints[point].append(using)
                startpoints.remove(potentialline.startpoint)
                # n += 1
                # if n**0.5 == round(n**0.5):
                #     img.alpha_composite(shadingcolors.convert("RGBA"))
                #     img.show()
        # Remove everything that's now filled in
        # shadingcolorspoints = getallalphapoints(shadingcolors)
        # print(potentialline.startpoint in shadingcolorspoints)
        # startpoints -= getallalphapoints(shadingcolors)
        # print(len(startpoints))
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
    shadingcolors.putalpha(img.split()[img.getbands().index("A")])
    img.alpha_composite(shadingcolors.convert("RGBA"))
    img.show()


def metallicdirectionalshading(img):
    """Draws gradient-y shading based on direction from the light source. Does this by repeatedly drawing the largest
    still-possible shortest-distance lines from inside the structure to an outline along the outside."""
    from ZachsStupidImageLibrary.analysis import get_all_opaque_pixels, getedgepixels, getcenterpixels

    # Maybe don't force this?
    # img = threshholdalpha(img, 127)

    # Get possible endpoints
    edgeendpoints = getedgepixels(img)
    centerendpoints = getcenterpixels(img)
    bothendpoints = centerendpoints | edgeendpoints

    # Get possible starting points
    startpoints = get_all_opaque_pixels(img)
    # startpoints -= bothendpoints

    # Get potential lines
    edgepotentiallines = {}
    centerpotentiallines = {}
    for potentiallines, endpoints in [
        (edgepotentiallines, edgeendpoints),
        (centerpotentiallines, centerendpoints),
    ]:
        for potentialline in getdistancestopoints(startpoints, endpoints):
            potentiallines[potentialline.startpoint] = potentialline
    allpotentiallines = {**edgepotentiallines, **centerpotentiallines}

    # Draw along the longest potential lines
    shadingcolors = Image.new("LA", img.size)
    draw = ImageDraw.Draw(shadingcolors)
    n = 0
    goalpoints = len(startpoints | bothendpoints)
    while goalpoints * 0.6 > len(get_all_opaque_pixels(shadingcolors)) and n < 100:
        print(len(get_all_opaque_pixels(shadingcolors)), goalpoints)
        # Get current longest
        # longest = -1
        # for startpoint in startpoints:
        #     potentialline = edgepotentiallines[startpoint]
        #     longest = max(longest, potentialline.dis)
        #     potentialline = centerpotentiallines[startpoint]
        #     longest = max(longest, potentialline.dis)
        # if longest <= 8:
        #     break
        # Draw on all of that length
        for startpoint in list(startpoints):
            edgepotentialline = edgepotentiallines[startpoint]
            centerpotentialline = centerpotentiallines[startpoint]
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
                draw.line((potentialline.startpoint, potentialline.endpoint), (using, 2))
                # draw.point(potentialline.startpoint, (using, 255))
                startpoints.remove(potentialline.startpoint)
                # n += 1
                # if n**0.5 == round(n**0.5):
                #     img.alpha_composite(shadingcolors.convert("RGBA"))
                #     img.show()
        # Remove everything that's now filled in
        # shadingcolorspoints = getallalphapoints(shadingcolors)
        # print(potentialline.startpoint in shadingcolorspoints)
        # startpoints -= getallalphapoints(shadingcolors)
        # print(len(startpoints))
        n += 1

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

    shadingcolors.putalpha(img.split()[img.getbands().index("A")])
    shadingcolors.show()
    img.alpha_composite(shadingcolors.convert("RGBA"))
    img.show()


def godeepdreamyourself(img, max_dim=None, steps=100):
    """A lazy wrapper for something made by somebody cooler than me."""
    import tensorflow as tf
    import numpy as np

    import IPython.display as display
    import PIL.Image

    # Normalize an image
    def deprocess(img):
        img = 255 * (img + 1.0) / 2.0
        return tf.cast(img, tf.uint8)

    # Display an image
    def show(img):
        display.display(PIL.Image.fromarray(np.array(img)))

    def backtopil(img):
        return PIL.Image.fromarray(np.array(img))

    # Downsizing the image makes it easier to work with.
    img = img.convert("RGB")
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    original_img = np.array(img)

    # show(original_img)
    display.display(display.HTML(
        'Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # Maximize the activations of these layers
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    def calc_loss(img, model):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)
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
        def __call__(self, img, steps, step_size):
            print("Tracing")
            loss = tf.constant(0.0)
            for n in tf.range(steps):
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img`
                    # `GradientTape` only watches `tf.Variable`s by default
                    tape.watch(img)
                    loss = calc_loss(img, self.model)

                # Calculate the gradient of the loss with respect to the pixels of the input image.
                gradients = tape.gradient(loss, img)

                # Normalize the gradients.
                gradients /= tf.math.reduce_std(gradients) + 1e-8

                # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
                # You can update the image by directly adding the gradients (because they're the same shape!)
                img = img + gradients * step_size
                img = tf.clip_by_value(img, -1, 1)

            print("Done tracing.")
            return loss, img

    deepdream = DeepDream(dream_model)

    def run_deep_dream_simple(img, steps=100, step_size=0.01):
        # Convert from uint8 to the range expected by the model.
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.convert_to_tensor(img)
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

            loss, img = deepdream(img, run_steps, tf.constant(step_size))

            display.clear_output(wait=True)
            show(deprocess(img))
            print("Step {}, loss {}".format(step, loss))

        result = deprocess(img)
        display.clear_output(wait=True)
        # show(result)

        im = backtopil(result)

        return im

    dream_img = run_deep_dream_simple(img=original_img,
                                      steps=steps, step_size=0.01)
    return dream_img


def draw_star(draw, xy, dir, points, innerradius, outerradius, fill):
    """Draw a star. Meant to invoke PIL's geometry drawing functions."""
    x, y = xy
    inneroffset = 360 / points / 2
    for d in range(points):
        workingdir = dir + (360 / points * d)
        draw.polygon([
            (x + lengthdir_x(innerradius, workingdir - inneroffset),
             y + lengthdir_y(innerradius, workingdir - inneroffset)),
            (x + lengthdir_x(outerradius, workingdir), y + lengthdir_y(outerradius, workingdir)),
            (x + lengthdir_x(innerradius, workingdir + inneroffset),
             y + lengthdir_y(innerradius, workingdir + inneroffset)),
        ],
            fill=fill)
    draw.ellipse((x - innerradius, y - innerradius, x + innerradius, y + innerradius), fill)


def repaint(img, function, growthchance=0.5, recalculationrate=10):
    """Takes an image(A), and creates a blank image(B). Until none exist, picks pixels that are transparent(alpha 0) in
    Image B but not image A, and runs function on Image B using the information from that pixel on Image A.
    I suppose it could be thought of repainting img using function as the brush.

    Parameters:
    img (PIL.Image): The original image.
    function (function): A function that takes the following arguments:
        img (PIL.ImageDraw.ImageDraw): Image B.
        xy (Tuple): Where to draw on Image B.
        size (int): How large the shape to draw on Image B is.
        color (Tuple): What color to draw on Image B.
    growthchance (float): Chance of higher sizes. Must be less than one.
    recalculationrate (int): How many "strokes" will be done at once before checking which pixels are still transparent.

    Returns:
    tuple: All coordinates to which function returned true."""
    # Make sure the growth chance isn't 1 or greater to avoid infinite loops
    if growthchance >= 1:
        raise Exception("Growth chance must be less than 1.")

    # Set shit up
    originalimg = img.convert("RGBA")
    newimg = Image.new("RGBA", originalimg.size)
    newimg.thumbnail()
    originalimgdata = originalimg.load()
    newimgdata = newimg.load()
    originaltransparentpixels = set(pixelfilter(lambda x: x[3] == 0, originalimg, imgdata=originalimgdata))

    # Main loop
    while True:
        # Get remaining transparent pixels
        remainingtransparentpixels = list(
            set(pixelfilter(lambda x: x[3] == 0, newimg, imgdata=newimgdata)) \
            - originaltransparentpixels
        )

        # Bail if finished
        if not remainingtransparentpixels:
            break

        # Iterate through them
        random.shuffle(remainingtransparentpixels)
        for xy in remainingtransparentpixels[:min(len(remainingtransparentpixels), recalculationrate)]:
            # Get size
            size = 1
            while random.random() <= growthchance:
                size += 1

            # Get color
            color = originalimgdata[xy[0], xy[1]]

            # Do the function
            function(newimg, xy, size, color)

    # Return
    return newimg


def simpleshape(imgordraw, xy, radius, color, shape, rotation=None):
    """Wrapper for more complicated shape drawing functions so they can be more easily interchanged.

    Parameters:
    img (PIL.ImageDraw.ImageDraw or PIL.Image): The Image or ImageDraw to draw on.
    xy (Tuple): Center point of the shape.
    radius (float): Distance from center the shape reaches.
    color (Tuple): Color/fill of the shape.
    rotation: Angle of the shape. Random if not provided.

    Returns:
    PIL.ImageDraw.ImageDraw or PIL.Image: img. Doesn't make a copy, only returns for convenience."""
    x, y = xy
    if rotation is None:
        rotation = random.random() * 360

    if type(imgordraw) == Image.Image:
        draw = ImageDraw.Draw(imgordraw)
    else:
        draw = imgordraw

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
    return imgordraw


def predictthumbnailsize(originalsize, newsize):
    """Figures out what size an image of originalsize would become if you used it to make a thumbnail of newsize.

    Parameters:
    originalsize (Tuple): Original size(duh).
    newsize (Tuple): New size(duh).

    Returns:
    Tuple: Predicted thumbnail size."""
    originalwidth, originalheight = originalsize
    x, y = map(math.floor, newsize)
    if x >= originalwidth and y >= originalheight:
        return originalsize

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    # preserve aspect ratio
    aspect = originalwidth / originalheight
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(
            x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n)
        )
    return x, y


def multiline_textsize_but_it_works(text, font, maxwidth=1500):
    # Should probably implement other parameters.
    img = Image.new('RGBA', (maxwidth, 1200), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((0, 0), text, font=font, fill="black", align="center")

    bbox = img.getbbox()
    if bbox:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    else:
        return 0, 0


def autobreakingtext(imgordraw, allowedwidth, xy, text, fill=None, font=None, anchor=None, spacing=4, align='left',
                     direction=None, features=None, language=None, stroke_width=0, stroke_fill=None,
                     embedded_color=False):
    """Draws PIL text and adds line breaks whenever it gets too wide.

    Parameters:
    originalsize (Tuple): Original size(duh).
    newsize (Tuple): New size(duh).

    Returns:
    Tuple: Predicted thumbnail size."""
    if type(imgordraw) == Image.Image:
        draw = ImageDraw.Draw(imgordraw)
    else:
        draw = imgordraw

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
        if multiline_textsize_but_it_works(tempteststring, font, maxwidth=allowedwidth + 20)[0] <= allowedwidth:
            teststring = tempteststring
            usingstring += " " + word
        else:
            teststring = word
            usingstring += "\n" + word

    # Actually draw
    draw.multiline_text(xy, usingstring, fill=fill, font=font)


def piltowand(img):
    stream = io.BytesIO()
    img.save(stream, format="PNG")
    return WandImage(blob=stream.getvalue())


def wandtopil(img):
    return Image.open(io.BytesIO(img.make_blob("png")))


def enlargablethumbnail(img, size, resample=None):
    """Same as PIL.Image.thumbnail, but can also grow to fit the required size."""
    if img.size[0] > size[0] or img.size[1] > size[1]:
        img.thumbnail(size, resample=resample)
        return img
    possiblemultipliers = sorted([
        size[0] / img.size[0],
        size[1] / img.size[1],
    ])[::-1]
    for multiplier in possiblemultipliers:
        newwidth = round(img.size[0] * multiplier)
        newheight = round(img.size[1] * multiplier)
        if newwidth <= size[0] and newheight <= size[1]:
            return img.resize((newwidth, newheight), resample=resample)


def squarecrop(img):
    """Adds extra size on both sides of an image to form a square."""
    if img.width > img.height:
        # These are separate so that one can be one pixel larger if it's an odd number
        extrasizeononeside = (img.width - img.height) // 2
        extrasizeontheother = (img.width - img.height) - extrasizeononeside
        img = img.crop((0, -extrasizeononeside, img.width, img.height + extrasizeontheother))
    elif img.width < img.height:
        # These are separate so that one can be one pixel larger if it's an odd number
        extrasizeononeside = (img.height - img.width) // 2
        extrasizeontheother = (img.height - img.width) - extrasizeononeside
        img = img.crop((-extrasizeononeside, 0, img.width + extrasizeontheother, img.height))
    return img


def textimage(text):
    """Generates a simple square image with text in the middle."""
    img = Image.new("RGBA", (1200, 1200))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((5, 5), text)
    img = croptocontent(img)
    img = outline(img, 1, (0, 0, 0))
    img = squarecrop(img)
    return img


def dynamiclysizedtextimage(text: str, size: tuple[int, int], font=None, fill="black"):
    """Generates an image with text whose lines are resized such that the longer lines are smaller."""
    width, height = size
    textimg = Image.new("RGBA", size)

    # Set up the lines
    textlinedata = []
    for pos, textline in enumerate(text.split("\n")):
        textlinedata.append({
            "s": textline,
            "pos": pos,
            "baby": match(r"^(\s|-|of|or|in|my|o|o'|the|\.|mr|mrs|\w{,2}\.)*$", textline, flags=IGNORECASE) is not None
        })
    textlinedata.sort(key=lambda x: -len(x["s"]))
    textlinedata.sort(key=lambda x: -x["baby"])
    print(textlinedata)

    # Generate thumbnails
    roomtaken = 0
    buildup = {}
    normallinesremaining = len(list(filter(lambda x: not x["baby"], textlinedata)))
    babylinesremaining = len(textlinedata) - normallinesremaining
    babylinedivider = 3
    for n, tld in enumerate(textlinedata):
        currentlinedivider = babylinedivider if tld["baby"] else 1
        roomremaining = height - roomtaken
        spaceforxlinesrequired = normallinesremaining + (babylinedivider / babylinedivider)

        textlineimg = Image.new("RGBA", (width * 18, height * 4))
        textlinedraw = ImageDraw.Draw(textlineimg)
        textlinedraw.text((textlineimg.size[0] // 2, textlineimg.size[1] // 2), tld["s"], font=font, fill=fill,
                          align="center")
        textlineimg = croptocontent(textlineimg)
        textlineimg.thumbnail((width, roomremaining / spaceforxlinesrequired // currentlinedivider),
                              resample=Image.LANCZOS)

        buildup[tld["pos"]] = textlineimg
        roomtaken += textlineimg.size[1]
        if tld["baby"]:
            babylinesremaining -= 1
        else:
            normallinesremaining -= 1

    # Combine them
    drawaty = 0
    for n in range(len(buildup)):
        textlineimg = buildup[n]
        textimg.paste(textlineimg, (abs(textlineimg.size[0] - textimg.size[0]) // 2, drawaty))
        drawaty += textlineimg.size[1] + 5

    return textimg


def split_image():
    pass


def shift_hue_by(image: Image, by: int) -> Image:
    image = image.convert("HSV")
    h, s, v = image.split()
    h = Image.eval(h, lambda x: (x + by) % 255)
    return Image.merge("HSV", (h, s, v))


def shift_hue_towards(image: Image, towards: int) -> Image:
    image = image.convert("HSV")
    average_hue = colors.average_color(image.getchannel("H"))
    by = towards - average_hue
    return shift_hue_by(image, by)


def shift_bands_by(image: Image, by: Iterable[int]) -> Image:
    bands = []
    for index, band in enumerate(image.split()):
        if image.mode[index] == "H":  # Pretty sure H is the only one that "loops"?
            band = Image.eval(band, lambda x: (x + by[index]) % 255)
        else:
            band = Image.eval(band, lambda x: min(255, max(0, (x + by[index]))))
        bands.append(band)
    return Image.merge(image.mode, bands)


def shift_bands_towards(image: Image, towards: Iterable[int]) -> Image:
    average_color = colors.average_color(image)
    return shift_bands_by(image, [x[0] - x[1] for x in zip(towards, average_color)])


def buttonize(img: Image, distance_from_edge: int = 5, outermost_color=(0, 0, 0), innermost_color=(255, 255, 255)):
    for current_distance_from_edge in list(range(distance_from_edge))[::-1]:
        img = inneroutline(img, current_distance_from_edge, colors.mergecolors(outermost_color, innermost_color,
                                                                               current_distance_from_edge / distance_from_edge))
    return img


def voronoi_edges(img: Image, points: Sequence[tuple[int, int]], color=(0, 0, 0), width=1):
    voronoi = spatial.Voronoi(points, furthest_site=False)
    draw = ImageDraw.Draw(img)
    for region in voronoi.regions:
        if -1 in region:
            continue
        points_on_line = []
        for vert_index in region:
            points_on_line.append(tuple(voronoi.vertices[vert_index]))
        if len(points_on_line) == 0:
            continue
        print(points_on_line)
        points_on_line.append((points_on_line[0]))
        draw.line(points_on_line, color, width)
    # for point in points:
    #     draw.ellipse([
    #         point[0]-5,
    #         point[1]-5,
    #         point[0]+5,
    #         point[1]+5,
    #     ], 0XFF0000)
    return img

if __name__ == "__main__":
    img = Image.new("RGB", (1272, 1272), 0xffffff)
    layers = 12
    sides = 16
    points = []
    for layer in range(1, layers):
        distance = (1272**2*2)**0.5/layers*layer/2
        for side in range(sides):
            angle = math.pi * 2 / sides * (side + (1/2*(layer%2)))
            points.append( (
                1272/2 + math.sin(angle) * distance,
                1272/2 + math.cos(angle) * distance,
            ))
    voronoi_edges(img, points, width=10)
    img.save("crazy.png")
    exit()
    path = r"S:\Models\Custom\Gloomhaven Tokens\buttonized\attempt"
    for file in os.listdir(path):
        img = Image.open(os.path.join(path, file))
        buttonize(img, 50).save(os.path.join(path, "modifiedd_" + file))
    exit()
    dynamiclysizedtextimage("test\nertdjhnguerdhjguierjhuijejsrih", (50, 50)).show()
    exit()
    # import numpy as np
    # import cv2
    #
    # v = cv2.imread(r'S:\Code\django\clock\app\static\app\wallpapers\chosen\gen\hippo.png', 0)
    # v = v.astype(np.uint)
    # s = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    # s = cv2.Laplacian(s, cv2.CV_16S, ksize=3)
    # s = cv2.convertScaleAbs(s)
    # cv2.imshow('nier',s)
    #
    # cv2.waitKey(0)
    #
    # import bezier
    # nodes1 = np.asfortranarray([
    #     [0.0, 0.5, 1.0],
    #     [0.0, 1.0, 0.0],
    # ])
    # curve1 = bezier.Curve(nodes1, degree=2)
    #
    # nodes2 = np.asfortranarray([
    # [0.0, 0.25, 0.5, 0.75, 1.0],
    # [0.0, 2.0, -2.0, 2.0, 0.0],
    # ])
    # curve2 = bezier.Curve.from_nodes(nodes2)
    # intersections = curve1.intersect(curve2)
    # intersections
    # np.asfortranarray([[0.31101776, 0.68898224, 0., 1.],
    # [0.31101776, 0.68898224, 0., 1.]])
    # s_vals = np.asfortranarray(intersections[0, :])
    # points = curve1.evaluate_multi(s_vals)
    #
    # import seaborn
    # import matplotlib.pyplot as plt
    # seaborn.set()
    # ax = curve1.plot(num_pts=256)
    # _ = curve2.plot(num_pts=256, ax=ax)
    # lines = ax.plot( points[0, :], points[1, :], marker = "o", linestyle = "None", color = "black")
    # _ = ax.axis("scaled")
    # _ = ax.set_xlim(-0.125, 1.125)
    # _ = ax.set_ylim(-0.0625, 0.625)
    # plt.show()
    #
    # nodes = np.asfortranarray([
    #
    #     [0.0, 0.625, 1.0],
    #
    #     [0.0, 0.5, 0.5],
    #
    # ])
    #
    # curve = bezier.Curve(nodes, degree=2)
    # curve.plot(5)
    # plt.show()
    # exit()
    img = Image.new("RGBA", (1200, 1200))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("segoescb.ttf", 120)
    draw.multiline_text((5, 5), "QBR", font=font, fill=(255, 0, 0))  # QWERTYUIOPqwertyuiop
    img = croptocontent(img)
    # img = outline(img, 1, (0, 0, 0))
    directionalshading(img)
    img.show()
    exit()
    path = r"S:\Code\django\clock\app\static\app\wallpapers\chosen\\"
    imgname = random.choice(os.listdir(path))
    img = Image.open(path + imgname).convert("RGBA")
    # img = enlargablethumbnail(img, (10000,10000), Image.LANCZOS)
    # # power = 8
    # # img.thumbnail((2**power, 2**power))
    # print(img.size)
    # exit()
    # allcolors = [(r,g,b) for r in range(0,256,4) for g in range(0,256,4) for b in range(0,256,4)]
    # colors.showpalettecube(allcolors)
    # colors.showpalettecube(allcolors, back=True)
    # common = colors.getmostcommoncolors(img, 1)
    common = colors.getmostrepresentativecolors(img)
    img.show()
    colors.show_palette_cube(common)
    colors.show_palette_cube(common, back=True)
    # colors.showpalette(colors.getmostrepresentativecolors(img, 0.5, 0.5), 1)
    # repaint(img, lambda newimg, xy, size, color:
    #         newimg.alpha_composite(
    #             shadow(
    #                 edgelight(
    #                 (
    #                     simpleshape
    #                     (
    #                         Image.new("RGBA", img.size),
    #                         xy,
    #                         size*2,
    #                         color,
    #                         SHAPE_DIAMOND,
    #                         rotation=90
    #                     )
    #                 ),
    #                 2
    #             ),
    #             2
    #         )
    # )).show()
