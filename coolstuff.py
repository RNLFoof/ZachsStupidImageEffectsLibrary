import os
import random

from PIL import Image, ImageChops, ImageMath, ImageFilter
from PIL import ImageDraw

import colors
from generalfunctions import lengthdir_x,lengthdir_y

def outline(img, radius, color, retonly=False, pattern=None, minradius=None, maxradius=None, skipchance=0,
            lowestradiusof=1, colorrandomize=0, positiondeviation=0, shape="circle", breeze={}):
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
                    for _ in range(lowestradiusof-1):
                        radius = min(radius, random.uniform(minradius, maxradius))
                    workingdev = random.uniform(-positiondeviation, positiondeviation)
                    # It's breezy.
                    if minradius == maxradius:
                        breezeradiusfactor = 1
                    else:
                        breezeradiusfactor = (1 - (radius-minradius) / (maxradius-minradius))
                    workingdev += workingdev*(breeze["left"] if workingdev < 0 else breeze["right"])\
                                  *breezeradiusfactor\
                                  * min([random.random() for _ in range(3)])
                    x += workingdev

                    workingdev = random.uniform(-positiondeviation, positiondeviation)
                    workingdev += workingdev*(breeze["up"] if workingdev < 0 else breeze["down"])\
                                  *breezeradiusfactor\
                                  * min([random.random() for _ in range(3)])
                    y += workingdev

                if colorrandomize:
                    currentcolor = tuple(min(255, max(0, c-colorrandomize+random.randint(0, colorrandomize*2))) for c in color)
                else:
                    currentcolor = color

                if radius > 0:
                    if shape == "star":
                        draw_star(outlinedraw, (x,y), random.random()*360, 5, radius*random.uniform(0.15, 0.6), radius, currentcolor)
                    elif shape == "circle":
                        outlinedraw.ellipse((x-radius, y-radius, x+radius, y+radius), currentcolor)
                    elif shape == "square":
                        outlinedraw.rectangle((x-radius, y-radius, x+radius, y+radius), fill=currentcolor)
                    elif shape == "hexagon":
                        outlinedraw.regular_polygon((x, y, radius), 6, fill=currentcolor)
                    elif shape == "spinny square":
                        outlinedraw.regular_polygon((x, y, radius), 4, rotation=random.random()*360, fill=currentcolor)
                    elif shape == "many shapes":
                        outlinedraw.regular_polygon((x, y, radius), random.choice([3,4,5,6,8]), rotation=random.random()*360, fill=currentcolor)
                    else:
                        raise Exception("Invalid shape")
        if pattern:
            outline = transferalpha(outline, pattern)

    if not retonly:
        outline.alpha_composite(img)
    return outline

def inneroutline(img, radius, color, retonly=False, pattern=None):
    # Get a white silhouette of the image
    a = roundalpha(img).getchannel("A")
    bwimg = Image.merge("RGBA", (a,a,a,a))

    # Get a white outline of the image
    outline(bwimg, 1, (0,0,0,0))
    thinblackoutline = bwimg.getchannel("R")
    two = Image.eval(thinblackoutline, lambda x: 255 - x)  # Thin white outline
    two = Image.merge("RGBA", (two,two,two,two))

    # Apply the outline to this line, get only the lowest alphas, and return
    inneroutline = outline(two, radius, color, retonly=True, pattern=pattern)
    inneroutline = lowestoftwoalphaas(inneroutline, img)
    if not retonly:
        img.alpha_composite(inneroutline)
        inneroutline = img
    return inneroutline

def pattern(color1, color2):
    dir = "images/patterns"
    file = random.choice(os.listdir(dir))
    ret = Image.open(os.path.join(dir, file)).convert("RGB")
    rgba = list(ret.split())
    for x in range(3):
        rgba[x] = Image.eval(rgba[x], lambda y: ((color1[x]-color2[x])/255*y)+color2[x])
    ret = Image.merge("RGB", rgba)
    return ret

def transferalpha(alphaman, colorman):
    colorman = colorman.resize(alphaman.size)
    a = alphaman.getchannel("A")
    r, g, b = colorman.split()[:3]
    return Image.merge("RGBA", (r,g,b,a))

def roundalpha(img):
    r,g,b,a = img.split()
    a = Image.eval(a, lambda x: round(x / 255) * 255)
    return Image.merge("RGBA", (r,g,b,a))

def threshholdalpha(img, threshhold):
    r,g,b,a = img.split()
    a = Image.eval(a, lambda x: 0 if x < threshhold else 255)
    return Image.merge("RGBA", (r,g,b,a))

def lowestoftwoalphaas(returnme, otherimage):
    a = returnme.getchannel("A")
    b = otherimage.getchannel("A")
    a = ImageMath.eval("convert(min(a, b), 'L')", a=a, b=b)
    r, b, g, nerd = returnme.split()
    return Image.merge("RGBA", (r, g, b ,a))

def indent(img):
    # Open
    indentsdir = "images/indents"
    while True:
        indentfile = random.choice(os.listdir(indentsdir))
        indentimg = Image.open(os.path.join(indentsdir, indentfile)).convert("L")
        if indentimg.size[0] >= img.size[0] and indentimg.size[1] >= img.size[1]:
            break
    if indentimg.size[0] > indentimg.size[1]:
        indentimg.thumbnail((999999, max(img.size)*2))
    else:
        indentimg.thumbnail((max(img.size)*2, 999999))

    # Crop
    startx = random.randint(0, indentimg.size[0]-img.size[0])
    starty = random.randint(0, indentimg.size[1]-img.size[1])
    indentimg = indentimg.crop((startx, starty, startx+img.size[0], starty+img.size[1]))

    # Get minmax
    mincolor, maxcolor = indentimg.getextrema()
    colordif = maxcolor-mincolor

    # Stretch the values
    disfromcenter = 255
    indentimg = Image.eval(indentimg, lambda x: ((x-mincolor)/colordif*disfromcenter)-(disfromcenter//2)+127)

    indentimgdata = indentimg.load()
    l = []
    for x in range(indentimg.size[0]):
        for y in range(indentimg.size[1]):
            l.append( [x, y, indentimgdata[x, y]] )
    l = sorted(l, key=lambda x: x[2])
    for n, z in enumerate(l):
        x, y, cum = z
        indentimgdata[x, y] = int(n/len(l)*255)
    # indentimg.show()

    indentimg = indentimg.convert("RGBA")
    r, g, b, a = indentimg.split()
    a = Image.eval(r, lambda x: ( (abs(x-127)/(disfromcenter//2))*2  # Convert black and white to distances from center
                                  *16**2  # Pronounce the tip
                                  /256*32  # max
                                  #  //16*16  # Round
                                  ))
    bwband = Image.eval(r, lambda x: round(x/255)*255 )
    indentimg = Image.merge("RGBA", (bwband,bwband,bwband,a))

    invertband = Image.eval(r, lambda x: 255-x )
    indentimginverted = Image.merge("RGBA", (invertband,invertband,invertband,a))
    indentimginverted = ImageChops.offset(indentimginverted, 5, 5)
    img = img.convert("RGBA")
    a = img.getchannel("A")
    img.alpha_composite(indentimg)
    r, g, b, WAAAAAA = img.split()
    img = Image.merge("RGBA", (r,g,b,a))

    return img

def offsetedge(img, xoff, yoff):
    img = roundalpha(img)
    a = img.getchannel("A")
    allwhiteimg = Image.new("L", a.size, 255)
    allblackimg = Image.new("L", a.size, 0)
    w = Image.merge("RGBA", (allwhiteimg,allwhiteimg,allwhiteimg, a))
    b = Image.merge("RGBA", (allblackimg,allblackimg,allblackimg, a))
    b = ImageChops.offset(b, xoff, yoff)
    w.alpha_composite(b)
    allblackimg = allblackimg.convert("RGBA")
    allblackimg.alpha_composite(w)
    thisbandisgoingtobeeveryband = allblackimg.getchannel("R")
    ret = Image.merge("RGBA", (thisbandisgoingtobeeveryband,thisbandisgoingtobeeveryband,thisbandisgoingtobeeveryband,thisbandisgoingtobeeveryband))
    return ret

def croptocontent(img):
    bb = img.getbbox()
    img = img.crop(bb)
    return img

def resizeandcrop(img, size):
    oldwidth, oldheight = img.size
    newwidth, newheight = size

    widthmultiplier = newwidth/oldwidth
    heightmultiplier = newheight/oldheight
    multiplier = max(widthmultiplier, heightmultiplier)

    stretchsize = (round(oldwidth*multiplier), round(oldheight*multiplier))
    stretchedimg = img.resize(stretchsize)

    stretchedwidth, stretchedheight = stretchedimg.size
    offsetx = abs(newwidth-stretchedwidth)//2
    offsety = abs(newheight-stretchedheight)//2
    croppedimg = stretchedimg.crop(box=(offsetx, offsety, offsetx+newwidth, offsety+newheight))
    return croppedimg

def shading(img, offx, offy, size, shrink, growback, color, alpha, blockermultiplier=1, blurring=False):
    hasntblurredyet = True
    # Get side to shade, and also the opposite side
    light = offsetedge(img, offx, offy)
    dark = lowestoftwoalphaas(Image.new("RGBA", light.size, (0,0,0,255)), offsetedge(img, -offx, -offy))
    # Make it bigger, one at a time, from both sides. The dark cancels out the light.
    # The lights from each step are added together.
    lighttomakeclonesof = light.copy()
    for x in range(1, size+1):
        workinglight = lighttomakeclonesof.copy()
        workingdark = dark.copy()
        workinglight = outline(workinglight, x, (255, 255, 255, 255))
        workingdark = outline(workingdark, x*blockermultiplier, (0, 0, 0, 255))
        workinglight.alpha_composite(workingdark)
        a = workinglight.getchannel("R")
        light.alpha_composite(Image.merge("RGBA", (a,a,a,a)))
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
        light = outline(light, growback, (255,255,255,255))
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
    return shading(img, 1, 1, size+disout, disout, 0, colors.randomwhite(), 192, blockermultiplier=blockermultiplier,
                   blurring=blurring)

def roundedlight(img, disout, size, blockermultiplier=1, blurring=False):
    return shading(img, 1, 1, size+disout, ((size+disout)/2)-1, disout/2-1, colors.randomwhite(), 127,
                   blockermultiplier=blockermultiplier, blurring=blurring)

def edgelight(img, size, blockermultiplier=1, blurring=False):
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

def draw_star(draw, xy, dir, sides, innerradius, outerradius, fill):
    x, y = xy
    inneroffset=360/sides/2
    for d in range(sides):
        workingdir = dir+(360/sides*d)
        draw.polygon([
            (x + lengthdir_x(innerradius, workingdir-inneroffset), y + lengthdir_y(innerradius, workingdir-inneroffset)),
            (x + lengthdir_x(outerradius, workingdir), y + lengthdir_y(outerradius, workingdir)),
            (x + lengthdir_x(innerradius, workingdir+inneroffset), y + lengthdir_y(innerradius, workingdir+inneroffset)),
             ],
                        fill=fill)
    draw.ellipse((x - innerradius, y - innerradius, x + innerradius, y + innerradius), fill)