import colorsys

import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageShow

import random as rando

def convert1to255(col):
    c = list(col)
    for n, x in enumerate(c):
        c[n] = round(x * 255)
    c = tuple(c)
    return c

def convert255to1(col):
    c = list(col)
    for n, x in enumerate(c):
        c[n] = x / 255
    c = tuple(c)
    return c

def randomwhite(takehuefrom=None):
    if takehuefrom:
        hue = colorsys.rgb_to_hsv(*convert255to1(takehuefrom))[0]
    else:
        hue = None

    if rando.randint(0, 1) and not (hue is not None and (hue < 180/360 or hue > 246/360)):
        if hue is None:
            hue = rando.uniform(180/360, 246/360)
        c = colorsys.hsv_to_rgb(
            hue,
            rando.random()*0.1,
            1 - rando.random()*0.03,
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

def randomblack():
    if rando.randint(0, 1):
        c = colorsys.hsv_to_rgb(
            rando.uniform(180/360, 246/360),
            rando.random()*0.1,
            rando.random()*0.03,
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

def mergecolors(col1, col2, amount=None):
    if amount is None:
        amount = rando.uniform(0.4, 0.6)
    newcolor = []
    for n,x in enumerate(col1):
        newcolor.append(round( x*(1-amount) + col2[n]*amount ))
    return tuple(newcolor)

def variantof(col):
    a = None
    if len(col) > 3:
        a = col[3]
        col = col[:3]
    h, s, v = colorsys.hsv_to_rgb(*convert255to1(col))
    choice = rando.randint(0, 2)
    if choice == 0:
        h = (h + rando.random()/5) % 1
    elif choice == 1:
        s = rando.random()
    else:
        v = rando.random()
    if a is not None:
        return convert1to255((h, s, v, a/255))
    else:
        return convert1to255((h, s, v))

def colormind(inp):
    print(inp)
    url = "http://colormind.io/api/"
    data = {
        "model": "default",
        "input": [list(x) if type(x) is tuple else x for x in inp]
    }
    data = str(data).replace("'",'"')
    #print("sending")
    r = requests.post(url, data=data)
    #print("hey guys")
    ret = [tuple(x) for x in r.json()["result"]]
    #print(ret)
    ret = ret*(1+len(inp)//5)
    return ret

def showpalette(cols):
    img = Image.new("RGB", (64*len(cols), 64))
    draw = ImageDraw.Draw(img)
    for n, x in enumerate(cols):
        draw.rectangle((n*64, 0, (n+1)*64, 64), tuple(x))
    ImageShow.show(img)