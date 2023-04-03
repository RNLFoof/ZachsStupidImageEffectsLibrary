from PIL import Image

from ZachsStupidImageLibrary.analysis import get_edge_points


def test_get_edge_points_at_image_border():
    image = Image.new("RGBA", (1, 1), 0xffffffff)
    assert get_edge_points(image) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }

def test_get_edge_points_one_pixel():
    image = Image.new("RGBA", (3, 3), 0)
    image.putpixel((1, 1), 0xffffffff)
    assert get_edge_points(image) == {
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    }

def test_get_edge_points_square():
    square_side = 10
    image = Image.new("RGBA", (square_side+2, square_side+2), 0)
    for x in [1, 2]:
        for y in [1, 2]:
            image.putpixel((x, y), 0xffffffff)
    lowest = 1
    highest = square_side-1
    assert get_edge_points(image) == set([
        (x, y)
        for x, y
        in zip(
            range(lowest, highest + 1),
            range(lowest, highest + 1),
        )
        if x in (lowest, highest) or y in (lowest, highest)
    ])
