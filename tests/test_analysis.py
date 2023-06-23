from PIL import Image
from pytest_assume.plugin import assume

from zsil.analysis import get_edge_points, get_center_pixels


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
    image = Image.new("RGBA", (square_side + 2, square_side + 2), 0)
    for x in [1, 2]:
        for y in [1, 2]:
            image.putpixel((x, y), 0xffffffff)
    lowest = 1
    highest = square_side - 1
    assert get_edge_points(image) == set([
        (x, y)
        for x, y
        in zip(
            range(lowest, highest + 1),
            range(lowest, highest + 1),
        )
        if x in (lowest, highest) or y in (lowest, highest)
    ])


def test_get_center_pixels():
    rectangle_image = Image.new("RGBA", (3, 5), 0xffffffff)
    with assume: assert get_center_pixels(rectangle_image) == {(1, 1), (2, 1), (3, 1)}
