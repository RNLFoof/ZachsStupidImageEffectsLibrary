import pytest
from numpy import pi

from zsil.internal import point_direction

tau = pi * 2


@pytest.mark.parametrize("expectations", [
    ((*(0, 0), *(1, 0)), 0 / 8 * tau),
    ((*(0, 0), *(1, -1)), 1 / 8 * tau),
    ((*(0, 0), *(0, -1)), 2 / 8 * tau),
    ((*(0, 0), *(-1, -1)), 3 / 8 * tau),
    ((*(0, 0), *(-1, 0)), 4 / 8 * tau),
    ((*(0, 0), *(-1, 1)), 5 / 8 * tau),
    ((*(0, 0), *(0, 1)), 6 / 8 * tau),
    ((*(0, 0), *(1, 1)), 7 / 8 * tau),

    ((*(1, 1), *(1 + 1, 0 + 1)), 0 / 8 * tau),
    ((*(1, 1), *(1 + 1, -1 + 1)), 1 / 8 * tau),
    ((*(1, 1), *(0 + 1, -1 + 1)), 2 / 8 * tau),
    ((*(1, 1), *(-1 + 1, -1 + 1)), 3 / 8 * tau),
    ((*(1, 1), *(-1 + 1, 0 + 1)), 4 / 8 * tau),
    ((*(1, 1), *(-1 + 1, 1 + 1)), 5 / 8 * tau),
    ((*(1, 1), *(0 + 1, 1 + 1)), 6 / 8 * tau),
    ((*(1, 1), *(1 + 1, 1 + 1)), 7 / 8 * tau),

    ((*(0, 0), *(0, 0)), 0),
])
def test_point_direction(expectations):
    input, output = expectations
    assert point_direction(*input) == output
