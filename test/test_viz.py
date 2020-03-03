import pytest
from supermariopy import viz
import numpy as np


def create_test_parts():
    """ create test parts in form of 10 squares aligned diagonally on an image
    Looks something like this
    ----
    |  |
    ----
        ----
        |  |
        ----
             ----
             |  |
             ----
                 ...
    """
    P = 10
    h = 50
    H = P * h
    x = np.zeros((1, H, H, P))
    for i in range(P):
        x[:, (h * i) : (h * (i + 1)), (h * i) : (h * (i + 1)), i] = 1
    return x, H


def test_show_active_parts():
    x, H = create_test_parts()
    x_rgb = viz.show_active_parts(x, [1], background_color=np.array([0.5, 0.5, 0.5]))

    assert x_rgb.shape == (1, H, H, 3)


def test_argmax_rgb():
    x, H = create_test_parts()
    m_rgb = viz.argmax_rgb(x) / 2 + 0.5
    assert m_rgb.shape == (1, H, H, 3)

