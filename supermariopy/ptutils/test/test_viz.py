import pytest
import numpy as np
from matplotlib import pyplot as plt
import torch
from supermariopy import ptutils
from supermariopy.ptutils import viz as ptviz


def test_argmax_rgb():
    P = 10
    h = 50
    H = P * h
    x = torch.zeros(1, P, H, H)
    for i in range(P):
        x[:, i, (h * i) : (h * (i + 1)), (h * i) : (h * (i + 1))] = i
    m_rgb = ptviz.argmax_rgb(x) / 2 + 0.5
    # plt.imshow(m_rgb.permute(0, 2, 3, 1).numpy().squeeze())
    # plt.show()
    assert m_rgb.shape == (1, 3, H, H)
