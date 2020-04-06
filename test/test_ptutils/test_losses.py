import pytest
import torch
from supermariopy.ptutils import losses as smptlosses
import numpy as np


def test_vgg19_loss():
    x = torch.zeros((1, 3, 224, 224))
    y = torch.zeros((1, 3, 224, 224))

    criterion = smptlosses.VGGLoss(None)
    l = criterion(x, y)
    assert np.allclose(l.numpy(), np.array([0]))


def test_vgg19_with_l1_loss():
    x = torch.zeros((1, 3, 224, 224))
    y = torch.zeros((1, 3, 224, 224))

    criterion = smptlosses.VGGLossWithL1(None)
    l = criterion(x, y)
    assert np.allclose(l.numpy(), np.array([0]))
