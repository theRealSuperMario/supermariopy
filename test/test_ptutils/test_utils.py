import pytest
import torch
import numpy as np
from supermariopy.ptutils import utils as ptu


def test_to_numpy():
    x = torch.zeros((128, 128))
    y = ptu.to_numpy(x)
    assert isinstance(y, np.ndarray)

    x = torch.zeros((128, 128), requires_grad=False)
    y = ptu.to_numpy(x)
    assert isinstance(y, np.ndarray)

    if torch.cuda.is_available():
        x = torch.zeros((128, 128), requires_grad=True)
        x = x.cuda()
        y = ptu.to_numpy(x)
        assert isinstance(y, np.ndarray)

    x = torch.zeros((1, 3, 128, 128), requires_grad=False)
    y = ptu.to_numpy(x)
    assert y.shape == (1, 3, 128, 128)
    y = ptu.to_numpy(x, permute=True)
    assert y.shape == (1, 128, 128, 3)


def test_to_torch():
    x = np.zeros((1, 128, 128, 3))
    y = ptu.to_torch(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 128, 128, 3)

    x = np.zeros((1, 128, 128, 3))
    y = ptu.to_torch(x, True)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 3, 128, 128)
