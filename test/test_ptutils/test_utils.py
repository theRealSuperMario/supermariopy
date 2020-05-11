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

    x = np.zeros((1, 128, 128, 3), dtype=np.float64)
    y = ptu.to_torch(x, True)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 3, 128, 128)
    assert y.dtype == torch.float32

    x = np.zeros((1, 128, 128, 3), dtype=np.float32)
    y = ptu.to_torch(x, True)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 3, 128, 128)
    assert y.dtype == torch.float32


def test_split_stack_reshape():
    x = torch.zeros((1, 24, 128, 128))
    y = ptu.split_stack_reshape(x, 3, 1, 1, 0)


def test_linear_variable():
    x = torch.tensor([0])
    v = ptu.linear_variable(x, 10, 100, 0, 1, 0.0, 1.0)
    assert v == 0.0

    x = torch.tensor([101])
    v = ptu.linear_variable(x, 10, 100, 0, 1, 0.0, 1.0)
    assert v == 1.0

    x = 0
    v = ptu.linear_variable(x, 10, 100, 0, 1, 0.0, 1.0)
    assert v == 0.0

    x = int(0)
    v = ptu.linear_variable(x, 10, 100, 0, 1, 0.0, 1.0)
    assert v == 0.0
