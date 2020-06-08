import pytest
import numpy as np


def test_one_hot():
    from supermariopy import numpyutils as npu

    a = np.arange(10)
    a_onehot = npu.one_hot(a, 10)
    assert a_onehot.shape == (10, 10)
    assert all([np.sum(a_onehot[:, i]) == 1 for i in range(10)])

    a = np.arange(10)
    a = np.stack([a] * 2, axis=0)
    a_onehot = npu.one_hot(a, 10)
    assert a_onehot.shape == (2, 10, 10)

    a = np.arange(10)
    a = np.stack([a] * 2, axis=0)
    a_onehot = npu.one_hot(a, 10, axis=0)
    assert a_onehot.shape == (10, 2, 10)

    a = np.arange(10)
    a = np.stack([a] * 2, axis=0)
    a_onehot = npu.one_hot(a, 10, axis=1)
    assert a_onehot.shape == (2, 10, 10)
