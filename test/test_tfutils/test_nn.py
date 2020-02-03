import pytest
import numpy as np
import tensorflow as tf


tf.enable_eager_execution()


def test_filltriangular():
    params = tf.reshape(tf.range(0, 6), (1, 6))
    L = tf.contrib.distributions.fill_triangular(params)
    # assert L.shape == ()
    assert L.shape == (1, 3, 3)
    assert np.allclose(L, np.array([[3, 0, 0], [5, 4, 0], [2, 1, 0]]))


def test_diag_part():
    params = tf.reshape(tf.range(1, 7), (1, 6))
    L = tf.contrib.distributions.fill_triangular(params)
    diag_L = tf.linalg.diag_part(L)
    assert diag_L.shape == (1, 3)
    assert np.allclose(np.squeeze(diag_L), np.array([4, 5, 1]))


def test_set_diag():
    params = tf.reshape(tf.range(1, 7), (1, 6))
    L = tf.contrib.distributions.fill_triangular(params)
    diag_L = tf.ones((1, 3), dtype=params.dtype) * 6
    M = tf.linalg.set_diag(L, diag_L)
    assert np.allclose(tf.linalg.diag_part(M), diag_L)
