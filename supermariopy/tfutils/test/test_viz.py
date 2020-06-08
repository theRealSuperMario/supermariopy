import tensorflow as tf
from supermariopy.tfutils import viz as tfviz
import numpy as np

tf.enable_eager_execution()


def test_argmax_rgb():
    P = 10
    h = 50
    H = P * h
    x = np.zeros((1, P, H, H))
    for i in range(P):
        x[:, i, (h * i) : (h * (i + 1)), (h * i) : (h * (i + 1))] = i
    m_rgb = tfviz.argmax_rgb(x) / 2 + 0.5
    assert m_rgb.shape == (1, 3, H, H)
