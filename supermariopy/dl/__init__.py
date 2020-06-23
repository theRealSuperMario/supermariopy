import tensorflow as tf
import numpy as np
import torch

tf.enable_eager_execution()


def tf_allclose_pt(t_tf, t_pt):
    """ convert both tensors to numpy and assert """
    t_tf = np.array(t_tf)
    t_pt = t_pt.numpy()
    return np.allclose(t_tf, t_pt)
