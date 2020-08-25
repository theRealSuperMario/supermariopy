import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from .. import imageutils
from ..tfutils import nn


def argmax_rgb(m, cmap=plt.cm.viridis):
    """Take argmax of m along dimension 1 and apply RGB colorcode on it

    Parameters
    ----------
    m : tf.Tensor
        Tensorflor tensor or numpy array as result of eager execution

    Returns
    -------
    tf.Tensor
        RGB mask tensor shaped [B, H, W, 3]
    """
    B, H, W, P = nn.shape_as_list(m)
    argmax_map = tf.arg_max(m, dimension=-1)
    colors = imageutils.make_colors(P, cmap=cmap)
    colors = colors.astype(np.float32)
    colors = tf.convert_to_tensor(colors)
    m_one_hot = tf.one_hot(argmax_map, P, axis=-1)
    mask_rgb = tf.einsum("bhwp,pc->bhwc", m_one_hot, colors)
    return mask_rgb
