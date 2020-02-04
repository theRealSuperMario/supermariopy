import tensorflow as tf
from matplotlib import pyplot as plt
from supermariopy.tfutils import nn
import numpy as np
from supermariopy import imageutils


def argmax_rgb(m, cmap=plt.cm.viridis):
    """Take argmax of m along dimension 1 and apply RGB colorcode on it
    
    Parameters
    ----------
    m : [type]
        [description]

    Returns
    -------
    np.array
        RGB mask tensor shaped [B, 3, H, W]
    """
    B, P, H, W = nn.shape_as_list(m)
    argmax_map = tf.arg_max(m, dimension=1)
    colors = imageutils.make_colors(P, cmap=cmap)
    colors = colors.astype(np.float32)
    colors = tf.convert_to_tensor(colors)
    m_one_hot = tf.one_hot(argmax_map, P, axis=-1)
    m_one_hot = tf.transpose(m_one_hot, (0, 3, 1, 2))
    mask_rgb = tf.einsum("bphw,pc->bchw", m_one_hot, colors)
    return mask_rgb
