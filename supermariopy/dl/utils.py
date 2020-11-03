import enum

import tensorflow as tf
import torch


class CHANNELS_2D(enum.IntEnum):
    """ IntEnum allows comparisons like CHANNELS_2D == 2 (--> true)
    See https://docs.python.org/3/library/enum.html"""

    torch2tf = 0
    tf2torch = 1


def cvt_channels2D(x, from_to: CHANNELS_2D):
    if isinstance(x, torch.Tensor):
        if from_to == CHANNELS_2D.torch2tf:
            x = x.permute((0, 2, 3, 1))
        elif from_to == CHANNELS_2D.tf2torch:
            x = x.permute((0, 3, 1, 2))
    elif isinstance(x, tf.Tensor):
        if from_to == CHANNELS_2D.torch2tf:
            x = tf.transpose(x, (0, 2, 3, 1))
        elif from_to == CHANNELS_2D.tf2torch:
            x = tf.transpose(x, (0, 3, 1, 2))
    return x
