import numpy as np
import torch
from matplotlib import pyplot as plt
from supermariopy import imageutils, ptutils
from supermariopy.ptutils import nn as ptnn


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
    B, P, H, W = ptutils.nn.shape_as_list(m)
    max_values, argmax_map = torch.max(m, dim=1)
    # if m.is_cuda:
    #     dtype = torch.cuda.FloatTensor
    # else:
    #     dtype = torch.FloatTensor
    colors = imageutils.make_colors(P, cmap=cmap)
    colors = colors.astype(np.float32)
    colors = torch.from_numpy(colors)
    colors = colors.to(m.device)
    m_one_hot = ptnn.to_one_hot(argmax_map, P)
    m_one_hot = m_one_hot.permute(0, 3, 1, 2)
    mask_rgb = torch.einsum("bphw,pc->bchw", m_one_hot, colors)
    return mask_rgb
