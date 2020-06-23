from matplotlib import pyplot as plt
import numpy as np
from supermariopy import imageutils
from supermariopy import numpyutils as npu


def argmax_rgb(m, cmap=plt.cm.viridis):
    """Take argmax of m along dimension 1 and apply RGB colorcode on it
    
    Parameters
    ----------
    m : np.ndarray
        tensor representing list of binary masks, shaped [B, H, W, P]

    Returns
    -------
    np.array
        RGB mask tensor shaped [B, H, W, 3]

    See also
    --------
    .. tfutils.viz.argmax_rgb 
    """
    B, H, W, P = m.shape
    argmax_map = np.argmax(m, axis=-1)
    colors = imageutils.make_colors(P, cmap=cmap)
    colors = colors.astype(np.float32)
    m_one_hot = npu.one_hot(argmax_map, P, axis=-1)
    mask_rgb = np.einsum("bhwp,pc->bhwc", m_one_hot, colors)
    return mask_rgb


def show_active_parts(
    parts,
    active_idx,
    inactive_parts_color=np.array([0.8, 0.8, 0.8]),
    background_color=np.array([1.0, 1.0, 1.0]),
    cmap=plt.cm.viridis,
):
    """Highlight active parts in color and set color of other parts to a gray value
    
    Parameters
    ----------
    parts : np.ndarray
        tensor of binary masks indicating mutually exclusive parts or segments of a segmentation.
        Shaped [N, H, W, P]
    active_idx : np.array or list
        list of indices indicating which part (indexing into last dimension of parts) are active.
    inactive_parts_color : np.ndarray, default np.array([0.8, 0.8, 0.8])
        Color value in range [0, 1] for inactive parts.
    background_color : np.ndarray, default np.array([1.0, 1.0, 1.0])
        Color value in range [0, 1] for background. Note that this only makes sense if parts does not provide a background itself.
        If part contains a background part, it will be assigned a background_color specified by the color map.
    """
    if not any([isinstance(active_idx, list), isinstance(active_idx, np.ndarray)]):
        raise TypeError()
    N, H, W, P = parts.shape
    colors = imageutils.make_colors(P, cmap)
    all_part_idx = set(range(P))
    inactive_part_idx = all_part_idx.difference(set(active_idx))
    colors[np.array(list(inactive_part_idx)), :] = inactive_parts_color.reshape((1, 3))

    background_part = 1.0 - np.clip(np.sum(parts, axis=-1), 0, 1)
    colors = np.insert(colors, 0, background_color.reshape(1, 3), axis=0)
    parts = np.insert(parts, 0, background_part, axis=-1)
    parts_rgb = np.einsum("nhwp,pc->nhwc", parts, colors)
    return parts_rgb


# TODO segmentation false color plot
