import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from typing import *
import seaborn as sns
import cv2
from supermariopy import imageutils

# https://github.com/ubernostrum/webcolors
import webcolors

import colorcet as cc

# https://colorcet.pyviz.org/user_guide/Categorical.html


def set_style():
    plt.style.use("seaborn-whitegrid")


NB_RC_PARAMS = {
    "figure.figsize": [5, 3],
    "figure.dpi": 220,
    "figure.autolayout": True,
    "legend.frameon": True,
}
BLOG_RC_PARAMS_5x4 = {
    "figure.figsize": [5, 4],
    "figure.dpi": 150,
    "figure.autolayout": True,
    "legend.frameon": True,
    "axes.titlesize": "xx-large",
    "axes.labelsize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "legend.fontsize": "x-large",
}
TIKZ_RC_PARAMS = {
    "pgf.rcfonts": False,
    "figure.figsize": [5, 3],
    "figure.dpi": 220,
    "figure.autolayout": True,
    "lines.linewidth": 2,
    "legend.frameon": True,
}


# TODO: define more color palettes
# TODO: implement some of these
# https://sk1project.net/palettes/

COLOR_PALETTE_4 = np.array(sns.color_palette("bright", 4))[:, :3]

"""
https://sk1project.net/palettes/ms-office-2013-primary-colors/
https://msdn.microsoft.com/en-us/library/office/dn684229.aspx
implemented only first 12 colors
"""
PALETTES = ["msoffice", "navy", "custom6"]


def get_palette(name, bytes=False):
    """Get color palette by name. See "plotting.PALETTES" for available palettes.
    
    Parameters
    ----------
    name : [type]
        [description]
    bytes : bool, optional
        if True, palettes are returned as bytes, by default False
    
    Returns
    -------
    np.ndarray
        palette
    """
    if name.lower() == "msoffice":
        palette = [
            r"#005A9E",
            r"#0A6332",
            r"#B83B1D",
            r"#19478A",
            r"#0072C6",
            r"#217346",
            r"#D24726",
            r"#2B579A",
            r"#2A8DD4",
            r"#439467",
            r"#F0623E",
            r"#3E6DB5",
        ]
        palette = np.array([webcolors.hex_to_rgb(c) for c in palette])
    elif name.lower() == "navy":
        palette = np.array(sns.light_palette("navy", reverse=False, n_colors=10 + 1))[
            :, :3
        ]
    elif name.lower() == "custom6":
        # blue, red, purple, green, yellow, black
        palette = [
            r"#000AC8",
            r"#dc0000",
            r"#E498FF",
            r"#99DE00",
            r"#FFFF3E",
            r"#000000",
        ]
        palette = np.array([webcolors.hex_to_rgb(c) for c in palette])
    if bytes:
        return palette
    else:
        return palette / 255.0


colors1 = get_palette("navy")
colors2 = np.array(sns.light_palette("red", reverse=False, n_colors=10 + 1))[:, :3]
colors3 = np.array(sns.light_palette("orange", reverse=False, n_colors=10 + 1))[:, :3]
colors4 = np.array(sns.light_palette("black", reverse=False, n_colors=10 + 1))[:, :3]
COLORS_GLASBEY_BW = cc.glasbey_bw


def imageStack_2_subplots(image_stack, axis=0):
    """utility function to plot a stack of images into a grid of subplots

    
    Parameters
    ----------
    image_stack : np.ndarray
        stack of images [N, H, W, C]
    axis : int, optional
        axis of image_stack along which to make subplots, by default 0
    
    Returns
    -------
    fig
    axes
    """
    image_stack = np.rollaxis(image_stack, axis)
    N_subplots = image_stack.shape[0]
    R = math.floor(math.sqrt(N_subplots))
    C = math.ceil(N_subplots / R)
    fig, axes = plt.subplots(R, C)
    axes = axes.ravel()
    for ax, img in zip(axes, image_stack):
        ax.imshow(img)
    return fig, axes


def imageList_2_subplots(*image_list):
    N_subplots = len(image_list)
    R = math.floor(math.sqrt(N_subplots))
    C = math.ceil(N_subplots / R)
    fig, axes = plt.subplots(R, C)
    axes = axes.ravel()
    for ax, img in zip(axes, image_list):
        ax.imshow(img)
        ax.grid(False)
        ax.set_axis_off()
    return fig, axes


def add_colorbars_to_axes(axes=None, loc="right", size="5%", pad=0.05, **kwargs):
    """add colorbars to each axis in the current figures list of axes
    
    Parameters
    ----------
    axes : list, optional
        list of axes. Will use all axes from current figure if None, by default None
    loc : str, optional
        where to put colorbar, by default "right"
    size : str, optional
        size of colorbar, by default "5%"
    pad : float, optional
        padding between canvas and colorbar, by default 0.05

    Returns
    -------
    list
        list of colorbars

    Examples
    --------

        plt.subplot(121); plt.imshow(np.arange(100).reshape((10,10)))
        plt.subplot(122); plt.imshow(np.arange(100).reshape((10,10)))
        add_colorbars_to_axes()
    """

    if axes is None:
        axes = plt.gcf().get_axes()
    cbars = []
    for ax in axes:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(loc, size=size, pad=pad)
        cbar = plt.colorbar(ax.images[0], cax=cax, **kwargs)
        cbars.append(cbar)
    return cbars


def set_all_axis_off(axes: List = None):
    """apply ax.set_axis_off() to all given axis.
    Apply it to all axes in current figure if no axes are provided

    Parameters
    ----------
    axes : list, optional
    list of axes where axis should be turned off, by default None

    Returns
    -------
    None

    Examples
    --------

        plt.subplot(121); plt.imshow(np.arange(100).reshape((10,10)))
        plt.subplot(122); plt.imshow(np.arange(100).reshape((10,10)))
        set_all_axis_off()
    """
    if axes is None:
        axes = plt.gcf().get_axes()
    for ax in axes:
        ax.set_axis_off()


def set_all_fontsize(axes: List = None, fs=3):
    if axes is None:
        axes = plt.gcf().get_axes()
    for ax in axes:
        change_fontsize(ax, fs)


def change_fontsize(ax, fs):
    """change fontsize on given axis. This effects the following properties of ax

    - ax.title
    - ax.xaxis.label
    - ax.yaxis.label
    - ax.get_xticklabels()
    - ax.get_yticklabels()
    
    Parameters
    ----------
    ax : mpl.axis
        [description]
    fs : float
        new font size
    
    Returns
    -------
    ax
        mpl.axis

    Examples
    --------

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(10), np.arange(10))
        change_fontsize(ax, 5)
    """
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fs)
    return ax


def change_linewidth(ax, lw=3):
    """change linewidth for each line plot in given axis
    
    Parameters
    ----------
    ax : mpl.axis
        axis to change fontsize of
    lw : float, optional
        [description], by default 3
    
    Returns
    -------
    ax
        mpl.axis 

    Examples
    --------

        fig, ax = plt.subplots(1, 1)
        x = np.arange(10)
        y = np.arange(10)
        ax.plot(x, y, x + 1, y, x -1, y )
        change_linewidth(ax, 3)
    """
    for item in ax.lines:
        item.set_linewidth(lw)
    return ax


def smooth(scalars, weight):
    """Smooth scalar array using some weight between [0, 1]. This is the tensorboard implementation
    
    For comparison with @ewma, see https://github.com/theRealSuperMario/notebook_collection

    Parameters
    ----------
    scalars : np.array
        array of scalars
    weight : float between 0, 1
        amount of smoothing
    
    Returns
    -------
    np.array
        smoothed array
    """
    if weight <= 0.0:  # no smoothing
        return scalars
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def ewma(x, alpha):
    """
    copied from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    Returns the exponentially weighted moving average of x.

    For comparison with @smooth, see https://github.com/theRealSuperMario/notebook_collection

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    np.array
        the exponentially weighted moving average
    """
    # Coerce x to an array
    if alpha <= 0.0:
        return x
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (alpha)
    p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0 ** p, 0)

    # Calculate the ewma
    return np.dot(w, x[:: np.newaxis]) / w.sum(axis=1)


def draw_keypoint_markers(
    img: np.ndarray,
    keypoints: np.ndarray,
    font_scale: float = 0.5,
    thickness: int = 2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    marker_list=["o", "v", "x", "+", "<", "-", ">", "c"],
) -> np.ndarray:
    """ Draw keypoints on image with markers
    
    Parameters
    ----------
    img : np.ndarray
        shaped [H, W, 3] array  in range [0, 1]
    keypoints : np.ndarray
        shaped [kp, 2] - array giving keypoint positions in range [-1, 1] for x and y. keypoints[:, 0] is x-coordinate (horizontal).
    font_scale : int, optional
        openCV font scale passed to 'cv2.putText', by default 1
    thickness : int, optional
        openCV font thickness passed to 'cv2.putText', by default 2
    font : cv2.FONT_xxx, optional
        openCV font, by default cv2.FONT_HERSHEY_SIMPLEX
    
    Examples
    --------

        from skimage import data
        astronaut = data.astronaut()
        keypoints = np.stack([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], axis=1)
        img_marked = draw_keypoint_markers(astronaut, keypoints, font_scale=2, thickness=3)
        plt.imshow(img_marked)
    """
    if not imageutils.is_in_range(img, [0, 1]):
        raise RangeError(img, "img", [0, 1])
    if img.shape[0] != img.shape[1]:
        raise ValueError("only square images are supported currently")

    img_marked = img.copy()
    keypoints = imageutils.convert_range(keypoints, [-1, 1], [0, img.shape[0] - 1])
    colors = imageutils.make_colors(
        keypoints.shape[0], bytes=False, cmap=plt.cm.inferno
    )
    for i, kp in enumerate(keypoints):
        text = marker_list[i % len(marker_list)]
        (label_width, label_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        textX = kp[0]
        textY = kp[1]
        font_color = colors[i]
        text_position = (
            textX - label_width / 2.0 - baseline,
            textY - label_height / 2.0 + baseline,
        )
        text_position = tuple([int(x) for x in text_position])
        img_marked = cv2.putText(
            img_marked,
            text,
            text_position,
            font,
            font_scale,
            font_color,
            thickness=thickness,
        )
    return img_marked


def plot_canvas(canvas, delta_x, delta_y, show_grid=True, fig=None, ax=None):
    """ plot a grid of images arranged in a canvas as obtained by `imageutils.batch_to_canvas` into the provided `ax` """
    nx = canvas.shape[1] // delta_x
    ny = canvas.shape[0] // delta_y

    if ax is None and fig is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(canvas, origin="lower")
    ax.set_xticks(np.arange(nx) * delta_x + delta_x // 2)
    ax.set_xlim(0, canvas.shape[1])
    ax.set_yticks(np.arange(ny) * delta_y + delta_y // 2)
    ax.set_ylim(0, canvas.shape[0])

    if show_grid:
        ax.set_xticks(np.arange(nx) * delta_x, minor=True)
        ax.set_yticks(np.arange(ny) * delta_y, minor=True)
        ax.grid(which="minor", color="#000000", linestyle="-")

    return fig, ax


import matplotlib.pyplot as plt


import io
import cv2


def figure_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), -1)[
        ..., :3
    ]  # no alpha
    image = image.reshape((1,) + image.shape)
    return image


def plot_bars(
    m, xticks=None, xticklabels=None, ylim=[0, 1], figsize=(5, 5), colors=None
):
    """ 
        Barplot of an array shaped [N, ] with labels xticklabels on y axis
        figsize (width, height)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if xticks is None:
        xticks = np.arange(len(m))
    if colors is None:
        colors = cc.glasbey_light[1 : (len(m) + 1)]  # no white foreground

    ax.bar(xticks, m, color=colors)
    ax.set_xticks(xticks)
    if xticklabels is None:
        pass
    else:
        ax.set_xticklabels(xticklabels)

    ax.set_ylim(ylim)
    ax.set_xlim([-1 + np.min(xticks), np.max(xticks) + 1])
    plt.tight_layout()

    return fig, ax
