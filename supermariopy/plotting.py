import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from typing import *
import seaborn as sns

# import colorcet as cc
# https://colorcet.pyviz.org/user_guide/Categorical.html

# TODO rename module to something like : plotting, matborn (seaborn + matplotlib)


def set_style():
    plt.style.use("seaborn-whitegrid")


NB_RC_PARAMS = {"figure.figsize": [5, 3], "figure.dpi": 220, "figure.autolayout": True}
TIKZ_RC_PARAMS = {
    "pgf.rcfonts": False,
    "figure.figsize": [5, 3],
    "figure.dpi": 220,
    "figure.autolayout": True,
    "lines.linewidth": 2,
}

colors1 = np.array(sns.light_palette("navy", reverse=False, n_colors=10 + 1))[:, :3]
colors2 = np.array(sns.light_palette("red", reverse=False, n_colors=10 + 1))[:, :3]
colors3 = np.array(sns.light_palette("orange", reverse=False, n_colors=10 + 1))[:, :3]
colors4 = np.array(sns.light_palette("black", reverse=False, n_colors=10 + 1))[:, :3]

# TODO: define more color palettes
# TODO: implement some of these
# https://sk1project.net/palettes/

COLOR_PALETTE_4 = np.array(sns.color_palette("bright", 4))[:, :3]


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


def add_colorbars_to_axes(
    axes=None, loc="right", size="5%", pad=0.05, **kwargs
) -> None:
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
    None

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


# TODO: listmap: shortcut for list(map(f, args))


def set_all_axis_off(axes: List = None) -> None:
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
    map(lambda x: change_fontsize(x, fs), axes)


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
    np.ndarray
        [description]
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
    ewma: numpy array
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

