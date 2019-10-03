import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from typing import *


def imageStack_2_subplots(image_stack, axis=0):
    """utility function to plot a stack of images into a grid of subplots

    # TODO: example
    
    Parameters
    ----------
    image_stack : [type]
        [description]
    axis : int, optional
        [description], by default 0
    
    Returns
    -------
    [type]
        [description]
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


def add_colorbars_to_axes(axes=None, loc="right", size="5%", pad=0.05) -> None:
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
    for ax in axes:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(loc, size=size, pad=pad)
        plt.colorbar(ax.images[0], cax=cax)


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
