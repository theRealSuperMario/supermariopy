import math
import subprocess
from typing import Callable, Iterable, Tuple

import cv2
import deprecation
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from . import __version__


class DimensionError(ValueError):
    def __init__(self, var, var_name, target_dimension):
        msg = "Variable '{}' has to have {} dimensions but has {} dimensions".format(
            var_name, target_dimension, len(var.shape)
        )
        super(DimensionError, self).__init__(msg)


class ShapeError(ValueError):
    def __init__(self, var, var_name, target_shape):
        msg = "Variable '{}' has to have shape {} but has {} shape".format(
            var_name, target_shape, var.shape
        )
        super(ShapeError, self).__init__(msg)


class RangeError(ValueError):
    def __init__(self, var, var_name, target_range):
        found_range = [var.min(), var.max()]
        msg = "Variable '{}' has to be in value range {} but has range {}".format(
            var_name, target_range, found_range
        )
        super(RangeError, self).__init__(msg)


class LengthError(ValueError):
    def __init__(self, var, var_name, target_len):
        msg = "Variable '{}' has to have length {} but has length {}".format(
            var_name, target_len, len(var)
        )
        super(LengthError, self).__init__(msg)


def is_in_range(array: np.ndarray, target_range: Iterable) -> bool:
    if len(target_range) != 2:
        raise LengthError(target_range, "target_range", 2)
    if target_range[0] > target_range[1]:
        raise ValueError("target_range[0] has to be smaller than target_range[1]")
    return array.min() >= target_range[0] and array.max() <= target_range[1]


def put_text(
    img: np.ndarray,
    text: str,
    loc: str = "center",
    font_scale: int = 1,
    thickness: int = 1,
    color: Tuple[int, int, int] = (-1, -1, -1),
) -> np.ndarray:
    """small utility function to put text into image using opencv

    Parameters
    ----------
    img : np.ndarray
        shaped [H, W, 3]
    text : str
        text to put in image
    loc : str, optional
        where to put the text. one of `center`, `top`, `topleft`, `bottomleft`,
        `bottom`, by default "center"
    font_scale : int, optional
        openCV font scale parameter. Int >= 1, by default 1
    thickness : int, optional
        openCV thickness parameter. Int >= 1, by default 1
    color : Tuple[int, int, int], optional
        openCV color parameter, by default (-1, -1, -1)

    Returns
    -------
    np.ndarray
        image with text
    """
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if loc == "center":
        # get coords based on boundary
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
    elif loc == "top":
        textX = (img.shape[1] - textsize[0]) // 2
        textY = textsize[1]
    elif loc == "topleft":
        textX = 0
        textY = textsize[1]
    elif loc == "topright":
        textX = img.shape[1] - textsize[0]
        textY = textsize[1]
    elif loc == "bottomleft":
        textX = 0
        textY = img.shape[0]
    elif loc == "bottomright":
        textX = img.shape[1] - textsize[0]
        textY = img.shape[0]
    elif loc == "bottom":
        textX = (img.shape[1] - textsize[0]) // 2
        textY = img.shape[0]
    font_color = color
    cv2.putText(img, text, (textX, textY), font, font_scale, font_color, thickness=3)
    return img


def make_colors(
    n_classes: int,
    cmap: Callable = plt.cm.inferno,
    bytes: bool = False,
    with_background=False,
    background_color=np.array([1, 1, 1]),
    background_id=0,
) -> np.ndarray:
    """make a color array using the specified colormap for `n_classes` classes

    # TODO: test new background functionality
    Parameters
    ----------
    n_classes: int
        how many classes there are in the mask
    cmap: Callable, optional, by default plt.cm.inferno
        matplotlib colormap handle
    bytes: bool, optional, by default False
        bytes option passed to `cmap`.
        Returns colors in range [0, 1] if False and range [0, 255] if True

    Returns
    -------
    colors: ndarray
        an array with shape [n_classes, 3] representing colors in the range [0, 1].

    """
    colors = cmap(np.linspace(0, 1, n_classes), alpha=False, bytes=bytes)[:, :3]
    if with_background:
        colors = np.insert(colors, background_id, background_color, axis=0)
    return colors


@deprecation.deprecated(
    deprecated_in="0.2",
    removed_in="0.3",
    current_version=__version__,
    details="Use the function plotting.draw_keypoint_markers",
)
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
        shaped [kp, 2] - array giving keypoint positions in range [-1, 1] for x and y.
        keypoints[:, 0] is x-coordinate (horizontal).
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
        img_marked = draw_keypoint_markers(astronaut,
                                           keypoints,
                                           font_scale=2,
                                           thickness=3)
        plt.imshow(img_marked)
    """
    if not is_in_range(img, [0, 1]):
        raise RangeError(img, "img", [0, 1])
    if img.shape[0] != img.shape[1]:
        raise ValueError("only square images are supported currently")

    img_marked = img.copy()
    keypoints = convert_range(keypoints, [-1, 1], [0, img.shape[0] - 1])
    colors = make_colors(keypoints.shape[0], bytes=False, cmap=plt.cm.inferno)
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


def convert_range(
    array: np.ndarray, input_range: Iterable[int], target_range: Iterable[int]
) -> np.ndarray:
    """convert range of array from input range to target range

    Parameters
    ----------
    array: np.ndarray
        array in any shape
    input_range: Iterable[int]
        range of array values in format [min, max]
    output_range: Iterable[int]
        range of rescaled array values in format [min, max]

    Returns
    -------
    np.ndarray
        rescaled array

    Examples
    --------
        t = imageutils.convert_range(np.array([-1, 1]), [-1, 1], [0, 1])
        assert np.allclose(t, np.array([0, 1]))
        t = imageutils.convert_range(np.array([0, 1]), [0, 1], [-1, 1])
        assert np.allclose(t, np.array([-1, 1]))
    """
    if input_range[1] <= input_range[0]:
        raise ValueError
    if target_range[1] <= target_range[0]:
        raise ValueError

    a = input_range[0]
    b = input_range[1]
    c = target_range[0]
    d = target_range[1]
    return (array - a) / (b - a) * (d - c) + c


def colorize_heatmaps(heatmaps: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """apply color to each heatmap and sum across heatmaps

    Parameters
    ----------
    heatmaps : np.ndarray
        heatmaps in shape [N, H, W, C].
        Each item along axis [0 and 3] is a heatmap with maximum 1 and minimum 0
    colors : np.ndarray
        array with colors in shape [C, 3]. use @make_colors.

    Returns
    -------
    np.ndarray
        colorized heatmaps in shape [N, H, W, 3]

    Examples
    --------
        n_parts = 10
        keypoints = np.stack([
            np.linspace(-1, 1, n_parts), np.linspace(-1, 1, n_parts)
            ], axis=1)
        heatmaps = imageutils.CRF().keypoints_to_heatmaps((h, w), keypoints, var=0.01)
        heatmaps_colorized = imageutils.colorize_heatmaps(
            heatmaps[np.newaxis, ...],
            imageutils.make_colors(n_parts)
        )
        plt.imshow(np.squeeze(heatmaps_colorized))
    """
    N, H, W, C = heatmaps.shape
    heatmaps_colorized = np.expand_dims(heatmaps, -1) * colors.reshape(
        (1, 1, 1, C, 3)
    )  # [N, H, W, C, 3]
    heatmaps_colorized = np.sum(heatmaps_colorized, axis=3)
    return heatmaps_colorized


def keypoints_to_heatmaps(
    img_shape: Tuple[int, int], keypoints: np.ndarray, var=0.01
) -> np.ndarray:
    """
    imgshape : tuple[int, int]
    keypoints : np.ndarray
        shaped [kp, 2]. Keypoints coordinates be in range (-1, 1).
        Keypoints[:, 0] is horizontal coordinate (x)
    var : float
        variance of blobs to create around keypoint.
        Relative to keypoints coordinates range (-1, 1)
    outputs : [img_shape[0], img_shape[1], kp]
    """
    if not is_in_range(keypoints, [-1, 1]):
        raise RangeError(keypoints, "keypoints", [-1, 1])
    if len(keypoints.shape) == 2:
        pass
    elif len(keypoints.shape) > 2:
        raise DimensionError(keypoints, "keypoints", 2)

    if keypoints.shape[-1] != 2:
        raise ValueError(
            "keypoints has to have shape [kp, 2], found {}".format(keypoints.shape)
        )
    kp, _ = keypoints.shape
    h, w = img_shape
    heatmaps = list(
        map(lambda x: _keypoint_to_heatmap(img_shape, x, var=var), keypoints)
    )
    heatmaps = np.stack(heatmaps, axis=-1)
    return heatmaps


def _keypoint_to_heatmap(
    img_shape: Tuple[int, int], keypoint: np.array, var=10.0
) -> np.ndarray:
    """
    imgshape : tuple[int, int]
    keypoint : np.array
        shaped [2]. Keypoints coordinates be in range (-1, 1)
    outputs : [img_shape[0], img_shape[1]]
    """
    if not is_in_range(keypoint, [-1, 1]):
        raise RangeError(keypoint, "keypoints", [-1, 1])
    keypoint_shape = keypoint.shape
    if len(keypoint_shape) != 1:
        raise DimensionError(keypoint, "keypoint", target_dimension=1)
    if len(keypoint) != 2:
        raise ValueError("keypoint has to have 2 elements")
    h, w = img_shape
    x, y = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    rv = multivariate_normal(keypoint, var)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    z = rv.pdf(pos)
    z /= z.max()
    return z


def tile(X, rows, cols):
    """Tile images for display.

    Parameters
    ----------
    X: np.ndarray
        tensor of images shaped [N, H, W, C]. Images have to be in range [-1, 1]
    rows: int
        number of rows for final canvas
    cols: int
        number of rows for final canvas

    Returns
    -------
    np.ndarray
        canvas with images as grid
    """
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype=X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx, ...]
                tiling[
                    i * X.shape[1] : (i + 1) * X.shape[1],  # noqa
                    j * X.shape[2] : (j + 1) * X.shape[2],  # noqa
                    :,
                ] = img
    return tiling


def batch_to_canvas(X, cols=None):
    """convert batch of images to canvas

    Parameters
    ----------
    X : np.ndarray
        tensor of images shaped [N, H, W, C]. Images have to be in range [-1, 1]
    cols : int, optional
        number of columns for the final canvas, by default None

    Returns
    -------
    np.ndarray
        canvas with images as grid
    """
    if len(X.shape) == 5:
        # tile
        oldX = np.array(X)
        n_tiles = X.shape[3]
        side = math.ceil(math.sqrt(n_tiles))
        X = np.zeros(
            (oldX.shape[0], oldX.shape[1] * side, oldX.shape[2] * side, oldX.shape[4]),
            dtype=oldX.dtype,
        )
        # cropped images
        for i in range(oldX.shape[0]):
            inx = oldX[i]
            inx = np.transpose(inx, [2, 0, 1, 3])
            X[i] = tile(inx, side, side)
    n_channels = X.shape[3]
    if n_channels > 4:
        X = X[:, :, :, :3]
    if n_channels == 1:
        X = np.tile(X, [1, 1, 1, 3])
    rc = math.sqrt(X.shape[0])
    if cols is None:
        rows = cols = math.ceil(rc)
    else:
        cols = max(1, cols)
        rows = math.ceil(X.shape[0] / cols)
    canvas = tile(X, rows, cols)
    return canvas


def vstack_paths(paths, out_path):
    cmd = ["convert", "-append"] + paths + [out_path]
    subprocess.call(cmd)


def hstack_paths(paths, out_path):
    cmd = ["convert", "+append"] + paths + [out_path]
    subprocess.call(cmd)


def rotate_bound(image, angle):
    """ Pad and then rotate to prevent image cropping

    References
    ----------
        https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/  # noqa
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# TODO: hstack with padding and vstack with padding


def hstack(*tlist, padding=0):
    """ layout N, H, W, C """
    # TODO: add padding option
    return np.concatenate(tlist, axis=2)


def vstack(*tlist, padding=0):
    """ layout N, H, W, C """
    # TODO: add padding option
    return np.concatenate(tlist, axis=1)
