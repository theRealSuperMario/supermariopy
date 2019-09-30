import cv2
import numpy as np
from typing import *
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


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
        where to put the text. one of `center`, `top`, `topleft`, `bottomleft`, `bottom`, by default "center"
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
        textX = textsize[0]
        textY = textsize[1]
    elif loc == "bottomleft":
        textX = img.shape[0] // 2
        textY = img.shape[0] - textsize[1]
    elif loc == "bottom":
        textX = (img.shape[1] - textsize[0]) // 2
        textY = img.shape[0] - textsize[1] // 2
    # TODO: add all possible palcement options
    font_color = color
    cv2.putText(img, text, (textX, textY), font, font_scale, font_color, thickness=3)
    return img


def make_colors(
    n_classes: int, cmap: Callable = plt.cm.inferno, bytes: bool = False
) -> np.ndarray:
    """make a color array using the specified colormap for `n_classes` classes

    Parameters
    ----------
    n_classes: int
        how many classes there are in the mask
    cmap: Callable, optional, by default plt.cm.inferno
        matplotlib colormap handle
    bytes: bool, optional, by default False
        bytes option passed to `cmap`. Returns colors in range [0, 1] if False and range [0, 255] if True

    Returns
    -------
    colors: ndarray
        an array with shape [n_classes, 3] representing colors in the range [0, 1].

    """
    colors = cmap(np.linspace(0, 1, n_classes), alpha=False, bytes=bytes)[:, :3]
    return colors


def draw_keypoint_markers(
    img: np.ndarray,
    keypoints: np.ndarray,
    font_scale: int = 1,
    thickness: int = 2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
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
    # TODO: check image range
    if not is_in_range(img, [0, 1]):
        raise RangeError(img, "img", [0, 1])
    if img.shape[0] != img.shape[1]:
        raise ValueError("only square images are supported currently")
    marker_list = ["o", "v", "x", "+", "<", "-", ">", "c"]
    img_marked = img.copy()
    keypoints = convert_range(keypoints, [-1, 1], [0, img.shape[0] - 1])
    colors = make_colors(keypoints.shape[0], bytes=False, cmap=plt.cm.Set1)
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
        heatmaps in shape [N, H, W, C]. Each item along axis [0 and 3] is a heatmap with maximum 1 and minimum 0
    colors : np.ndarray
        array with colors in shape [C, 3]. use @make_colors.

    Returns
    -------
    np.ndarray
        colorized heatmaps in shape [N, H, W, 3]
    
    Examples
    --------
        n_parts = 10
        keypoints = np.stack([np.linspace(-1, 1, n_parts), np.linspace(-1, 1, n_parts)], axis=1)
        heatmaps = imageutils.CRF().keypoints_to_heatmaps((h, w), keypoints, var=0.01)
        heatmaps_colorized = imageutils.colorize_heatmaps(heatmaps[np.newaxis, ...], imageutils.make_colors(n_parts))
        plt.imshow(np.squeeze(heatmaps_colorized))
    """
    N, H, W, C = heatmaps.shape
    heatmaps_colorized = np.expand_dims(heatmaps, -1) * colors.reshape(
        (1, 1, 1, C, 3)
    )  # [N, H, W, C, 3]
    heatmaps_colorized = np.sum(heatmaps_colorized, axis=3)
    return heatmaps_colorized


def np_map_fn(func: Callable, data: Tuple) -> Tuple:
    """map func along axis 0 of each item in data.

    # TODO: fails when tuple has length 1

    Similar to tf.map_fn
    
    Parameters
    ----------
    func : Callab
        function to map to the items in data
    data : Tuple[np.ndarray]
        
    Returns
    -------
    Tuple[np.ndarray]
        function `func` applied to each element in `data`

    Examples
    --------
        data = (np.arange(10).reshape(10, 1), np.arange(10)[::-1].reshape(10, 1))
        output = np_map_fn(lambda x: (x[0]**2, x[1]**2), data)
        output[0].squeeze()
        >>> array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        output = np_map_fn(lambda x: (x[0]**2, x[1]**2), data)
        output[0].shape
        >>> (10, 1)
    """
    generator = zip(*map(lambda x: [x[i, ...] for i in range(x.shape[0])], data))
    # (data[0][0], data[0][1], ...), (data[1][0], data[1][1], ...), ...
    outputs = map(func, generator)
    outputs = list(map(np.stack, zip(*outputs)))
    return outputs


def keypoints_to_heatmaps(
    img_shape: Tuple[int, int], keypoints: np.ndarray, var=0.01
) -> np.ndarray:
    """
    imgshape : tuple[int, int]
    keypoints : np.ndarray
        shaped [kp, 2]. Keypoints coordinates be in range (-1, 1). keypoints[:, 0] is horizontal coordinate (x)
    var : float
        variance of blobs to create around keypoint. relative to keypoints coordinates range (-1, 1)
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

