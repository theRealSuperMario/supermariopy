import cv2
import numpy as np

COLOR_DICT = {"white": (1, 1, 1), "black": (0, 0, 0), "red": (1, 0, 0)}


def put_text(img, text, loc="center", font_scale=1, thickness=1, color=(-1, -1, -1)):
    """
    small utility function to put text into image using opencv
    # TODO: complete docstring
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


def convert_range(image, input_range, target_range):
    # TODO: doc

    """
    TEST:
    t = imageutils.convert_range(np.array([-1, 1]), [-1, 1], [0, 1])
    assert np.allclose(t, np.array([0, 1]))
    t = imageutils.convert_range(np.array([0, 1]), [0, 1], [-1, 1])
    assert np.allclose(t, np.array([-1, 1]))
    """
    if input_range[1] <= input_range[0]:
        raise ValueError
    if target_range[1] <= target_range[0]:
        raise ValueError
    # offset = target_range[0] - input_range[0]
    # scale = (target_range[1] - target_range[0]) / (input_range[1] - input_range[0])

    a = input_range[0]
    b = input_range[1]
    c = target_range[0]
    d = target_range[1]
    return (image - a) / (b - a) * (d - c) + c
