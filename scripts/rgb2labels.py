import numpy as np
from matplotlib import pyplot as plt

"""

https://stackoverflow.com/questions/42750910/convert-rgb-image-to-index-image/62980021#62980021
convert semantic labels from RGB coding to index coding
Steps:
1. define COLORS (see below)
2. hash colors
3. run rgb2index(segmentation_rgb)

see example below
TODO: apparently, using cv2.LUT is much simpler (and maybe faster?)
"""


COLORS = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0]])
W = np.power(255, [0, 1, 2])

HASHES = np.sum(W * COLORS, axis=-1)
HASH2COLOR = {h: c for h, c in zip(HASHES, COLORS)}
HASH2IDX = {h: i for i, h in enumerate(HASHES)}


def rgb2index(segmentation_rgb):
    """
    turn a 3 channel RGB color to 1 channel index color
    """
    s_shape = segmentation_rgb.shape
    s_hashes = np.sum(W * segmentation_rgb, axis=-1)
    print(np.unique(segmentation_rgb.reshape((-1, 3)), axis=0))
    func = lambda x: HASH2IDX[int(x)]  # noqa
    segmentation_idx = np.apply_along_axis(func, 0, s_hashes.reshape((1, -1)))
    segmentation_idx = segmentation_idx.reshape(s_shape[:2])
    return segmentation_idx


segmentation = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]] * 3).reshape((3, 3, 3))
segmentation_idx = rgb2index(segmentation)
print(segmentation)
print(segmentation_idx)


fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(segmentation)
axes[0].set_title("Segmentation RGB")
axes[1].imshow(segmentation_idx)
axes[1].set_title("Segmentation IDX")
plt.show()
