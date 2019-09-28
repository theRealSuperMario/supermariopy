import pytest
import numpy as np
from matplotlib import pyplot as plt


class Test_Imageutils:
    @pytest.mark.mpl_image_compare
    def test_put_text(self):
        from skimage import data
        from supermariopy.imageutils import put_text
        from matplotlib import pyplot as plt

        astronaut = data.astronaut()
        annotated = put_text(astronaut.copy(), "hi", loc="center")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(annotated)
        return fig

    def test_make_colors_01(self):
        from supermariopy.imageutils import make_colors
        from matplotlib import pyplot as plt

        colors = make_colors(10)
        expected = [0, 1]
        assert isinstance(colors.flatten()[0], np.floating)
        assert colors.min() >= expected[0] and colors.max() <= expected[1]
        assert colors.shape == (10, 3)

    def test_make_colors_bytes(self):
        from supermariopy.imageutils import make_colors
        from matplotlib import pyplot as plt

        colors = make_colors(10, bytes=True)
        expected = [0, 255]
        assert isinstance(colors.flatten()[0], np.integer)
        assert colors.min() >= expected[0] and colors.max() <= expected[1]
        assert colors.shape == (10, 3)

    @pytest.mark.mpl_image_compare
    def test_draw_keypoint_markers(self):
        from skimage import data
        from supermariopy.imageutils import draw_keypoint_markers
        from matplotlib import pyplot as plt

        astronaut = data.astronaut()
        keypoints = np.stack([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], axis=1)
        img_marked = draw_keypoint_markers(
            astronaut, keypoints, font_scale=2, thickness=3
        )
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_marked)
        ax.plot(np.arange(512), np.arange(512))
        return fig

    @pytest.mark.mpl_image_compare
    def test_colorize_heatmaps(self):
        from supermariopy.imageutils import (
            keypoints_to_heatmaps,
            make_colors,
            colorize_heatmaps,
        )
        from matplotlib import pyplot as plt

        keypoints = np.stack([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], axis=1)
        heatmaps = keypoints_to_heatmaps((512, 512), keypoints)
        colors = make_colors(keypoints.shape[0])
        img_marked = colorize_heatmaps(heatmaps[np.newaxis, ...], colors)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(np.squeeze(img_marked))
        return fig

    # def test_dimension_error(self):
    #     import numpy as np
    #     from supermariopy.imageutils import DimensionError

    #     a = np.arange(10)
    #     with pytest.raises(DimensionError):
    #         raise DimensionError(a, "a", 1)

    #     with pytest.raises(DimensionError):
    #         raise DimensionError(a.reshape((1, 1, 10)), "a", 3)

    # def test_shape_error(self):
    #     import numpy as np
    #     from supermariopy.imageutils import ShapeError

    #     a = np.arange(10)
    #     with pytest.raises(DimensionError):
    #         raise Shape(a, "a", 1)

    #     with pytest.raises(DimensionError):
    # raise DimensionError(a.reshape((1, 1, 10)), "a", 3)

