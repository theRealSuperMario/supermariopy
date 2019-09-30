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
        from supermariopy.imageutils import draw_keypoint_markers, convert_range
        from matplotlib import pyplot as plt

        astronaut = data.astronaut()
        astronaut = convert_range(astronaut, [0, 255], [0, 1])
        keypoints = np.stack([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], axis=1)
        img_marked = draw_keypoint_markers(
            astronaut,
            keypoints,
            font_scale=2,
            thickness=3,
            marker_list=["1", "2", "3", "x", "o", "v"],
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

    def test_keypoints_to_heatmaps(self):
        from supermariopy.imageutils import keypoints_to_heatmaps, RangeError

        keypoints = np.stack([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], axis=1)
        heatmaps = keypoints_to_heatmaps((512, 512), keypoints)
        assert heatmaps.shape == (512, 512, 10)

        keypoints = np.stack([np.arange(10), np.arange(10)], axis=1)
        with pytest.raises(RangeError):
            heatmaps = keypoints_to_heatmaps((512, 512), keypoints)

    def test_is_in_range(self):
        from supermariopy.imageutils import is_in_range, LengthError

        a = np.array([0, 1])
        assert is_in_range(a, [0, 1])
        assert is_in_range(a, [-1, 1])
        assert not is_in_range(a, [0.5, 1])

        with pytest.raises(LengthError):
            is_in_range(a, [0, 1, 2])

        with pytest.raises(ValueError):
            is_in_range(a, [1, 0])

    def test_convert_range(self):
        from supermariopy.imageutils import convert_range

        a = np.array([0, 1])
        with pytest.raises(ValueError):
            convert_range(a, [1, 0], [0, 1])

        with pytest.raises(ValueError):
            convert_range(a, [0, 1], [1, 0])

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


class Test_crf:
    @pytest.mark.mpl_image_compare
    def test_segmentationFromKeypoints(self):
        from supermariopy.crf import SegmentationFromKeypoints
        from supermariopy import imageutils
        from skimage import data

        n_keypoints = 10
        var = 0.05
        keypoints = np.stack(
            [np.linspace(-1, 1, n_keypoints), np.linspace(-1, 1, n_keypoints)], axis=1
        )

        img = data.astronaut()
        segmentation_algorithm = SegmentationFromKeypoints(var)
        labels = segmentation_algorithm(img, keypoints)
        labels_rgb = imageutils.make_colors(n_keypoints + 1)[labels]
        heatmaps = imageutils.keypoints_to_heatmaps(img.shape[:2], keypoints, var)
        heatmaps_rgb = imageutils.colorize_heatmaps(
            heatmaps[np.newaxis, ...], imageutils.make_colors(n_keypoints)
        )
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(labels_rgb)
        axes[0].set_axis_off()

        axes[1].imshow(np.squeeze(heatmaps_rgb))
        axes[1].set_axis_off()
        return fig

    def test_segmentationFromKeypoints_lowRangeError(self):
        from supermariopy.crf import SegmentationFromKeypoints
        from supermariopy import imageutils
        from skimage import data

        n_keypoints = 10
        var = 0.05
        keypoints = np.stack(
            [np.linspace(-1, 1, n_keypoints), np.linspace(-1, 1, n_keypoints)], axis=1
        )

        img = data.astronaut()
        img = imageutils.convert_range(img, [0, 255], [0, 1])
        segmentation_algorithm = SegmentationFromKeypoints(var)
        with pytest.warns(Warning):
            labels = segmentation_algorithm(img, keypoints)


class Test_denseposelib:
    @pytest.mark.parametrize(
        "in_shape,out_shape", [((256, 256), (128, 128)), ((3, 256, 256), (3, 128, 128))]
    )
    def test_resize_labels(self, in_shape, out_shape):
        from supermariopy.denseposelib import resize_labels

        labels = np.random.randint(0, 10, in_shape)
        resized = resize_labels(labels, out_shape[-2:])
        assert resized.shape == out_shape

    def test_compute_iou(self):
        from supermariopy.denseposelib import compute_iou

        A = np.ones((10, 10, 1), dtype=np.int)
        B = np.ones((10, 10, 1), dtype=np.int)
        B[:5, :5] = 0

        iou, unique_labels = compute_iou(A, B)

        assert (float(iou[unique_labels == 1])) == 0.75

