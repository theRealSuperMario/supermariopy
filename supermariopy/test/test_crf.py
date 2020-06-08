import pytest
import numpy as np
from matplotlib import pyplot as plt


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
