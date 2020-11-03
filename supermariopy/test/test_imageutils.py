import numpy as np
import pytest


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

    @pytest.mark.mpl_image_compare
    def test_put_text_at_all_locations(self):
        from skimage import data
        from supermariopy.imageutils import put_text
        from matplotlib import pyplot as plt

        astronaut = data.astronaut()
        locs = [
            "center",
            "topleft",
            "topright",
            "top",
            "bottomleft",
            "bottomright",
            "bottom",
        ]
        fig, axes = plt.subplots(1, len(locs), figsize=(6 * len(locs), 6))
        for ax, loc in zip(axes.ravel(), locs):
            annotated = put_text(
                astronaut.copy(), "hi", loc=loc, font_scale=5, color=(255, 255, 255)
            )
            ax.imshow(annotated)
        return fig

    def test_make_colors_01(self):
        from supermariopy.imageutils import make_colors

        colors = make_colors(10)
        expected = [0, 1]
        assert isinstance(colors.flatten()[0], np.floating)
        assert colors.min() >= expected[0] and colors.max() <= expected[1]
        assert colors.shape == (10, 3)

    def test_make_colors_bytes(self):
        from supermariopy.imageutils import make_colors

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

    def test_batch_to_canvas(self):
        from supermariopy.imageutils import batch_to_canvas

        x = np.ones((9, 100, 100, 3))
        canvas = batch_to_canvas(x)
        assert canvas.shape == (300, 300, 3)

        canvas = batch_to_canvas(x, cols=5)
        assert canvas.shape == (200, 500, 3)

        canvas = batch_to_canvas(x, cols=1)
        assert canvas.shape == (900, 100, 3)

        canvas = batch_to_canvas(x, cols=0)
        assert canvas.shape == (900, 100, 3)

        canvas = batch_to_canvas(x, cols=None)
        assert canvas.shape == (300, 300, 3)

    def test_stack_images(self, tmpdir):
        import cv2
        import os

        def setup_tempdir(tmpdir):
            a = np.zeros((100, 100, 3), dtype=np.float32)
            b = np.zeros((100, 100, 3), dtype=np.float32)
            cv2.imwrite(os.path.join(tmpdir, "a.png"), a)
            cv2.imwrite(os.path.join(tmpdir, "b.png"), b)

        setup_tempdir(tmpdir)
        from supermariopy.imageutils import vstack_paths, hstack_paths

        paths = [os.path.join(tmpdir, "a.png"), os.path.join(tmpdir, "b.png")]
        opath = os.path.join(tmpdir, "out.png")
        vstack_paths(paths, opath)
        probe = cv2.imread(opath, -1)
        assert (200, 100) == probe.shape

        paths = [os.path.join(tmpdir, "a.png"), os.path.join(tmpdir, "b.png")]
        opath = os.path.join(tmpdir, "out.png")
        hstack_paths(paths, opath)
        probe = cv2.imread(opath, -1)
        assert (100, 200) == probe.shape
