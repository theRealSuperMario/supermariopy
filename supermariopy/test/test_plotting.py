import numpy as np
import pytest
import skimage
from matplotlib import pyplot as plt
from supermariopy import imageutils
from supermariopy import plotting as smplt


class Test_Plotting:
    @pytest.mark.mpl_image_compare
    def test_add_colorbars_to_axes(self):
        plt.subplot(121)
        plt.imshow(np.arange(100).reshape((10, 10)))
        plt.subplot(122)
        plt.imshow(np.arange(100).reshape((10, 10)))
        smplt.add_colorbars_to_axes()
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_set_all_axis_off(self):
        plt.subplot(121)
        plt.imshow(np.arange(100).reshape((10, 10)))
        plt.subplot(122)
        plt.imshow(np.arange(100).reshape((10, 10)))
        smplt.set_all_axis_off()
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_imageStack_2_subplots(self):
        images = np.stack([np.arange(100).reshape((10, 10))] * 3)
        fig, axes = smplt.imageStack_2_subplots(images, axis=0)
        return fig

    @pytest.mark.mpl_image_compare
    def test_change_linewidth(self):
        fig, ax = plt.subplots(1, 1)
        x = np.arange(10)
        y = np.arange(10)
        ax.plot(x, y, x + 1, y, x - 1, y)
        smplt.change_linewidth(ax, 3)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_change_fontsize(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(10), np.arange(10))
        smplt.change_fontsize(ax, 5)
        return plt.gcf()

    def test_colorpalettes(self):
        name = "msoffice"
        palette = smplt.get_palette(name, bytes=False)
        assert all((palette >= 0.0).ravel()) and all((palette <= 1.0).ravel())
        palette = smplt.get_palette(name, bytes=True)
        assert all((palette >= 0.0).ravel()) and all((palette <= 255.0).ravel())

        name = "sns_coolwarm"
        palette = smplt.get_palette(name, bytes=False)
        assert all((palette >= 0.0).ravel()) and all((palette <= 1.0).ravel())
        palette = smplt.get_palette(name, bytes=True)
        assert all((palette >= 0.0).ravel()) and all((palette <= 255.0).ravel())

        name = "plt_coolwarm"
        palette = smplt.get_palette(name, bytes=False)
        assert all((palette >= 0.0).ravel()) and all((palette <= 1.0).ravel())
        palette = smplt.get_palette(name, bytes=True)
        assert all((palette >= 0.0).ravel()) and all((palette <= 255.0).ravel())

    @pytest.mark.mpl_image_compare
    def test_plot_canvas(self):
        image = np.ones((128, 128, 3), dtype=np.uint8)
        image_stack = np.stack([image * i for i in range(25)], axis=0)

        canvas = imageutils.batch_to_canvas(image_stack)
        fig, ax = smplt.plot_canvas(canvas, 128, 128)
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_to_image(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(np.arange(10), np.arange(10))
        image = smplt.figure_to_image(fig)
        assert image.shape == (1, 500, 500, 3)

    @pytest.mark.mpl_image_compare
    def test_plot_bars(self):
        m = np.arange(20)
        fig, ax = smplt.plot_bars(m)
        return fig

    @pytest.mark.mpl_image_compare
    def test_overlay_boxes_without_labels(self):
        fig, ax = plt.subplots(1, 1)

        image = skimage.data.astronaut()
        bboxes = [np.array([0, 0, 50, 50])]

        overlay = smplt.overlay_boxes_without_labels(image, bboxes)

        ax.imshow(overlay)
        return fig
