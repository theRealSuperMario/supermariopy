import pytest
import numpy as np


class Test_Plotting:
    @pytest.mark.mpl_image_compare
    def test_add_colorbars_to_axes(self):
        from supermariopy.plotting import add_colorbars_to_axes
        from matplotlib import pyplot as plt

        plt.subplot(121)
        plt.imshow(np.arange(100).reshape((10, 10)))
        plt.subplot(122)
        plt.imshow(np.arange(100).reshape((10, 10)))
        add_colorbars_to_axes()
        plt.show()
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_set_all_axis_off(self):
        from supermariopy.plotting import set_all_axis_off
        from matplotlib import pyplot as plt

        plt.subplot(121)
        plt.imshow(np.arange(100).reshape((10, 10)))
        plt.subplot(122)
        plt.imshow(np.arange(100).reshape((10, 10)))
        set_all_axis_off()
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_imageStack_2_subplots(self):
        from supermariopy.plotting import imageStack_2_subplots
        from matplotlib import pyplot as plt

        images = np.stack([np.arange(100).reshape((10, 10))] * 3)
        fig, axes = imageStack_2_subplots(images, axis=0)
        return fig

    @pytest.mark.mpl_image_compare
    def test_change_linewidth(self):
        from supermariopy.plotting import change_linewidth
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1)
        x = np.arange(10)
        y = np.arange(10)
        ax.plot(x, y, x + 1, y, x - 1, y)
        change_linewidth(ax, 3)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_change_fontsize(self):
        from supermariopy.plotting import change_fontsize
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(10), np.arange(10))
        change_fontsize(ax, 5)
        return plt.gcf()
