import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


def get_test_image():
    """Test image representing coordinate grid

        --------------------
        |  |  |  |  |  |  |
        --------------------
        |  |  |  |  |  |  |
        --------------------
        |  |  |  |  |  |  |
        --------------------
        |  |  |  |  |  |  |
        --------------------

    Returns
    -------
    """
    N = 256
    d = 25
    image = np.zeros((1, 3, N, N), dtype=np.float32)
    image[:, :, ::d, :] = 1.0
    image[:, :, :, ::d] = 1.0
    image = torch.from_numpy(image)
    image = torch.nn.functional.interpolate(image, (N, N))
    return image


class Test_TPS:
    @pytest.mark.mpl_image_compare
    def test_tps_params(self):
        from supermariopy.ptutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        bs = 1
        scal = 1.0
        tps_scal = 0.05
        rot_scal = 0.1
        off_scal = 0.15
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[-1], image.shape[1]
        )

        assert t_images.shape == image.shape

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0, ...]).permute((1, 2, 0)))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]).permute((1, 2, 0)))

        return plt.gcf()
