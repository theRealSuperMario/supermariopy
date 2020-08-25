import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

tf.enable_eager_execution()
tf.random.set_random_seed(42)


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
    image = np.zeros((1, N, N, 3), dtype=np.float32)
    image[:, ::d, :, :] = 1.0
    image[:, :, ::d, :] = 1.0
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (N, N))
    return image


class Test_TPS:
    @pytest.mark.mpl_image_compare
    def test_tps_params(self):
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
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
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_tps_no_transform_params(self):
        from supermariopy.tfutils.tps import (
            make_input_tps_param,
            ThinPlateSpline,
            tps_parameters,
            no_transformation_parameters,
        )

        tf.random.set_random_seed(42)
        trf_args = no_transformation_parameters(1)
        tps_param_dic = tps_parameters(**trf_args)
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.parametrize("tps_scal", [0.05, 0.1, 0.15, 0.2, 0.25])
    @pytest.mark.mpl_image_compare
    def test_tps_param__tps_scal(self, tps_scal):
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
        bs = 1
        scal = 1.0
        rot_scal = 0.1
        off_scal = 0.1
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.parametrize("rot_scal", [0.05, 0.1, 0.15, 0.2, 0.25])
    @pytest.mark.mpl_image_compare
    def test_tps_param__rot_scal(self, rot_scal):
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
        bs = 1
        scal = 1.0
        tps_scal = 0.1
        off_scal = 0.1
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.parametrize("off_scal", [0.05, 0.1, 0.15, 0.2, 0.25])
    @pytest.mark.mpl_image_compare
    def test_tps_param__off_scal(self, off_scal):
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
        bs = 1
        scal = 1.0
        tps_scal = 0.1
        rot_scal = 0.1
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.parametrize("augm_scal", [1.2, 1.1, 1.0, 0.9, 0.8])
    @pytest.mark.mpl_image_compare
    def test_tps_param__augm_scal(self, augm_scal):
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
        bs = 1
        scal = 1.0
        tps_scal = 0.1
        rot_scal = 0.1
        off_scal = 0.1
        scal_var = 0.05
        # augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.parametrize("scal", [1.2, 1.1, 1.0, 0.9, 0.8])
    @pytest.mark.mpl_image_compare
    def test_tps_param__scal(self, scal):
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
        bs = 1
        # scal = 1.0
        tps_scal = 0.1
        rot_scal = 0.1
        off_scal = 0.1
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )
        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_tps_single_image(self):
        """
        Test single image
        Parameters
        ----------

        Returns
        -------
        """
        from supermariopy.tfutils.tps import (
            tps_parameters,
            make_input_tps_param,
            ThinPlateSpline,
        )

        tf.random.set_random_seed(42)
        bs = 1
        scal = 1.0
        tps_scal = 0.1
        rot_scal = 0.1
        off_scal = 0.1
        scal_var = 0.05
        augm_scal = 1.0

        tps_param_dic = tps_parameters(
            bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
        )

        image = get_test_image()

        coord, vector = make_input_tps_param(tps_param_dic)
        t_images, t_mesh = ThinPlateSpline(
            image, coord, vector, image.shape[1], image.shape[-1]
        )

        with plt.rc_context({"figure.figsize": [10, 5]}):
            plt.subplot(121)
            plt.imshow(np.squeeze(image[0]))
            plt.subplot(122)
            plt.imshow(np.squeeze(t_images[0]))
        return plt.gcf()
