import tensorflow as tf
from supermariopy.tfutils import losses as smlosses
import numpy as np
from tensorflow.contrib.keras.api.keras.applications.vgg19 import VGG19


class TestVGG19Features:
    def test_eager(self):
        tf.enable_eager_execution()
        vgg = smlosses.VGG19Features(
            session=None, default_gram=0.0, original_scale=True, eager=True
        )

        from skimage import data

        x = data.astronaut().astype(np.float32).reshape((1, 512, 512, 3))
        y = data.astronaut().astype(np.float32).reshape((1, 512, 512, 3))
        y += tf.random_normal(y.shape, dtype=tf.float32)
        loss = vgg.make_loss_op(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
        assert loss


class Test_PerceptualVGG:
    def test_eager(self):
        tf.enable_eager_execution()
        vgg = VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        perceptual_vgg = smlosses.PerceptualVGG(vgg=vgg, eager=True)

        from skimage import data
        from supermariopy.tfutils import image

        x = data.astronaut().astype(np.float32).reshape((1, 512, 512, 3))
        x = image.resize_bilinear(x, [224, 224])
        y = data.astronaut().astype(np.float32).reshape((1, 512, 512, 3))
        y = image.resize_bilinear(y, [224, 224])
        loss = perceptual_vgg.loss(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
        assert all([np.allclose(l, np.array([0])) for l in loss])

    def test_session(self):
        vgg = VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        session = tf.compat.v1.Session()
        perceptual_vgg = smlosses.PerceptualVGG(vgg=vgg, eager=False, session=session)

        from skimage import data
        from supermariopy.tfutils import image

        x = (
            data.astronaut()
            .astype(np.float32)
            .reshape((1, 512, 512, 3))[:, :224, :224, :]
        )
        y = (
            data.astronaut()
            .astype(np.float32)
            .reshape((1, 512, 512, 3))[:, :224, :224, :]
        )

        x_ph = tf.placeholder(shape=(1, 224, 224, 3), dtype=tf.float32)
        y_ph = tf.placeholder(shape=(1, 224, 224, 3), dtype=tf.float32)

        loss = perceptual_vgg.loss(x_ph, y_ph)
        session.run(tf.initialize_all_variables())
        loss_v = session.run(loss, {x_ph: x, y_ph: y})
        assert all([np.allclose(l, np.array([0])) for l in loss_v])
