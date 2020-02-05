import tensorflow as tf
from supermariopy.tfutils import losses as smlosses
import numpy as np


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
