import numpy as np
import torch
import torch.nn as nn


class TestVGGFeatures:
    def test_forward(self):
        import tensorflow as tf
        from supermariopy.tfutils.losses import VGG19Features
        from skimage import data

        xph = tf.placeholder(tf.float32, shape=(None, 512, 512, 3))

        torch.manual_seed(7)

        x = data.astronaut().astype(np.float32).reshape((1, 512, 512, 3))
        x = x / 255.0 * 2 - 1.0
        x = np.transpose(x, (0, 3, 1, 2))

        x_tf = np.transpose(x, (0, 2, 3, 1))
        x = torch.from_numpy(x)

        with tf.Session() as sess:
            net = VGG19Features(session=sess, original_scale=True)
            features_op = net.make_features_op(xph)

            features_tf = sess.run(features_op, {xph: x_tf})

        vgg = nn.VGGFeatures(original_scale=True)
        features_pt = vgg.make_features_op(x)

        features_pt = [fpt.permute(0, 2, 3, 1).detach().numpy() for fpt in features_pt]
        for ftf, fpt in zip(features_tf, features_pt):
            # checks.append(np.allclose(ftf, fpt, atol=1.0e-1))
            assert ftf.shape == fpt.shape

        # TODO: find a way to check this
        # TODO: maybe I need to use the weights from the keras library
        # assert checks.all()

        # TODO: compare each features individually
