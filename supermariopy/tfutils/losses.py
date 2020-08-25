from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.applications.vgg19 import VGG19
from tensorflow.contrib.keras.api.keras.models import Model

from ..tfutils import image


"""
In part from
https://github.com/pesser/edflow/blob/dev/edflow/applications/tf_perceptual_loss.py
"""


def _ll_loss(target, reconstruction, log_variance, calibrate):
    dim = np.prod(target.shape.as_list()[1:])
    variance = tf.exp(log_variance)
    log2pi = np.log(2.0 * np.pi)
    e = tf.reduce_mean(tf.square(target - reconstruction))  # noqa
    l = 0.5 * dim * (e / variance + log_variance + log2pi)  # noqa
    # if calibrate:
    #     calibrate_op = tf.assign(log_variance, tf.log(e))
    # else:
    #     calibrate_op = tf.no_op()
    return l, calibrate


class VGG19Features(object):
    def __init__(
        self,
        session,
        feature_layers=None,
        feature_weights=None,
        gram_weights=None,
        default_gram=0.1,
        original_scale=False,
        eager=False,
    ):
        if eager:
            pass
        else:
            K.set_session(session)
        self.base_model = VGG19(include_top=False, weights="imagenet")
        if feature_layers is None:
            feature_layers = [
                "input_1",
                "block1_conv2",
                "block2_conv2",
                "block3_conv2",
                "block4_conv2",
                "block5_conv2",
            ]
        self.layer_names = [layer.name for layer in self.base_model.layers]
        for k in feature_layers:
            if k not in self.layer_names:
                raise KeyError(
                    "Invalid layer {}. Available layers: {}".format(k, self.layer_names)
                )
        self.feature_layers = feature_layers
        features = [self.base_model.get_layer(k).output for k in feature_layers]
        self.model = Model(inputs=self.base_model.input, outputs=features)
        if feature_weights is None:
            feature_weights = len(feature_layers) * [1.0]
        if gram_weights is None:
            gram_weights = len(feature_layers) * [default_gram]
        elif isinstance(gram_weights, (int, float)):
            gram_weights = len(feature_layers) * [gram_weights]
        self.feature_weights = feature_weights
        self.gram_weights = gram_weights
        assert len(self.feature_weights) == len(features)
        self.use_gram = np.max(self.gram_weights) > 0.0
        self.original_scale = original_scale

        self.variables = self.base_model.weights

    def extract_features(self, x):
        """x should be rgb in [-1,1]."""
        x = self.preprocess_input(x)
        features = self.model.predict(x)
        return features

    def make_features_op(self, x):
        """y should be rgb tensors in [-1, 1]. This function is just used for testing"""
        if self.original_scale:
            x = tf.image.resize_bilinear(x, [256, 256])
            x = x[:, 16:239, 16:239, :]

        x = self.preprocess_input(x)
        x_features = self.model(x)
        return x_features

    def grams(self, fs):
        gs = list()
        for f in fs:
            bs, h, w, c = f.shape.as_list()
            bs = -1 if bs is None else bs
            f = tf.reshape(f, [bs, h * w, c])
            ft = tf.transpose(f, [0, 2, 1])
            g = tf.matmul(ft, f)
            g = g / (4.0 * h * w)
            gs.append(g)
        return gs

    def make_loss_op(self, x, y):
        """x, y should be rgb tensors in [-1,1]. Uses l1 and spatial average."""
        if self.original_scale:
            xy = tf.concat([x, y], axis=0)
            xy = tf.image.resize_bilinear(xy, [256, 256])
            bs = tf.shape(xy)[0]
            xy = tf.random_crop(xy, [bs, 224, 224, 3])
            x, y = tf.split(xy, 2, 0)

        x = self.preprocess_input(x)
        x_features = self.model(x)

        y = self.preprocess_input(y)
        y_features = self.model(y)

        x_grams = self.grams(x_features)
        y_grams = self.grams(y_features)

        losses = [
            tf.reduce_mean(tf.abs(xf - yf)) for xf, yf in zip(x_features, y_features)
        ]
        gram_losses = [
            tf.reduce_mean(tf.abs(xg - yg)) for xg, yg in zip(x_grams, y_grams)
        ]

        for i in range(len(losses)):
            losses[i] = self.feature_weights[i] * losses[i]
            gram_losses[i] = self.gram_weights[i] * gram_losses[i]
        loss = tf.add_n(losses)
        if self.use_gram:
            loss = loss + tf.add_n(gram_losses)

        self.losses = losses
        self.gram_losses = gram_losses

        return loss

    def make_nll_op(self, x, y, log_variances, gram_log_variances=None, calibrate=True):
        """x, y should be rgb tensors in [-1,1]. This version treats every
        layer independently."""
        use_gram = gram_log_variances is not None
        if self.original_scale:
            xy = tf.concat([x, y], axis=0)
            xy = image.resize_bilinear(xy, [256, 256])
            bs = tf.shape(xy)[0]
            xy = tf.random_crop(xy, [bs, 224, 224, 3])
            x, y = tf.split(xy, 2, 0)

        x = self.preprocess_input(x)
        x_features = self.model(x)

        y = self.preprocess_input(y)
        y_features = self.model(y)

        if use_gram:
            x_grams = self.grams(x_features)
            y_grams = self.grams(y_features)

        if len(log_variances) == 1:
            log_variances = len(x_features) * [log_variances[0]]

        feature_ops = [
            _ll_loss(xf, yf, logvar, calibrate=calibrate)
            for xf, yf, logvar in zip(x_features, y_features, log_variances)
        ]
        losses = [f[0] for f in feature_ops]
        self.losses = losses
        calibrations = [f[1] for f in feature_ops]
        self.calibrations = calibrations
        if use_gram:
            gram_ops = [
                _ll_loss(xg, yg, glogvar)
                for xg, yg, glogvar in zip(x_grams, y_grams, gram_log_variances)
            ]
            gram_losses = [g[0] for g in gram_ops]
            self.gram_losses = gram_losses
            gram_calibrations = [g[1] for g in gram_ops]
            self.gram_calibrations = gram_calibrations

        loss = tf.add_n(losses)
        if use_gram:
            loss = loss + tf.add_n(gram_losses)

        return loss

    def make_l1_nll_op(self, x, y, log_variance):
        """x, y should be rgb tensors in [-1,1]. Uses make_loss_op to compute
        version compatible with previous experiments."""

        rec_loss = 1e-3 * self.make_loss_op(x, y)
        dim = np.prod(x.shape.as_list()[1:])
        log_gamma = log_variance
        gamma = tf.exp(log_gamma)
        log2pi = np.log(2.0 * np.pi)
        likelihood = 0.5 * dim * (rec_loss / gamma + log_gamma + log2pi)

        return likelihood

    def make_style_op(self, x, y):
        __feature_weights = self.feature_weights
        __gram_weights = self.gram_weights
        self.feature_weights = [0.01 for _ in __feature_weights]
        self.gram_weights = [1.0 for _ in __gram_weights]
        loss = self.make_loss_op(x, y)
        self.feature_weights = __feature_weights
        self.gram_weights = __gram_weights
        return loss

    def preprocess_input(self, x):
        """Preprocesses a tensor encoding a batch of images.

        1. Transform range [-1, 1] to [0, 255.0]
        2. center around imagenet mean RGB values

        Parameters
        ----------
        x : tf.Tenser
            input tensor, 4D in [-1,1]

        Returns
        -------
        Preprocessed tensor : tf.Tensor
        """
        # from [-1, 1] to [0, 255.0]
        x = (x + 1.0) / 2.0 * 255.0
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x = x - np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3))
        return x


def l1_loss(x, y):
    diff = tf.math.abs(x - y)
    return tf.reduce_mean(diff)


class PerceptualVGG(object):
    def __init__(
        self,
        vgg,
        feature_weights=[1.0] * 6,
        use_gram=False,
        gram_weights=[0.1] * 6,
        eager=False,
        session=None,
    ):
        self.vgg = vgg
        self.feature_weights = feature_weights
        self.gram_weights = gram_weights
        self.use_gram = use_gram
        self.target_layers = [
            # "input_1",
            "block1_conv2",
            "block2_conv2",
            "block3_conv2",
            "block4_conv2",
            "block5_conv2",
        ]

        if eager:
            pass
        else:
            K.set_session(session)
        self.layer_names = [layer.name for layer in self.vgg.layers]
        for k in self.target_layers:
            if k not in self.layer_names:
                raise KeyError(
                    "Invalid layer {}. Available layers: {}".format(k, self.layer_names)
                )
        features = [self.vgg.get_layer(k).output for k in self.target_layers]
        self.model = Model(inputs=self.vgg.input, outputs=features)
        self.variables = self.vgg.weights

    def grams(self, fs: List):
        gs = list()
        for f in fs:
            bs, h, w, c = f.shape.as_list()
            bs = -1 if bs is None else bs
            f = tf.reshape(f, [bs, h * w, c])
            ft = tf.transpose(f, [0, 2, 1])
            g = tf.matmul(ft, f)
            g = g / (4.0 * h * w)
            gs.append(g)
        return gs

    def forward(self, x):
        """ x in range [0, 255] and shaped [B, 224, 224, 3]"""

        # RGB --> BGR
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x = x - np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3))
        # Divide not by standard distribution

        out = {"input": x}

        out_ = self.model(x)
        for i, k in enumerate(self.target_layers):
            out[k] = out_[i]
        return out

    def loss(self, target: tf.Tensor, pred: tf.Tensor):
        VGGOutput = self.target_layers
        # weights = self.feature_weights
        target_feats = self.forward(target)
        target_feats = [target_feats[k] for k in VGGOutput]
        pred_feats = self.forward(pred)
        pred_feats = [pred_feats[k] for k in VGGOutput]

        criterion = l1_loss

        losses = [
            tf.expand_dims(criterion(xf, yf), axis=-1)
            for xf, yf in zip(target_feats, pred_feats)
        ]

        if self.use_gram:
            target_grams = self.grams(target_feats)
            pred_grams = self.grams(pred_feats)
            gram_losses = [
                tf.expand_dims(criterion(xf, yf), axis=-1)
                for xf, yf in zip(target_grams, pred_grams)
            ]
            losses = losses + gram_losses

        return losses
