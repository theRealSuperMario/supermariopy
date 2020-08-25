import os

import numpy as np
import tensorflow as tf

from ...tfutils import nn

tf.enable_eager_execution()


def test_filltriangular():
    params = tf.reshape(tf.range(0, 6), (1, 6))
    L = tf.contrib.distributions.fill_triangular(params)
    # assert L.shape == ()
    assert L.shape == (1, 3, 3)
    assert np.allclose(L, np.array([[3, 0, 0], [5, 4, 0], [2, 1, 0]]))


def test_diag_part():
    params = tf.reshape(tf.range(1, 7), (1, 6))
    L = tf.contrib.distributions.fill_triangular(params)
    diag_L = tf.linalg.diag_part(L)
    assert diag_L.shape == (1, 3)
    assert np.allclose(np.squeeze(diag_L), np.array([4, 5, 1]))


def test_set_diag():
    params = tf.reshape(tf.range(1, 7), (1, 6))
    L = tf.contrib.distributions.fill_triangular(params)
    diag_L = tf.ones((1, 3), dtype=params.dtype) * 6
    M = tf.linalg.set_diag(L, diag_L)
    assert np.allclose(tf.linalg.diag_part(M), diag_L)


class Test_FullLatentDistribution:
    def test_n_parameters(self):
        dim = 10
        assert nn.FullLatentDistribution.n_parameters(dim) == 65

    def test_sample(self):
        dim = 10
        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = tf.random_normal((1, n_parameters), dtype=tf.float32)
        distr = nn.FullLatentDistribution(parameters, dim)
        assert distr.sample().shape == (1, dim)

        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = tf.random_normal((10, 1, 1, n_parameters), dtype=tf.float32)
        latent = nn.FullLatentDistribution(parameters, dim, False)
        sample = latent.sample()
        assert sample.shape == (10, 1, 1, dim)

    def test_kl(self):
        dim = 10
        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = tf.random_normal((1, n_parameters), dtype=tf.float32)
        distr = nn.FullLatentDistribution(parameters, dim)
        distr.mean = tf.zeros((1, dim), dtype=parameters.dtype)
        distr.L = tf.linalg.set_diag(
            tf.zeros((1, dim, dim), dtype=parameters.dtype),
            tf.ones((1, dim), dtype=parameters.dtype),
        )
        distr.log_diag_L = tf.zeros((1, dim), dtype=parameters.dtype)
        assert np.allclose(distr.kl(), np.array([0]))


class Test_MeanFieldDistribution:
    def test_kl_improper_gmrf(self):
        dim = (128, 128, 1)
        parameters = tf.zeros((1,) + dim)
        mfd = nn.MeanFieldDistribution(parameters, dim)
        kl = mfd.kl_improper_gmrf()
        assert np.allclose(kl, np.array([0]))

    def test_sample(self):
        dim = (128, 128, 1)
        parameters = tf.zeros((1,) + dim)
        mfd = nn.MeanFieldDistribution(parameters, dim)
        s = mfd.sample()
        assert s.shape == parameters.shape


def test_ema(tmpdir):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    class Model(tf.keras.Model):
        def __init__(self):
            super(Model, self).__init__()
            self.ema = nn.EMA()
            self.ema.init_var("foo", 0.5)

        def call(self, *args):
            pass

    model = Model()
    checkpoint_path = str(tmpdir.mkdir("checkpoints").join("model"))
    tfcheckpoint = tf.train.Checkpoint(model=model)
    tfcheckpoint.write(checkpoint_path)

    model.ema.update("foo", 1.0)

    tfcheckpoint.restore(checkpoint_path)

    assert model.ema.get_value("foo") == 0.5
