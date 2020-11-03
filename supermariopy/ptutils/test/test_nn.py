import numpy as np
import pytest
import torch
from supermariopy.ptutils import nn


def test_spatial_softmax():
    t = torch.rand(1, 10, 128, 128)
    probs = nn.spatial_softmax(t)
    assert nn.shape_as_list(probs) == [1, 10, 128, 128]
    assert np.allclose(torch.sum(probs, dim=[2, 3]).numpy(), np.ones((1, 10)))


def test_grad():
    t = torch.rand(1, 10, 128, 128, dtype=torch.float32)
    g = nn.grad(t)
    assert np.allclose(nn.shape_as_list(g), [1, 20, 128, 128])


def test_mumford_shah():
    t = torch.rand(1, 10, 128, 128, dtype=torch.float32)
    alpha = 1.0
    lambda_ = 1.0
    r, s, c = nn.mumford_shah(t, alpha, lambda_)
    assert True


def test_filltriangular():
    params = torch.range(0, 5).view(1, 6)
    dim = 3
    L = nn.fill_triangular(params, dim)
    # assert L.shape == ()
    assert L.shape == (1, 3, 3)
    assert np.allclose(L, np.array([[3, 0, 0], [5, 4, 0], [2, 1, 0]]))


def test_diag_part():
    params = torch.range(1, 6).view(1, 6)
    dim = 3
    L = nn.fill_triangular(params, dim)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    assert diag_L.shape == (1, 3)
    assert np.allclose(diag_L.numpy().squeeze(), np.array([4, 5, 1]))


def test_set_diag():
    with torch.enable_grad():
        params = torch.range(1, 6, requires_grad=True).view(1, 6)
        dim = 3
        L = nn.fill_triangular(params, dim)
        diag_L = torch.ones((1, 3), dtype=params.dtype, requires_grad=True) * 6
        M = nn.set_diag(L, diag_L)
        loss = M.sum()
        loss.backward()
    assert M.grad_fn  # is not None
    assert L.grad_fn  # is not None
    assert np.allclose(nn.diag_part(M).detach(), diag_L.detach())


class Test_FullLatentDistribution:
    def test_n_parameters(self):
        dim = 10
        assert nn.FullLatentDistribution.n_parameters(dim) == 65

    def test_sample(self):
        dim = 10
        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = torch.rand((1, n_parameters), dtype=torch.float32)
        distr = nn.FullLatentDistribution(parameters, dim)
        assert distr.sample().shape == (1, dim)

        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = torch.rand(10, n_parameters, 1, 1)
        latent = nn.FullLatentDistribution(parameters, dim, False)
        sample = latent.sample()
        assert sample.shape == (10, dim, 1, 1)

    def test_kl(self):
        dim = 10
        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = torch.rand((1, n_parameters), dtype=torch.float32)
        distr = nn.FullLatentDistribution(parameters, dim)
        distr.mean = torch.zeros((1, dim), dtype=parameters.dtype)
        distr.L = nn.set_diag(
            torch.zeros((1, dim, dim), dtype=parameters.dtype),
            torch.ones((1, dim), dtype=parameters.dtype),
        )
        distr.log_diag_L = torch.zeros((1, dim), dtype=parameters.dtype)
        assert np.allclose(distr.kl(), np.array([0]))

    @pytest.mark.cuda
    def test_cuda(self):
        # TODO: test this
        pass

    def test_tf_implementation(self):
        dim = 10
        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = torch.rand((1, n_parameters), dtype=torch.float32)
        distr = nn.FullLatentDistribution(parameters, dim)

        kl_pt = distr.kl()

        import tensorflow as tf

        tf.enable_eager_execution()
        from supermariopy.tfutils import nn as tfnn

        distr_tf = tfnn.FullLatentDistribution(
            tf.convert_to_tensor(parameters.numpy()), dim
        )
        kl_tf = distr_tf.kl()

        assert np.allclose(kl_tf, kl_pt)


class Test_MeanFieldDistribution:
    def test_kl_improper_gmrf(self):
        dim = (1, 128, 128)
        parameters = torch.zeros((1,) + dim)
        mfd = nn.MeanFieldDistribution(parameters, dim)
        kl = mfd.kl_improper_gmrf()
        assert np.allclose(kl, np.array([0]))

    def test_sample(self):
        dim = (1, 128, 128)
        parameters = torch.zeros((1,) + dim)
        mfd = nn.MeanFieldDistribution(parameters, dim)
        s = mfd.sample()
        assert s.shape == parameters.shape

    @pytest.mark.cuda
    def test_cuda(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                return super(Foo, self).__init__()

            def forward(self, x):
                gmrf = nn.MeanFieldDistribution(x, True)
                return gmrf.sample()

        assert torch.cuda.is_available()
        device = torch.device("cuda")
        model = Foo()
        with torch.autograd.set_detect_anomaly(True):
            with torch.cuda.device(device):
                p = torch.rand(1, 10, 128, 128, requires_grad=True)
                p = p.to(device)
                model = model.to(device)
                sample = model(p)

                # device.type is {cuda, cpu}
                assert sample.is_cuda
                assert p.is_cuda
                loss = sample.mean()
                loss.backward()


class Test_to_one_hot:
    @pytest.mark.cuda
    def test_cuda(self):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        with torch.cuda.device(device):
            x = torch.zeros(1, 128, 128)
            x[:, :50, :50] = 1
            x = x.to(device)
            y = nn.to_one_hot(x, 2)
            assert x.is_cuda
            assert y.is_cuda
            assert y.shape == (1, 128, 128, 2)

    def test_cpu(self):
        x = torch.zeros(1, 128, 128)
        x[:, :50, :50] = 1
        y = nn.to_one_hot(x, 2)
        assert not x.is_cuda
        assert not y.is_cuda
        assert y.shape == (1, 128, 128, 2)


def test_image_gradient():
    import tensorflow as tf

    tf.enable_eager_execution()
    x_tf = tf.random.normal((1, 128, 128, 10))
    x_np = np.array(x_tf)
    x_pt = torch.from_numpy(x_np)
    x_pt = x_pt.permute(0, 3, 1, 2)

    g_tf = tf.image.image_gradients(x_tf)
    g_pt = nn.image_gradient(x_pt)
    g_pt = [g.permute(0, 2, 3, 1) for g in g_pt]
    assert np.allclose(np.array(g_tf[0]), np.array(g_pt[0]))
    assert np.allclose(np.array(g_tf[1]), np.array(g_pt[1]))


def test_hloss():
    import tensorflow as tf

    tf.enable_eager_execution()

    logits = torch.randn(1, 2, 128, 128)
    probs = torch.nn.functional.softmax(logits, dim=1)

    l_tf = logits.permute(0, 2, 3, 1).numpy()
    p_tf = probs.permute(0, 2, 3, 1).numpy()

    h_pt = nn.HLoss()(logits)
    h_tf = tf.nn.softmax_cross_entropy_with_logits_v2(p_tf, l_tf)
    h_tf = tf.reduce_sum(h_tf, axis=(1, 2))
    assert np.allclose(h_pt, h_tf)


def test_probs_to_mu_sigma():
    from supermariopy.tfutils import nn as tfnn
    import tensorflow as tf

    tf.enable_eager_execution()

    _means = [30, 50, 70]

    means = tf.ones((3, 1, 2), dtype=tf.float32) * np.array(_means).reshape((3, 1, 1))
    stds = tf.concat(
        [
            tf.ones((1, 1, 1), dtype=tf.float32) * 5,
            tf.ones((1, 1, 1), dtype=tf.float32) * 10,
        ],
        axis=-1,
    )

    blob = tfnn.tf_hm(means, 100, 100, stds)

    mu, sigma = tfnn.probs_to_mu_sigma(blob)

    pt_blob = tf.transpose(blob, (0, 3, 1, 2))
    pt_blob = torch.from_numpy(np.array(pt_blob))

    mupt, sigmapt = nn.probs_to_mu_sigma(pt_blob)

    assert np.allclose(mupt, mu)
    assert np.allclose(sigmapt, sigma, rtol=1.0e-2)


def test_flip():
    c = torch.rand(1, 10)
    c_inv = nn.flip(c, 1)

    assert np.allclose(c.numpy()[:, ::-1], c_inv)


def test_init():
    assert False


def test_convbnrelu():
    N = 1
    H = 128
    W = 128
    C = 10

    x = torch.ones((N, C, H, W))
    c_bn_relu = nn.ConvBnRelu(C, 256)(x)
    assert list(c_bn_relu.shape) == [1, 256, H, W]
