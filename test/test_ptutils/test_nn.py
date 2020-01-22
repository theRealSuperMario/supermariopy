import pytest
from supermariopy.ptutils import nn
import torch
import numpy as np


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


def test_full_latent_distribution():
    dim = 10
    n_parameters = nn.FullLatentDistribution.n_parameters(dim)
    parameters = torch.rand(10, n_parameters)
    latent = nn.FullLatentDistribution(parameters, dim, False)


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
        assert nn.FullLatentDistribution.n_parameters(dim) == 55

    def test_sample(self):
        dim = 10
        n_parameters = nn.FullLatentDistribution.n_parameters(dim)
        parameters = torch.rand((1, n_parameters), dtype=torch.float32)
        distr = nn.FullLatentDistribution(parameters, dim)
        assert distr.sample().shape == (1, dim)

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

