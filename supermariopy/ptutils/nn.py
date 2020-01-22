import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional
import numpy as np

## TORCHVISION
import torchvision.models as models
from torchvision import transforms, utils, datasets

# TODO: add flake8 check
CHANNELS_FIRST = True


def shape_as_list(t):
    """Return shape of tensor as list."""
    return list(t.shape)


def spatial_softmax(features):
    """Apply softmax on flattened spatial dimensions and then unflatten again.

    Parameters
    ----------
    input : input tensor of shape :math: `(\text{minibatch}, H, W, \text{in\_channels})`
        [description]
    """
    if not CHANNELS_FIRST:
        raise NotImplementedError
    N, C, H, W = shape_as_list(features)
    probs = functional.softmax(features.view(N * C, H * W), dim=-1)
    probs = probs.view(N, C, H, W)
    return probs


def softmax(x, spatial=False):
    if spatial:
        return spatial_softmax(x)
    else:
        return functional.softmax(x, dim=-1)


def reshape_4D(x):
    """Reshapes 5D tensor [N, C, H, W, P] to 4D tensor [N, C * P, H, W]."""
    N, C, H, W, P = shape_as_list(x)
    out = torch.permute(x, [0, 1, 4, 2, 3])
    out = out.view(N, C * P, H, W)
    return out


difference1d = np.float32([0.0, 0.5, -0.5])


def fd_kernel(n):
    ffd = np.zeros([3, 3, n, n * 2])
    for i in range(n):
        ffd[1, :, i, 2 * i + 0] = difference1d
        ffd[:, 1, i, 2 * i + 1] = difference1d
    ffd = np.transpose(ffd, axes=(3, 2, 0, 1))
    return 0.5 * ffd


def grad(x):
    """Channelwise FD gradient for cell size of one."""
    n = shape_as_list(x)[1]
    kernel = fd_kernel(n)
    g = torch.nn.functional.conv2d(
        input=x, weight=torch.Tensor(kernel), stride=2 * [1,], padding=1
    )
    return g


def squared_grad(x):
    """Pointwise squared L2 norm of gradient assuming cell size of one."""
    s = shape_as_list(x)
    gx = grad(x)
    gx = gx.view(s[0], s[1], 2, s[2], s[3])
    return torch.sum(gx, axis=2)


def mumford_shah(x, alpha, lambda_):
    g = squared_grad(x)
    r, _ = torch.min(alpha * g, lambda_)
    smoothness_cost = torch.where(
        (alpha * g) < lambda_, r, torch.zeros_like(g, dtype=g.dtype)
    )
    contour_cost = torch.where(
        (alpha * g) >= lambda_, r, torch.zeros_like(g, dtype=g.dtype)
    )
    return r, smoothness_cost, contour_cost


def fill_triangular(x, dim):
    """equivalent of tfd.fill_triangular"""
    N = shape_as_list(x)[0]
    xc = torch.cat([x[:, dim:], x.flip(dims=[1])], dim=1)
    y = xc.view(N, dim, dim)
    return torch.tril(y)


def set_diag(L, diag_L):
    """fill diagonal of L with elements from diag_L.
    equivalent to tf.linalg.set_diag(L, diag_L).

    L: shape [N, d, d]
    diag_L: shape [N, d]
    """
    d = shape_as_list(L)[2]
    M = L
    M[:, torch.arange(0, d), torch.arange(0, d)] = diag_L
    return M


def diag_part(x):
    """equivalent of tf.linalg.diag_part"""
    diag_L = torch.diagonal(x, dim1=-2, dim2=-1)
    return diag_L


class FullLatentDistribution(object):
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = shape_as_list(self.parameters)
        if len(ps) != 2:
            assert len(ps) == 4
            assert ps[1] == 4
            self.expand_dims = True
            self.parameters = torch.squeeze(self.parameters, axis=[1, 2])
            ps = shape_as_list(self.parameters)
        else:
            self.expand_dims = False

        assert len(ps) == 2
        self.batch_size = ps[0]

        event_dim = self.dim
        n_L_parameters = (event_dim * (event_dim + 1)) // 2

        size_splits = [event_dim, n_L_parameters]

        self.mean, self.L = torch.split(self.parameters, size_splits, dim=1)
        self.L = fill_triangular(self.L, self.dim)
        diag_L = diag_part(self.L)
        self.log_diag_L = diag_L
        diag_L = torch.exp(diag_L)
        row_weights = np.array([np.sqrt(i + 1) for i in range(event_dim)])
        row_weights = np.reshape(row_weights, [1, event_dim, 1])
        row_weights = torch.Tensor(row_weights)
        self.L = self.L / row_weights
        self.L = set_diag(self.L, diag_L)

        self.Sigma = torch.matmul(self.L, self.L.transpose(1, 2))  # L x L^T

        ms = shape_as_list(self.mean)
        self.event_axes = list(range(1, len(ms)))
        self.event_shape = ms[1:]
        assert len(self.event_shape) == 1, self.event_shape

    @staticmethod
    def n_parameters(dim):
        return dim + (dim * (dim + 1)) // 2

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * torch.randn([self.batch_size, self.dim, 1])
            eps = torch.matmul(self.L, eps)
            eps = torch.squeeze(eps, dim=-1)
            out = self.mean + eps
        if self.expand_dims:
            out = torch.unsqueeze(out, dim=1)
            out = torch.unsqueeze(out, dim=1)
        return out

    def kl(self, other=None):
        if other is not None:
            raise NotImplementedError("Only KL to standard normal is implemented")
        # TODO: add mathy docstring
        delta = self.mean ** 2
        diag_covar = torch.sum(self.L ** 2, dim=2)
        logdet = 2.0 * self.log_diag_L

        kl = 0.5 * torch.sum(diag_covar - 1.0 + delta - logdet, dim=self.event_axes)

        # average across batches
        kl = torch.mean(kl)
        return kl
