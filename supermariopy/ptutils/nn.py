import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional
from torch.nn import functional as F
import numpy as np
import torchvision

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
    r = torch.min(alpha * g, torch.ones_like(g) * lambda_)
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


def straight_through_estimator(y_hard, y):
    # from https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
    return (y_hard.float() - y).detach() + y


class FullLatentDistribution(torch.nn.Module):
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = shape_as_list(self.parameters)
        if len(ps) != 2:
            assert len(ps) == 4
            self.expand_dims = True
            self.parameters = self.parameters.view(ps[0], ps[1])
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
        row_weights = row_weights.to(self.L.device)
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
            eps = eps.to(self.mean.device)
            eps = torch.matmul(self.L, eps)
            eps = torch.squeeze(eps, dim=-1)
            out = self.mean + eps
        if self.expand_dims:
            out = torch.unsqueeze(out, dim=-1)
            out = torch.unsqueeze(out, dim=-1)
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
        # kl = torch.mean(kl)
        return kl


class LatentDistribution(torch.nn.Module):
    def __init__(self, parameters, dim, stochastic=True):
        """Implements normal Gaussian distribution with diagonal covariance matrix.
        
        Parameters
        ----------
        nn : [type]
            [description]
        parameters : [type]
            [description]
        dim : [type]
            [description]
        stochastic : bool, optional
            [description], by default True
        """
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = shape_as_list(self.parameters)
        if len(ps) != 2:
            assert len(ps) == 4
            self.expand_dims = True
            self.parameters = self.parameters.view(ps[0], ps[1])
            ps = shape_as_list(self.parameters)
        else:
            self.expand_dims = False

        assert len(ps) == 2
        self.batch_size = ps[0]

        self.mean, self.log_var = torch.split(self.parameters, 2, dim=1)

        ms = shape_as_list(self.mean)
        self.event_axes = list(range(1, len(ms)))
        self.event_shape = ms[1:]
        assert len(self.event_shape) == 1, self.event_shape

    @staticmethod
    def n_parameters(dim):
        return 2 * dim

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * torch.randn([self.batch_size, self.dim, 1])
            eps = eps * torch.exp(0.5 * self.logvar)
            eps = eps.to(self.mean.device)
            out = self.mean + eps
        if self.expand_dims:
            out = torch.unsqueeze(out, dim=-1)
            out = torch.unsqueeze(out, dim=-1)
        return out

    def kl(self, other=None):
        if other is not None:
            raise NotImplementedError("Only KL to standard normal is implemented")
        # TODO: add mathy docstring
        delta = self.mean ** 2

        kl = -0.5 * torch.sum(
            1 + self.log_var - delta - torch.exp(self.log_var), dim=self.event_axes
        )

        # average across batches
        # kl = torch.mean(kl)
        return kl


class MeanFieldDistribution(torch.nn.Module):
    def __init__(self, parameters, stochastic=True):
        super(MeanFieldDistribution, self).__init__()
        self.parameters = parameters
        self.stochastic = stochastic

        ps = shape_as_list(self.parameters)

        assert len(ps) == 4
        self.batch_size = ps[0]
        self.event_axes = [1, 2, 3]

        self.mean = self.parameters
        self.shape = self.mean.shape

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * torch.randn(self.shape, device=self.mean.device)
            out = self.mean + eps
        return out

    def forward(self):
        return self.sample()

    def kl_improper_gmrf(self):
        # TODO use symmetric stencil
        dx, dy = image_gradient(self.mean)
        grad_squared = dy ** 2 + dy ** 2
        kl = 0.5 * grad_squared
        kl = torch.sum(kl, dim=self.event_axes)
        return kl


def image_gradient(x):
    # TODO: test against tensorflow implementation
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dy, dx = right - left, bottom - top
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dy[:, :, :, -1] = 0
    dx[:, :, -1, :] = 0

    return dx, dy


def to_one_hot(y, n_dims):
    """Creates one_hot representation of x with `n_dims` = `n_classes`.
    
    Parameters
    ----------
    y : torch.Tensor
        tensor with class indices
    n_dims : int
        number of classes or dimensions
    Returns
    -------
    torch.Tensor
        one_hot tensor. Note that new dimension is added at the end.

    References
    ----------
    [1] : https://gist.github.com/NegatioN/acbd8bb6be866ce1831b2d073fd7c450
    """

    if y.is_cuda:
        dtype = torch.cuda.FloatTensor
        long_dtype = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        long_dtype = torch.LongTensor
    scatter_dim = len(y.size())
    y_tensor = y.type(long_dtype).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)

    y_one_hot = zeros.scatter(scatter_dim, y_tensor, 1)
    return y_one_hot


def flip(x, dim):
    """Reverse tensor order aling given dimension.
    Drop-in replacement for tf.reverse(x, axis)
    
    Parameters
    ----------
    x : torch.Tensor
        tensor to flip
    dim : int
        dimension along which to flip
    
    Returns
    -------
    torch.Tensor
        flipped tensor

    References
    ----------

    [1] : https://github.com/pytorch/pytorch/issues/229
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


class HLoss(torch.nn.Module):
    def __init__(self):
        """
        Entropy loss function.
        
        References
        ----------
        [1] https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2
        """
        super(HLoss, self).__init__()

    def forward(self, x):
        """return entropy of x
        
        Parameters
        ----------
        x : torch.Tensor
            unnormalized log-probabilities, shaped [N, C, H, W]
        
        Returns
        -------
        torch.Tensor
            entropy for each item in the batch, shaped [N, C, H, W]
        """
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=(1, 2, 3))
        return b


class EMA(torch.nn.Module):
    def __init__(self, mu):
        r"""Implements exponential weighted moving average smoothing.
        
        Parameters
        ----------
        torch : [type]
            [description]
        mu : float
            ratio to sample previous value from

        # TODO: add math
            x_{avg} = x_{avg} * \mu + (1 - \mu) * x

        Examples
        --------
             ema = EMA(0.999)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.register(name, param.data)

            # in batch training loop
            # for batch in batches:
                optimizer.step()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data = ema(name, param.data)

        References
        ----------
        [1] : https://gist.github.com/jojonki/d78034ebb0bc798774d660458b3846e6
        """
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.shadow[name] * self.mu + (1.0 - self.mu) * x
        self.shadow[name] = new_average.clone()
        return new_average


def probs_to_mu_sigma(probs):
    """Calculate mean and covariance matrix for each channel of probs
    tensor of keypoint probabilites [N, C, H, W]
    mean calculated on a grid of scale [-1, 1]
    
    Parameters
    ----------
    probs : torch.Tensor
        tensor of shape [N, C, H, W] where each channel along axis 1 is interpreted as a probability density.
    
    Returns
    -------
    mu : torch.Tensor
        tensor of shape [N, C, 2] representing partwise mean coordinates of x and y for each item in the batch
    sigma : torch.Tensor
        tensor of shape [N, C, 2, 2] representing covariance matrix for each item in the batch
    """
    bn, nk, h, w = shape_as_list(probs)
    y_t = tile(torch.linspace(-1, 1, h).view(h, 1), w, 1)
    x_t = tile(torch.linspace(-1, 1, w).view(1, w), h, 0)
    y_t = torch.unsqueeze(y_t, dim=-1)
    x_t = torch.unsqueeze(x_t, dim=-1)

    meshgrid = torch.cat([y_t, x_t], dim=-1)
    if probs.is_cuda:
        meshgrid = meshgrid.to(probs.device)
    mu = torch.einsum("ijl,akij->akl", meshgrid, probs)
    mu_out_prod = torch.einsum("akm,akn->akmn", mu, mu)
    mesh_out_prod = torch.einsum("ijm,ijn->ijmn", meshgrid, meshgrid)
    sigma = torch.einsum("ijmn,akij->akmn", mesh_out_prod, probs) - mu_out_prod
    return mu, sigma


def tile(a, n_tile, dim):
    """Equivalent of numpy or tensorflow tile.
    
    References
    ----------
    [1] : https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    order_index = order_index.to(a.device)
    return torch.index_select(a, dim, order_index)


def crop_bounding_boxes(image_t, boundingboxes_t):
    """A differentiable version of bounding box cropping.

    Note that if the number of bounding boxes per image is different, the output tensors have different sizes
    
    Parameters
    ----------
    image_t : torch.tensor
        Tensor with a batch of images, shaped [N, C, H, W]
    boundingboxes_t : torch.tensor
        Tensor with a batch of bounding box coordinates, shaped [N, N_boxes, 4]. 
        First 2 indicate top left corner, last 2 indicate bottom right corner (x_top, y_top, x_bottom, y_bottom)
    """

    image_stack = []
    for image, box_coords in zip(image_t, boundingboxes_t):
        crops = []
        for coords in box_coords:
            x_min, y_min, x_max, ymax = coords
            crops.append(image[:, x_min:x_max, y_min:y_max])
        image_stack.append(torch.stack(crops))
    return image_stack


def overlay_boxes_without_labels(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (torch.tensor): image tensor shaped [1, 3, H, W]
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    image = image[0, ...]
    image = image.permute(1, 2, 0)
    image = image.numpy()
    boxes = predictions.bbox

    colors = [[255, 0, 0]] * len(boxes)

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image.get()


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
            From https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
            Implements the following mapping

            Inputs: x

            y := x
            --> y:= relu(bn(conv(y)))
            --> y:= bn(conv(y))
            --> y:= y + x
            --> y:= relu(y)
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlock_NoBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Adapted from 
        https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
        Implements the following mapping

        Inputs: x

        y := x
        --> y:= relu(conv(y)))
        --> y:= conv(y)
        --> y:= y + x
        --> y:= relu(y)
        """
        super(ResidualBlock_NoBN, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1, pad=0):
        """conv without bias
        
        Parameters
        ----------
        torch : [type]
            [description]
        in_channels : [type]
            [description]
        out_channels : [type]
            [description]
        kernel_size : int, optional
            [description], by default 1
        strides : int, optional
            [description], by default 1
        pad : int, optional
            [description], by default 0
        """
        super(ConvBnRelu, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=pad, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.nn.ReLU()(self.bn(self.conv(x)))


def part_map_to_mu_L_inv(part_maps, scal):
    """
    Calculate mean for each channel of part_maps
    :param part_maps: tensor of part map activations [bn, h, w, n_part]
    :return: mean calculated on a grid of scale [-1, 1]
    """
    bn, nk, h, w = list(part_maps.shape)
    y_t = tile(torch.linspace(-1.0, 1.0, h).view([h, 1]), w, dim=1)
    x_t = tile(torch.linspace(-1.0, 1.0, w).view([1, w]), h, dim=0)
    y_t = torch.unsqueeze(y_t, axis=-1)
    x_t = torch.unsqueeze(x_t, axis=-1)
    meshgrid = torch.cat([y_t, x_t], axis=-1)

    mu = torch.einsum("ijl,akij->akl", meshgrid, part_maps)
    mu_out_prod = torch.einsum("akm,akn->akmn", mu, mu)

    mesh_out_prod = torch.einsum("ijm,ijn->ijmn", meshgrid, meshgrid)
    stddev = torch.einsum("ijmn,akij->akmn", mesh_out_prod, part_maps) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12

    a = torch.sqrt(
        a_sq + eps
    )  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = torch.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = torch.zeros_like(a)

    det = torch.unsqueeze(torch.unsqueeze(a * c, dim=-1), dim=-1)
    row_1 = torch.unsqueeze(
        torch.cat([torch.unsqueeze(c, dim=-1), torch.unsqueeze(z, dim=-1)], dim=-1),
        dim=-2,
    )
    row_2 = torch.unsqueeze(
        torch.cat([torch.unsqueeze(-b, dim=-1), torch.unsqueeze(a, dim=-1)], dim=-1),
        dim=-2,
    )

    L_inv = (
        scal / (det + eps) * torch.cat([row_1, row_2], dim=-2)
    )  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    return mu, L_inv


# TODO: small funciton to calculate padding sizes
# https://github.com/pytorch/pytorch/issues/3867
def _get_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding



def meshgrid(image_height, image_width):
    y_coords = 2.0 * torch.arange(image_height).unsqueeze(
        1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
    x_coords = 2.0 * torch.arange(image_width).unsqueeze(
        0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

    coords = torch.stack((y_coords, x_coords), dim=0)