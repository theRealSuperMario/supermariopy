import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd


def shape_as_list(t):
    return list(t.shape)


def tf_meshgrid(h, w):
    #     xs = np.linspace(-1.0,1.0,w)
    xs = np.arange(0, w)
    ys = np.arange(0, h)
    #     ys = np.linspace(-1.0,1.0,h)
    xs, ys = np.meshgrid(xs, ys)
    meshgrid = np.stack([xs, ys], 2)
    meshgrid = meshgrid.astype(np.float32)
    return meshgrid

    # def tf_hm(h, w, mu, L):
    """
    Returns Gaussian densitiy function based on μ and L for each batch index and part
    L is the cholesky decomposition of the covariance matrix : Σ = L L^T

    Parameters
    ----------
    h : int
        heigh ot output map
    w : int
        width of output map
    mu : tensor
        mean of gaussian part and batch item. Shape [b, p, 2]. Mean in range [-1, 1] with respect to height and width
    L : tensor
        cholesky decomposition of covariance matrix for each batch item and part. Shape [b, p, 2, 2]
    order:

    Returns
    -------
    density : tensor
        gaussian blob for each part and batch idx. Shape [b, h, w, p]

    Example
    -------

    .. code-block:: python

        from matplotlib import pyplot as plt
        tf.enable_eager_execution()
        import numpy as np
        import tensorflow as tf
        import tensorflow.contrib.distributions as tfd

        # create Target Blobs
        _means = [-0.5, 0, 0.5]
        means = tf.ones((3, 1, 2), dtype=tf.float32) * np.array(_means).reshape((3, 1, 1))
        means = tf.concat([means, means, means[::-1, ...]], axis=1)
        means = tf.reshape(means, (-1, 2))

        var_ = 0.1
        rho = 0.5
        cov = [[var_, rho * var_],
               [rho * var_, var_]]
        scale = tf.cholesky(cov)
        scale = tf.stack([scale] * 3, axis=0)
        scale = tf.stack([scale] * 3, axis=0)
        scale = tf.reshape(scale, (-1, 2, 2))

        mvn = tfd.MultivariateNormalTriL(
            loc=means,
            scale_tril=scale)

        h = 100
        w = 100
        y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
        x_t = tf.tile(tf.reshape(tf.linspace(-1., 1., w), [1, w]), [h, 1])
        y_t = tf.expand_dims(y_t, axis=-1)
        x_t = tf.expand_dims(x_t, axis=-1)
        meshgrid = tf.concat([y_t, x_t], axis=-1)
        meshgrid = tf.expand_dims(meshgrid, 0)
        meshgrid = tf.expand_dims(meshgrid, 3)  # 1, h, w, 1, 2

        blob = mvn.prob(meshgrid)
        blob = tf.reshape(blob, (100, 100, 3, 3))
        blob = tf.transpose(blob, perm=[2, 0, 1, 3])

        # Estimate mean and L
        norm_const = np.sum(blob, axis=(1, 2), keepdims=True)
        mu, L = nn.probs_to_mu_L(blob / norm_const, 1, inv=False)

        bn, h, w, nk = blob.get_shape().as_list()

        # Estimate blob based on mu and L
        estimated_blob = nn.tf_hm(h, w, mu, L)

        # plot
        fig, ax = plt.subplots(2, 3, figsize=(9, 6))
        for b in range(len(_means)):
            ax[0, b].imshow(np.squeeze(blob[b, ...]))
            ax[0, b].set_title("target_blobs")
            ax[0, b].set_axis_off()

        for b in range(len(_means)):
            ax[1, b].imshow(np.squeeze(estimated_blob[b, ...]))
            ax[1, b].set_title("estimated_blobs")
            ax[1, b].set_axis_off()

    """
    assert len(mu.get_shape().as_list()) == 3
    assert len(L.get_shape().as_list()) == 4
    assert mu.get_shape().as_list()[-1] == 2
    assert L.get_shape().as_list()[-1] == 2
    assert L.get_shape().as_list()[-2] == 2

    b, p, _ = mu.get_shape().as_list()
    mu = tf.reshape(mu, (b * p, 2))
    L = tf.reshape(L, (b * p, 2, 2))

    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)
    meshgrid = tf.expand_dims(meshgrid, 0)
    meshgrid = tf.expand_dims(meshgrid, 3)  # 1, h, w, 1, 2

    probs = mvn.prob(meshgrid)
    probs = tf.reshape(probs, (h, w, b, p))
    probs = tf.transpose(probs, perm=[2, 0, 1, 3])  # move part axis to the back
    return probs


def tf_hm(P, h, w, stddev, exp=True):
    """Coordinates to Heatmap Layer

    P   : float Tensor
          xy coordinates of points in coordinate system [-1, 1]
          shape [num_batch, n_points, 2]
    h   : int
          Output height
    w   : int
          Output width
    stddev: float
            Standard deviation of heatmap, i.e. width with respect to [-1, 1]
            [num_batch, n_points, 2]

    Returns
        : float Tensor
          Heatmap with values in [0, 1]
          shape [num_batch, h, w, n_points]

    Examples
        B = 2
        H = 20
        W = 20
        parts = 2
        means = np.array([[10, 10], [10, 15]], dtype=np.float32) # means in [0, H] space
        means = ( means - np.array([H, W]) / 2 ) / np.array([H, W]) # means in [-1, 1] space
        variances = np.array([[3, 1], [1, 3]], dtype=np.float32) / ((np.array([H, W]) / 2)) # variance in [-1, 1] space
        variances = tf.reshape(variances, (1, 2, 2))
        points = tf.reshape(means, (1, 2, 2))
        points = tf.concat([points] * B, 0)
        variances = tf.concat([variances] * B, 0)
        heatmap = tf_hm(points, H, W, variances)
        for i in range(2):
            fig, ax = plt.subplots()
            ax.imshow(np.squeeze(heatmap[0, :, :, i]))

    """
    meshgrid = tf_meshgrid(h, w)
    assert meshgrid.shape == (h, w, 2)
    meshgrid = np.reshape(meshgrid, [1, h, w, 1, 2])  # b,h,w,p,2
    P = tf.expand_dims(P, 1)
    P = tf.expand_dims(P, 2)  # b,h,w,p,2
    d = tf.square(meshgrid - P)
    stddev = tf.expand_dims(stddev, 1)
    stddev = tf.expand_dims(stddev, 2)  # b,h,w,p,2
    d = -d / (2 * stddev ** 2)
    logits = tf.reduce_sum(d, 4)  # b,h,w,p
    if exp:
        heat = tf.exp(logits)
    #     heat /= 2.0 * math.pi * stddev[:,:, 0] * stddev[:, : 1]
    else:
        heat = 1 / (1 - logits)
    return heat


class FullLatentDistribution(object):
    # TODO: write some comment on where this comes from
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = self.parameters.shape.as_list()
        if len(ps) != 2:
            assert len(ps) == 4
            assert ps[1] == ps[2] == 1
            self.expand_dims = True
            self.parameters = tf.squeeze(self.parameters, axis=[1, 2])
            ps = self.parameters.shape.as_list()
        else:
            self.expand_dims = False

        assert len(ps) == 2
        self.batch_size = ps[0]

        event_dim = self.dim
        n_L_parameters = (event_dim * (event_dim + 1)) // 2

        size_splits = [event_dim, n_L_parameters]

        self.mean, self.L = tf.split(self.parameters, size_splits, axis=1)
        # L is Cholesky parameterization
        self.L = tf.contrib.distributions.fill_triangular(self.L)
        # make sure diagonal entries are positive by parameterizing them
        # logarithmically
        diag_L = tf.linalg.diag_part(self.L)
        self.log_diag_L = diag_L  # keep for later computation of logdet
        diag_L = tf.exp(diag_L)
        # scale down then set diags
        row_weights = np.array([np.sqrt(i + 1) for i in range(event_dim)])
        row_weights = np.reshape(row_weights, [1, event_dim, 1])
        self.L = self.L / row_weights
        self.L = tf.linalg.set_diag(self.L, diag_L)
        self.Sigma = tf.matmul(self.L, self.L, transpose_b=True)  # L times L^t

        ms = self.mean.shape.as_list()
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
            eps = noise_level * tf.random_normal([self.batch_size, self.dim, 1])
            eps = tf.matmul(self.L, eps)
            eps = tf.squeeze(eps, axis=-1)
            out = self.mean + eps
        if self.expand_dims:
            out = tf.expand_dims(out, axis=1)
            out = tf.expand_dims(out, axis=1)
        return out

    def kl(self, other=None):
        if other is not None:
            raise NotImplemented("Only KL to standard normal is implemented.")

        delta = tf.square(self.mean)
        diag_covar = tf.reduce_sum(tf.square(self.L), axis=2)
        logdet = 2.0 * self.log_diag_L

        kl = 0.5 * tf.reduce_sum(
            diag_covar - 1.0 + delta - logdet, axis=self.event_axes
        )
        kl = tf.reduce_mean(kl)
        return kl


class MeanFieldDistribution(object):
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = self.parameters.shape.as_list()

        assert len(ps) == 4
        self.batch_size = ps[0]
        self.event_axes = [1, 2, 3]

        event_dim = self.dim
        self.mean = self.parameters
        self.shape = tf.shape(self.mean)

    @staticmethod
    def n_parameters(dim):
        return dim

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * tf.random_normal(self.shape)
            out = self.mean + eps
        return out

    def kl_improper_gmrf(self):
        # TODO use symmetric stencil
        dy, dx = tf.image.image_gradients(self.mean)
        grad_squared = tf.square(dy) + tf.square(dx)
        kl = 0.5 * grad_squared
        kl = tf.reduce_sum(kl, axis=self.event_axes)
        kl = tf.reduce_mean(kl)
        return kl


def probs_to_mu_sigma(probs):
    """Calculate mean and covariance matrix for each channel of probs
    tensor of keypoint probabilites [bn, h, w, n_kp]
    mean calculated on a grid of scale [-1, 1]

    Parameters
    ----------
    probs: tensor
        tensor of shape [b, h, w, k] where each channel along axis 3 is interpreted as an unnormalized probability density.
    scaling_factor : tensor
        tensor of shape [b, 1, 1, k] representing normalizing the normalizing constant of the density

    Returns
    -------
    mu : tensor
        tensor of shape [b, k, 2] representing partwise mean coordinates of x and y for each item in the batch
    sigma : tensor
        tensor of shape [b, k, 2, 2] representing covariance matrix for each item in the batch

    Example
    -------
        norm_const = np.sum(blob, axis=(1, 2), keepdims=True)
        mu, sigma = nn.probs_to_mu_sigma(blob / norm_const, tf.ones_like(norm_const))
    """
    (
        bn,
        h,
        w,
        nk,
    ) = (
        probs.get_shape().as_list()
    )  # todo instead of calulating sequrity measure from amplitude one could alternativly calculate it by letting the network predict a extra paremeter also one could do
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)

    mu = tf.einsum("ijl,aijk->akl", meshgrid, probs)
    mu_out_prod = tf.einsum(
        "akm,akn->akmn", mu, mu
    )  # todo incosisntent ordereing of mu! compare with cross_V2

    mesh_out_prod = tf.einsum(
        "ijm,ijn->ijmn", meshgrid, meshgrid
    )  # todo efficient (expand_dims)
    sigma = tf.einsum("ijmn,aijk->akmn", mesh_out_prod, probs) - mu_out_prod
    return mu, sigma


