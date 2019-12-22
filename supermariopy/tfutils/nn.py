import numpy as np
import tensorflow as tf


def tf_meshgrid(h, w):
    #     xs = np.linspace(-1.0,1.0,w)
    xs = np.arange(0, w)
    ys = np.arange(0, h)
    #     ys = np.linspace(-1.0,1.0,h)
    xs, ys = np.meshgrid(xs, ys)
    meshgrid = np.stack([xs, ys], 2)
    meshgrid = meshgrid.astype(np.float32)
    return meshgrid


def tf_hm(h, w, mu, L):
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
