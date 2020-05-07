from supermariopy.ptutils import nn as smptnn
import tfpyth
import tensorflow as tf
import warnings
import torch

"""Functions to establish compatibility with other frameworks such as Tensorflow"""


def torch_tile_nd(a, n_tile_nd):
    for dim, nn_tile in enumerate(n_tile_nd):
        a = smptnn.tile(a, nn_tile, dim)
    return a


def torch_gather_nd(params, indices):
    """Gather values from params given a multidimensinonal list of indices

    Parameters
    ----------
    params : torch.Tensor
        d-dimensional tensor
    indices : torch.Tensor
        multidimensional list of d-dimensional indices

    Returns
    -------
    torch.Tensor
        Gathered Tensor

    Examples
    --------

        4D example
        params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
        indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
        returns: tensor shaped [m_1, m_2, m_3, m_4]

        ND_example
        params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
        indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
        returns: tensor shaped [m_1, ..., m_1]

    References
    ----------
    [1] : https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/26
    """
    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


def torch_gather(params, indices):
    """ assumes params = (N, d) and indices = (N, ) shaped arrays"""
    indices = torch_astype(indices, torch.int64)
    out = params[indices, :]
    return out


# def torch_slice(input_, begin, size):
#     """dirty wrapper for tf.slice to use with pytorch.
#       https://discuss.pytorch.org/t/tensor-slice-in-pytorch/1449
#     """

#     warnings.warn("Implemted using tfpyth, thus tensorflow is called in the back")

#     def func(input_):
#         return tf.slice(input_, begin, size)

#     out = tfpyth.wrap_torch_from_tensorflow(
#         func,
#         ["input_"],
#         # input_shapes=[params_shape, indices_shape],
#         # input_dtypes=[tf.float32, tf.int32],
#     )(input_)
#     out = out.to(input_.device)
#     return out


def torch_random_uniform(shape, lower, upper, dtype):
    t = torch.empty(list(shape), dtype=dtype).uniform_(lower, upper)
    return t


def torch_astype(x, dtype):
    return x.type(dtype)


def torch_stop_gradient(x):
    return x.detach()


def torch_reshape(x, shape):
    return x.view(shape)


def torch_image_random_contrast(image, lower, upper, seed=None):
    contrast_factor = torch.distributions.uniform.Uniform(lower, upper).sample()
    return torch_image_adjust_contrast(image, contrast_factor)


def torch_image_adjust_contrast(image, contrast_factor):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_contrast(img_pil, contrast_factor)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_image_random_brightness(image, max_delta, seed=None):
    delta = torch.distributions.uniform.Uniform(-max_delta, max_delta).sample()
    return torch_image_adjust_brightness(image, delta)


def torch_image_adjust_brightness(image, delta):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_brightness(img_pil, delta)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_image_random_saturation(image, lower, upper, seed=None):
    saturation_factor = torch.distributions.uniform.Uniform(lower, upper).sample()
    return torch_image_adjust_saturation(image, saturation_factor)


def torch_image_adjust_saturation(image, saturation_factor):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_saturation(img_pil, saturation_factor)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_image_random_hue(image, max_delta, seed=None):
    delta = torch.distributions.uniform.Uniform(-max_delta, max_delta).sample()
    return torch_image_adjust_saturation(image, delta)


def torch_image_adjust_hue(image, delta):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_hue(img_pil, delta)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_sigmoid_cross_entropy_with_logits(logits, labels):
    """always use this function with keyword arguments

    Parameters
    ----------
    logits : [type]
        [description]
    labels : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    References
    ----------

    [1]: https://discuss.pytorch.org/t/equivalent-of-tensorflows-sigmoid-cross-entropy-with-logits-in-pytorch/1985/13

    # TODO: check if reduction mode matches tf.nn.sigmoid_cross_entropy_with_logits
    """
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    return criterion(logits, labels)


def torch_unravel_index(index, shape):
    """
    x = torch.arange(30).view(10, 3)
    for i in range(x.numel()):
        assert i == x[unravel_index(i, x.shape)]

    https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/2

    Parameters
    ----------
    index : [type]
        [description]
    shape : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

