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


def torch_gather_nd(params, indices, params_shape, indices_shape):
    """dirty wrapper for tf.gather_nd to use with pytorch"""
    warnings.warn("Implemted using tfpyth, thus tensorflow is called in the back")

    if any([params.is_cuda, indices.is_cuda, params_shape.is_cuda, indices_shape.is_cuda]):
        params = params.cuda()
        incices = indices.cuda()
        params_shape = params_shape.cuda()
        indices_shape = indices_shape.cuda()

    def func(params, indices):
        return tf.gather_nd(params, indices)

    out = tfpyth.wrap_torch_from_tensorflow(
        func,
        ["params", "indices"],
        input_shapes=[params_shape, indices_shape],
        input_dtypes=[tf.float32, tf.int32],
    )(params, indices)
    return out


def torch_gather(params, indices, params_shape, indices_shape):
    """dirty wrapper for tf.gather_nd to use with pytorch"""
    warnings.warn("Implemted using tfpyth, thus tensorflow is called in the back")

    def func(params, indices):
        return tf.gather(params, indices)

    out = tfpyth.wrap_torch_from_tensorflow(
        func,
        ["params", "indices"],
        input_shapes=[params_shape, indices_shape],
        input_dtypes=[tf.float32, tf.int32],
    )(params, indices)
    out = out.to(params.device)
    return out


def torch_slice(input_, begin, size):
    """dirty wrapper for tf.slice to use with pytorch"""
    warnings.warn("Implemted using tfpyth, thus tensorflow is called in the back")

    def func(input_):
        return tf.slice(input_, begin, size)

    out = tfpyth.wrap_torch_from_tensorflow(
        func,
        ["input_"],
        # input_shapes=[params_shape, indices_shape],
        # input_dtypes=[tf.float32, tf.int32],
    )(input_)
    out = out.to(input_.device)
    return out


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
