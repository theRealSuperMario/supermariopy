import torch
from supermariopy.ptutils import nn as ptnn
from supermariopy.ptutils import utils as ptu
from supermariopy.ptutils import compat as ptcompat

"""
Note:
1. there are other tps implementations out there, such as 
    * https://github.com/cheind/py-thin-plate-spline
    * https://github.com/WarBean/tps_stn_pytorch/blob/master/tps_grid_gen.py
2.  I decided not to use them, but rather reimplement the one provided by https://github.com/CompVis/unsupervised-disentangling 
    in pytorch. This way, I have a TPS implementation with consistent behvior in pytorch and tensorflow.
"""


def pt_rotation_matrix(rotation):
    a = torch.unsqueeze(torch.cos(rotation), dim=0)
    b = torch.unsqueeze(torch.sin(rotation), dim=0)
    row_1 = torch.cat([a, -b], dim=1)
    row_2 = torch.cat([b, a], dim=1)
    mat = torch.cat([row_1, row_2], dim=0)
    return mat


def tps_parameters(
    batch_size, scal, tps_scal, rot_scal, off_scal, scal_var, rescal=1, augm_scal=1.0
):
    coord = torch.tensor(
        [
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [-0.5, 0.5],
                [0.5, 0.5],
                [0.2, -0.2],
                [-0.2, 0.2],
                [0.2, 0.2],
                [-0.2, -0.2],
            ]
        ],
        dtype=torch.float32,
    )
    coord = ptcompat.torch_tile_nd(coord, [batch_size, 1, 1])
    shape = ptnn.shape_as_list(coord)
    coord = coord + ptcompat.torch_random_uniform(shape, -0.2, 0.2, dtype=torch.float32)
    vector = ptcompat.torch_random_uniform(
        shape, -tps_scal, tps_scal, dtype=torch.float32
    )

    offset = ptcompat.torch_random_uniform(
        [batch_size, 1, 2], -off_scal, off_scal, dtype=torch.float32
    )
    offset_2 = ptcompat.torch_random_uniform(
        [batch_size, 1, 2], -off_scal, off_scal, dtype=torch.float32
    )
    t_scal = ptcompat.torch_random_uniform(
        [batch_size, 2],
        scal * (1.0 - scal_var),
        scal * (1.0 + scal_var),
        dtype=torch.float32,
    )
    t_scal = t_scal * rescal

    rot_param = ptcompat.torch_random_uniform(
        [batch_size, 1], -rot_scal, rot_scal, dtype=torch.float32
    )
    rot_mat = torch.stack([pt_rotation_matrix(r) for r in rot_param], dim=0)
    parameter_dict = {
        "coord": coord,
        "vector": vector,
        "offset": offset,
        "offset_2": offset_2,
        "t_scal": t_scal,
        "rot_mat": rot_mat,
        "augm_scal": augm_scal,
    }
    return parameter_dict


def static_param_2d(param):
    bn, d_1 = ptnn.shape_as_list(param)
    param = param[::2]
    param = ptcompat.torch_tile_nd(param, [1, 2])
    param = ptcompat.torch_reshape(param, [bn, d_1])

    return param


def static_param_3d(param):
    bn, d_1, d_2 = ptnn.shape_as_list(param)
    param = param[::2]
    param = ptcompat.torch_tile_nd(param, [1, 2, 1])
    param = ptcompat.torch_reshape(param, [bn, d_1, d_2])
    return param


def make_input_tps_param(tps_parameter_dict, move_point=None, scal_point=None):
    coord = tps_parameter_dict["coord"]
    vector = tps_parameter_dict["vector"]
    offset = tps_parameter_dict["offset"]
    offset_2 = tps_parameter_dict["offset_2"]
    rot_mat = tps_parameter_dict["rot_mat"]
    t_scal = tps_parameter_dict["t_scal"]

    scaled_coord = torch.einsum("bk,bck->bck", t_scal, coord + vector - offset) + offset
    t_vector = (
        torch.einsum("blk,bck->bcl", rot_mat, scaled_coord - offset_2)
        + offset_2
        - coord
    )
    if move_point is not None and scal_point is not None:
        coord = torch.einsum("bk,bck->bck", scal_point, coord + move_point)
        t_vector = torch.einsum("bk,bck->bck", scal_point, t_vector)

    else:
        assert move_point is None and scal_point is None
    return coord, t_vector


def no_transformation_parameters(batch_size):
    """create TPS transformation parameters that produce the identity (roughly).
    Should be used with @make_input_tps_param

    Parameters
    ----------
    batch_size: int

    Returns
    -------
    tps_transform_args: dict
        dict with the following values:
            tps_transform_args["scal"] = 1.0
            tps_transform_args["tps_scal"] = 0.0
            tps_transform_args["rot_scal"] = 0.0
            tps_transform_args["off_scal"] = 0.0
            tps_transform_args["scal_var"] = 0.0
            tps_transform_args["augm_scal"] = 0.0
            tps_transform_args["batch_size"] = batch_size
    """
    tps_transform_args = {}

    tps_transform_args["scal"] = 1.0
    tps_transform_args["tps_scal"] = 0.0
    tps_transform_args["rot_scal"] = 0.0
    tps_transform_args["off_scal"] = 0.0
    tps_transform_args["scal_var"] = 0.0
    tps_transform_args["augm_scal"] = 0.0
    tps_transform_args["batch_size"] = batch_size

    return tps_transform_args


def adapt_tps_for_crop(tps_param, move_point, scal_point):
    """
    # TODO: what does this?
    :param center_point: b, 1, 2
    :param tps_param:
    :return:
    """
    move_point = -move_point
    scal_point = 1.0 / scal_point
    crop_coord, t_vector_coord = make_input_tps_param(tps_param, move_point, scal_point)
    return crop_coord, t_vector_coord


def ThinPlateSpline(U, coord, vector, out_size, n_c, move=None, scal=None):
    U = U.permute((0, 2, 3, 1))  # NCHW -> NHWC
    coord = ptnn.flip(coord, -1)
    vector = ptnn.flip(vector, -1)
    num_batch, height, width, _ = ptnn.shape_as_list(U)
    channels = n_c
    out_height = out_size
    out_width = out_size
    height_f = float(height)
    width_f = float(width)
    num_point = ptnn.shape_as_list(coord)[1]

    def _repeat(x, n_repeats):
        rep = torch.unsqueeze(
            torch.ones(torch.stack([torch.tensor([n_repeats])])), dim=1
        )
        rep = rep.permute([1, 0])
        rep = ptcompat.torch_astype(rep, torch.int32)
        x = torch.matmul(ptcompat.torch_reshape(x, (-1, 1)), rep)
        return ptcompat.torch_reshape(x, [-1])

    def _interpolate(im, y, x):
        # constants
        y = ptcompat.torch_astype(y, torch.float32)
        x = ptcompat.torch_astype(x, torch.float32)

        zero = torch.zeros([], dtype=torch.int32)
        max_y = int(height - 1)
        max_x = int(width - 1)

        # scale indices from aprox [-1, 1] to [0, width/height]

        y = (y + 1) * height_f / 2.0
        x = (x + 1) * width_f / 2.0

        y = ptcompat.torch_reshape(y, [-1])
        x = ptcompat.torch_reshape(x, [-1])

        # do sampling
        y0 = ptcompat.torch_astype(torch.floor(y), torch.int32)
        y1 = y0 + 1
        x0 = ptcompat.torch_astype(torch.floor(x), torch.int32)
        x1 = x0 + 1

        y0 = y0.clamp(zero, max_y)
        y1 = y1.clamp(zero, max_y)
        x0 = x0.clamp(zero, max_x)
        x1 = x1.clamp(zero, max_x)

        base = _repeat(
            torch.range(start=0, end=num_batch - 1, dtype=torch.int32) * width * height,
            out_height * out_width,
        )
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = ptcompat.torch_reshape(im, [-1, channels])
        # im_flat = tf.reshape(im, [-1, channels])
        im_flat = ptcompat.torch_astype(im_flat, torch.float32)
        Ia = ptcompat.torch_gather(im_flat, idx_a, im_flat.shape, idx_a.shape)
        Ib = ptcompat.torch_gather(im_flat, idx_b, im_flat.shape, idx_b.shape)
        Ic = ptcompat.torch_gather(im_flat, idx_c, im_flat.shape, idx_c.shape)
        Id = ptcompat.torch_gather(im_flat, idx_d, im_flat.shape, idx_d.shape)

        # and finally calculate interpolated values
        x0_f = ptcompat.torch_astype(x0, torch.float32)
        x1_f = ptcompat.torch_astype(x1, torch.float32)
        y0_f = ptcompat.torch_astype(y0, torch.float32)
        y1_f = ptcompat.torch_astype(y1, torch.float32)

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width, coord):
        x_t = ptcompat.torch_tile_nd(
            ptcompat.torch_reshape(torch.linspace(-1.0, 1.0, width), [1, width]),
            [height, 1],
        )
        y_t = ptcompat.torch_tile_nd(
            ptcompat.torch_reshape(torch.linspace(-1.0, 1.0, height), [height, 1]),
            [1, width],
        )
        x_t_flat = ptcompat.torch_reshape(x_t, (1, 1, -1))
        y_t_flat = ptcompat.torch_reshape(y_t, (1, 1, -1))

        px = torch.unsqueeze(coord[:, :, 0], 2)  # [bn, pn, 1]
        py = torch.unsqueeze(coord[:, :, 1], 2)  # [bn, pn, 1]

        d2 = (x_t_flat - px) ** 2 + (y_t_flat - py) ** 2
        r = d2 * torch.log(d2 + 1.0e-6)  # [bn, pn, h*w]

        x_t_flat_g = ptcompat.torch_tile_nd(x_t_flat, [num_batch, 1, 1])  # [bn, 1, h*w]
        y_t_flat_g = ptcompat.torch_tile_nd(y_t_flat, [num_batch, 1, 1])  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        grid = torch.cat([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [bn, 3+pn, h*w]
        return grid

    def _transform(T, coord, move, scal):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        grid = _meshgrid(out_height, out_width, coord)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        x_s = ptcompat.torch_slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = ptcompat.torch_slice(T_g, [0, 1, 0], [-1, 1, -1])

        if move is not None and scal is not None:
            off_y = torch.unsqueeze(move[:, :, 0], dim=-1)
            off_x = torch.unsqueeze(move[:, :, 1], dims=-1)
            scal_y = torch.unsqueeze(torch.unsqueeze(scal[:, 0], dim=-1), dim=-1)
            scal_x = torch.unsqueeze(torch.unsqueeze(scal[:, 1], dim=-1), dim=-1)
            y = y_s * scal_y + off_y
            x = x_s * scal_x + off_x

        else:
            assert move is None and scal is None
            y = y_s
            x = x_s

        return y, x

    def _solve_system(coord, vector):
        ones = torch.ones([num_batch, num_point, 1], dtype=torch.float32)
        p = torch.cat([ones, coord], 2)  # [bn, pn, 3]

        p_1 = ptcompat.torch_reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = ptcompat.torch_reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum((p_1 - p_2) ** 2, 3)  # [bn, pn, pn]
        r = d2 * torch.log(d2 + 1.0e-6)  # Kernel [bn, pn, pn]

        zeros = torch.zeros([num_batch, 3, 3], dtype=torch.float32)
        W_0 = torch.cat([p, r], 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat([zeros, p.permute((0, 2, 1))], 2)  # [bn, 3, pn+3]
        W = torch.cat([W_0, W_1], 1)  # [bn, pn+3, pn+3]
        W_inv = torch.inverse(W)

        tp = torch.nn.functional.pad(
            coord + vector, (0, 0, 0, 3, 0, 0), mode="constant"
        )  # [bn, pn+3, 2]
        T = torch.matmul(W_inv, tp)
        T = T.permute([0, 2, 1])
        return T

    T = _solve_system(coord, vector)
    y, x = _transform(T, coord, move, scal)
    input_transformed = _interpolate(U, y, x)
    output = ptcompat.torch_reshape(
        input_transformed, [num_batch, out_height, out_width, channels]
    )
    y = ptcompat.torch_reshape(y, [num_batch, out_height, out_width, 1])
    x = ptcompat.torch_reshape(x, [num_batch, out_height, out_width, 1])
    t_arr = torch.cat([y, x], dim=-1)
    output = output.permute((0, 3, 1, 2))  # NHWC --> NCHW
    return output, t_arr
