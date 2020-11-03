import numpy as np
import tensorflow as tf
import torch
from supermariopy.ptutils import tps as pt_tps
from supermariopy.tfutils import tps as tf_tps

tf.enable_eager_execution()


def test_tps_parameter_compatiblity():
    bs = 1
    scal = 1.0
    tps_scal = 0.05
    rot_scal = 0.1
    off_scal = 0.15
    scal_var = 0.05
    augm_scal = 1.0

    tps_param_dic_tf = tf_tps.tps_parameters(
        bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
    )

    tps_param_dic_pt = pt_tps.tps_parameters(
        bs, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal
    )

    keys = ["coord", "vector"]
    for k in keys:
        tps_param_dic_pt[k].numpy().shape == tps_param_dic_tf[k].shape


def test_input_tps_param_compatibility():
    batch_size = 1
    tf.set_random_seed(42)
    no_transform_params_tf = tf_tps.no_transformation_parameters(batch_size)
    tps_params = tf_tps.tps_parameters(**no_transform_params_tf)
    coords_tf, t_vector_tf = tf_tps.make_input_tps_param(tps_params)

    torch.manual_seed(42)
    no_transform_params_pt = pt_tps.no_transformation_parameters(batch_size)
    tps_params = pt_tps.tps_parameters(**no_transform_params_pt)
    coords_pt, t_vector_pt = pt_tps.make_input_tps_param(tps_params)

    assert np.allclose(coords_tf, coords_pt.numpy())
    assert np.allclose(t_vector_tf, t_vector_pt.numpy())


def test_tf_rotation_matrix():
    rotation_angle = np.array([[45.0]])

    r_matrix_tf = tf_tps.tf_rotation_matrix(tf.convert_to_tensor(rotation_angle))
    r_matrix_pt = pt_tps.pt_rotation_matrix(torch.from_numpy(rotation_angle))

    assert np.allclose(r_matrix_pt.numpy(), r_matrix_tf)
