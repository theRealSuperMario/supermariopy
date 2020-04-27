def test_torch_image_random_contrast():
    f_tf = tf.image.random_contrast
    img = astronaut()[np.newaxis, ...] / 255.0
    img = img.astype(np.float32)
    args_tf = (img.copy(), 0.3, 0.7)
    f_pt = ops.torch_image_random_contrast
    args_pt = (tfpyth.th_NHWC_to_NCHW(torch.from_numpy(img.copy())), 0.3, 0.7)
    torch.backends.cudnn.deterministic = True
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.set_random_seed(seed)
    tf.enable_eager_execution()
    out_tf = f_tf(*args_tf)
    tf.disable_eager_execution()
    out_pt = f_pt(*args_pt)
    assert np.allclose(out_tf.shape, tfpyth.th_NCHW_to_NHWC(out_pt).numpy().shape)


def test_tile_nd():
    a = torch.zeros((10, 1, 1, 1))
    b = ops_pt.tile_nd(a, [1, 10, 20, 30])
    assert b.shape == (10, 10, 20, 30)
