import tensorflow as tf


def resize_bilinear(x, shape):
    """
    Raises a warning if tensorflow version is too in order to buggy behavior

    References
    ----------
    [1]: https://github.com/tensorflow/tensorflow/issues/6720
    [2]: https://github.com/tensorflow/tensorflow/issues/33691
    """
    tf_version = tf.__version__
    major_version, minor_version, _ = tf_version.split(".")
    version = int(major_version) * 100 + int(minor_version)
    if version < 114:  # 1.14
        raise NotImplementedError(
            "Resize bilinear is buggy for tensorflow version below 1.14"
        )
    elif version >= 114 and version < 115:  # 114
        return tf.image.resize_bilinear(x, shape, align_corners=True)
    elif version >= 115:
        return tf.image.resize_bilinear(x, shape, align_corners=True)
