import tensorflow as tf
from supermariopy.tfutils import nn as smnn
from tensorflow.keras.layers import Layer

from enum import Enum


class SPADEParamFreeNormType(Enum):
    INSTANCE_NORM = 1
    SYNC_BATCH_NORM = 2
    BATCH_NORM = 3


class SPADEResnetBlock(Layer):
    def __init__(
        self,
        n_channels_x_in,
        n_channels_x_out,
        use_spectral_norm=False,
        spade_norm=SPADEParamFreeNormType.BATCH_NORM,
    ):
        r"""Implementation of SPADE Residual Block. 
        Applies SPADE on x and adds a skip. Does this two times, thus it has two SPADE blocks.

        x-->SPADE1-->SPADE2--> + --> y
        |                      |
        ---------skip-----------

        Each SPADE block implements a functional like f: x, segmap --> y
        `x` is assumed to be a learned activation, which is then mapped to `y` conditioned on a spatial semantic map `segmap`.

        If `n_channels_x_in` and `n_channels_x_out` is not the same, the skip connection is replaced by a third SPADE block.

        Parameters
        ----------
        n_channels_x_in : int
            number of feature channels of input.
        n_channels_x_out : int
            number of final output feature channels.
        use_spectral_norm : bool, optional
            not implemented yet, by default False
        spade_norm : SPADEParamFreeNormType, optional
            which norm each SPADE block uses, by default SPADEParamFreeNormType.BATCH_NORM

        Returns
        -------
        tf.Tensor
            y

        Raises
        ------
        NotImplementedError


        References
        ----------
        .. [1] https://github.com/NVlabs/SPADE/blob/master/models/networks/architecture.py
        .. [2] https://arxiv.org/pdf/1903.07291.pdf
        """
        super().__init__()
        # Attributes
        self.learned_shortcut = n_channels_x_in != n_channels_x_out
        n_channels_middle = min(n_channels_x_in, n_channels_x_out)

        # create conv layers
        self.conv_0 = tf.layers.Conv2D(
            n_channels_middle, kernel_size=[3, 3], padding="SAME"
        )
        self.conv_1 = tf.layers.Conv2D(
            n_channels_x_out, kernel_size=[3, 3], padding="SAME"
        )
        if self.learned_shortcut:
            self.conv_s = tf.layers.Conv2D(
                n_channels_x_out, kernel_size=[1, 1], use_bias=False
            )

        # apply spectral norm if specified
        if use_spectral_norm:
            raise NotImplementedError
            # self.conv_0 = spectral_norm(self.conv_0)
            # self.conv_1 = spectral_norm(self.conv_1)
            # if self.learned_shortcut:
            #     self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(n_channels_x=n_channels_x_in, norm_type=spade_norm)
        self.norm_1 = SPADE(n_channels_x=n_channels_middle, norm_type=spade_norm)
        if self.learned_shortcut:
            self.norm_s = SPADE(n_channels_x=n_channels_x_in, norm_type=spade_norm)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def call(self, x, segmap):
        x_s = self.shortcut(x, segmap)
        dx = self.conv_0(self.actvn(self.norm_0(x, segmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, segmap)))
        out = x_s + dx
        return out

    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, segmap))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return tf.nn.leaky_relu(x, 2e-1)


class SPADE(Layer):
    def __init__(
        self,
        n_channels_x=64,
        n_channels_hidden=128,
        norm_type=SPADEParamFreeNormType.BATCH_NORM,
        kernel_size=[3, 3],
    ):
        """SPADE operation.
        Each SPADE block implements a functional like f: x, segmap --> y
        `x` is assumed to be a learned activation, which is then mapped to `y` conditioned on a spatial semantic map `segmap`.

        Parameters
        ----------
        n_channels_x : int, optional
            number of feature channels of input x, by default 64
        n_channels_hidden : int, optional
            number of hidden feature channels, by default 128
        norm_type : SPADEParamFreeNormType, optional
            internal normalization for x, by default SPADEParamFreeNormType.BATCH_NORM
        kernel_size : list, optional
            by default [3, 3]

        Raises
        ------
        NotImplementedError
            [description]
        ValueError
            [description]
        """
        super().__init__()
        if norm_type == SPADEParamFreeNormType.INSTANCE_NORM:
            # TODO: implement instance norm
            param_free_norm = tf.contrib.layers.instance_norm
        elif norm_type == SPADEParamFreeNormType.SYNC_BATCH_NORM:
            raise NotImplementedError
            # param_free_norm = SynchronizedBatchNorm2d: only means that statistics are synchronized across multiple GPUs
        elif norm_type == SPADEParamFreeNormType.BATCH_NORM:
            param_free_norm = tf.layers.batch_normalization
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE" % norm_type
            )
        self.normalize = param_free_norm

        self.shared_mlp = tf.layers.Conv2D(
            n_channels_hidden, kernel_size, padding="SAME"
        )
        self.gamma_conv = tf.layers.Conv2D(n_channels_x, kernel_size, padding="SAME")
        self.beta_conv = tf.layers.Conv2D(n_channels_x, kernel_size, padding="SAME")

    def call(self, x, segmap):
        normalized = self.normalize(x)
        segmap = tf.image.resize_nearest_neighbor(segmap, size=x.shape[1:3])
        actv = tf.nn.relu(self.shared_mlp(segmap))
        gamma = self.gamma_conv(actv)
        beta = self.beta_conv(actv)

        out = normalized * (1 + gamma) + beta
        return out

