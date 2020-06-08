import pytest
import tensorflow as tf

tf.enable_eager_execution()

from supermariopy.tfutils import layers as smlayers
from supermariopy.tfutils import nn as smnn


class Test_SPADE:
    def test_simple(self):
        x = tf.random_normal((2, 128, 128, 3))
        m = tf.random.uniform((2, 128, 128), minval=0, maxval=9, dtype=tf.int32)
        m = tf.one_hot(m, 10)
        spaded = smlayers.SPADE(n_channels_x=x.shape[-1])(x, m)
        assert spaded.shape == x.shape


class Test_SpadeResBlock:
    def test_no_shortcut(self):
        x = tf.random_normal((2, 128, 128, 3))
        m = tf.random.uniform((2, 128, 128), minval=0, maxval=9, dtype=tf.int32)
        m = tf.one_hot(m, 10)
        block = smlayers.SPADEResnetBlock(n_channels_x_in=3, n_channels_x_out=3)
        spaded = block(x, m)
        assert spaded.shape == x.shape

    def test_with_shortcut(self):
        x = tf.random_normal((2, 128, 128, 3))
        m = tf.random.uniform((2, 128, 128), minval=0, maxval=9, dtype=tf.int32)
        m = tf.one_hot(m, 10)
        block = smlayers.SPADEResnetBlock(n_channels_x_in=3, n_channels_x_out=64)
        spaded = block(x, m)
        assert smnn.shape_as_list(spaded) == [2, 128, 128, 64]


class Test_ResidualBlock:
    def test(self):
        x = tf.random_normal((2, 128, 128, 3))
        res_block = smlayers.ResidualBlock(filters=[64, 64, 3])
        y = res_block(x)
        assert smnn.shape_as_list(y) == [2, 128, 128, 3]

        res_block = smlayers.ResidualBlock(filters=[3], kernel_sizes=[(3, 3)])
        y = res_block(x)
        assert smnn.shape_as_list(y) == [2, 128, 128, 3]

        res_block = smlayers.ResidualBlock(
            filters=[3], kernel_sizes=[(3, 3)], use_batch_norm=False
        )
        y = res_block(x)
        assert smnn.shape_as_list(y) == [2, 128, 128, 3]
