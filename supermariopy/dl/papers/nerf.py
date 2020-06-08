import tensorflow as tf


class Embedder:
    def __init__(self, **kwargs):
        r"""
        num_freqs is :\math: L parameter in paper
        max_freq_log2 is :\math: L-1 parameter in paper

        kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": 9,
            "num_freqs": 10,
            "log_sampling": True,
            "periodic_fns": [tf.math.sin, tf.math.cos],
        }

        References
        ----------
        [1]: https://github.com/bmild/nerf/blob/8edde335d2b18188769850b03c45515352d66b31/run_nerf_helpers.py#L22
        """
        self.kwargs = kwargs
        self.create_embedding_fn()

    @staticmethod
    def get_default_kwargs():
        kwargs = {
            "include_input": True,
            "input_dims": 3,
            "max_freq_log2": 9,
            "num_freqs": 10,
            "log_sampling": True,
            "periodic_fns": [tf.math.sin, tf.math.cos],
        }
        return kwargs

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** tf.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.0 ** 0.0, 2.0 ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """
        Examples
        --------
            _input = tf.ones((100, 100, 2))
            embeder.embed(_input).shape >>> (100, 100, 2)
        """
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


import numpy as np


def get_positional_encodings(inputs, **kwargs):
    r"""
    Use this snippet for old non-eager tensorflow.
    Calculate positional encoding :math:`\gamma`

    .. math:: 

        \gamma(p) = (\sin(2^{0} \pi p), \cos(2^{0} \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))

    num_freqs is :\math: L parameter in paper
    max_freq_log2 is :\math: L-1 parameter in paper
    
    kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": 9,
        "num_freqs": 10,
        "log_sampling": True,
        "periodic_fns": [tf.math.sin, tf.math.cos],
    }
    Examples
    --------
        _input = tf.ones((100, 100, 2))
        kwargs = Embedder.get_default_kwargs()
        get_positional_encodings(_input, **kwargs).shape >>> (100, 100, 2)
    """
    embed_fns = []
    d = kwargs["input_dims"]
    out_dim = 0
    if kwargs["include_input"]:
        embed_fns.append(lambda x: x)
        out_dim += d

    max_freq = kwargs["max_freq_log2"]
    N_freqs = kwargs["num_freqs"]

    if kwargs["log_sampling"]:
        freq_bands = 2.0 ** np.linspace(0.0, max_freq, N_freqs, dtype=np.float32)
    else:
        freq_bands = np.linspace(2.0 ** 0.0, 2.0 ** max_freq, N_freqs, dtype=np.float32)

    for freq in freq_bands:
        for p_fn in kwargs["periodic_fns"]:
            embed_fns.append(
                lambda x, p_fn=p_fn, freq=freq: p_fn(x * tf.constant(freq))
            )
            out_dim += d

    embed_fns = embed_fns
    out_dim = out_dim
    return tf.concat([fn(inputs) for fn in embed_fns], -1)
